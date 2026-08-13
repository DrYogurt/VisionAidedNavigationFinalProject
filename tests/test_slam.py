import unittest
import numpy as np

from src.environment import EnvironmentConfig, Environment
from src.models import (
    SE2Utils, MotionModel, GeometricObservationModel,
    ViewpointDependentSemanticModel, PassiveSemanticModel
)
from src.slam_engine import SemanticSLAMEngine, HypothesisComponent
from src.metrics import MetricsCalculator

class TestSemanticSLAM(unittest.TestCase):

    def test_se2_kinematics(self):
        # Test identity composition
        x1 = np.array([1.0, 2.0, np.pi/4])
        x_id = np.array([0.0, 0.0, 0.0])
        self.assertTrue(np.allclose(SE2Utils.compose(x1, x_id), x1))

        # Test relative pose and inverse composition
        x2 = np.array([4.0, 6.0, np.pi/2])
        rel = SE2Utils.relative_pose(x2, x1)
        recovered_x2 = SE2Utils.compose(x1, rel)
        self.assertTrue(np.allclose(x2[:2], recovered_x2[:2], atol=1e-5))

        # Test angle wrapping
        angle = 3 * np.pi
        self.assertAlmostEqual(SE2Utils.wrap_angle(angle), -np.pi)

    def test_geometric_observation_model(self):
        geo_model = GeometricObservationModel(np.diag([0.1, 0.05]))
        robot = np.array([0.0, 0.0, 0.0])
        obj = np.array([3.0, 4.0, 0.0])

        obs = geo_model.observe(robot, obj)
        self.assertAlmostEqual(obs[0], 5.0)  # sqrt(3^2 + 4^2) = 5
        self.assertAlmostEqual(obs[1], np.atan2(4.0, 3.0))

        # Test likelihood is positive
        lh = geo_model.likelihood(obs, robot, obj)
        self.assertGreater(lh, 0.0)

    def test_viewpoint_dependent_semantic_model(self):
        sem_model = ViewpointDependentSemanticModel(num_classes=2, alpha=0.25, precision_k=15.0)
        obj = np.array([0.0, 0.0, 0.0])

        # Eq. (18): theta=0 and theta=pi for class 1 (zero-indexed here).
        np.testing.assert_allclose(
            sem_model.expected_semantic_vector(np.array([1.0, 0.0, 0.0]), obj, class_m=0),
            np.array([0.75, 0.25]),
        )
        np.testing.assert_allclose(
            sem_model.expected_semantic_vector(np.array([-1.0, 0.0, 0.0]), obj, class_m=0),
            np.array([1.0, 0.0]),
            atol=1e-12,
        )

        R = 15.0 * np.array([[1.0, -0.5], [0.0, 1.0]])
        np.testing.assert_allclose(
            sem_model.semantic_covariance(np.zeros(3), np.zeros(3)),
            np.linalg.inv(R.T @ R),
        )

    def test_semantic_jacobians_match_finite_difference(self):
        model = ViewpointDependentSemanticModel(num_classes=3)
        robot = np.array([1.0, 2.0, 0.5])
        obj = np.array([4.0, 5.0, -0.5])
        exact_x, exact_o = model.jacobians(robot, obj, class_m=1)
        epsilon = 1e-6
        finite_x = np.zeros_like(exact_x)
        finite_o = np.zeros_like(exact_o)
        for index in range(3):
            delta = np.zeros(3)
            delta[index] = epsilon
            finite_x[:, index] = (
                model.expected_semantic_vector(robot + delta, obj, 1)
                - model.expected_semantic_vector(robot - delta, obj, 1)
            ) / (2.0 * epsilon)
            finite_o[:, index] = (
                model.expected_semantic_vector(robot, obj + delta, 1)
                - model.expected_semantic_vector(robot, obj - delta, 1)
            ) / (2.0 * epsilon)
        np.testing.assert_allclose(exact_x, finite_x, atol=1e-8)
        np.testing.assert_allclose(exact_o, finite_o, atol=1e-8)

    def test_metrics_calculator(self):
        # Create dummy hypotheses
        h1 = HypothesisComponent((0,1,2,3,4,0), ((0,),), 0.8, np.zeros(21), np.eye(21)*0.01)
        h2 = HypothesisComponent((0,1,2,3,4,1), ((1,),), 0.2, np.ones(21)*0.5, np.eye(21)*0.04)

        entropy = MetricsCalculator.da_weight_entropy([h1, h2])
        self.assertGreater(entropy, 0.0)

        cov_det = MetricsCalculator.position_cov_determinant([h1, h2])
        self.assertAlmostEqual(cov_det, 0.0001, places=5)  # 0.01 * 0.01 = 0.0001

        max_err = MetricsCalculator.max_weight_error([h1, h2], np.zeros(3))
        self.assertAlmostEqual(max_err, 0.0)

    def test_slam_engine_step(self):
        config = EnvironmentConfig(
            num_objects=2,
            num_classes_in_scene=1,
            num_classes_in_model=2,
            num_samples=200,
            sensor_range=100.0,
            sigma_p=np.diag([0.01, 0.01, 0.01]),
        )
        env = Environment(config, seed=42)
        engine = SemanticSLAMEngine(config, env.objects, mode="viewpoint_dependent")

        self.assertEqual(len(engine.hypotheses), 1)
        self.assertEqual(engine.hypotheses[0].class_assignments, (-1, -1))

        rng = np.random.default_rng(42)
        action = np.array([1.0, 0.0, 0.1])
        z_geo = np.array([4.8, 0.1])
        z_sem = np.array([0.8, 0.2])
        observations = [(0, z_geo, z_sem)]

        engine.step(action, observations, rng)
        self.assertGreater(len(engine.hypotheses), 0)
        self.assertTrue(all(len(h.da_history) == 1 for h in engine.hypotheses))
        self.assertTrue(all(len(h.da_history[0]) == 1 for h in engine.hypotheses))

    def test_geometric_baseline_keeps_classes_marginalized(self):
        config = EnvironmentConfig(
            num_objects=1,
            num_classes_in_model=3,
            num_samples=100,
            sensor_range=100.0,
            sigma_p=np.diag([0.01, 0.01, 0.01]),
        )
        env = Environment(config, seed=7)
        engine = SemanticSLAMEngine(config, env.objects, mode="geometric_only")
        obj = env.objects[0]
        z_geo = engine.geo_model.observe(config.initial_robot_pose, obj.gt_pose)
        engine.step(
            np.zeros(3),
            [(obj.id, z_geo, np.zeros(3))],
            np.random.default_rng(7),
        )
        self.assertEqual(engine.hypotheses[0].class_assignments, (-1,))

    def test_visibility_factor(self):
        config = EnvironmentConfig(num_objects=1, sensor_range=5.0, sensor_fov=np.pi / 2.0)
        env = Environment(config, seed=1)
        engine = SemanticSLAMEngine(config, env.objects, mode="geometric_only")
        robots = np.array([[0.0, 0.0, 0.0]] * 3)
        objects = np.array([[4.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
        np.testing.assert_array_equal(engine._batch_visibility(robots, objects), [1.0, 0.0, 0.0])

    def test_data_association_is_injective_within_time_step(self):
        config = EnvironmentConfig(
            num_objects=2,
            num_classes_in_model=1,
            num_samples=50,
            sensor_range=100.0,
            max_hypotheses=100,
            sigma_p=np.diag([0.01, 0.01, 0.01]),
        )
        env = Environment(config, seed=13)
        engine = SemanticSLAMEngine(config, env.objects, mode="geometric_only")
        observations = [
            (
                obj.id,
                engine.geo_model.observe(config.initial_robot_pose, obj.gt_pose),
                np.ones(1),
            )
            for obj in env.objects
        ]
        engine.step(np.zeros(3), observations, np.random.default_rng(13))
        self.assertTrue(engine.hypotheses)
        for hypothesis in engine.hypotheses:
            beta_vector = hypothesis.da_history[-1]
            self.assertEqual(len(beta_vector), len(set(beta_vector)))

    def test_visibility_sample_collapse_uses_measurement_recovery(self):
        config = EnvironmentConfig(
            num_objects=1,
            num_classes_in_model=1,
            num_samples=20,
            sensor_range=100.0,
            sigma_p=np.diag([0.01, 0.01, 0.01]),
        )
        env = Environment(config, seed=23)
        engine = SemanticSLAMEngine(config, env.objects, mode="geometric_only")
        engine._batch_visibility = lambda robot_poses, obj_poses: np.zeros(len(robot_poses))
        obj = env.objects[0]
        z_geo = engine.geo_model.observe(config.initial_robot_pose, obj.gt_pose)

        engine.step(
            np.zeros(3),
            [(obj.id, z_geo, np.ones(1))],
            np.random.default_rng(23),
        )

        self.assertTrue(engine.hypotheses)
        self.assertEqual(engine.visibility_recoveries, 1)

if __name__ == "__main__":
    unittest.main()
