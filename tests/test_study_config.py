import unittest
from collections import Counter

import numpy as np

from src.environment import Environment, EnvironmentConfig
from src.experiment import (
    FIXED_MODEL_CLASS_COUNT,
    ExperimentRunner,
    build_study_config,
)


class TestStudyConfiguration(unittest.TestCase):
    def test_m_is_actual_class_count_and_model_always_has_five_classes(self):
        base = EnvironmentConfig(num_objects=6, num_classes_in_model=5)

        for scene_class_count in range(1, 6):
            with self.subTest(M=scene_class_count):
                config = build_study_config(base, scene_class_count, trials_per_env=3)
                environment = Environment(config, seed=100)
                counts = Counter(obj.gt_class for obj in environment.objects)

                self.assertEqual(config.num_classes_in_scene, scene_class_count)
                self.assertEqual(
                    config.num_classes_in_model, FIXED_MODEL_CLASS_COUNT
                )
                self.assertEqual(set(counts), set(range(scene_class_count)))
                self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_scene_cannot_exceed_fixed_model_vocabulary(self):
        base = EnvironmentConfig(num_objects=6, num_classes_in_model=5)
        with self.assertRaises(ValueError):
            build_study_config(base, 6, trials_per_env=1)

    def test_geometric_baseline_is_invariant_to_scene_class_count(self):
        base = EnvironmentConfig(
            num_objects=6,
            num_classes_in_model=5,
            num_trials=1,
            num_steps=2,
            num_samples=20,
            max_hypotheses=20,
        )
        results = []
        for scene_class_count in (1, 5):
            config = build_study_config(base, scene_class_count, trials_per_env=1)
            environment = Environment(config, seed=101)
            runner = ExperimentRunner(config, seed=101, env_instance=environment)
            results.append(
                runner.run_comparative_experiment(
                    modes=["geometric_only"], workers=1
                )["geometric_only"]
            )

        for metric in results[0]:
            if metric in {"step_time", "trial_time"}:
                continue
            np.testing.assert_array_equal(results[0][metric], results[1][metric])


if __name__ == "__main__":
    unittest.main()
