import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from typing import Callable, Dict, List, Tuple, Optional
from tqdm import tqdm

from src.environment import EnvironmentConfig, Environment
from src.models import (
    SE2Utils, GeometricObservationModel, ViewpointDependentSemanticModel
)
from src.slam_engine import SemanticSLAMEngine
from src.metrics import MetricsCalculator
from src.data_store import VersionedDataStore


def _generate_observations(
    config: EnvironmentConfig,
    objects,
    robot_gt_pose: np.ndarray,
    trial_rng: np.random.Generator,
    geo_model: Optional[GeometricObservationModel] = None,
    semantic_model: Optional[ViewpointDependentSemanticModel] = None,
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """Generate one time step of observations without shared mutable state."""
    if geo_model is None:
        geo_model = GeometricObservationModel(config.sigma_v_geo)
    if semantic_model is None:
        semantic_model = ViewpointDependentSemanticModel(
            config.num_classes_in_model,
            config.semantic_alpha,
            config.semantic_k,
        )
    observations = []
    for obj in objects:
        dx = obj.gt_pose[0] - robot_gt_pose[0]
        dy = obj.gt_pose[1] - robot_gt_pose[1]
        distance = np.hypot(dx, dy)
        relative_bearing = SE2Utils.wrap_angle(
            np.arctan2(dy, dx) - robot_gt_pose[2]
        )
        in_fov = (
            config.sensor_fov >= 2.0 * np.pi
            or abs(relative_bearing) <= config.sensor_fov / 2.0
        )
        if distance > config.sensor_range or not in_fov:
            continue

        z_geo = geo_model.observe(robot_gt_pose, obj.gt_pose)
        z_geo += trial_rng.multivariate_normal(np.zeros(2), config.sigma_v_geo)
        z_geo[1] = SE2Utils.wrap_angle(z_geo[1])

        semantic_mean = semantic_model.expected_semantic_vector(
            robot_gt_pose, obj.gt_pose, obj.gt_class
        )
        semantic_covariance = semantic_model.semantic_covariance(
            robot_gt_pose, obj.gt_pose
        )
        z_sem = semantic_mean + trial_rng.multivariate_normal(
            np.zeros(config.num_classes_in_model), semantic_covariance
        )
        observations.append((obj.id, z_geo, z_sem))
    return observations


def _run_trial(
    config: EnvironmentConfig,
    objects,
    gt_track: Tuple[np.ndarray, np.ndarray],
    mode: str,
    trial_idx: int,
):
    """Run one independent trial; module scope keeps it process-pickleable."""
    trial_start_time = time.perf_counter()
    obs_rng = np.random.default_rng(2000 + trial_idx)
    engine_rng = np.random.default_rng(3000 + trial_idx)
    gt_poses, controls = gt_track
    engine = SemanticSLAMEngine(config, objects, mode=mode)
    geo_model = GeometricObservationModel(config.sigma_v_geo)
    semantic_model = ViewpointDependentSemanticModel(
        config.num_classes_in_model,
        config.semantic_alpha,
        config.semantic_k,
    )

    result = {
        "da_entropy": np.zeros(config.num_steps + 1),
        "cov_det": np.zeros(config.num_steps + 1),
        "max_weight_err": np.zeros(config.num_steps + 1),
        "avg_weight_err": np.zeros(config.num_steps + 1),
        "num_hypotheses": np.zeros(config.num_steps + 1),
        "step_time": np.zeros(config.num_steps),
        "trial_time": np.zeros(1),
        "est_trajectories": np.zeros((config.num_steps + 1, 3)),
    }

    top_hypothesis = max(engine.hypotheses, key=lambda h: h.weight)
    result["est_trajectories"][0] = top_hypothesis.get_robot_pose()
    result["da_entropy"][0] = MetricsCalculator.da_weight_entropy(engine.hypotheses)
    result["cov_det"][0] = MetricsCalculator.position_cov_determinant(engine.hypotheses)
    result["max_weight_err"][0] = MetricsCalculator.max_weight_error(
        engine.hypotheses, gt_poses[0]
    )
    result["avg_weight_err"][0] = MetricsCalculator.average_estimation_error(
        engine.hypotheses, gt_poses[0]
    )
    result["num_hypotheses"][0] = MetricsCalculator.num_active_hypotheses(
        engine.hypotheses
    )

    for step_idx in range(config.num_steps):
        robot_gt_pose = gt_poses[step_idx + 1]
        observations = _generate_observations(
            config,
            objects,
            robot_gt_pose,
            obs_rng,
            geo_model,
            semantic_model,
        )
        start_time = time.perf_counter()
        engine.step(controls[step_idx], observations, engine_rng)
        result["step_time"][step_idx] = time.perf_counter() - start_time

        top_hypothesis = max(engine.hypotheses, key=lambda h: h.weight)
        result["est_trajectories"][step_idx + 1] = top_hypothesis.get_robot_pose()
        result["da_entropy"][step_idx + 1] = MetricsCalculator.da_weight_entropy(
            engine.hypotheses
        )
        result["cov_det"][step_idx + 1] = MetricsCalculator.position_cov_determinant(
            engine.hypotheses
        )
        result["max_weight_err"][step_idx + 1] = MetricsCalculator.max_weight_error(
            engine.hypotheses, robot_gt_pose
        )
        result["avg_weight_err"][step_idx + 1] = MetricsCalculator.average_estimation_error(
            engine.hypotheses, robot_gt_pose
        )
        result["num_hypotheses"][step_idx + 1] = MetricsCalculator.num_active_hypotheses(
            engine.hypotheses
        )

    result["trial_time"][0] = time.perf_counter() - trial_start_time
    return mode, trial_idx, result

class ExperimentRunner:
    """Orchestrates multi-trial semantic SLAM benchmark experiments."""

    def __init__(self, config: EnvironmentConfig, seed: int = 42, env_instance: Environment = None):
        self.config = config
        if env_instance is not None:
            self.env = env_instance
        else:
            self.env = Environment(config, seed=seed)
        self.gt_tracks = self.env.generate_tracks()

        self.geo_obs_model = GeometricObservationModel(config.sigma_v_geo)
        self.sem_classifier = ViewpointDependentSemanticModel(
            config.num_classes_in_model,
            config.semantic_alpha,
            config.semantic_k,
        )

    def generate_observations(self, robot_gt_pose: np.ndarray, trial_rng: np.random.Generator) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        """Generates geometric and M-dimensional semantic observations for objects visible from robot_gt_pose."""
        return _generate_observations(
            self.config,
            self.env.objects,
            robot_gt_pose,
            trial_rng,
            self.geo_obs_model,
            self.sem_classifier,
        )

    def run_comparative_experiment(
        self,
        modes: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        workers: int = 1,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Run independent mode/trial jobs, optionally in worker processes."""
        if modes is None:
            modes = ["viewpoint_dependent", "geometric_only"]
        if workers < 1:
            raise ValueError("workers must be at least one")

        results = {}
        for mode in modes:
            results[mode] = {
                "da_entropy": np.zeros((self.config.num_trials, self.config.num_steps + 1)),
                "cov_det": np.zeros((self.config.num_trials, self.config.num_steps + 1)),
                "max_weight_err": np.zeros((self.config.num_trials, self.config.num_steps + 1)),
                "avg_weight_err": np.zeros((self.config.num_trials, self.config.num_steps + 1)),
                "num_hypotheses": np.zeros((self.config.num_trials, self.config.num_steps + 1)),
                "step_time": np.zeros((self.config.num_trials, self.config.num_steps)),
                "trial_time": np.zeros((self.config.num_trials, 1)),
                "est_trajectories": np.zeros((self.config.num_trials, self.config.num_steps + 1, 3)),
            }

        def store_trial(mode: str, trial_idx: int, trial_result: dict):
            for metric, values in trial_result.items():
                results[mode][metric][trial_idx] = values

        jobs = [
            (mode, trial_idx)
            for mode in modes
            for trial_idx in range(self.config.num_trials)
        ]
        completed_by_mode = {mode: 0 for mode in modes}

        if workers == 1:
            for mode, trial_idx in jobs:
                completed_mode, completed_trial, trial_result = _run_trial(
                    self.config,
                    self.env.objects,
                    self.gt_tracks[trial_idx],
                    mode,
                    trial_idx,
                )
                store_trial(completed_mode, completed_trial, trial_result)
                completed_by_mode[completed_mode] += 1
                if progress_callback is not None:
                    progress_callback(
                        completed_mode,
                        completed_by_mode[completed_mode],
                        self.config.num_trials,
                    )
        else:
            effective_workers = min(workers, len(jobs))
            with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                futures = [
                    executor.submit(
                        _run_trial,
                        self.config,
                        self.env.objects,
                        self.gt_tracks[trial_idx],
                        mode,
                        trial_idx,
                    )
                    for mode, trial_idx in jobs
                ]
                for future in as_completed(futures):
                    completed_mode, completed_trial, trial_result = future.result()
                    store_trial(completed_mode, completed_trial, trial_result)
                    completed_by_mode[completed_mode] += 1
                    if progress_callback is not None:
                        progress_callback(
                            completed_mode,
                            completed_by_mode[completed_mode],
                            self.config.num_trials,
                        )

        return results

    @staticmethod
    def run_multi_environment_study(
        base_config: EnvironmentConfig,
        num_environments: int = 20,
        trials_per_env: int = 50,
        class_counts: Optional[List[int]] = None,
        modes: Optional[List[str]] = None,
        workers: int = 1,
    ) -> Dict[int, Dict[int, dict]]:
        """
        Runs the study across independently seeded environments, with versioned
        caching keyed by the full experiment configuration and source code.

        Returns nested dictionary: multi_env_data[M][env_id] = {results, runner}
        """
        if class_counts is None:
            class_counts = [1, 2, 3, 4, 5]
        if modes is None:
            modes = ["viewpoint_dependent", "geometric_only"]
        store = VersionedDataStore()
        multi_env_data = {M: {} for M in class_counts}

        print(f"\n=========================================================================")
        print(f"  RUNNING MULTI-ENVIRONMENT STUDY ({num_environments} UNIQUE ENVS x {trials_per_env} TRIALS)")
        print(f"=========================================================================")

        experiments_per_environment = len(class_counts) * len(modes) * trials_per_env
        for env_id in range(num_environments):
            env_seed = 100 + env_id
            environment_progress = tqdm(
                total=experiments_per_environment,
                desc=f"Environment {env_id + 1:02d}/{num_environments:02d}",
                unit="experiment",
                leave=True,
                dynamic_ncols=True,
            )

            try:
                for M in class_counts:
                    scaled_config = EnvironmentConfig(
                        num_objects=base_config.num_objects,
                        # Match the paper's same-class scene while varying M,
                        # the number of candidate classes and M^N realizations.
                        num_classes_in_scene=1,
                        num_classes_in_model=M,
                        sensor_range=base_config.sensor_range,
                        sensor_fov=base_config.sensor_fov,
                        num_trials=trials_per_env,
                        num_steps=base_config.num_steps,
                        num_samples=base_config.num_samples,
                        pruning_ratio=base_config.pruning_ratio,
                        max_hypotheses=base_config.max_hypotheses,
                        semantic_alpha=base_config.semantic_alpha,
                        semantic_k=base_config.semantic_k,
                        sigma_o=base_config.sigma_o.copy(),
                        sigma_p=base_config.sigma_p.copy(),
                        sigma_w=base_config.sigma_w.copy(),
                        sigma_v_geo=base_config.sigma_v_geo.copy(),
                        initial_robot_pose=base_config.initial_robot_pose.copy(),
                    )

                    env = Environment.create_random_environment(scaled_config, env_seed)
                    runner = ExperimentRunner(scaled_config, seed=env_seed, env_instance=env)

                    config_dict = {
                        "num_objects": scaled_config.num_objects,
                        "num_classes_in_scene": scaled_config.num_classes_in_scene,
                        "num_classes_in_model": M,
                        "sensor_range": scaled_config.sensor_range,
                        "sensor_fov": scaled_config.sensor_fov,
                        "num_trials": trials_per_env,
                        "num_steps": scaled_config.num_steps,
                        "num_samples": scaled_config.num_samples,
                        "pruning_ratio": scaled_config.pruning_ratio,
                        "max_hypotheses": scaled_config.max_hypotheses,
                        "semantic_alpha": scaled_config.semantic_alpha,
                        "semantic_k": scaled_config.semantic_k,
                        "sigma_o": scaled_config.sigma_o.tolist(),
                        "sigma_p": scaled_config.sigma_p.tolist(),
                        "sigma_w": scaled_config.sigma_w.tolist(),
                        "sigma_v_geo": scaled_config.sigma_v_geo.tolist(),
                        "initial_robot_pose": scaled_config.initial_robot_pose.tolist(),
                        "modes": list(modes),
                    }

                    def update_progress(mode: str, trial: int, total_trials: int):
                        environment_progress.set_postfix_str(
                            f"M={M} {mode} trial={trial}/{total_trials}",
                            refresh=False,
                        )
                        environment_progress.update(1)

                    # Cache identity includes every result-affecting parameter.
                    if store.exists(env_id, M, trials_per_env, config_dict):
                        cached_payload = store.load(env_id, M, trials_per_env, config_dict)
                        mode_results = cached_payload["results"]
                        environment_progress.set_postfix_str(f"M={M} cached", refresh=False)
                        environment_progress.update(len(modes) * trials_per_env)
                    else:
                        mode_results = runner.run_comparative_experiment(
                            modes=modes,
                            progress_callback=update_progress,
                            workers=workers,
                        )
                        store.save(
                            env_id, M, trials_per_env, env.objects,
                            mode_results, config_dict,
                        )

                    multi_env_data[M][env_id] = {
                        "runner": runner,
                        "modes": mode_results,
                    }
            finally:
                environment_progress.close()

        return multi_env_data
