import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class EnvironmentConfig:
    """
    =============================================================================
    EXPERIMENT HYPERPARAMETERS & CONFIGURATION VARIABLES
    =============================================================================
    """
    num_objects: int = 6           # N: Total number of objects in the scene
    num_classes_in_scene: int = 1  # Distinct ground-truth classes represented in the scene
    num_classes_in_model: int = 2  # M: candidate classes represented by the hybrid belief
    sensor_range: float = 6.0      # Maximum object-detection range [m]
    sensor_fov: float = 2.0 * np.pi # Camera field of view [rad]; 2*pi preserves the paper simulation's omnidirectional gate
    num_trials: int = 50           # Number of sampled ground truth tracks
    num_steps: int = 10            # T: Time steps per trajectory
    num_samples: int = 1000        # Ns: Pose samples per step for weight calculation
    pruning_ratio: float = 150.0   # Threshold tau = w_max / pruning_ratio for pruning
    max_hypotheses: Optional[int] = 100 # Beam cap applied after relative-weight pruning
    semantic_alpha: float = 0.25   # Paper Eq. (18) viewpoint-dependence strength
    semantic_k: float = 15.0       # Paper semantic precision parameter K

    # Covariance Matrices
    sigma_o: np.ndarray = field(
        default_factory=lambda: np.diag([0.05, 0.05, 0.5e-3])
    )
    sigma_p: np.ndarray = field(
        default_factory=lambda: np.diag([100.0, 100.0, 0.04])
    )
    sigma_w: np.ndarray = field(
        default_factory=lambda: np.diag([0.75e-3, 0.75e-3, 0.25e-3])
    )
    sigma_v_geo: np.ndarray = field(
        default_factory=lambda: np.diag([0.1, 0.05])
    )

    initial_robot_pose: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )

    def __post_init__(self):
        if self.num_objects < 1:
            raise ValueError("num_objects must be positive")
        if self.num_classes_in_scene < 1:
            raise ValueError("num_classes_in_scene must be positive")
        if self.num_classes_in_model < self.num_classes_in_scene:
            raise ValueError("the model must contain every ground-truth class")
        if self.sensor_range <= 0.0:
            raise ValueError("sensor_range must be positive")
        if not 0.0 < self.sensor_fov <= 2.0 * np.pi:
            raise ValueError("sensor_fov must lie in (0, 2*pi]")
        if self.num_samples < 1:
            raise ValueError("num_samples must be positive")
        if self.num_trials < 1 or self.num_steps < 1:
            raise ValueError("num_trials and num_steps must be positive")
        if self.pruning_ratio <= 1.0:
            raise ValueError("pruning_ratio must be greater than one")
        if self.max_hypotheses is not None and self.max_hypotheses < 1:
            raise ValueError("max_hypotheses must be positive when specified")
        if not 0.0 <= self.semantic_alpha <= 1.0:
            raise ValueError("semantic_alpha must lie in [0, 1]")
        if self.semantic_k <= 0.0:
            raise ValueError("semantic_k must be positive")

@dataclass
class ObjectLandmark:
    id: int
    gt_pose: np.ndarray        # Ground truth pose [x, y, theta]
    gt_class: int              # Ground truth class c_i in {0, ..., M-1}
    prior_pose_mean: np.ndarray # Sampled prior mean given sigma_o

class Environment:
    """Simulates 2D environment with stationary objects and ground truth tracks."""

    def __init__(self, config: EnvironmentConfig, seed: int = 42, custom_objects: List[ObjectLandmark] = None):
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        if custom_objects is not None:
            self.objects = custom_objects
        else:
            self.objects = self._generate_objects()

    def _generate_objects(self) -> List[ObjectLandmark]:
        """Generate N stationary objects with reproducible seed layout."""
        objects = []
        # Spatial workspace region around robot path: highly ambiguous cluster
        for i in range(self.config.num_objects):
            x_gt = self.rng.uniform(-3.0, 3.0)
            y_gt = self.rng.uniform(2.0, 4.0)
            th_gt = self.rng.uniform(-np.pi, np.pi)

            gt_pose = np.array([x_gt, y_gt, th_gt], dtype=np.float64)

            # Ensure classes are somewhat evenly distributed
            gt_class = int(i % self.config.num_classes_in_scene)

            prior_noise = self.rng.multivariate_normal(
                np.zeros(3), self.config.sigma_o
            )
            prior_mean = gt_pose + prior_noise
            prior_mean[2] = (prior_mean[2] + np.pi) % (2 * np.pi) - np.pi

            objects.append(ObjectLandmark(
                id=i,
                gt_pose=gt_pose,
                gt_class=gt_class,
                prior_pose_mean=prior_mean
            ))
        return objects

    @staticmethod
    def create_random_environment(config: EnvironmentConfig, env_seed: int) -> 'Environment':
        """Creates a unique randomized environment instance given an env_seed."""
        return Environment(config, seed=env_seed)

    def generate_tracks(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate ground truth tracks starting at initial_robot_pose.
        Returns list of (gt_poses, controls) tuples.
        """
        tracks = []
        base_speed = 1.2

        for trial in range(self.config.num_trials):
            trial_rng = np.random.default_rng(self.seed * 100 + trial)
            gt_poses = np.zeros((self.config.num_steps + 1, 3))
            controls = np.zeros((self.config.num_steps, 3))

            gt_poses[0] = self.config.initial_robot_pose.copy()

            for k in range(self.config.num_steps):
                v_x = base_speed + 0.1 * trial_rng.standard_normal()
                v_y = 0.05 * trial_rng.standard_normal()
                w_theta = 0.05 * np.sin(k * 0.5) + 0.02 * trial_rng.standard_normal()

                action = np.array([v_x, v_y, w_theta])
                controls[k] = action

                curr_pose = gt_poses[k]
                c_th = np.cos(curr_pose[2])
                s_th = np.sin(curr_pose[2])
                R = np.array([[c_th, -s_th], [s_th, c_th]])

                d_pos = R @ action[:2]
                next_x = curr_pose[0] + d_pos[0]
                next_y = curr_pose[1] + d_pos[1]
                next_th = curr_pose[2] + action[2]

                w_noise = trial_rng.multivariate_normal(
                    np.zeros(3), self.config.sigma_w
                )
                next_pose = np.array([next_x, next_y, next_th]) + w_noise
                next_pose[2] = (next_pose[2] + np.pi) % (2 * np.pi) - np.pi
                gt_poses[k+1] = next_pose

            tracks.append((gt_poses, controls))

        return tracks
