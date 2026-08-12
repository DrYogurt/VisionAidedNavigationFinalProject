import numpy as np
from typing import Tuple, Optional

class SE2Utils:
    """Utilities for SE(2) transformations and angle wrapping."""

    @staticmethod
    def wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def wrap_angles(angles: np.ndarray) -> np.ndarray:
        """Wrap array of angles to [-pi, pi]."""
        return (angles + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def compose(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """SE(2) pose composition: x1 (+) x2."""
        c1, s1 = np.cos(x1[2]), np.sin(x1[2])
        x = x1[0] + x2[0] * c1 - x2[1] * s1
        y = x1[1] + x2[0] * s1 + x2[1] * c1
        th = SE2Utils.wrap_angle(x1[2] + x2[2])
        return np.array([x, y, th])

    @staticmethod
    def relative_pose(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """SE(2) relative pose: x1 (-) x2."""
        dx = x1[0] - x2[0]
        dy = x1[1] - x2[1]
        c2, s2 = np.cos(x2[2]), np.sin(x2[2])

        rx = dx * c2 + dy * s2
        ry = -dx * s2 + dy * c2
        rth = SE2Utils.wrap_angle(x1[2] - x2[2])
        return np.array([rx, ry, rth])

class MotionModel:
    """Robot Motion / Process Model: P(x_{k+1} | x_k, a_k) = N(f(x_k, a_k), Sigma_w)."""

    def __init__(self, sigma_w: np.ndarray):
        self.sigma_w = sigma_w

    def propagate(self, x: np.ndarray, action: np.ndarray) -> np.ndarray:
        return SE2Utils.compose(x, action)

    def jacobian_x(self, x: np.ndarray, action: np.ndarray) -> np.ndarray:
        c = np.cos(x[2])
        s = np.sin(x[2])
        ax, ay = action[0], action[1]

        F_x = np.eye(3)
        F_x[0, 2] = -ax * s - ay * c
        F_x[1, 2] = ax * c - ay * s
        return F_x

class GeometricObservationModel:
    """Geometric Measurement Model: range and bearing z_geo = [range, bearing]."""

    def __init__(self, sigma_v_geo: np.ndarray):
        self.sigma_v_geo = sigma_v_geo
        self.inv_sigma_v_geo = np.linalg.inv(sigma_v_geo)
        self.det_sigma_v_geo = np.linalg.det(sigma_v_geo)
        self.norm_const = 1.0 / (2.0 * np.pi * np.sqrt(self.det_sigma_v_geo))

    def observe(self, robot_pose: np.ndarray, obj_pose: np.ndarray) -> np.ndarray:
        dx = obj_pose[0] - robot_pose[0]
        dy = obj_pose[1] - robot_pose[1]
        range_val = np.sqrt(dx**2 + dy**2)
        bearing = SE2Utils.wrap_angle(np.atan2(dy, dx) - robot_pose[2])
        return np.array([range_val, bearing])

    def jacobians(self, robot_pose: np.ndarray, obj_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dx = obj_pose[0] - robot_pose[0]
        dy = obj_pose[1] - robot_pose[1]
        q = dx**2 + dy**2
        r = np.sqrt(q)
        if r < 1e-6:
            r = 1e-6
            q = 1e-12

        H_x = np.array([
            [-dx / r, -dy / r, 0.0],
            [dy / q, -dx / q, -1.0]
        ])

        H_o = np.array([
            [dx / r, dy / r, 0.0],
            [-dy / q, dx / q, 0.0]
        ])

        return H_x, H_o

    def likelihood(self, z_geo: np.ndarray, robot_pose: np.ndarray, obj_pose: np.ndarray) -> float:
        z_hat = self.observe(robot_pose, obj_pose)
        diff = z_geo - z_hat
        diff[1] = SE2Utils.wrap_angle(diff[1])

        exponent = -0.5 * float(diff.T @ self.inv_sigma_v_geo @ diff)
        return self.norm_const * np.exp(exponent)

    def batch_likelihood(self, z_geo: np.ndarray, robot_poses: np.ndarray, obj_poses: np.ndarray) -> np.ndarray:
        dx = obj_poses[:, 0] - robot_poses[:, 0]
        dy = obj_poses[:, 1] - robot_poses[:, 1]
        r = np.sqrt(dx**2 + dy**2)
        bearing = SE2Utils.wrap_angles(np.atan2(dy, dx) - robot_poses[:, 2])

        diff_r = z_geo[0] - r
        diff_b = SE2Utils.wrap_angles(z_geo[1] - bearing)

        inv_var_r = self.inv_sigma_v_geo[0, 0]
        inv_var_b = self.inv_sigma_v_geo[1, 1]

        mahal = diff_r**2 * inv_var_r + diff_b**2 * inv_var_b
        return self.norm_const * np.exp(-0.5 * mahal)

class ViewpointDependentSemanticModel:
    """Paper-compatible viewpoint-dependent classifier model.

    For two classes this implements Eq. (18) exactly (up to a permutation
    selecting the hypothesized true class).  For M > 2, the paper does not
    define a model; this implementation uses the symmetric extension that
    distributes the Eq. (18) error mass uniformly over the other classes.
    """

    def __init__(self, num_classes: int = 2, alpha: float = 0.25, precision_k: float = 15.0):
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if precision_k <= 0.0:
            raise ValueError("precision_k must be positive")
        self.num_classes = num_classes
        self.alpha = alpha
        self.precision_k = precision_k

    @staticmethod
    def relative_view_angle(robot_pose: np.ndarray, obj_pose: np.ndarray) -> float:
        """Angle from the object's heading to the camera, as in paper Eq. (18)."""
        dx = robot_pose[0] - obj_pose[0]
        dy = robot_pose[1] - obj_pose[1]
        return SE2Utils.wrap_angle(np.atan2(dy, dx) - obj_pose[2])

    def expected_semantic_vector(self, robot_pose: np.ndarray, obj_pose: np.ndarray, class_m: int) -> np.ndarray:
        if not 0 <= class_m < self.num_classes:
            raise ValueError("class_m is outside the modeled class set")
        if self.num_classes == 1:
            return np.ones(1)

        theta = self.relative_view_angle(robot_pose, obj_pose)
        error_mass = self.alpha * np.cos(theta / 2.0) ** 2
        h_c = np.full(self.num_classes, error_mass / (self.num_classes - 1))
        h_c[class_m] = 1.0 - error_mass
        return h_c

    def semantic_covariance(self, robot_pose: np.ndarray, obj_pose: np.ndarray) -> np.ndarray:
        """Constant paper covariance for M=2 and a documented isotropic M-class extension."""
        if self.num_classes == 2:
            R = self.precision_k * np.array([[1.0, -0.5], [0.0, 1.0]])
            return np.linalg.inv(R.T @ R)
        variance = 1.0 / (self.precision_k ** 2)
        return variance * np.eye(self.num_classes)

    def jacobians(self, robot_pose: np.ndarray, obj_pose: np.ndarray, class_m: int) -> Tuple[np.ndarray, np.ndarray]:
        dx = robot_pose[0] - obj_pose[0]
        dy = robot_pose[1] - obj_pose[1]
        q = max(dx**2 + dy**2, 1e-12)
        theta = self.relative_view_angle(robot_pose, obj_pose)

        d_theta_d_x = np.array([-dy / q, dx / q, 0.0])
        d_theta_d_o = np.array([dy / q, -dx / q, -1.0])

        dh_d_theta = np.zeros(self.num_classes)
        if self.num_classes > 1:
            d_error_d_theta = -0.5 * self.alpha * np.sin(theta)
            dh_d_theta.fill(d_error_d_theta / (self.num_classes - 1))
            dh_d_theta[class_m] = -d_error_d_theta

        return (
            np.outer(dh_d_theta, d_theta_d_x),
            np.outer(dh_d_theta, d_theta_d_o),
        )

    def likelihood(self, z_sem: np.ndarray, robot_pose: np.ndarray, obj_pose: np.ndarray, class_m: int) -> float:
        h_c = self.expected_semantic_vector(robot_pose, obj_pose, class_m)
        Sigma_c = self.semantic_covariance(robot_pose, obj_pose)
        diff = z_sem - h_c
        inv_Sigma_c = np.linalg.inv(Sigma_c)
        det_Sigma_c = np.linalg.det(Sigma_c)

        exponent = -0.5 * float(diff.T @ inv_Sigma_c @ diff)
        norm_const = 1.0 / (np.power(2.0 * np.pi, self.num_classes / 2.0) * np.sqrt(det_Sigma_c))
        return norm_const * np.exp(exponent)

    def batch_likelihood(self, z_sem: np.ndarray, robot_poses: np.ndarray, obj_poses: np.ndarray, class_m: int) -> np.ndarray:
        """Vectorized likelihood for Monte Carlo component-weight propagation."""
        n_samples = len(robot_poses)
        if self.num_classes == 1:
            means = np.ones((n_samples, 1))
        else:
            dx = robot_poses[:, 0] - obj_poses[:, 0]
            dy = robot_poses[:, 1] - obj_poses[:, 1]
            theta = SE2Utils.wrap_angles(np.atan2(dy, dx) - obj_poses[:, 2])
            error_mass = self.alpha * np.cos(theta / 2.0) ** 2
            means = np.repeat((error_mass / (self.num_classes - 1))[:, None], self.num_classes, axis=1)
            means[:, class_m] = 1.0 - error_mass

        covariance = self.semantic_covariance(robot_poses[0], obj_poses[0])
        inverse = np.linalg.inv(covariance)
        determinant = np.linalg.det(covariance)
        normalizer = 1.0 / (
            np.power(2.0 * np.pi, self.num_classes / 2.0) * np.sqrt(determinant)
        )
        residuals = z_sem - means
        mahalanobis = np.einsum("ni,ij,nj->n", residuals, inverse, residuals)
        return normalizer * np.exp(-0.5 * mahalanobis)

class PassiveSemanticModel:
    """Optional viewpoint-independent semantic ablation (not the paper baseline)."""

    def __init__(self, num_classes: int = 2, alpha: float = 0.25, precision_k: float = 15.0):
        self.num_classes = num_classes
        self.alpha = alpha
        self._paper_model = ViewpointDependentSemanticModel(num_classes, alpha, precision_k)
        self.covariance = self._paper_model.semantic_covariance(np.zeros(3), np.zeros(3))
        self.inverse = np.linalg.inv(self.covariance)
        self.norm_const = 1.0 / (
            np.power(2.0 * np.pi, num_classes / 2.0) * np.sqrt(np.linalg.det(self.covariance))
        )

    def expected_semantic_vector(self, class_m: int) -> np.ndarray:
        """Constant expected classification vector independent of relative pose/viewpoint."""
        if self.num_classes == 1:
            return np.ones(1)
        error_mass = self.alpha / 2.0
        h_c = np.full(self.num_classes, error_mass / (self.num_classes - 1))
        h_c[class_m] = 1.0 - error_mass
        return h_c

    def likelihood(self, z_sem: np.ndarray, robot_pose: np.ndarray, obj_pose: np.ndarray, class_m: int) -> float:
        h_c = self.expected_semantic_vector(class_m)
        diff = z_sem - h_c
        exponent = -0.5 * float(diff.T @ self.inverse @ diff)
        return self.norm_const * np.exp(exponent)

    def batch_likelihood(self, z_sem: np.ndarray, robot_poses: np.ndarray, obj_poses: np.ndarray, class_m: int) -> np.ndarray:
        """Vectorized batch likelihood for N_s samples."""
        h_c = self.expected_semantic_vector(class_m)
        diff = z_sem - h_c
        mahal = float(diff.T @ self.inverse @ diff)
        l_val = self.norm_const * np.exp(-0.5 * mahal)
        return np.full(len(robot_poses), l_val)
