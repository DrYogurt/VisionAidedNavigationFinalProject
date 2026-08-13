import numpy as np
from typing import List, Tuple

from src.environment import EnvironmentConfig, ObjectLandmark
from src.models import (
    SE2Utils, MotionModel, GeometricObservationModel,
    ViewpointDependentSemanticModel, PassiveSemanticModel
)

class HypothesisComponent:
    """
    Represents a discrete-continuous hybrid hypothesis component:
    (C, beta_{1:k}) with weight w and continuous Gaussian belief b[X_k]^C_{beta_{1:k}}.
    """
    def __init__(self,
                 class_assignments: Tuple[int, ...],
                 da_history: Tuple[Tuple[int, ...], ...],
                 weight: float,
                 mu: np.ndarray,
                 cov: np.ndarray):
        self.class_assignments = class_assignments
        self.da_history = da_history
        self.weight = weight
        self.mu = mu.copy()
        self.cov = cov.copy()

    def get_robot_pose(self) -> np.ndarray:
        return self.mu[:3].copy()

    def get_robot_cov(self) -> np.ndarray:
        return self.cov[:3, :3].copy()

    def get_obj_pose(self, obj_id: int) -> np.ndarray:
        idx = 3 + 3 * obj_id
        return self.mu[idx:idx+3].copy()

    def get_obj_cov(self, obj_id: int) -> np.ndarray:
        idx = 3 + 3 * obj_id
        return self.cov[idx:idx+3, idx:idx+3].copy()


class SemanticSLAMEngine:
    """
    EKF approximation to Algorithm 1's hybrid semantic SLAM belief.

    Discrete class and data-association hypotheses follow the paper. Each
    continuous factor-graph component is approximated by a joint EKF over the
    current robot pose and stationary objects rather than iSAM2 smoothing.
    """

    def __init__(self,
                 config: EnvironmentConfig,
                 objects: List[ObjectLandmark],
                 mode: str = "viewpoint_dependent"):
        self.config = config
        self.objects = objects
        self.mode = mode
        self.num_objects = config.num_objects
        self.num_classes = config.num_classes_in_model

        self.motion_model = MotionModel(config.sigma_w)
        self.geo_model = GeometricObservationModel(config.sigma_v_geo)

        if mode == "viewpoint_dependent":
            self.sem_model = ViewpointDependentSemanticModel(
                config.num_classes_in_model,
                config.semantic_alpha,
                config.semantic_k,
            )
        elif mode == "passive":
            self.sem_model = PassiveSemanticModel(
                config.num_classes_in_model,
                config.semantic_alpha,
                config.semantic_k,
            )
        elif mode == "geometric_only":
            self.sem_model = None
        else:
            raise ValueError(f"unknown inference mode: {mode}")

        self.hypotheses: List[HypothesisComponent] = []
        # Counts rare beam/Monte-Carlo support collapses where the hard
        # visibility gate rejects every candidate for an observation.  In
        # that case the measurement likelihood is used without the gate so
        # the filter can recover instead of terminating the whole study.
        self.visibility_recoveries = 0
        self._initialize_beliefs()

    def _initialize_beliefs(self):
        state_dim = 3 + 3 * self.num_objects
        mu_init = np.zeros(state_dim)
        cov_init = np.zeros((state_dim, state_dim))

        mu_init[:3] = self.config.initial_robot_pose
        cov_init[:3, :3] = self.config.sigma_p

        for i, obj in enumerate(self.objects):
            idx = 3 + 3 * i
            mu_init[idx:idx+3] = obj.prior_pose_mean
            cov_init[idx:idx+3, idx:idx+3] = self.config.sigma_o

        # -1 denotes an unobserved class variable marginalized under the
        # uniform prior.  It is expanded lazily when an association first
        # makes that object's class relevant (Sec. IV-D of the paper).
        initial_C = tuple(-1 for _ in self.objects)

        comp = HypothesisComponent(
            class_assignments=initial_C,
            da_history=(),
            weight=1.0,
            mu=mu_init,
            cov=cov_init
        )
        self.hypotheses = [comp]

    def step(self, action: np.ndarray, observations: List[Tuple[int, np.ndarray, np.ndarray]], rng: np.random.Generator):
        """
        Executes one step of Algorithm 1:
        1. Propagate continuous belief according to motion model.
        2. Procedure PROPWEIGHTS: Calculate & propagate weights via Monte Carlo sampling.
        3. Normalize and Procedure PRUNEANDNORMALIZE.
        4. EKF continuous factor updates for surviving hypotheses.
        """
        if not self.hypotheses:
            return

        # Step 1: Propagate continuous beliefs according to motion model
        for comp in self.hypotheses:
            F_x = self.motion_model.jacobian_x(comp.mu[:3], action)
            comp.mu[:3] = self.motion_model.propagate(comp.mu[:3], action)

            comp.cov[:3, :3] = F_x @ comp.cov[:3, :3] @ F_x.T + self.config.sigma_w

            for j in range(self.num_objects):
                idx = 3 + 3 * j
                comp.cov[:3, idx:idx+3] = F_x @ comp.cov[:3, idx:idx+3]
                comp.cov[idx:idx+3, :3] = comp.cov[:3, idx:idx+3].T

        if not observations:
            return

        # Condition on the independent factors in Z_k sequentially. Class
        # variables are expanded lazily, and beta_k is committed to history
        # as one vector after every detection at this time has been handled.
        working = [(comp, tuple()) for comp in self.hypotheses]
        for _landmark_gt_id, z_geo, z_sem in observations:
            next_working = []
            visibility_fallback = []
            candidate_betas = list(range(self.num_objects))

            for comp, beta_prefix in working:
                for beta_k in candidate_betas:
                    # A time-step observation set contains at most one
                    # detection per physical object, so beta_k is injective.
                    # Enforcing this removes impossible duplicate assignments
                    # and reduces N^n_k candidates to permutations.
                    if beta_k in beta_prefix:
                        continue
                    assigned_class = comp.class_assignments[beta_k]
                    if self.mode == "geometric_only":
                        class_options = [(assigned_class, 1.0)]
                    elif assigned_class >= 0:
                        class_options = [(assigned_class, 1.0)]
                    else:
                        class_prior = 1.0 / self.num_classes
                        class_options = [(class_m, class_prior) for class_m in range(self.num_classes)]

                    for class_m, class_prior in class_options:
                        psi, measurement_psi = self._prop_weights_sampling(
                            comp, beta_k, z_geo, z_sem, class_m, rng
                        )
                        target = next_working
                        if psi <= 0.0 or not np.isfinite(psi):
                            # The hard visibility indicator is estimated with
                            # finite samples. Beam pruning can leave no sample
                            # inside that indicator even though the observed
                            # geometric/semantic measurement has finite
                            # support. Keep these candidates only as a
                            # last-resort recovery set for this factor.
                            psi = measurement_psi
                            target = visibility_fallback
                        if psi <= 0.0 or not np.isfinite(psi):
                            continue

                        classes = list(comp.class_assignments)
                        if class_m >= 0:
                            classes[beta_k] = class_m
                        new_comp = HypothesisComponent(
                            class_assignments=tuple(classes),
                            da_history=comp.da_history,
                            weight=comp.weight * class_prior * psi,
                            mu=comp.mu,
                            cov=comp.cov,
                        )
                        self._ekf_update(new_comp, beta_k, z_geo, z_sem)
                        target.append((new_comp, beta_prefix + (beta_k,)))

            if not next_working:
                if not visibility_fallback:
                    raise RuntimeError("all hypotheses received zero measurement likelihood")
                next_working = visibility_fallback
                self.visibility_recoveries += 1

            # Incremental threshold pruning keeps the Python implementation
            # tractable for multi-detection steps. With pruning disabled this
            # sequential factorization is equivalent to the joint likelihood.
            surviving_components = self._prune_and_normalize([item[0] for item in next_working])
            surviving_ids = {id(comp) for comp in surviving_components}
            working = [item for item in next_working if id(item[0]) in surviving_ids]

        for comp, beta_vector in working:
            comp.da_history = comp.da_history + (beta_vector,)
        self.hypotheses = [comp for comp, _ in working]

    def _prune_and_normalize(self, hypotheses: List[HypothesisComponent]) -> List[HypothesisComponent]:
        total_weight = sum(h.weight for h in hypotheses)
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            raise RuntimeError("hypothesis weights cannot be normalized")
        for hypothesis in hypotheses:
            hypothesis.weight /= total_weight

        max_weight = max(h.weight for h in hypotheses)
        threshold = max_weight / self.config.pruning_ratio
        surviving = [h for h in hypotheses if h.weight >= threshold]
        surviving.sort(key=lambda h: h.weight, reverse=True)
        if self.config.max_hypotheses is not None:
            surviving = surviving[:self.config.max_hypotheses]

        surviving_total = sum(h.weight for h in surviving)
        for hypothesis in surviving:
            hypothesis.weight /= surviving_total
        return surviving

    def _prop_weights_sampling(self,
                              comp: HypothesisComponent,
                              beta_k: int,
                              z_geo: np.ndarray,
                              z_sem: np.ndarray,
                              class_m: int,
                              rng: np.random.Generator) -> Tuple[float, float]:
        """
        Procedure PROPWEIGHTS:
        Vectorized sampling of Ns poses {x_k^(i), X_{beta_k}^{o(i)}} from continuous belief b^-[x_k, X_{beta_k}^o]
        and evaluates average observation likelihood psi.
        """
        N_s = self.config.num_samples
        idx_obj = 3 + 3 * beta_k

        indices = [0, 1, 2, idx_obj, idx_obj+1, idx_obj+2]
        joint_mu = comp.mu[indices]
        joint_cov = comp.cov[np.ix_(indices, indices)]

        joint_cov = 0.5 * (joint_cov + joint_cov.T) + 1e-8 * np.eye(6)

        try:
            samples = rng.multivariate_normal(joint_mu, joint_cov, size=N_s)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("failed to sample the propagated Gaussian component") from exc

        robot_samples = samples[:, :3]
        obj_samples = samples[:, 3:]

        l_geo = self.geo_model.batch_likelihood(z_geo, robot_samples, obj_samples)

        if self.mode != "geometric_only" and self.sem_model is not None and class_m >= 0:
            l_sem = self.sem_model.batch_likelihood(z_sem, robot_samples, obj_samples, class_m)
        else:
            l_sem = np.ones(N_s)

        visible = self._batch_visibility(robot_samples, obj_samples)
        measurement_likelihoods = l_geo * l_sem
        gated_psi = float(np.mean(measurement_likelihoods * visible))
        measurement_psi = float(np.mean(measurement_likelihoods))
        return gated_psi, measurement_psi

    def _batch_visibility(self, robot_poses: np.ndarray, obj_poses: np.ndarray) -> np.ndarray:
        """Object-observation factor P(beta_k | x_k, X^o_beta_k)."""
        dx = obj_poses[:, 0] - robot_poses[:, 0]
        dy = obj_poses[:, 1] - robot_poses[:, 1]
        in_range = np.hypot(dx, dy) <= self.config.sensor_range
        if self.config.sensor_fov >= 2.0 * np.pi:
            return in_range.astype(float)
        bearing = SE2Utils.wrap_angles(np.arctan2(dy, dx) - robot_poses[:, 2])
        in_fov = np.abs(bearing) <= self.config.sensor_fov / 2.0
        return (in_range & in_fov).astype(float)

    def _ekf_update(self, comp: HypothesisComponent, beta_k: int, z_geo: np.ndarray, z_sem: np.ndarray):
        """Extended Kalman Filter update for geometric and semantic observation of landmark beta_k."""
        robot_pose = comp.get_robot_pose()
        obj_pose = comp.get_obj_pose(beta_k)

        # 1. Geometric Factor Update
        z_hat = self.geo_model.observe(robot_pose, obj_pose)
        H_x, H_o = self.geo_model.jacobians(robot_pose, obj_pose)

        state_dim = len(comp.mu)
        H_geo = np.zeros((2, state_dim))
        H_geo[:, :3] = H_x
        idx_obj = 3 + 3 * beta_k
        H_geo[:, idx_obj:idx_obj+3] = H_o

        y_geo = z_geo - z_hat
        y_geo[1] = SE2Utils.wrap_angle(y_geo[1])

        S_geo = H_geo @ comp.cov @ H_geo.T + self.config.sigma_v_geo
        K_geo = comp.cov @ H_geo.T @ np.linalg.inv(S_geo)

        comp.mu = comp.mu + K_geo @ y_geo
        comp.mu[2] = SE2Utils.wrap_angle(comp.mu[2])
        for j in range(self.num_objects):
            o_idx = 3 + 3 * j + 2
            comp.mu[o_idx] = SE2Utils.wrap_angle(comp.mu[o_idx])

        I = np.eye(state_dim)
        comp.cov = (I - K_geo @ H_geo) @ comp.cov @ (I - K_geo @ H_geo).T + K_geo @ self.config.sigma_v_geo @ K_geo.T

        # 2. Semantic Factor Update (for Viewpoint-Dependent mode)
        if self.mode == "viewpoint_dependent" and self.sem_model is not None:
            class_m = comp.class_assignments[beta_k]
            if class_m < 0:
                return
            robot_pose = comp.get_robot_pose()
            obj_pose = comp.get_obj_pose(beta_k)

            z_sem_hat = self.sem_model.expected_semantic_vector(robot_pose, obj_pose, class_m)
            H_x_sem, H_o_sem = self.sem_model.jacobians(robot_pose, obj_pose, class_m)

            H_sem = np.zeros((self.num_classes, state_dim))
            H_sem[:, :3] = H_x_sem
            H_sem[:, idx_obj:idx_obj+3] = H_o_sem

            y_sem = z_sem - z_sem_hat
            Sigma_c = self.sem_model.semantic_covariance(robot_pose, obj_pose)

            S_sem = H_sem @ comp.cov @ H_sem.T + Sigma_c
            try:
                K_sem = comp.cov @ H_sem.T @ np.linalg.inv(S_sem)
                comp.mu = comp.mu + K_sem @ y_sem
                comp.mu[2] = SE2Utils.wrap_angle(comp.mu[2])
                for j in range(self.num_objects):
                    o_idx = 3 + 3 * j + 2
                    comp.mu[o_idx] = SE2Utils.wrap_angle(comp.mu[o_idx])
                comp.cov = (I - K_sem @ H_sem) @ comp.cov @ (I - K_sem @ H_sem).T + K_sem @ Sigma_c @ K_sem.T
            except np.linalg.LinAlgError as exc:
                raise RuntimeError("semantic EKF innovation covariance is singular") from exc
