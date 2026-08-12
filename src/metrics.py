import numpy as np
from typing import List, Dict
from src.slam_engine import HypothesisComponent

class MetricsCalculator:
    """Calculates quantitative performance metrics specified in Experiment Specification Sheet."""

    @staticmethod
    def da_weight_entropy(hypotheses: List[HypothesisComponent]) -> float:
        """
        Entropy over Data Association Weights H(w):
        H(w) = - sum_{i=1}^{N_k} w_i * log(w_i)
        Measures data association ambiguity.
        """
        # Eq. (8): marginalize class realizations before computing entropy of
        # the data-association posterior P(beta_1:k | H_k).
        weights_by_da = {}
        for hypothesis in hypotheses:
            if hypothesis.weight > 0.0:
                weights_by_da[hypothesis.da_history] = (
                    weights_by_da.get(hypothesis.da_history, 0.0)
                    + hypothesis.weight
                )
        weights = np.array(list(weights_by_da.values()))
        if len(weights) == 0:
            return 0.0
        weights = weights / np.sum(weights)
        # Add epsilon to prevent log(0)
        eps = 1e-15
        entropy = -float(np.sum(weights * np.log(weights + eps)))
        return max(0.0, entropy)

    @staticmethod
    def position_cov_determinant(hypotheses: List[HypothesisComponent]) -> float:
        """
        Position Covariance Determinant det(Sigma):
        Determinant of (2x2 position covariance) for highest weight realization at time step k.
        """
        if not hypotheses:
            return 0.0
        top_h = max(hypotheses, key=lambda h: h.weight)
        pos_cov = top_h.get_robot_cov()[:2, :2]
        return float(np.linalg.det(pos_cov))

    @staticmethod
    def max_weight_error(hypotheses: List[HypothesisComponent], gt_pose: np.ndarray) -> float:
        """
        Maximum Weight Estimation Error (x~^{w_max}):
        Euclidean distance from ground truth to highest weight estimate.
        """
        if not hypotheses:
            return 0.0
        top_h = max(hypotheses, key=lambda h: h.weight)
        est_pos = top_h.get_robot_pose()[:2]
        gt_pos = gt_pose[:2]
        return float(np.linalg.norm(est_pos - gt_pos))

    @staticmethod
    def average_estimation_error(hypotheses: List[HypothesisComponent], gt_pose: np.ndarray) -> float:
        """
        Average Estimation Error (x~^{w-avg}):
        Weighted average of estimation errors across all active hypotheses.
        """
        if not hypotheses:
            return 0.0
        weights = np.array([h.weight for h in hypotheses])
        total_w = np.sum(weights)
        if total_w == 0:
            return 0.0
        norm_weights = weights / total_w

        gt_pos = gt_pose[:2]
        errors = [np.linalg.norm(h.get_robot_pose()[:2] - gt_pos) for h in hypotheses]
        weighted_err = float(np.sum(norm_weights * np.array(errors)))
        return weighted_err

    @staticmethod
    def num_active_hypotheses(hypotheses: List[HypothesisComponent]) -> int:
        """Returns number of surviving active hypotheses N_k after pruning."""
        return len(hypotheses)
