import os
import argparse
from typing import Optional
import numpy as np
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from src.environment import EnvironmentConfig
from src.experiment import ExperimentRunner


def mean_and_95_ci(values: np.ndarray, axis: int = 0):
    """Return a mean and normal-approximation CI, with a safe n=1 fallback."""
    values = np.asarray(values, dtype=float)
    mean = np.mean(values, axis=axis)
    count = values.shape[axis]
    if count < 2:
        return mean, np.zeros_like(mean)
    ci = 1.96 * np.std(values, axis=axis, ddof=1) / np.sqrt(count)
    return mean, ci

def configure_hyperparameters(
    num_objects: int = 6,
    num_model_classes: int = 5,
    num_trials: int = 50,
    num_steps: int = 10,
    num_samples: int = 1000,
    pruning_ratio: float = 150.0,
    max_hypotheses: Optional[int] = 100,
    semantic_alpha: float = 0.25,
    semantic_k: float = 15.0,
    sensor_range: float = 6.0,
    sensor_fov: float = 2.0 * np.pi,
) -> EnvironmentConfig:
    sigma_o = np.diag([0.05, 0.05, 0.5e-3])
    sigma_p = np.diag([100.0, 100.0, 0.04])
    sigma_w = np.diag([0.75e-3, 0.75e-3, 0.25e-3])
    sigma_v_geo = np.diag([0.1, 0.05])

    config = EnvironmentConfig(
        num_objects=num_objects,
        num_classes_in_scene=1,
        num_classes_in_model=num_model_classes,
        num_trials=num_trials,
        num_steps=num_steps,
        num_samples=num_samples,
        pruning_ratio=pruning_ratio,
        max_hypotheses=max_hypotheses,
        semantic_alpha=semantic_alpha,
        semantic_k=semantic_k,
        sensor_range=sensor_range,
        sensor_fov=sensor_fov,
        sigma_o=sigma_o,
        sigma_p=sigma_p,
        sigma_w=sigma_w,
        sigma_v_geo=sigma_v_geo
    )
    return config

def setup_results_dir() -> str:
    res_dir = os.path.join(os.path.dirname(__file__), "results", "v9")
    os.makedirs(res_dir, exist_ok=True)
    return res_dir

def plot_tractability_table_graph(multi_env_data: dict, output_dir: str):
    """
    Generates a dedicated 4-panel graph plotting all metrics from the Class Scaling & Tractability Study:
    - Active Hypotheses (N_k) vs M
    - Inference Time per Step (ms) vs M
    - Fixed belief class realizations (5^N) vs M
    - Final Pose Estimation Error (m) vs M (Viewpoint-Dependent vs geometric-only)
    """
    classes = sorted(list(multi_env_data.keys()))
    first_class = classes[0]
    first_environment = next(iter(multi_env_data[first_class].values()))
    num_objects = first_environment["runner"].config.num_objects

    active_hyps = []
    times_ms = []
    model_class_count = first_environment["runner"].config.num_classes_in_model
    realizations = [model_class_count ** num_objects for _M in classes]
    vp_err_mean = []
    pas_err_mean = []

    for M in classes:
        env_dict = multi_env_data[M]
        m_hyps = []
        m_times = []
        m_vp_errs = []
        m_pas_errs = []

        for env_id, data in env_dict.items():
            vp_res = data["modes"]["viewpoint_dependent"]
            pas_res = data["modes"]["geometric_only"]

            m_hyps.append(np.mean(vp_res["num_hypotheses"]))
            m_times.append(np.mean(vp_res["step_time"]) * 1000.0)
            m_vp_errs.extend(vp_res["max_weight_err"][:, -1])
            m_pas_errs.extend(pas_res["max_weight_err"][:, -1])

        active_hyps.append(np.mean(m_hyps))
        times_ms.append(np.mean(m_times))
        vp_err_mean.append(np.mean(m_vp_errs))
        pas_err_mean.append(np.mean(m_pas_errs))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.rcParams.update({'font.size': 12})

    # Panel 1: Active Hypotheses
    axes[0, 0].plot(classes, active_hyps, 'o-', color='#1f77b4', linewidth=2.5, markersize=8)
    axes[0, 0].set_title("Active Hypotheses (N_k) vs. Scene Class Count M")
    axes[0, 0].set_xlabel("Actual Classes Present (M)")
    axes[0, 0].set_ylabel("Avg Active Hypotheses (N_k)")
    axes[0, 0].set_xticks(classes)
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Inference Time
    axes[0, 1].plot(classes, times_ms, 's-', color='#2ca02c', linewidth=2.5, markersize=8)
    axes[0, 1].set_title("Inference Speed (ms/step) vs. Scene Class Count M")
    axes[0, 1].set_xlabel("Actual Classes Present (M)")
    axes[0, 1].set_ylabel("Time per Step (ms)")
    axes[0, 1].set_xticks(classes)
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # Panel 3: the five-class belief space is fixed for every scene condition.
    axes[1, 0].plot(classes, realizations, '^--', color='#9467bd', linewidth=2.5, markersize=8)
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title(f"Fixed Belief Class Realizations ({model_class_count}^N) [log scale]")
    axes[1, 0].set_xlabel("Actual Classes Present (M)")
    axes[1, 0].set_ylabel(f"Belief Realizations ({model_class_count}^N)")
    axes[1, 0].set_xticks(classes)
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # Panel 4: Final Pose Estimation Error
    axes[1, 1].plot(classes, vp_err_mean, 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label='Viewpoint-Dependent')
    axes[1, 1].plot(classes, pas_err_mean, 's--', color='#ff7f0e', linewidth=2.5, markersize=8, label='Geometric-only DA-BSP')
    axes[1, 1].set_title("Pose Error vs. Scene Class Count M")
    axes[1, 1].set_xlabel("Actual Classes Present (M)")
    axes[1, 1].set_ylabel("Final Pose Error (meters)")
    axes[1, 1].set_xticks(classes)
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend()

    plt.suptitle(
        "Actual-Class Diversity Study with a Fixed Five-Class Belief (M=1..5)",
        fontsize=16,
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "tractability_table_graph.png"), dpi=300)
    plt.close()

def plot_accuracy_vs_classes(multi_env_data: dict, output_dir: str):
    """Plot pose error against the number of actual classes present."""
    classes = sorted(list(multi_env_data.keys()))
    vp_means, vp_stds = [], []
    pas_means, pas_stds = [], []

    for M in classes:
        env_dict = multi_env_data[M]
        vp_environment_means = []
        pas_environment_means = []

        for env_id, data in env_dict.items():
            vp_res = data["modes"]["viewpoint_dependent"]
            pas_res = data["modes"]["geometric_only"]
            vp_environment_means.append(np.mean(vp_res["max_weight_err"][:, -1]))
            pas_environment_means.append(np.mean(pas_res["max_weight_err"][:, -1]))

        vp_mean, vp_ci = mean_and_95_ci(vp_environment_means)
        pas_mean, pas_ci = mean_and_95_ci(pas_environment_means)
        vp_means.append(vp_mean)
        vp_stds.append(vp_ci)
        pas_means.append(pas_mean)
        pas_stds.append(pas_ci)

    plt.figure(figsize=(9, 5.5))
    plt.rcParams.update({'font.size': 12})

    plt.errorbar(classes, vp_means, yerr=vp_stds, fmt='-o', color='#1f77b4',
                 linewidth=2.5, markersize=8, capsize=5, label='Viewpoint-Dependent Semantic SLAM')
    plt.errorbar(classes, pas_means, yerr=pas_stds, fmt='--s', color='#ff7f0e',
                 linewidth=2.5, markersize=8, capsize=5, label='Geometric-only DA-BSP')

    num_environments = len(next(iter(multi_env_data.values())))
    num_trials = next(iter(next(iter(multi_env_data.values())).values()))["runner"].config.num_trials
    plt.title(f"Accuracy vs. Actual Classes Present M ({num_environments} Envs x {num_trials} Trials)")
    plt.xlabel("Actual Classes Present (M)")
    plt.ylabel("Final Pose Estimation Error (meters)")
    plt.xticks(classes)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_vs_actual_classes.png"), dpi=300)
    plt.close()

def plot_metrics_per_m(multi_env_data: dict, output_dir: str):
    """Generates 1x5 subplots for DA Entropy H(w) and det(Sigma) for each M across environments."""
    classes = sorted(list(multi_env_data.keys()))
    sample_env = multi_env_data[classes[0]][0]["modes"]["viewpoint_dependent"]["da_entropy"]
    steps = np.arange(sample_env.shape[1])

    # 1. DA Entropy H(w)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=True)
    for idx, M in enumerate(classes):
        ax = axes[idx]
        vp_ent_list = [data["modes"]["viewpoint_dependent"]["da_entropy"] for data in multi_env_data[M].values()]
        pas_ent_list = [data["modes"]["geometric_only"]["da_entropy"] for data in multi_env_data[M].values()]

        vp_by_environment = np.array([np.mean(values, axis=0) for values in vp_ent_list])
        pas_by_environment = np.array([np.mean(values, axis=0) for values in pas_ent_list])
        vp_m, vp_s = mean_and_95_ci(vp_by_environment, axis=0)
        pas_m, pas_s = mean_and_95_ci(pas_by_environment, axis=0)

        ax.plot(steps, vp_m, 'b-', linewidth=2.0, label='Viewpoint-Dep')
        ax.fill_between(steps, vp_m - vp_s, vp_m + vp_s, color='blue', alpha=0.15)
        ax.plot(steps, pas_m, 'r--', linewidth=2.0, label='Geometric-only')
        ax.fill_between(steps, pas_m - pas_s, pas_m + pas_s, color='red', alpha=0.15)

        ax.set_title(f"M = {M} Actual Classes")
        ax.set_xlabel("Time Step (k)")
        if idx == 0:
            ax.set_ylabel("DA Weight Entropy H(w)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9)

    fig.suptitle(
        "DA Ambiguity vs. Actual Scene Classes (Fixed Five-Class Belief)",
        x=0.5, y=0.98, fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(output_dir, "da_entropy_comparison_all_M.png"), dpi=300)
    plt.close(fig)

    # 2. Position Covariance Det det(Sigma)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=True)
    for idx, M in enumerate(classes):
        ax = axes[idx]
        vp_det_list = [data["modes"]["viewpoint_dependent"]["cov_det"] for data in multi_env_data[M].values()]
        pas_det_list = [data["modes"]["geometric_only"]["cov_det"] for data in multi_env_data[M].values()]

        vp_concat = np.concatenate(vp_det_list, axis=0)
        pas_concat = np.concatenate(pas_det_list, axis=0)

        vp_m = np.mean(vp_concat, axis=0)
        pas_m = np.mean(pas_concat, axis=0)

        ax.plot(steps, vp_m, 'b-', linewidth=2.0, label='Viewpoint-Dep')
        ax.plot(steps, pas_m, 'r--', linewidth=2.0, label='Geometric-only')
        ax.set_yscale('log')
        ax.set_title(f"M = {M} Actual Classes")
        ax.set_xlabel("Time Step (k)")
        if idx == 0:
            ax.set_ylabel("det(Σ_pos) [log scale]")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9)

    fig.suptitle(
        "Localization Uncertainty vs. Actual Scene Classes (Fixed Five-Class Belief)",
        x=0.5, y=0.98, fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(
        os.path.join(output_dir, "pose_covariance_det_comparison_all_M.png"),
        dpi=300,
    )
    plt.close(fig)

def plot_trajectories_for_each_m(multi_env_data: dict, output_dir: str):
    """
    Generates trajectory visualizations for each actual scene class count M.
    """
    class_colors = ['#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf']

    for M, env_dict in multi_env_data.items():
        sample_env_id = 0
        data = env_dict[sample_env_id]
        runner = data["runner"]
        objects = runner.env.objects
        gt_tracks = runner.gt_tracks

        vp_res = data["modes"]["viewpoint_dependent"]
        pas_res = data["modes"]["geometric_only"]

        vp_errs = vp_res["max_weight_err"][:, -1]
        best_trial_idx = int(np.argmin(vp_errs))
        median_trial_idx = int(np.argsort(vp_errs)[len(vp_errs) // 2])

        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        plt.rcParams.update({'font.size': 11})

        def plot_landmarks(ax):
            for obj in objects:
                c_color = class_colors[obj.gt_class % len(class_colors)]
                ax.plot(obj.gt_pose[0], obj.gt_pose[1], 'k*', markersize=14, zorder=5)
                ax.plot(obj.gt_pose[0], obj.gt_pose[1], 'o', color=c_color, markersize=8, zorder=6,
                        label=f'Obj {obj.id} (Class {obj.gt_class})')

        # Subplot 1: Best Trial
        ax = axes[0]
        plot_landmarks(ax)
        gt_best = gt_tracks[best_trial_idx][0]
        vp_best = vp_res["est_trajectories"][best_trial_idx]
        pas_best = pas_res["est_trajectories"][best_trial_idx]

        ax.plot(gt_best[:, 0], gt_best[:, 1], 'k--', linewidth=2.5, label='Ground Truth Path')
        ax.plot(vp_best[:, 0], vp_best[:, 1], 'b-o', linewidth=2.0, label='Viewpoint-Dep (Best)')
        ax.plot(pas_best[:, 0], pas_best[:, 1], 'r--s', linewidth=2.0, label='Geometric-only (Best)')
        ax.set_title(f"M={M} Actual Classes: Best Trial (Env #0, Trial #{best_trial_idx})")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.grid(True, linestyle="--", alpha=0.5)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=9)

        # Subplot 2: Median Trial
        ax = axes[1]
        plot_landmarks(ax)
        gt_med = gt_tracks[median_trial_idx][0]
        vp_med = vp_res["est_trajectories"][median_trial_idx]
        pas_med = pas_res["est_trajectories"][median_trial_idx]

        ax.plot(gt_med[:, 0], gt_med[:, 1], 'k--', linewidth=2.5, label='Ground Truth Path')
        ax.plot(vp_med[:, 0], vp_med[:, 1], 'b-o', linewidth=2.0, label='Viewpoint-Dep (Median)')
        ax.plot(pas_med[:, 0], pas_med[:, 1], 'r--s', linewidth=2.0, label='Geometric-only (Median)')
        ax.set_title(f"M={M} Actual Classes: Median Trial (Env #0, Trial #{median_trial_idx})")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.grid(True, linestyle="--", alpha=0.5)

        # Subplot 3: Combined Overlay
        ax = axes[2]
        plot_landmarks(ax)
        num_t = len(gt_tracks)
        for i in range(num_t):
            vp_path = vp_res["est_trajectories"][i]
            pas_path = pas_res["est_trajectories"][i]

            lbl_vp = 'Viewpoint-Dep (All Trials)' if i == 0 else ""
            lbl_pas = 'Geometric-only (All Trials)' if i == 0 else ""

            ax.plot(vp_path[:, 0], vp_path[:, 1], 'b-', alpha=0.3, label=lbl_vp)
            ax.plot(pas_path[:, 0], pas_path[:, 1], 'r--', alpha=0.2, label=lbl_pas)

        ax.set_title(f"M={M} Actual Classes: Combined Overlay ({num_t} Trials)")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.grid(True, linestyle="--", alpha=0.5)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"trajectory_visualization_M{M}.png"), dpi=300)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Multi-Class Viewpoint-Dependent Semantic SLAM Study")
    parser.add_argument("--num-environments", type=int, default=20, help="Number of unique environments (Default: 20)")
    parser.add_argument("--trials", type=int, default=50, help="Number of Monte Carlo trials per environment (Default: 50)")
    parser.add_argument("--steps", type=int, default=10, help="Number of time steps per trajectory (Default: 10)")
    parser.add_argument("--num-objects", type=int, default=6, help="Number of objects N (Default: 6)")
    parser.add_argument("--num-samples", type=int, default=1000, help="Pose samples Ns for weight calculation (Default: 1000)")
    parser.add_argument("--pruning-ratio", type=float, default=150.0, help="Pruning ratio threshold max_w / ratio (Default: 150.0)")
    parser.add_argument("--max-hypotheses", type=int, default=100,
                        help="Beam cap after relative-weight pruning (Default: 100). Use 0 to disable.")
    parser.add_argument("--semantic-alpha", type=float, default=0.25,
                        help="Viewpoint dependence alpha from paper Eq. (18).")
    parser.add_argument("--semantic-k", type=float, default=15.0,
                        help="Semantic precision K used by the paper.")
    parser.add_argument("--sensor-range", type=float, default=6.0,
                        help="Maximum object detection range in metres.")
    parser.add_argument("--sensor-fov-deg", type=float, default=360.0,
                        help="Object detector field of view in degrees.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel trial worker processes. Use 1 for serial execution.")

    args = parser.parse_args()

    if plt is None:
        parser.error("matplotlib is required; install dependencies with 'python -m pip install -r requirements.txt'")
    if args.workers < 1:
        parser.error("--workers must be at least one")
    if args.max_hypotheses == 0:
        args.max_hypotheses = None

    print(f"=== Multi-Class Semantic SLAM Study ({args.num_environments} Unique Envs x {args.trials} Trials) ===")

    output_dir = setup_results_dir()

    config = configure_hyperparameters(
        num_objects=args.num_objects,
        num_model_classes=5,
        num_trials=args.trials,
        num_steps=args.steps,
        num_samples=args.num_samples,
        pruning_ratio=args.pruning_ratio,
        max_hypotheses=args.max_hypotheses,
        semantic_alpha=args.semantic_alpha,
        semantic_k=args.semantic_k,
        sensor_range=args.sensor_range,
        sensor_fov=np.deg2rad(args.sensor_fov_deg),
    )

    # 1. Run multi-environment study across 20 environments and M in [1, 2, 3, 4, 5]
    print("Starting fixed-five-class-belief experiment suite (v9.0.0)")
    print("M denotes actual classes present; the belief always models 5 classes")
    print("Results will be versioned and cached in results/data_v9")

    multi_env_data = ExperimentRunner.run_multi_environment_study(
        base_config=config,
        num_environments=args.num_environments,
        trials_per_env=args.trials,
        actual_class_counts=[1, 2, 3, 4, 5],
        modes=["viewpoint_dependent", "geometric_only"],
        workers=args.workers,
    )

    # 2. Generate publication graphics & plots
    print("\n[Graphics] Saving publication plots to 'results/'...")
    plot_tractability_table_graph(multi_env_data, output_dir)
    plot_accuracy_vs_classes(multi_env_data, output_dir)
    plot_metrics_per_m(multi_env_data, output_dir)
    plot_trajectories_for_each_m(multi_env_data, output_dir)

    print(f"\nExecution complete! Versioned raw data saved to 'results/data_v9/'. Artifacts written to: {output_dir}")

if __name__ == "__main__":
    main()
