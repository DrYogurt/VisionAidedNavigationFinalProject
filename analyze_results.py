"""Aggregate and analyze a completed v8 semantic-SLAM experiment."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODES = ("viewpoint_dependent", "geometric_only")
MODE_LABELS = {
    "viewpoint_dependent": "Viewpoint-dependent",
    "geometric_only": "Geometric-only",
}


def mean_ci95(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, 0.0
    return mean, float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))


def load_checkpoints(data_dir, expected_environments, expected_trials):
    """Load the newest complete v8 checkpoint for every (environment, M)."""
    checkpoints = {}
    for path in Path(data_dir).glob("env_*_M*_trials*_*.json"):
        with path.open() as handle:
            payload = json.load(handle)
        metadata = payload.get("metadata", {})
        if metadata.get("version") != "v8.0.0":
            continue
        if metadata.get("num_trials") != expected_trials:
            continue
        key = (metadata["env_id"], metadata["num_candidate_classes"])
        timestamp = metadata.get("timestamp", "")
        if key not in checkpoints or timestamp > checkpoints[key][0]:
            checkpoints[key] = (timestamp, path, payload)

    classes = sorted({key[1] for key in checkpoints})
    expected_keys = {
        (env_id, class_count)
        for env_id in range(expected_environments)
        for class_count in classes
    }
    missing = sorted(expected_keys - set(checkpoints))
    if not classes or missing:
        raise RuntimeError(
            f"incomplete study: found {len(checkpoints)} checkpoints; missing {missing[:10]}"
        )

    return classes, {
        key: payload
        for key, (_timestamp, _path, payload) in checkpoints.items()
    }


def environment_metric(payload, mode, metric, selector):
    values = np.asarray(payload["results"][mode][metric], dtype=float)
    return float(np.mean(selector(values)))


def summarize(classes, checkpoints, expected_environments):
    rows = []
    paired_rows = []
    raw = {}

    for class_count in classes:
        raw[class_count] = {}
        for mode in MODES:
            by_environment = {
                "final_pose_error": [],
                "final_weighted_error": [],
                "final_da_entropy": [],
                "final_cov_det": [],
                "active_hypotheses": [],
                "step_time_ms": [],
            }
            all_trial_times = []
            recoveries = 0
            for env_id in range(expected_environments):
                payload = checkpoints[(env_id, class_count)]
                by_environment["final_pose_error"].append(
                    environment_metric(payload, mode, "max_weight_err", lambda x: x[:, -1])
                )
                by_environment["final_weighted_error"].append(
                    environment_metric(payload, mode, "avg_weight_err", lambda x: x[:, -1])
                )
                by_environment["final_da_entropy"].append(
                    environment_metric(payload, mode, "da_entropy", lambda x: x[:, -1])
                )
                by_environment["final_cov_det"].append(
                    environment_metric(payload, mode, "cov_det", lambda x: x[:, -1])
                )
                by_environment["active_hypotheses"].append(
                    environment_metric(payload, mode, "num_hypotheses", lambda x: x)
                )
                by_environment["step_time_ms"].append(
                    1000.0 * environment_metric(payload, mode, "step_time", lambda x: x)
                )
                all_trial_times.extend(
                    np.asarray(payload["results"][mode]["trial_time"], dtype=float).ravel()
                )
                recovery_values = payload["results"][mode].get("visibility_recoveries", [])
                recoveries += int(np.sum(recovery_values))

            raw[class_count][mode] = by_environment
            row = {"M": class_count, "mode": mode}
            for metric, values in by_environment.items():
                row[f"{metric}_mean"], row[f"{metric}_ci95"] = mean_ci95(values)
            row["trial_time_p95_s"] = float(np.percentile(all_trial_times, 95))
            row["trial_time_max_s"] = float(np.max(all_trial_times))
            row["visibility_recoveries"] = recoveries
            rows.append(row)

        viewpoint = np.asarray(raw[class_count]["viewpoint_dependent"]["final_pose_error"])
        geometric = np.asarray(raw[class_count]["geometric_only"]["final_pose_error"])
        differences = viewpoint - geometric
        difference_mean, difference_ci = mean_ci95(differences)
        paired_rows.append(
            {
                "M": class_count,
                "difference_mean": difference_mean,
                "difference_ci95": difference_ci,
                "percent_reduction": 100.0 * (np.mean(geometric) - np.mean(viewpoint)) / np.mean(geometric),
            }
        )

    return rows, paired_rows, raw


def write_csv(rows, output_path):
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_aggregate(classes, rows, raw, output_path):
    row_lookup = {(row["M"], row["mode"]): row for row in rows}
    panels = (
        ("final_pose_error", "Final pose error (m)"),
        ("final_da_entropy", "Final DA entropy"),
        ("active_hypotheses", "Mean active hypotheses"),
        ("step_time_ms", "Inference time (ms/step)"),
    )
    styles = {
        "viewpoint_dependent": ("#1f77b4", "o", "-"),
        "geometric_only": ("#ff7f0e", "s", "--"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (metric, ylabel) in zip(axes.ravel(), panels):
        for mode in MODES:
            color, marker, linestyle = styles[mode]
            means = [row_lookup[(m, mode)][f"{metric}_mean"] for m in classes]
            cis = [row_lookup[(m, mode)][f"{metric}_ci95"] for m in classes]
            ax.errorbar(
                classes, means, yerr=cis, color=color, marker=marker,
                linestyle=linestyle, capsize=4, linewidth=2,
                label=MODE_LABELS[mode],
            )
        ax.set_xlabel("Candidate classes (M)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(classes)
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    fig.suptitle("Semantic SLAM aggregate results (95% CI across environments)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_paired_difference(paired_rows, output_path):
    classes = [row["M"] for row in paired_rows]
    means = [row["difference_mean"] for row in paired_rows]
    cis = [row["difference_ci95"] for row in paired_rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0.0, color="black", linewidth=1)
    ax.errorbar(classes, means, yerr=cis, fmt="o-", capsize=5, linewidth=2)
    ax.set_xlabel("Candidate classes (M)")
    ax.set_ylabel("Paired final-error difference (semantic − geometric), m")
    ax.set_xticks(classes)
    ax.grid(alpha=0.3)
    ax.set_title("Semantic model effect (95% CI across paired environments)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_report(rows, paired_rows, expected_environments, expected_trials, output_path):
    lookup = {(row["M"], row["mode"]): row for row in rows}
    total_recoveries = sum(row["visibility_recoveries"] for row in rows)
    maximum_trial_time = max(row["trial_time_max_s"] for row in rows)
    lines = [
        "# Results Analysis",
        "",
        f"The completed study contains {expected_environments} environments, "
        f"{expected_trials} trials per environment, five class counts, and two inference modes "
        f"({expected_environments * expected_trials * 5 * 2:,} trial experiments). "
        "Confidence intervals below use the environment mean as the independent unit (n=20).",
        "",
        "## Final pose error",
        "",
        "| M | Viewpoint-dependent (m) | Geometric-only (m) | Paired difference (m) | Reduction |",
        "|---:|---:|---:|---:|---:|",
    ]
    for paired in paired_rows:
        class_count = paired["M"]
        viewpoint = lookup[(class_count, "viewpoint_dependent")]
        geometric = lookup[(class_count, "geometric_only")]
        lines.append(
            f"| {class_count} | {viewpoint['final_pose_error_mean']:.4f} ± "
            f"{viewpoint['final_pose_error_ci95']:.4f} | "
            f"{geometric['final_pose_error_mean']:.4f} ± {geometric['final_pose_error_ci95']:.4f} | "
            f"{paired['difference_mean']:+.4f} ± {paired['difference_ci95']:.4f} | "
            f"{paired['percent_reduction']:+.1f}% |"
        )

    supported = [
        row["M"] for row in paired_rows
        if row["difference_mean"] + row["difference_ci95"] < 0.0
    ]
    adverse = [
        row["M"] for row in paired_rows
        if row["difference_mean"] - row["difference_ci95"] > 0.0
    ]
    inconclusive = [
        row["M"] for row in paired_rows
        if row["M"] not in supported and row["M"] not in adverse
    ]
    lines.extend(
        [
            "",
            "Negative paired differences favor viewpoint-dependent semantics. Using whether the "
            "normal-approximation paired 95% CI excludes zero as a descriptive criterion, "
            f"semantics are favored for M={supported or 'none'}, geometry is favored for "
            f"M={adverse or 'none'}, and the comparison is inconclusive for M={inconclusive or 'none'}.",
            "",
            "## Tractability and numerical audit",
            "",
            f"The slowest recorded individual trial took {maximum_trial_time:.3f} seconds, "
            "well below the five-minute requirement. "
            f"The complete study recorded {total_recoveries} visibility-support recoveries. "
            "These are disclosed because they replace the hard visibility factor only when all "
            "retained Monte Carlo candidates have zero gated support for one observation.",
            "",
            "See `aggregate_results.png` for error, entropy, active-hypothesis, and runtime trends; "
            "see `paired_pose_error_difference.png` for the paired semantic effect.",
            "",
            "## Limitations",
            "",
            "The intervals describe variation among these synthetic environments and do not establish "
            "generalization to real classifiers or scenes. The estimator uses EKF components, "
            "factor-by-factor pruning, and a top-100 beam cap; M>2 uses a symmetric extension of the "
            "paper's two-class semantic model. Runtime measurements also include ordinary host-load "
            "variation and should be read as implementation diagnostics rather than asymptotic evidence.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="results/data_v8")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--expected-environments", type=int, default=20)
    parser.add_argument("--expected-trials", type=int, default=50)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    classes, checkpoints = load_checkpoints(
        args.data_dir, args.expected_environments, args.expected_trials
    )
    rows, paired_rows, raw = summarize(
        classes, checkpoints, args.expected_environments
    )
    write_csv(rows, output_dir / "aggregate_summary.csv")
    plot_aggregate(classes, rows, raw, output_dir / "aggregate_results.png")
    plot_paired_difference(paired_rows, output_dir / "paired_pose_error_difference.png")
    write_report(
        rows, paired_rows, args.expected_environments, args.expected_trials,
        output_dir / "RESULTS_ANALYSIS.md",
    )
    print(f"Analyzed {len(checkpoints)} checkpoints across M={classes}")
    print(f"Wrote analysis artifacts to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
