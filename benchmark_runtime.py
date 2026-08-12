"""Benchmark one production-size trial for every class count and mode."""

import argparse
import time

from src.environment import Environment, EnvironmentConfig
from src.experiment import _run_trial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seconds", type=float, default=300.0)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--max-hypotheses", type=int, default=100)
    args = parser.parse_args()

    failures = []
    print("M,mode,seconds", flush=True)
    for num_classes in range(1, 6):
        config = EnvironmentConfig(
            num_objects=6,
            num_classes_in_scene=1,
            num_classes_in_model=num_classes,
            num_trials=1,
            num_steps=10,
            num_samples=args.num_samples,
            pruning_ratio=150.0,
            max_hypotheses=args.max_hypotheses,
        )
        environment = Environment(config, seed=100)
        track = environment.generate_tracks()[0]
        for mode in ("viewpoint_dependent", "geometric_only"):
            start = time.perf_counter()
            _, _, result = _run_trial(
                config, environment.objects, track, mode, trial_idx=0
            )
            elapsed = time.perf_counter() - start
            measured = float(result["trial_time"][0])
            print(f"{num_classes},{mode},{measured:.3f}", flush=True)
            if elapsed > args.max_seconds:
                failures.append((num_classes, mode, elapsed))

    if failures:
        details = ", ".join(
            f"M={m} {mode}: {seconds:.1f}s"
            for m, mode, seconds in failures
        )
        raise SystemExit(
            f"runtime budget of {args.max_seconds:.1f}s exceeded: {details}"
        )


if __name__ == "__main__":
    main()
