# Vision-Aided Navigation Final Project

This repository implements an EKF-based approximation to *Data Association
Aware Semantic Mapping and Localization via a Viewpoint-Dependent Classifier
Model*. It compares viewpoint-dependent semantic SLAM with the paper's
geometry-only passive DA-BSP baseline.

## Current v9 experiment

The current research variable is the number of distinct ground-truth object
classes actually present in each environment:

```text
M = num_classes_in_scene ∈ {1, 2, 3, 4, 5}
num_classes_in_model = 5 for every condition
```

There are six objects. Classes are assigned deterministically and as evenly as
possible, so the five conditions contain:

```text
M=1: [0, 0, 0, 0, 0, 0]
M=2: [0, 1, 0, 1, 0, 1]
M=3: [0, 1, 2, 0, 1, 2]
M=4: [0, 1, 2, 3, 0, 1]
M=5: [0, 1, 2, 3, 4, 0]
```

The estimator always considers five candidate classes, giving a fixed nominal
belief class space of `5^6 = 15,625` realizations. Varying `M` therefore changes
only actual scene diversity, not the belief vocabulary size.

The completed v9 run uses 20 environments × 50 trials × 5 scene-class counts ×
2 inference modes, or 10,000 trial experiments. All 100 checkpoints passed the
v9 integrity checks. Results under `results/data_v8` answer a different
question—one actual class with a varying belief vocabulary—and must not be
reused as v9 evidence.

## Completed v9 results

Viewpoint-dependent semantics reduced final pose error relative to the paired
geometric-only baseline for every actual scene-class count. Normal-approximation
95% confidence intervals use the 20 environment means as independent units.

| M | Semantic pose error (m) | Geometric pose error (m) | Paired difference (m) | Reduction |
|---:|---:|---:|---:|---:|
| 1 | 0.6526 ± 0.0557 | 0.7777 ± 0.1001 | -0.1251 ± 0.0632 | 16.1% |
| 2 | 0.6541 ± 0.0564 | 0.7777 ± 0.1001 | -0.1237 ± 0.0668 | 15.9% |
| 3 | 0.6813 ± 0.0698 | 0.7777 ± 0.1001 | -0.0965 ± 0.0556 | 12.4% |
| 4 | 0.6633 ± 0.0665 | 0.7777 ± 0.1001 | -0.1144 ± 0.0626 | 14.7% |
| 5 | 0.6889 ± 0.0686 | 0.7777 ± 0.1001 | -0.0888 ± 0.0529 | 11.4% |

Negative paired differences favor semantics; all five paired intervals exclude
zero. Pose error is not monotonic in scene diversity. In contrast, final DA
entropy decreases from 2.665 at `M=1` to 0.474 at `M=5`, mean active hypotheses
decrease from 32.59 to 4.88, and semantic inference time decreases from 244.3
to 91.1 ms/step. The geometric outputs are identical across `M`, which is the
expected control behavior because that mode ignores class labels. No visibility
recovery occurred, and the slowest individual trial took 9.963 seconds.

- [Full generated analysis](results/milkjug_v9/v9/RESULTS_ANALYSIS.md)
- [Aggregate results](results/milkjug_v9/v9/aggregate_results.png)
- [Paired pose-error effect](results/milkjug_v9/v9/paired_pose_error_difference.png)
- [Accuracy versus actual classes](results/milkjug_v9/v9/accuracy_vs_actual_classes.png)
- [DA entropy](results/milkjug_v9/v9/da_entropy_comparison_all_M.png)
- [Pose covariance](results/milkjug_v9/v9/pose_covariance_det_comparison_all_M.png)
- [Tractability](results/milkjug_v9/v9/tractability_table_graph.png)
- Trajectories: [M=1](results/milkjug_v9/v9/trajectory_visualization_M1.png),
  [M=2](results/milkjug_v9/v9/trajectory_visualization_M2.png),
  [M=3](results/milkjug_v9/v9/trajectory_visualization_M3.png),
  [M=4](results/milkjug_v9/v9/trajectory_visualization_M4.png), and
  [M=5](results/milkjug_v9/v9/trajectory_visualization_M5.png)

## Reproduce the experiment

Create an environment and run the tests:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

Run the full v9 study:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/vision-slam-mpl \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -u run_experiments.py \
  --num-environments 20 --trials 50 --steps 10 --num-objects 6 \
  --num-samples 1000 --pruning-ratio 150 --max-hypotheses 100 \
  --workers 8
```

Independent `(mode, trial)` jobs run in worker processes. Memory use grows
approximately linearly with worker count because every worker owns its
hypothesis trees. Fixed per-trial seeds make scientific outputs independent of
process scheduling. Limiting BLAS threads avoids nested oversubscription.

Completed environment/class blocks are cached under `results/data_v9`. Cache
identity includes every result-affecting configuration value and a source-code
fingerprint. A stopped run resumes at the next incomplete `(environment, M)`
block.

## Analyze results

After all 100 v9 checkpoints complete, regenerate the aggregate CSV, paired
analysis, Markdown report, and summary figures with:

```bash
.venv/bin/python analyze_results.py
```

Raw checkpoints are written to `results/data_v9`; built-in and aggregate v9
figures and reports are written to `results/v9`.

Confidence intervals use the 20 environment means as independent units rather
than treating 1,000 correlated trials as independent replicates. The analyzer
validates that every checkpoint uses five model classes and contains exactly
the requested set of actual classes before producing results.

Benchmark one production-sized trial for every actual scene-class count and
mode with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python benchmark_runtime.py --max-seconds 300
```

## Implementation notes and paper alignment

The engine represents a hybrid belief. Discrete components contain object-class
assignments and data-association histories; each component carries a joint EKF
over the robot and stationary objects. The paper instead uses a per-hypothesis
factor graph with iSAM2, so numerical equivalence should not be claimed.

Within each timestep, associations are injective because the simulator emits
at most one observation per physical object. Unobserved object classes remain
marginalized and are expanded lazily. Components are first pruned by the
paper-style maximum-weight ratio of 150 and then by a documented top-100 beam
cap that bounds runtime.

The belief always has five classes. Consequently, its semantic likelihood is
always the documented five-class symmetric extension that distributes Equation
18's error mass among the other four classes. Even when `M=2` actual classes
are present, this is not the paper's exact two-class likelihood. The baseline
uses geometry and visibility only, with no semantic substitute.

The hard visibility factor is estimated with finite Monte Carlo samples. If
every retained candidate for one observation has zero gated support but finite
geometric/semantic measurement likelihood, the engine falls back to that
measurement likelihood and increments `visibility_recoveries`.

See [EXPERIMENT.md](EXPERIMENT.md) for the research question, complete setup,
metrics, and limitations.

## Archived v8 figures

The figures below are retained for provenance but represent the superseded v8
question (one actual class, variable belief vocabulary). They must not be cited
as results of the current v9 experiment.

- [v8 aggregate results](results/milkjug_full/aggregate_results.png)
- [v8 paired effect](results/milkjug_full/paired_pose_error_difference.png)
- [v8 accuracy](results/milkjug_full/accuracy_vs_num_classes.png)
- [v8 tractability](results/milkjug_full/tractability_table_graph.png)
- [v8 DA entropy](results/milkjug_full/da_entropy_comparison_all_M.png)
- [v8 covariance](results/milkjug_full/pose_covariance_det_comparison_all_M.png)
- [v8 trajectories: M=1](results/milkjug_full/trajectory_visualization_M1.png)
- [v8 trajectories: M=2](results/milkjug_full/trajectory_visualization_M2.png)
- [v8 trajectories: M=3](results/milkjug_full/trajectory_visualization_M3.png)
- [v8 trajectories: M=4](results/milkjug_full/trajectory_visualization_M4.png)
- [v8 trajectories: M=5](results/milkjug_full/trajectory_visualization_M5.png)
