# Results Analysis

The completed study contains 20 environments, 50 trials per environment, five actual scene-class counts, and two inference modes (10,000 trial experiments). M is the number of distinct ground-truth classes present; every estimator belief models five candidate classes. Confidence intervals below use the environment mean as the independent unit (n=20).

## Final pose error

| M | Viewpoint-dependent (m) | Geometric-only (m) | Paired difference (m) | Reduction |
|---:|---:|---:|---:|---:|
| 1 | 0.6526 ± 0.0557 | 0.7777 ± 0.1001 | -0.1251 ± 0.0632 | +16.1% |
| 2 | 0.6541 ± 0.0564 | 0.7777 ± 0.1001 | -0.1237 ± 0.0668 | +15.9% |
| 3 | 0.6813 ± 0.0698 | 0.7777 ± 0.1001 | -0.0965 ± 0.0556 | +12.4% |
| 4 | 0.6633 ± 0.0665 | 0.7777 ± 0.1001 | -0.1144 ± 0.0626 | +14.7% |
| 5 | 0.6889 ± 0.0686 | 0.7777 ± 0.1001 | -0.0888 ± 0.0529 | +11.4% |

Negative paired differences favor viewpoint-dependent semantics. Using whether the normal-approximation paired 95% CI excludes zero as a descriptive criterion, semantics are favored for M=[1, 2, 3, 4, 5], geometry is favored for M=none, and the comparison is inconclusive for M=none.

## Tractability and numerical audit

The slowest recorded individual trial took 9.963 seconds, well below the five-minute requirement. The complete study recorded 0 visibility-support recoveries. These are disclosed because they replace the hard visibility factor only when all retained Monte Carlo candidates have zero gated support for one observation.

See `aggregate_results.png` for error, entropy, active-hypothesis, and runtime trends; see `paired_pose_error_difference.png` for the paired semantic effect.

## Limitations

The intervals describe variation among these synthetic environments and do not establish generalization to real classifiers or scenes. The estimator uses EKF components, factor-by-factor pruning, and a top-100 beam cap. Its fixed five-class likelihood is a symmetric extension of the paper's two-class semantic model. Runtime measurements also include ordinary host-load variation and should be read as implementation diagnostics rather than asymptotic evidence.
