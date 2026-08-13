# Research Question and Hypothesis

This project asks whether semantic observations from a viewpoint-dependent
classifier improve simultaneous localization and mapping when data association
(DA) is uncertain. It follows *Data Association Aware Semantic Mapping and
Localization via a Viewpoint-Dependent Classifier Model* and extends the
paper's class-realization experiment. Each scene contains six objects with the
same ground-truth class, while inference treats every object's class as an
independent latent variable. The candidate class count is varied over
$M\in\{1,2,3,4,5\}$, corresponding to a nominal $M^6$ class-realization space.

The hypothesis is that viewpoint-dependent semantic evidence will reduce DA
ambiguity, localization uncertainty, and pose error relative to the
geometry-only passive DA-BSP baseline. The original expectation was that more
candidate classes would increase the number of active hypotheses, while
likelihood pruning would keep the represented subset tractable. The completed
study refines that expectation: semantic discrimination causes a sharp drop in
active hypotheses from the no-information $M=1$ control to $M=2$, after which
hypothesis count grows moderately with $M$.

# Implementation and Experimental Setup

The estimator represents a hybrid discrete-continuous belief. Discrete
hypotheses contain object-class assignments and the complete DA history; every
hypothesis carries a joint EKF approximation over the current robot pose and
stationary object poses. This differs from the paper's per-hypothesis
factor-graph/iSAM2 smoother and is an important approximation.

At each observation, candidate associations are scored with the range-bearing
measurement likelihood, a visibility factor, and, for the active method, the
semantic likelihood. Associations within one timestep are injective because
the simulator generates at most one detection per physical object. Classes
remain marginalized under a uniform prior until an object becomes relevant to
an association hypothesis. Components are normalized and pruned by the
paper-style maximum-weight ratio of 150, followed by a documented top-100 beam
cap.

The active method uses the paper's viewpoint-dependent classifier mean. For
$M=2$, the implementation follows Equation 18 with the paper's view-angle
convention, precision, and covariance. For $M>2$, a symmetric extension divides
the Equation 18 error mass uniformly among the other $M-1$ classes. The passive
baseline uses only geometry and visibility; it is not a viewpoint-independent
semantic classifier.

The visibility factor is a hard Monte Carlo indicator. Finite samples and beam
pruning can rarely leave no gated candidate for an observation even though its
measurement likelihood remains finite. In that case the implementation uses
the finite geometric/semantic measurement likelihood without the visibility
gate and records a `visibility_recoveries` event. This path is used only when
all gated candidates are zero.

The final study used 20 independently seeded environments, 50 tracks per
environment, 10 timesteps per track, six objects, 1,000 marginal-likelihood
samples, and both inference modes for every $M$. This yields
$20\times50\times5\times2=10{,}000$ trial experiments. `(mode, trial)` jobs ran
in two worker processes with deterministic observation and inference seeds.
Completed `(environment, M)` blocks were cached, allowing the repaired run to
resume without recomputing successful blocks.

# Evaluation Metrics

The primary accuracy metric is the final Euclidean position error of the
highest-weight hypothesis. The implementation also reports posterior-weighted
position error, DA-posterior entropy after marginalizing class realizations,
the determinant of the highest-weight robot position covariance, and the number
of active joint class/DA hypotheses after pruning. Runtime per step and per
trial are implementation tractability diagnostics.

For inference comparisons, each of the 20 environment-level means is treated
as an independent unit. Reported intervals are normal-approximation 95%
confidence intervals across those environment means. The semantic and
geometric errors are also compared as paired environment-level differences.
This avoids treating 1,000 correlated trials as independent replicates.

# Results

The full v8 experiment completed all 100 environment/class checkpoints and all
10,000 trial experiments. Final position error was:

| $M$ | Viewpoint-dependent (m) | Geometric-only (m) | Paired semantic − geometric (m) | Reduction |
|---:|---:|---:|---:|---:|
| 1 | 0.7746 ± 0.1164 | 0.7746 ± 0.1164 | +0.0000 ± 0.0000 | 0.0% |
| 2 | 0.6036 ± 0.0426 | 0.7615 ± 0.1038 | −0.1579 ± 0.0693 | 20.7% |
| 3 | 0.6651 ± 0.0672 | 0.7639 ± 0.1042 | −0.0988 ± 0.0540 | 12.9% |
| 4 | 0.6949 ± 0.0724 | 0.7502 ± 0.0982 | −0.0552 ± 0.0510 | 7.4% |
| 5 | 0.6526 ± 0.0557 | 0.7777 ± 0.1001 | −0.1251 ± 0.0632 | 16.1% |

Negative paired differences favor semantics. The paired 95% interval excludes
zero for $M=2,3,4,5$ and is exactly zero for $M=1$, where there is no class
information. The strongest observed reduction was 20.7% at $M=2$, the class
count defined directly by the paper.

The semantic method also reduced final DA entropy. For $M=2,3,4,5$, its mean
final entropies were 2.069, 2.525, 2.628, and 2.665, compared with 3.793, 3.761,
3.782, and 3.749 for geometry-only inference. Mean active hypothesis counts
were 22.35, 30.26, 31.87, and 32.59 for semantics versus approximately 60 for
the geometric baseline. The $M=1$ control maintained 59.55 hypotheses in both
modes. Thus semantics did not merely improve the selected trajectory; it
concentrated the represented DA posterior and allowed more aggressive
likelihood pruning.

Mean viewpoint-dependent inference time ranged from 109 to 205 ms per step,
while the geometric baseline ranged from 156 to 159 ms per step. The semantic
method was fastest at $M=2$ because it retained far fewer components, then grew
more expensive as $M$ increased. The slowest recorded complete trial took
7.783 seconds, comfortably below the five-minute requirement. Two
visibility-support recoveries were recorded across 10,000 trials: one in the
$M=2$ geometric condition and one in the $M=3$ geometric condition.

The raw checkpoints, aggregate CSV, report, experiment log, and all figures are
available locally under `results/milkjug_full/`. `aggregate_results.png` shows
the main error, entropy, hypothesis-count, and timing trends;
`paired_pose_error_difference.png` shows the paired accuracy effect.

# Interpretation and Limitations

The completed results support the research hypothesis within this simulator.
When at least two classes are modeled, viewpoint-dependent semantics reduce DA
ambiguity and final localization error relative to geometry alone. The exact
match at $M=1$ is a useful control: without discriminative class information,
the active and passive methods produce the same scientific outputs. The large
improvement at $M=2$ most directly supports the paper-aligned comparison. The
continued benefit at $M=3..5$ suggests that the symmetric extension remains
useful in this synthetic setting, although it is not a model supplied or
validated by the paper.

The experiment does not show monotonic accuracy improvement as $M$ increases.
$M$ changes both the class hypothesis space and the multiclass semantic
likelihood; after $M=2$, added class ambiguity partly offsets the available
semantic information. The top-100 cap also means active-hypothesis counts
describe the bounded implementation rather than the full posterior or the
nominal $M^6$ space.

Other limitations are the EKF continuous-state approximation, factor-by-factor
threshold pruning, synthetic geometry and semantic measurements, finite Monte
Carlo marginalization, and the two disclosed visibility-support recoveries.
Normal-approximation intervals summarize variation across 20 generated
environments but do not prove generalization to real scenes. Runtime depends on
the tested hardware and process load and is not evidence of asymptotic scaling.
Future work should use an incremental factor-graph smoother, calibrated real
classifier outputs, more environments, and sensitivity studies over beam size,
sample count, pruning ratio, and recovery policy.
