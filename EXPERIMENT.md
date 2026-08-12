# Research Question and Hypothesis

This project studies whether semantic observations from a viewpoint-dependent
classifier improve simultaneous localization and mapping (SLAM) when data
association (DA) is uncertain. It follows the setting of *Data Association
Aware Semantic Mapping and Localization via a Viewpoint-Dependent Classifier
Model*, while extending the class-realization experiment. In each environment,
all $N=6$ mapped objects have the same ground-truth class, but inference treats
each object's class as independently unknown. The number of candidate classes
$M$ is varied from 1 through 5. The central question is
whether explicitly maintaining uncertainty over both class realization and DA
remains tractable as $M$ grows, and whether viewpoint-dependent semantics
provide an advantage over passive, geometry-only DA-SLAM.

We hypothesize that the viewpoint-dependent semantic model will reduce robot
pose and data-association uncertainty relative to the geometric-only baseline,
especially in geometrically ambiguous parts of a track. We also expect the
number of active discrete hypotheses to grow with the number of candidate class
realizations, but that likelihood-based pruning will retain a manageable subset
of the nominal hypothesis space.

# Implementation and Experimental Setup

The corrected implementation represents a hybrid belief: discrete hypotheses
cover object-class realizations and the sequence of data associations, while
each hypothesis carries an EKF approximation to its continuous robot and object
state. This EKF is an approximation to the paper's per-hypothesis factor-graph
belief, and should be interpreted accordingly. At every observation, candidate
associations are scored using the geometric measurement likelihood, the
visibility/observation model, and—when enabled—the semantic likelihood. The
resulting components are normalized and pruned using the paper-style
maximum-weight ratio of 150, followed by a documented top-100 beam cap. Within
one time step, DA assignments are constrained to be injective because the
simulator produces at most one detection per physical object. Classes of
objects that have not yet been observed under a DA hypothesis remain collapsed
under their uniform prior.

The active method uses a viewpoint-dependent classifier. For $M=2$, its mean
semantic observation follows the paper's Equation 18, including the intended
view-angle convention and parameterization. For $M>2$, the implementation
uses a documented symmetric extension that distributes Equation 18's
viewpoint-dependent error mass uniformly among the other $M-1$ classes. Semantic
measurements are generated and evaluated with the same model and covariance.
The baseline is passive DA-BSP: it uses only geometry and the observation model
for inference, not a viewpoint-independent semantic substitute.

For each value of $M\in\{1,2,3,4,5\}$, the experiment uses 20 randomly
generated environments, 50 robot tracks per environment, 10 time steps per
track, and 1,000 samples for Monte Carlo marginal-likelihood estimation. The
same process, pose, object, and range-bearing noise assumptions are used across
the active and passive methods so that their comparison isolates the effect of
semantic information. Independent `(mode, trial)` jobs may be distributed
across worker processes; fixed per-trial random seeds make scientific outputs
independent of scheduling order.

# Evaluation Metrics

Following the paper, evaluation reports entropy of the DA posterior after
marginalizing class realizations, the determinant of the highest-weight robot
position covariance, Euclidean pose error for the highest-weight component,
and posterior-weighted average pose error. The implementation additionally
records the number of active joint class/DA hypotheses and runtime per step as
tractability diagnostics. Runtime is not by itself evidence of theoretical
scaling. Results are stratified by $M$ for both methods.

# Results

Corrected numerical results are pending a complete rerun of the experiment.
They must not be replaced by values from the repository's existing v1--v7
artifacts: those legacy runs used a mathematically different formulation,
including ground-truth class initialization and an incompatible semantic model.
Consequently, their plotted errors, hypothesis counts, and runtime measurements
are not valid results for this corrected class-realization study.

Once rerun, the primary result should compare active and passive uncertainty
curves across $M=1\ldots5$, together with active-hypothesis counts after
pruning. Any claim that pruning makes the corrected model tractable should be
grounded in these new measurements, rather than in the nominal $M^N$ count or
the old fixed component cap.

# Interpretation and Limitations

A lower posterior uncertainty for the active method would support the claim
that viewpoint-aware semantic predictions help resolve association ambiguity.
An increase in active hypotheses with $M$ would be expected; the important
question is whether pruning preserves useful posterior mass while keeping the
actual number of maintained components practical. Although all six objects have
the same ground-truth class, the estimator does not know that constraint and
represents independent object classes, giving the nominal $M^N$ realization
space studied by the paper.

The conclusions are limited by the EKF continuous-state approximation,
factor-by-factor threshold pruning within multi-detection time steps, the
synthetic sensor and visibility models, finite Monte Carlo sampling, and the
symmetric multiclass extension, which is not an empirical classifier supplied
by the original paper. The Equation 18 comparison is exact only for two
classes; results for $M>2$ evaluate the stated extension. A future study could
use incremental factor-graph smoothing, real calibrated classifier data, and
incremental class-hypothesis merging to test larger regimes.
