# Research Question and Hypothesis

This project asks how increasing the number of distinct object classes actually
present in a scene affects viewpoint-dependent semantic SLAM when the
estimator's class vocabulary remains fixed. The scene contains six stationary
objects, and the actual class count is varied over
$M\in\{1,2,3,4,5\}$. Every inference condition maintains the same five-class
belief. Thus $M$ denotes environmental class diversity, not the number of
candidate classes in the belief.

The hypothesis is that viewpoint-dependent semantic evidence will reduce data
association (DA) ambiguity, localization uncertainty, and pose error relative
to geometry-only passive DA-BSP, and that its usefulness may change as the six
objects become more class-diverse. Because the model vocabulary and nominal
$5^6$ class-realization space are fixed, differences across $M$ should not be
attributed to changing belief dimensionality.

# Implementation and Experimental Setup

The estimator represents a hybrid discrete-continuous belief. Discrete
hypotheses contain object-class assignments and complete DA histories; every
hypothesis carries a joint EKF approximation over the robot pose and stationary
object poses. This differs from the paper's per-hypothesis factor-graph/iSAM2
smoother and remains an important approximation.

At each observation, candidate associations are scored with the range-bearing
measurement likelihood, a visibility factor, and, for the active method, a
semantic likelihood. Associations within one timestep are injective because
the simulator emits at most one detection per physical object. Object classes
remain marginalized under a uniform five-class prior until they become
relevant to an association hypothesis. Components are normalized and pruned by
the paper-style maximum-weight ratio of 150, followed by a top-100 beam cap.

For every condition, `num_classes_in_model=5`. The semantic observation is
therefore always five-dimensional and uses the documented symmetric extension
of Equation 18, distributing viewpoint-dependent error mass uniformly among
the other four modeled classes. The geometry-only baseline ignores semantic
measurements. Since the paper defines the likelihood for two modeled classes,
none of these fixed-five-class conditions is an exact reproduction of its
two-class observation model.

The actual scene setting is `num_classes_in_scene=M`. Ground-truth classes are
assigned round-robin by object ID, guaranteeing that every class is represented
and making frequencies as even as six objects permit. Across $M=1..5$, the
assignments are `[0,0,0,0,0,0]`, `[0,1,0,1,0,1]`,
`[0,1,2,0,1,2]`, `[0,1,2,3,0,1]`, and `[0,1,2,3,4,0]`.

The full v9 design uses 20 independently seeded environments, 50 tracks per
environment, 10 timesteps per track, six objects, 1,000 Monte Carlo samples,
five actual class counts, and two inference modes. This yields 10,000 trial
experiments. Independent `(mode, trial)` jobs use fixed random seeds. The worker
count is configurable; the current full run on `milkjug` uses eight worker
processes. Completed `(environment, M)` blocks are stored in the incompatible
v9 cache so no v8 result can be loaded accidentally.

The hard visibility factor can rarely lose all retained Monte Carlo support.
When all gated candidates for an observation are zero but the measurement
likelihood is finite, the engine uses the ungated measurement likelihood and
records a `visibility_recoveries` event instead of aborting the study.

# Evaluation Metrics

The primary metric is final Euclidean robot-position error for the
highest-weight hypothesis. Additional metrics are posterior-weighted position
error, entropy of the DA posterior after class marginalization, determinant of
the highest-weight robot position covariance, active joint class/DA hypothesis
count, inference time per step, complete-trial runtime, and visibility recovery
count.

Each environment mean is treated as one independent unit. Results report
normal-approximation 95% confidence intervals across 20 environment means and
paired environment-level semantic-minus-geometric differences. The analyzer
rejects incomplete data, a model class count other than five, or any checkpoint
whose objects do not contain exactly classes $0..M-1$.

# Results

Corrected v9 numerical results are pending completion of the fresh experiment
and analysis. The completed v8 results and figures describe one actual class
with a varying belief class count; they answer a different question and must
not be substituted or pooled with v9.

# Interpretation and Limitations

If viewpoint-dependent inference reduces paired pose error and DA entropy as
actual class diversity increases, that would support the claim that semantic
diversity helps resolve geometric association ambiguity under a fixed belief
model. A flat geometric baseline across $M$ would be expected because object
positions, tracks, noise dimensions, and geometric inference are held fixed;
deviations should be checked for stochastic or implementation effects.

The experiment remains limited by its EKF approximation, factor-by-factor
pruning, top-100 beam cap, synthetic observations, deterministic round-robin
class balance, finite Monte Carlo sampling, and recovery policy. The fixed
five-class semantic likelihood is an extension rather than the paper's exact
two-class model. Results over 20 simulated environments do not establish
generalization to real classifiers or scenes, and runtime is an implementation
diagnostic rather than asymptotic evidence.
