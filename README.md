# Vision-Aided Navigation Final Project

This repository implements an EKF-based approximation to *Data Association
Aware Semantic Mapping and Localization via a Viewpoint-Dependent Classifier
Model*. It compares:

- viewpoint-dependent semantic inference using the paper's Eq. (18) when
  `M=2`; and
- the paper's geometric-only passive DA-BSP baseline.

The extension varies the number of candidate classes `M=1..5` for six
same-ground-truth-class objects. Object classes are latent: the engine lazily
expands the uniform class prior when an object first becomes relevant under a
data-association hypothesis. For `M>2`, the paper does not specify a classifier
model, so the code uses a documented symmetric extension that distributes Eq.
(18)'s error mass uniformly among the other classes.

## Run

```bash
python -m pip install -r requirements.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python run_experiments.py --num-environments 20 --trials 50 --steps 10 \
  --num-samples 1000 --pruning-ratio 150 --workers 4
```

Trials are independent and can run in parallel worker processes. Start with
`--workers 2` or `--workers 4`; each worker maintains its own hypothesis trees,
so memory use grows approximately linearly with the worker count. Use
`--workers 1` for deterministic debugging and memory-constrained machines. The
thread-limit environment variables prevent each process from starting its own
pool of BLAS threads.

The default `--max-hypotheses 100` beam cap is applied after the paper's
relative-weight pruning and bounds per-trial runtime. Pass `--max-hypotheses 0`
only for uncapped research runs; ambiguous six-object cases can otherwise grow
too large to finish. Corrected outputs are cached under `results/data_v8`; the
cache identity includes the full configuration and a source-code fingerprint.
If finite-sample beam pruning leaves no candidate inside the hard visibility
gate, the engine records a `visibility_recoveries` event and falls back to the
finite geometric/semantic measurement likelihood for that observation. This
prevents a rare particle-support collapse from aborting a multi-hour study.

Run the dependency-free test suite with:

```bash
python -m unittest discover -s tests -v
```

Verify that one production-size trial for every `M` and mode stays below five
minutes on a target machine with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python benchmark_runtime.py --max-seconds 300
```

After all 100 environment/class checkpoints finish, generate an aggregate CSV,
paired analysis, Markdown report, and publication figures with:

```bash
python analyze_results.py
```

Confidence intervals in this post-processing step use the 20 environment means
as independent units rather than treating the 1,000 correlated trials as
independent replicates.

## Scope and provenance

The continuous belief uses one EKF per discrete hypothesis rather than the
paper's GTSAM/iSAM2 factor-graph smoother. See [EXPERIMENT.md](EXPERIMENT.md)
for the experiment design and limitations.

Files under `results/data_v1` through `results/data_v7` are legacy artifacts
from an earlier, incompatible formulation. They must not be reported as results
of the bounded experiment; v8 results require a fresh run.
