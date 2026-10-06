# Refactor record

Starting commit: `dfd8f921dfec1245680bbac1cf941a30845e48a5`. The starting working tree was clean.

The original suite ran 221 tests: 199 passed, 18 retired-workflow tests were skipped, and four missing-document checks errored. The present suite replaces notebook/manuscript assertions with installed-interface and source-manifest checks.

## Source disposition

The physical model, current design, regression/projection, candidate generation, optimization routes, parallel runtime, audits, and reporting were extracted into their dedicated source packages. The large study runner was split into workflow stages and runtime utilities. Four plotting scripts became callable reporting modules.

Removed code comprises the retired workflow/surrogate/NLP stack, the notebook builder, hardcoded preflight runners, narrative tooling, chart overrides, historical forks/migrations, repeated holdout timing/equivalence sweeps, and unused deterministic search. The original workbook is preserved.

## Test disposition

Retained tests were renamed and imported from their actual module owners. Model/design tests retain physical and random-generator invariants; current projection/active-set/smooth/trust tests retain numerical checks. Runner tests retain fixed-attempt generation, provenance, advisory gates, eleven-case scheduling, exact replay, recovery, timing, and same-run resumption. Reporting tests use current artifact layouts. New tests cover installation, source closure, output placement, old-format rejection, and atomic publication failures.

Deleted tests exercised retired APIs, historical migrations/forks, the absent notebook/manuscript, surrogate continuation, and obsolete multistart searches. No numerical tolerance was relaxed.

## Verification scope

Run unit checks and the real reduced integration workload. Preserve execution logs and reduced-run artifacts beneath `results`. Do not execute the full production workload or claim complete numerical equivalence to an unavailable historical baseline.

## Detailed source inventory

| Original source | Current destination or deletion reason |
|---|---|
| `closed_loop/__init__.py` | Lightweight package initializer; eager legacy workflow import removed |
| `closed_loop/config.py` | Package configuration and central paths |
| `closed_loop/model.py` | Plant definitions, operating point, and exact model |
| `closed_loop/design.py` | SplitMix64 retained; obsolete design APIs removed |
| `closed_loop/manuscript_v3.py` | Design, generation, training, assessment, and physical validation |
| `closed_loop/projection.py` | Regression and physical projection; unused search removed |
| `closed_loop/v3_replacement_generation.py` | Fixed candidate generation and provenance |
| `closed_loop/v3_smooth.py` | Single-start mechanistic optimization and continuation |
| `closed_loop/v3_surrogate_nlp.py` | Expression graph, exact-QP search, result types, and certification; old solver executor removed |
| `closed_loop/v3_active_set.py` | Active-set sensitivities and local refinement |
| `closed_loop/v3_trust.py` | Development-only trust calibration |
| `closed_loop/v3_parallel.py` | Spawn-safe resumable batches |
| `closed_loop/v3_reporting.py` | Current reporting tables; obsolete readers removed |
| `closed_loop/workflow.py`, `surrogate.py`, `nlp.py` | Retired implementation removed; useful assertions transferred |
| `closed_loop/v3_derivative_audit.py` | Unreachable executor removed; live derivatives checked independently in tests |
| `scripts/run_article_v3_5000.py` | Thin CLI, workflow stages, contracts, checkpoints, and reference validation |
| `scripts/run_closed_loop.py` | Compatibility wrapper removed |
| `scripts/build_main_closed_loop_v3.py` | Notebook dependency removed; figures now part of complete execution |
| `scripts/generate_composite_two_route_charts.py` | Eight current comparison figures |
| `scripts/plot_surrogate_emulation_quality.py` | Three emulation figures |
| `scripts/plot_reporting_insights.py` | Five insight figures using current tables |
| `scripts/plot_nominal_optimum_parity.py` | Two selected-decision parity figures |
| `scripts/audit_article_narrative_values.py` | Historical narrative tooling removed |
| `scripts/materialize_r1_smooth_nlp_chart_candidate.py` | Archived chart override removed |
| `scripts/reinterpret_article_v3_no_minimum_srt.py` | Historical readjudication removed |
| `scripts/resume_v3_preflight.py`, `run_v3_optimization_phase.py` | Hardcoded preflight workflows removed |

## Detailed test inventory

| Original test group | Current coverage |
|---|---|
| Closed-loop model/design | Mechanistic model and sampling tests |
| Closed-loop surrogate/NLP | Useful assertions covered by current projection, conditioning, mechanistic optimization, and derivatives |
| Closed-loop workflow | Atomic publication tests; retired skips and bundle API removed |
| Notebook/manuscript | Package/path checks and current assessment tests |
| Article runner contracts/optimization | Run contracts, generation, assessment, and optimization pipeline |
| Article run forks | Removed; old-format/source mismatch rejection remains covered |
| Projection/active-set | Current projection and active-set checks |
| Parallel/replacement generation | Spawn-safe execution and fixed candidate generation |
| Reporting/smooth/SRT/conditioning/trust | Current reporting, mechanistic optimization, engineering eligibility, conditioning, and trust tests |

## Completed verification

- 134 tests passed with no skips: 133 unit tests and one real reduced-workload integration test.
- The real run attempted 80 development and 20 holdout candidates. It accepted 76 and 17 respectively, retaining every rejection without replacement.
- Both routes were attempted across the nominal case and all ten robustness cases.
- Results are in `results/validation_refactor_final_20261007`.
- The run wrote 34 tables and 28 exported files for eighteen logical figures. The current reporting manifest verifies 70 artifacts.
- Scientific status is `complete_with_validation_failures`. The small sample is not claimed to establish production-scale scientific eligibility.
- Completed-run resumption and reporting-only regeneration passed without repeating the completed primary searches.
- Independent ten-layer constraint Jacobian and objective Hessian checks passed at the existing derivative tolerance.
- All seven retained scientific dependency versions match the original lockfile.
- Static undefined-name/import checks passed, and package help works from another working directory.
- Earlier interrupted validation attempts were preserved. The source-change guard rejected an in-progress run after implementation changes.
- Ignored bytecode from the former source folders was archived under `.tools/retired_bytecode`; those old folders no longer exist in the source tree.
- No full 10000-candidate production execution or complete historical numerical-output comparison was performed.

## Eight-figure reporting revision

The reporting workflow now publishes only the eight untitled numbered PNGs
specified by `generate-reference-result-charts`. The three supplemental chart
modules were removed. The established layouts were reused, with target-owned
data, complete N/S1?S10 coverage, chart metadata, and no timing disclaimers.
Missing required responses stop publication. The earlier 18-figure validation
record above describes the prior refactor, not the current figure selection.

Verification of the eight-figure revision: all 139 unit tests passed. The
standalone generator ran against the supplied reference run into
`.tools/reference-chart-preview`. It produced exactly eight nonempty PNGs,
zero SVG/PDF exports, eight resolving index rows, and the package README and
numerical metadata. Representative accuracy, parity, control, profile,
objective, and time figures were inspected. Every scientific input used by
the generator retained its original file hash.

The reference-data holdout composite nRMSE is 0.04690 for raw predictions and
0.04306 for projected predictions. Mean location R? is 0.925245 and 0.937632
respectively. These values describe the supplied target, not a newly executed
production workload.
