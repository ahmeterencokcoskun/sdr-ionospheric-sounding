# E2E Test Infra: sdr-ionospheric-sounding

## Test Philosophy
- Opaque-box, requirement-driven testing derived from `ORIGINAL_REQUEST.md`.
- No internal mock reliance; tests exercise the real pipeline and physical models locally and offline.
- Methodology: Category-Partition + Boundary Value Analysis (BVA) + Pairwise Combinatorial Testing + Real-World Workload Testing (Tiers 1-4) + White-box Adversarial Coverage (Tier 5).

## Feature Inventory & Test Coverage Mapping
| # | Feature | Requirement | Tier 1 (Coverage) | Tier 2 (Boundary) | Tier 3 (Cross-Feature) | Tier 4 (Workloads) |
|---|---------|-------------|:-----------------:|:-----------------:|:---------------------:|:------------------:|
| 1 | Plasma Physics Engine | R2, R3 | >=5 tests | >=5 tests | ✓ | ✓ |
| 2 | Haselgrove Ray Tracing | R2, R3 | >=5 tests | >=5 tests | ✓ | ✓ |
| 3 | Breit-Tuve FMCW Sounder | R2, R3 | >=5 tests | >=5 tests | ✓ | ✓ |
| 4 | True-Height Inversion Engine | R1, R3 | >=5 tests | >=5 tests | ✓ | ✓ |
| 5 | Watterson Multipath & LDPC | R2, R3 | >=5 tests | >=5 tests | ✓ | ✓ |
| 6 | DSP Doppler & WSPR Drift | R2, R3 | >=5 tests | >=5 tests | ✓ | ✓ |
| 7 | CLI & Pipeline Integration | R1, R4 | >=5 tests | >=5 tests | ✓ | ✓ |

## Test Architecture
- Test runner: `.venv/bin/pytest tests/ -v`
- Pass/Fail semantics: 100% passing tests, exit code 0, zero warnings on critical assertions.
- Dedicated files:
  - `tests/test_pipeline.py`: Baseline regression tests (6 tests).
  - `tests/test_academic_validation.py`: Analytical baselines (Budden parabolic, linear layer, Bouguer invariant, space escape, Breit-Tuve positivity, Appleton-Hartree asymptotics, Martyn equivalence, LDPC waterfall).
  - `tests/test_e2e_suite.py`: Multi-tier requirement-driven E2E tests.

## Coverage Goals & Thresholds
- Total Feature count $N = 7$ primary feature groups.
- Tier 1: >= 35 test cases (>= 5 per feature group).
- Tier 2: >= 35 boundary and corner cases.
- Tier 3: >= 7 cross-feature interaction scenarios.
- Tier 4: >= 5 realistic end-to-end scientific sounding workflows.
- Total test target: >= 80 test assertions across the suite.
