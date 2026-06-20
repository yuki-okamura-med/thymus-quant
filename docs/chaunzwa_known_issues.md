# Chaunzwa Known Issues

Chaunzwa quantification is experimental in this package version. It is exposed
for API smoke testing and historical comparison, not as a validated primary
endpoint for collaborative release.

Known numerical and interpretation issues intentionally deferred:

- EM likelihood is not recomputed after the final update.
- Responsibility calculations can underflow for difficult HU distributions.
- One-component model interpretation is not settled.
- HU outliers and histogram ranges need formal release criteria.
- Non-converged models are not yet promoted to a hard failure policy.
- `bayesian_delta_hu` can materially affect adjusted ETV.
- Posterior volume semantics need clearer scientific validation.
- GMM component-to-tissue assignment is heuristic and not release-validated.

Do not add golden tests that freeze Chaunzwa pTT, GMM component assignment, or
component means as scientifically correct values until these issues are resolved.
