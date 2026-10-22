"""Joint data-fit scaling of identified winners only; no catalogue amplitudes.

Only one nonnegative scalar per winning waveform is fitted. No unmatched or
duplicate component participates, and no extra Gaussian shrinkage penalty is
added here. The separate RJMCMC amplitude prior is unchanged.
"""
import numpy as np
from scipy.optimize import lsq_linear


def joint_amplitude_only_refit(data, sources, association, covariance,
                              duration, sampling_rate,
                              *, inner_product):
    """Return nonnegative scales and residual statistics.

    Keep only the association's one-to-one winners in the joint solve. All shapes,
    the round-start data and covariance remain fixed. Nonwinners receive scale one
    so their stored parameters are not changed; the controller excludes them from
    subtraction. The solve uses the explicit whitened design.
    """
    data = np.asarray(data)
    sources = np.asarray(sources)
    covariance = np.asarray(covariance)
    count = len(sources)
    matched_ids = np.asarray(association['matched_dwd_id'], dtype=np.int64)
    best_ids = np.asarray(association['best_template_id'], dtype=np.int64)
    best_overlap = np.asarray(association['best_overlap'], dtype=float)
    selected = matched_ids >= 0
    threshold = float(association['minimum_overlap'])
    duplicate = (~selected & (best_overlap > threshold)
                 & np.isin(best_ids, matched_ids[selected]))
    fit_indices = np.flatnonzero(selected)
    audit = dict(method='joint_nonnegative_winner_only_fixed_shape_scale',
        prior_penalty='none_in_post_identification_refit',
        selected_component_indices=fit_indices.copy(),
        nuisance_component_indices=np.empty(0, dtype=np.int64),
        excluded_component_indices=np.flatnonzero(~selected),
        discarded_duplicate_indices=np.flatnonzero(duplicate),
        fitted_component_indices=fit_indices,
        matched_dwd_id=matched_ids[selected], fit_source_count=len(fit_indices),
        solved=False)
    scales = np.ones(count)
    if not selected.any():
        audit['scale_factors'] = scales.copy()
        return scales, audit

    def product(first, second):
        return float(inner_product(first, second, covariance, duration, sampling_rate))

    lower = np.linalg.cholesky(covariance)

    def whiten(values):
        white = np.linalg.solve(lower, np.swapaxes(values, -1, -2)[..., None])[..., 0]
        white *= np.sqrt(4/(duration*sampling_rate**2))
        return np.concatenate((white.real.reshape(*white.shape[:-2], -1),
                               white.imag.reshape(*white.shape[:-2], -1)), axis=-1)

    fitted = sources[fit_indices]
    design, target = whiten(fitted).T, whiten(data)
    norm = np.linalg.norm(design, axis=0)
    # Solve the explicit whitened design, not normal equations. This also
    # supports nonunique/collinear winner shapes without a hidden ridge term.
    normalized_design = design/norm
    result = lsq_linear(normalized_design, target, bounds=(0., np.inf), method='bvls',
                        tol=1.e-12, lsq_solver='exact', max_iter=max(100, 10*len(fitted)))
    # Apply the fitted nonnegative amplitude scales.
    bounded_x = np.maximum(result.x, 0.)
    fitted_scales = bounded_x / norm
    before = data - fitted.sum(axis=0)
    after = data - np.einsum('n,ncf->cf', fitted_scales, fitted)
    objective_before, objective_after = product(before, before), product(after, after)
    scales[fit_indices] = fitted_scales
    audit.update(solved=True, scale_factors=scales.copy(),
        objective_before=objective_before, objective_after=objective_after,
        data_norm_before=product(before, before), data_norm_after=product(after, after),
        solver_iterations=int(result.nit),
        zero_scale_count=int(np.count_nonzero(fitted_scales == 0.)))
    return scales, audit
