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
    """Return nonnegative scales and an audit, leaving every input unchanged.

    Keep only the association's one-to-one winners in the joint solve. All shapes,
    the round-start data and covariance remain fixed. Nonwinners receive scale one
    so their stored parameters are not changed; the controller excludes them from
    subtraction. A rank-deficient design is reported, not artificially regularized.
    """
    data = np.asarray(data)
    sources = np.asarray(sources)
    covariance = np.asarray(covariance)
    count = len(sources)
    if (data.ndim != 2 or data.shape[0] != 3
            or sources.shape != (count, *data.shape)
            or covariance.shape != (data.shape[-1], 3, 3)):
        raise ValueError('Amplitude refit requires (3,F) data, (N,3,F) sources and (F,3,3) covariance')
    if not all(np.isfinite(value).all() for value in (data, sources, covariance)):
        raise ValueError('Non-finite amplitude-refit inputs')
    if (not np.isfinite(duration) or duration <= 0
            or not np.isfinite(sampling_rate) or sampling_rate <= 0):
        raise ValueError('Duration and sampling rate must be finite and positive')
    matched_ids = np.asarray(association['matched_dwd_id'], dtype=np.int64)
    best_ids = np.asarray(association['best_template_id'], dtype=np.int64)
    best_overlap = np.asarray(association['best_overlap'], dtype=float)
    if any(value.shape != (count,) for value in (matched_ids, best_ids, best_overlap)):
        raise ValueError('Association/component size mismatch')
    selected = matched_ids >= 0
    if len(np.unique(matched_ids[selected])) != selected.sum():
        raise ValueError('Amplitude refit requires one winner per matched DWD ID')
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

    covariance_scale = max(float(np.max(abs(covariance))), np.finfo(float).tiny)
    if not np.allclose(covariance, covariance.conj().swapaxes(-1, -2),
                       rtol=1.e-10, atol=1.e-12*covariance_scale):
        raise ValueError('Amplitude-refit covariance must be Hermitian')
    try:
        lower = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError('Amplitude-refit covariance must be positive definite') from error

    def whiten(values):
        white = np.linalg.solve(lower, np.swapaxes(values, -1, -2)[..., None])[..., 0]
        white *= np.sqrt(4/(duration*sampling_rate**2))
        return np.concatenate((white.real.reshape(*white.shape[:-2], -1),
                               white.imag.reshape(*white.shape[:-2], -1)), axis=-1)

    fitted = sources[fit_indices]
    design, target = whiten(fitted).T, whiten(data)
    norm = np.linalg.norm(design, axis=0)
    if not np.isfinite(norm).all() or np.any(norm <= 0):
        raise ValueError('Amplitude refit has a zero/invalid waveform norm')
    # Solve the explicit whitened design, not normal equations. This also
    # supports nonunique/collinear winner shapes without a hidden ridge term.
    normalized_design = design/norm
    result = lsq_linear(normalized_design, target, bounds=(0., np.inf), method='bvls',
                        tol=1.e-12, lsq_solver='exact', max_iter=max(100, 10*len(fitted)))
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f'Amplitude-only refit failed: {result.message}')
    # BVLS can return a negative last-bit value at an active zero bound even
    # when it reports success. Correct only floating-point roundoff; meaningful
    # bound violations still fail. Recheck KKT and the objective on this feasible
    # vector below, exactly as for every other solve.
    roundoff = 32*np.finfo(float).eps*max(1., float(np.max(abs(result.x))))
    if np.any(result.x < -roundoff):
        raise RuntimeError('Amplitude-only refit violated its nonnegative bound')
    bounded_x = np.maximum(result.x, 0.)
    fitted_scales = bounded_x / norm
    rhs = normalized_design.T @ target
    gradient = normalized_design.T @ (normalized_design @ bounded_x-target)
    free = bounded_x > 1.e-10 * max(1., float(np.max(bounded_x)))
    kkt = np.where(free, abs(gradient), np.maximum(-gradient, 0.))
    relative_kkt = float(np.max(kkt) / max(1., float(np.max(abs(rhs)))))
    if relative_kkt > 1.e-8:
        raise RuntimeError(f'Amplitude-only refit failed KKT check: {relative_kkt}')

    before = data - fitted.sum(axis=0)
    after = data - np.einsum('n,ncf->cf', fitted_scales, fitted)
    objective_before, objective_after = product(before, before), product(after, after)
    if objective_after > objective_before + 1.e-9 * max(1., abs(objective_before)):
        raise RuntimeError('Amplitude refit increased its winner-only data objective')
    scales[fit_indices] = fitted_scales
    singular_values = np.linalg.svd(normalized_design, compute_uv=False)
    rank = int(np.linalg.matrix_rank(normalized_design))
    denominator = max(singular_values[-1], np.finfo(float).tiny)
    condition = (float(singular_values[0]/denominator)
                 if denominator > singular_values[0]/np.finfo(float).max
                 else float(np.finfo(float).max))
    audit.update(solved=True, scale_factors=scales.copy(),
        negative_bound_roundoff_count=int(np.count_nonzero(result.x < 0)),
        negative_bound_roundoff_tolerance=roundoff,
        minimum_raw_normalized_scale=float(np.min(result.x)),
        prior_precision_diagonal=np.zeros(len(fit_indices)),
        objective_before=objective_before, objective_after=objective_after,
        data_norm_before=product(before, before), data_norm_after=product(after, after),
        relative_kkt_residual=relative_kkt, solver_iterations=int(result.nit),
        zero_scale_count=int(np.count_nonzero(fitted_scales == 0.)),
        scaled_design_condition_number=condition, design_rank=rank,
        nonunique_scales_possible=bool(rank < len(fit_indices)),
        ill_conditioned=bool(rank < len(fit_indices) or condition >= 1.e12))
    return scales, audit
