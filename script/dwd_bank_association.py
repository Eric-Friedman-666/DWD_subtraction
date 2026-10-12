"""Post-inference waveform identification against a complete labelled bank.

Injection information is deliberately absent from assign_bank_ids. The separate
classify_injection_recovery function is for controlled-test validation only.
"""
import numpy as np


def waveform_overlap_matrix(recovered, templates, inverse_covariance, duration, sampling_rate):
    """Signed O, with all complex XYZ cross terms; no phase maximization."""
    recovered = np.asarray(recovered, dtype=np.complex128)
    templates = np.asarray(templates, dtype=np.complex128)
    factor = 4.0 / (duration * sampling_rate**2)
    cross = factor * np.einsum(
        'aik,kij,bjk->ab', recovered.conj(), inverse_covariance, templates, optimize=True
    ).real
    recovered_norm = factor * np.einsum(
        'aik,kij,ajk->a', recovered.conj(), inverse_covariance, recovered, optimize=True
    ).real
    template_norm = factor * np.einsum(
        'bik,kij,bjk->b', templates.conj(), inverse_covariance, templates, optimize=True
    ).real
    denominator = np.sqrt(np.maximum(recovered_norm[:, None] * template_norm[None], 0.0))
    overlaps = np.zeros_like(cross)
    np.divide(cross, denominator, out=overlaps, where=denominator > 0.0)
    return np.clip(overlaps, -1.0, 1.0)


def assign_overlap_matrix(overlaps, template_ids, minimum_overlap=0.9):
    """Highest-overlap pair wins; a displaced component tries its next choice.

    Sorting all eligible edges in decreasing O implements this greedy rule. It
    is deliberately not a maximum-total-overlap/Hungarian assignment. Ties use
    recovered row, then ascending bank ID. The acceptance inequality is strict.
    """
    overlaps = np.asarray(overlaps, dtype=np.float64)
    template_ids = np.asarray(template_ids, dtype=np.int64)
    count, bank_count = overlaps.shape
    assigned_index = np.full(count, -1, dtype=np.int64)
    assigned_id = np.full(count, -1, dtype=np.int64)
    assigned_overlap = np.zeros(count)
    assigned_rank = np.full(count, -1, dtype=np.int64)
    rank_order = np.lexsort((np.broadcast_to(template_ids, overlaps.shape), -overlaps), axis=1)
    best_index = rank_order[:, 0] if bank_count else np.full(count, -1, dtype=np.int64)
    best_overlap = overlaps[np.arange(count), best_index] if bank_count else np.zeros(count)
    best_id = template_ids[best_index] if bank_count else np.full(count, -1, dtype=np.int64)
    used = np.zeros(bank_count, dtype=bool)
    accepted_order = []
    rows, columns = np.nonzero(np.isfinite(overlaps) & (overlaps > minimum_overlap))
    priority = np.lexsort((template_ids[columns], rows, -overlaps[rows, columns]))
    for edge in priority:
        row, column = int(rows[edge]), int(columns[edge])
        if assigned_index[row] >= 0 or used[column]:
            continue
        assigned_index[row] = column
        assigned_id[row] = template_ids[column]
        assigned_overlap[row] = overlaps[row, column]
        assigned_rank[row] = int(np.flatnonzero(rank_order[row] == column)[0]) + 1
        used[column] = True
        accepted_order.append((row, column))
    return dict(
        overlap_matrix=overlaps, template_ids=template_ids,
        ranked_template_indices=rank_order, matched_template_index=assigned_index,
        matched_dwd_id=assigned_id, matched_overlap=assigned_overlap,
        matched_rank=assigned_rank, bank_confirmed=assigned_index >= 0,
        best_template_index=best_index, best_template_id=best_id, best_overlap=best_overlap,
        accepted_pair_order=np.asarray(accepted_order, dtype=np.int64).reshape(-1, 2),
        minimum_overlap=float(minimum_overlap),
    )


def assign_bank_ids(recovered, templates, template_ids, inverse_covariance,
                    duration, sampling_rate, minimum_overlap=0.9):
    overlaps = waveform_overlap_matrix(recovered, templates, inverse_covariance,
                                      duration, sampling_rate)
    return assign_overlap_matrix(overlaps, template_ids, minimum_overlap)


def classify_injection_recovery(association, injected_ids):
    """Only now compare assigned bank IDs to the controlled injection list."""
    injected_ids = np.asarray(injected_ids, dtype=np.int64)
    assigned = association['matched_dwd_id']
    confirmed = association['bank_confirmed']
    real = confirmed & np.isin(assigned, injected_ids)
    labels = np.full(len(assigned), 'ghost_unmatched', dtype='U24')
    labels[confirmed & ~real] = 'ghost_not_injected'
    labels[real] = 'real_detection'
    detected_lookup = {int(source_id): row for row, source_id in enumerate(assigned) if real[row]}
    injection_to_recovery = np.array([detected_lookup.get(int(source_id), -1)
                                      for source_id in injected_ids], dtype=np.int64)
    return dict(
        recovered_status=labels, real_detection=real, ghost=~real,
        unmatched_ghost=~confirmed, noninjected_ghost=confirmed & ~real,
        injection_to_recovery=injection_to_recovery,
        lurking_dwd_id=injected_ids[injection_to_recovery < 0],
    )
