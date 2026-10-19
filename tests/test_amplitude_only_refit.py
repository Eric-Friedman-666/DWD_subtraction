"""CPU-only numerical checks; run on the server, never touch production files."""
import ast
import pickle
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from script.dwd_amplitude_refit import joint_amplitude_only_refit


def inner(first, second, covariance, duration, rate):
    weighted = np.linalg.solve(covariance, second.T[..., None])[..., 0]
    return (4/(duration*rate**2) * np.sum(first.T.conj()*weighted)).real


class AmplitudeOnlyTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261018)

    def waveforms(self, n, bins=64):
        return self.rng.normal(size=(n, 3, bins)) + 1j*self.rng.normal(size=(n, 3, bins))

    def association(self, ids, *, best=None, overlap=None):
        ids = np.asarray(ids)
        return dict(matched_dwd_id=ids, best_template_id=ids if best is None else np.asarray(best),
                    best_overlap=np.ones(len(ids)) if overlap is None else np.asarray(overlap),
                    minimum_overlap=.9)

    def fit(self, data, sources, association=None, covariance=None):
        if association is None:
            association = self.association(np.arange(len(sources)))
        if covariance is None:
            covariance = np.broadcast_to(np.eye(3), (data.shape[-1], 3, 3)).copy()
        return joint_amplitude_only_refit(data, sources, association, covariance,
            4., 1., inner_product=inner)

    def test_one_tenth_component_is_rescaled_from_data(self):
        source = self.waveforms(1)
        scale, audit = self.fit(10*source[0], source)
        np.testing.assert_allclose(scale, [10.], atol=1.e-10)
        self.assertTrue(audit['solved'])

    def test_twelve_correlated_sources_are_fitted_jointly(self):
        sources = self.waveforms(12)
        sources[1:] += .8*sources[0]
        truth = np.linspace(.15, 4., 12)
        data = np.einsum('n,ncf->cf', truth, sources)
        independent = np.array([np.vdot(h, data).real/np.vdot(h, h).real for h in sources])
        self.assertGreater(np.max(abs(independent-truth)), .5)
        scale, audit = self.fit(data, sources)
        np.testing.assert_allclose(scale, truth, atol=1.e-10)
        self.assertEqual(audit['fit_source_count'], 12)

    def test_ten_groups_remove_duplicates_before_joint_fit(self):
        truth = self.waveforms(10)
        sources = np.repeat(truth/3, 3, axis=0)
        assigned = np.full(30, -1)
        assigned[::3] = np.arange(10)
        association = self.association(assigned, best=np.repeat(np.arange(10), 3))
        scale, audit = self.fit(truth.sum(axis=0), sources, association)
        np.testing.assert_allclose(scale[::3], 3., atol=1.e-10)
        np.testing.assert_array_equal(scale.reshape(10, 3)[:, 1:], 1.)
        self.assertEqual(len(audit['discarded_duplicate_indices']), 20)
        self.assertEqual(audit['fit_source_count'], 10)
        self.assertEqual(len(audit['nuisance_component_indices']), 0)

    def test_unmatched_and_duplicate_never_participate_in_fit(self):
        sources = self.waveforms(2)
        sources[1] += .8*sources[0]
        all_sources = np.stack((.1*sources[0], .1*sources[0], sources[1]))
        association = self.association([4, -1, -1], best=[4, 4, 4], overlap=[1., .99, .7])
        data = sources[0] + 2*sources[1]
        scale, audit = self.fit(data, all_sources, association)
        expected = np.vdot(all_sources[0], data).real/np.vdot(all_sources[0], all_sources[0]).real
        np.testing.assert_allclose(scale, [expected, 1., 1.], atol=1.e-10)
        np.testing.assert_array_equal(audit['fitted_component_indices'], [0])
        np.testing.assert_array_equal(audit['excluded_component_indices'], [1, 2])
        self.assertEqual(len(audit['nuisance_component_indices']), 0)

    def test_exact_overlap_threshold_is_not_a_duplicate_match(self):
        sources = np.zeros((2, 3, 2), complex)
        sources[0, 0, 0], sources[1, 0, 1] = 1., 1.
        association = self.association([4, -1], best=[4, 4], overlap=[1., .9])
        data = sources[0] + 2*sources[1]
        scale, audit = self.fit(data, sources, association)
        np.testing.assert_allclose(scale, [1., 1.], atol=1.e-10)
        self.assertEqual(len(audit['discarded_duplicate_indices']), 0)
        np.testing.assert_array_equal(audit['excluded_component_indices'], [1])
        self.assertEqual(len(audit['nuisance_component_indices']), 0)

    def test_boundary_solution_is_not_clipped_unconstrained_fit(self):
        sources = np.zeros((2, 3, 2), complex)
        sources[0, 0] = [1., 0.]
        sources[1, 0] = [.8, .6]
        scale, audit = self.fit(-sources[0]+sources[1], sources)
        np.testing.assert_allclose(scale, [0., .2], atol=1.e-12)
        self.assertEqual(audit['zero_scale_count'], 1)

    def test_no_extra_post_selection_amplitude_shrinkage(self):
        sources = np.zeros((1, 3, 1), complex)
        sources[0, 0, 0] = 2.
        scale, audit = self.fit(10*sources[0], sources)
        np.testing.assert_allclose(scale, [10.], atol=1.e-12)
        np.testing.assert_array_equal(audit['prior_precision_diagonal'], [0.])
        self.assertEqual(audit['prior_penalty'], 'none_in_post_identification_refit')

    def test_column_rescaling_preserves_fitted_waveform(self):
        sources = self.waveforms(3)
        data = np.einsum('n,ncf->cf', [2., .5, 4.], sources)
        first, _ = self.fit(data, sources)
        factor = np.array([.01, 10., 100.])
        second, _ = self.fit(data, sources*factor[:, None, None])
        np.testing.assert_allclose(second*factor, first, rtol=1.e-10)

    def test_complex_cross_channel_covariance_and_immutable_inputs(self):
        sources = self.waveforms(4)
        matrix = np.array([[2, .3+.2j, .1], [.3-.2j, 1.5, -.2j], [.1, .2j, 1.]])
        covariance = np.broadcast_to(matrix, (64, 3, 3)).copy()
        data = np.einsum('n,ncf->cf', [1., 3., .2, 2.], sources)
        before = [item.copy() for item in (data, sources, covariance)]
        scale, _ = self.fit(data, sources, covariance=covariance)
        np.testing.assert_allclose(scale, [1., 3., .2, 2.], atol=1.e-10)
        for actual, original in zip((data, sources, covariance), before):
            np.testing.assert_array_equal(actual, original)

    def test_no_matches_is_noop(self):
        sources = self.waveforms(2)
        scale, audit = self.fit(sources.sum(axis=0), sources, self.association([-1, -1]))
        np.testing.assert_array_equal(scale, [1., 1.])
        self.assertFalse(audit['solved'])

    def test_nonfinite_input_fails_closed(self):
        sources = self.waveforms(1)
        sources[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            self.fit(np.zeros((3, 64)), sources)

    def test_duplicate_assigned_ids_fail_closed(self):
        sources = self.waveforms(2)
        with self.assertRaisesRegex(ValueError, 'one winner'):
            self.fit(sources.sum(axis=0), sources, self.association([3, 3]))

    def test_zero_waveform_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'zero/invalid'):
            self.fit(np.zeros((3, 4)), np.zeros((1, 3, 4)))

    def test_collinear_winners_report_nonuniqueness_without_hidden_ridge(self):
        shape = self.waveforms(1)[0]
        sources = np.stack((shape, shape))
        scale, audit = self.fit(10*shape, sources)
        np.testing.assert_allclose(scale.sum(), 10., atol=1.e-10)
        self.assertEqual(audit['design_rank'], 1)
        self.assertTrue(audit['nonunique_scales_possible'])
        self.assertTrue(audit['ill_conditioned'])

    def test_bad_covariance_fails_closed(self):
        source = self.waveforms(1)
        covariance = np.broadcast_to(np.eye(3), (64, 3, 3)).copy().astype(complex)
        covariance[:, 0, 1] = .2j
        with self.assertRaisesRegex(ValueError, 'Hermitian'):
            self.fit(source[0], source, covariance=covariance)
        covariance[:, 1, 0] = -.2j
        covariance[:, 0, 0] = -1.
        with self.assertRaisesRegex(ValueError, 'positive definite'):
            self.fit(source[0], source, covariance=covariance)

    def test_many_excluded_correlated_shapes_do_not_change_winner_solution(self):
        truth = self.waveforms(10)
        data = truth.sum(axis=0)
        candidates = np.concatenate((truth/5, self.waveforms(25)))
        association = self.association(np.r_[np.arange(10), np.full(25, -1)],
            best=np.r_[np.arange(10), np.zeros(25, dtype=int)],
            overlap=np.r_[np.ones(10), np.full(25, .8)])
        scale, audit = self.fit(data, candidates, association)
        np.testing.assert_allclose(scale[:10], 5., atol=1.e-10)
        np.testing.assert_array_equal(scale[10:], 1.)
        self.assertEqual(audit['fit_source_count'], 10)

    def controller_hook(self, enabled):
        """Execute the actual controller's matching/refit loop on a closed fixture."""
        from script.dwd_bank_association import assign_bank_ids
        tree = ast.parse((Path(__file__).resolve().parents[1]/'eric_subtraction_all.py').read_text())
        loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
            and isinstance(node.target, ast.Tuple)
            and [item.id for item in node.target.elts] == ['block_index', 'table'])
        truth = self.waveforms(10)
        recovered = np.repeat(truth/3, 3, axis=0)
        parameters = np.arange(30*8, dtype=float).reshape(30, 8)/100
        parameters[:, 0] = .1
        coefficients = np.tile([.1, 0., 0., 0.], (30, 1))
        bases = np.zeros((30, 4, 3, 64), complex)
        bases[:, 0] = recovered/.1
        table = np.zeros((10, 11))
        table[:, 0] = np.arange(100, 110)
        covariance = np.broadcast_to(np.eye(3), (64, 3, 3)).copy()
        observation = np.vstack((np.arange(64), truth.sum(axis=0)))
        original_observation, original_covariance = observation.copy(), covariance.copy()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/'round_01_block_00000'
            checkpoint.mkdir()
            result = checkpoint/'result.pkl'
            result.write_bytes(pickle.dumps((parameters, table, np.arange(64),
                dict(fit_result=dict(coefficients=coefficients), bases=bases))))
            original_checkpoint = result.read_bytes()
            scope = dict(np=np, pickle=pickle, work=Path(directory), label='01',
                selected_block_indices=[0], block_template_tables=[table],
                template_number=np.zeros(1), recovered_number=np.zeros(1),
                identified_number=np.zeros(1), ghost_number=np.zeros(1),
                effective_noise_covariance=covariance, lisa_data_fd=observation,
                reference_waveform_cache={0: dict(retained_bins=np.arange(64),
                    dwd_id=table[:, 0].astype(int), waveforms=truth)},
                reconstruct_waveform=lambda a, h: np.einsum('na,nacf->ncf', a, h),
                assign_bank_ids=assign_bank_ids, joint_amplitude_only_refit=joint_amplitude_only_refit,
                refit_identified_amplitudes=enabled, inner_product=inner,
                amplitude_prior_sigma=.1, T_obs=4., fs=1.,
                proposals=[], proposal_ids=[], proposal_blocks=[], amplitude_refits={})
            exec(compile(ast.Module(body=[loop], type_ignores=[]), 'actual_controller_hook', 'exec'), scope)
            self.assertEqual(result.read_bytes(), original_checkpoint)
        np.testing.assert_array_equal(observation, original_observation)
        np.testing.assert_array_equal(covariance, original_covariance)
        self.assertEqual(len(scope['proposals'][0]), 10)
        np.testing.assert_array_equal(scope['proposal_ids'][0], np.arange(100, 110))
        return scope, parameters

    def test_actual_controller_hook_changes_only_amplitude_for_ten_ids(self):
        scope, original = self.controller_hook(True)
        accepted = scope['proposals'][0]
        np.testing.assert_array_equal(accepted[:, 1:], original[::3, 1:])
        self.assertTrue(np.all(accepted[:, 0] > .2))
        audit = scope['amplitude_refits']['block_00000']
        self.assertEqual(audit['fit_source_count'], 10)
        self.assertEqual(len(audit['discarded_duplicate_indices']), 20)
        self.assertEqual(len(audit['nuisance_component_indices']), 0)
        np.testing.assert_array_equal(audit['scale_factors'][np.arange(30)%3 != 0], 1.)

    def test_disabled_controller_hook_preserves_old_amplitudes(self):
        scope, original = self.controller_hook(False)
        np.testing.assert_array_equal(scope['proposals'][0], original[::3])
        self.assertEqual(scope['amplitude_refits'], {})

    def test_formal_controller_enables_winner_only_refit_by_default(self):
        source = Path(__file__).resolve().parents[1]/'eric_subtraction_all.py'
        tree = ast.parse(source.read_text())
        setting = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'refit_identified_amplitudes'
                    for target in node.targets))
        self.assertIs(ast.literal_eval(setting.value), True)
        self.assertIn('joint_nonnegative_winner_only_fixed_shape_scale', source.read_text())

    def test_last_bit_negative_at_zero_bound_is_rechecked_as_feasible(self):
        import script.dwd_amplitude_refit as refit
        sources = self.waveforms(2)
        original = refit.lsq_linear
        def tiny_negative(*args, **kwargs):
            result = original(*args, **kwargs)
            result.x[1] = -8*np.finfo(float).eps*max(1., np.max(abs(result.x)))
            return result
        with patch.object(refit, 'lsq_linear', tiny_negative):
            scale, audit = self.fit(2*sources[0], sources)
        np.testing.assert_allclose(scale, [2., 0.], atol=1.e-12)
        self.assertEqual(audit['negative_bound_roundoff_count'], 1)
        self.assertLess(audit['relative_kkt_residual'], 1.e-8)

    def test_meaningful_negative_bound_violation_still_fails(self):
        import script.dwd_amplitude_refit as refit
        sources = self.waveforms(2)
        original = refit.lsq_linear
        def negative(*args, **kwargs):
            result = original(*args, **kwargs)
            result.x[1] = -1.e-6*max(1., np.max(abs(result.x)))
            return result
        with patch.object(refit, 'lsq_linear', negative):
            with self.assertRaisesRegex(RuntimeError, 'nonnegative bound'):
                self.fit(2*sources[0], sources)


if __name__ == '__main__':
    unittest.main(verbosity=2)
