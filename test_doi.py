import unittest
from unittest.mock import patch

import numpy as np

from pytem import compute_doi, invert_joint
import pytem.inversion as inversion


class TestComputeDoi(unittest.TestCase):
    def test_weighted_column_sums_and_interpolation(self):
        result = compute_doi([[0.6, -0.3, 0.05], [-0.4, 0.2, -0.05]], [10., 20.])
        np.testing.assert_allclose(result['sensitivity'], [1., 0.5, 0.1])
        np.testing.assert_allclose(result['cumulative'], [1.6, 0.6, 0.1])
        np.testing.assert_allclose(result['layer_tops'], [0., 10., 30.])
        self.assertAlmostEqual(result['standard'], 8.)
        self.assertAlmostEqual(result['conservative'], 4.)
        self.assertFalse(result['standard_capped'])

    def test_surface_and_bottom_limits(self):
        surface = compute_doi(np.zeros((2, 3)), [10., 20.])
        self.assertEqual(surface['standard'], 0.)
        self.assertFalse(surface['standard_capped'])
        bottom = compute_doi([[0., 0., 2.]], [10., 20.])
        self.assertEqual(bottom['standard'], 30.)
        self.assertTrue(bottom['standard_capped'])
        self.assertTrue(bottom['conservative_capped'])
        halfspace = compute_doi([[2.]], [])
        self.assertEqual(halfspace['standard'], 0.)
        self.assertTrue(halfspace['standard_capped'])

    def test_plateau_and_noise_scaling(self):
        result = compute_doi([[1., 0., 0.1]], [10., 20.])
        self.assertAlmostEqual(result['standard'], 3.)
        noisier = compute_doi(np.array([[1., 0., 0.1]]) / 2, [10., 20.])
        self.assertLessEqual(noisier['standard'], result['standard'])
        self.assertLessEqual(result['conservative'], result['standard'])

    def test_invalid_inputs(self):
        cases = [([], []), ([[np.nan]], []), ([[np.inf]], []),
                 ([[1., 2.]], []), ([[1., 2.]], [-1.]),
                 ([[1., 2.]], [np.nan]), ([1., 2.], [10.])]
        for jacobian, thicknesses in cases:
            with self.subTest(jacobian=jacobian, thicknesses=thicknesses):
                with self.assertRaises(ValueError):
                    compute_doi(jacobian, thicknesses)
        for standard, conservative in [(0., 1.2), (1.2, 0.8), (np.nan, 1.2), (0.8, np.inf)]:
            with self.subTest(standard=standard, conservative=conservative):
                with self.assertRaises(ValueError):
                    compute_doi([[1.]], [], standard, conservative)


class TestJointDoi(unittest.TestCase):
    def test_invalid_predictions_preserve_fit_without_doi(self):
        fit = [{'M': np.eye(2), 'obs': np.ones(2), 'noise': np.full(2, .05)}]
        for invalid in [-1., 0., np.nan, np.inf]:
            with self.subTest(prediction=invalid), \
                    patch.object(inversion, 'fwd_circle_offset',
                                 return_value=-np.array([invalid, 1.])), \
                    patch.object(inversion, 'getJ_ana', return_value=np.eye(2)) as jacobian:
                args = (fit, np.array([10.]), np.ones(2),
                        np.array([1e-5, 1e-4]), 3., 'circle_offset')
                with np.errstate(invalid='ignore'):
                    baseline = invert_joint(*args, maxit=0, verbose=False)
                    baseline_calls = jacobian.call_count
                    jacobian.reset_mock()
                    result = invert_joint(*args, maxit=0, calc_doi=True, verbose=False)
                self.assertEqual(jacobian.call_count, baseline_calls)
                np.testing.assert_equal(result['predicted'], baseline['predicted'])
                np.testing.assert_equal(result['resistivities'], baseline['resistivities'])
                np.testing.assert_equal(result['rms'], baseline['rms'])
                self.assertFalse(result['converged'])
                self.assertFalse(result['doi']['valid'])
                self.assertIn('nonfinite/nonpositive predictions', result['doi']['reason'])
                self.assertTrue(np.isnan(result['doi']['standard']))
                self.assertTrue(np.isnan(result['doi']['conservative']))

    def test_receiver_filter_reaches_forward_jacobian_and_doi(self):
        from pytem import cascade_filter

        receiver = cascade_filter(250000., 800000.)
        fit = [{'M': np.eye(2), 'obs': np.ones(2), 'noise': np.full(2, .05)}]
        for geometry, forward_name in [('circle_offset', 'fwd_circle_offset'),
                                       ('square_offset', 'fwd_square_offset')]:
            with self.subTest(geometry=geometry), \
                    patch.object(inversion, forward_name, return_value=-np.ones(2)) as forward, \
                    patch.object(inversion, 'getJ_ana', return_value=np.eye(2)) as jacobian:
                result = invert_joint(fit, np.array([10.]), np.ones(2),
                                      np.array([1e-5, 1e-4]), 10., geometry,
                                      calc_doi=True, system_filter=receiver, verbose=False)
                for call in forward.call_args_list + jacobian.call_args_list:
                    self.assertIs(call.kwargs['system_filter'], receiver)
                self.assertIsNotNone(result['doi'])
                self.assertEqual(jacobian.call_count, 2)

    def test_gate_weights_backend_and_optional_cost(self):
        fit = [
            {'M': np.array([[1., 0.]]), 'obs': np.array([2.]), 'noise': np.array([.2])},
            {'M': np.array([[0., 2.]]), 'obs': np.array([8.]), 'noise': np.array([1.6])},
        ]
        args = (fit, np.array([10., 20.]), np.ones(3), np.array([1e-5, 1e-4]),
                10., 'square_offset')
        jacobian = np.array([[.2, -.1, .02], [.4, -.2, .04]])
        with patch.object(inversion, 'fwd_square_offset', return_value=-np.array([2., 4.])), \
                patch.object(inversion, 'getJ_ana', return_value=jacobian) as get_jacobian:
            baseline = invert_joint(*args, verbose=False)
            baseline_calls = get_jacobian.call_count
            get_jacobian.reset_mock()
            result = invert_joint(*args, verbose=False, calc_doi=True, transform='euler',
                                  doi_threshold=.6, doi_conservative_threshold=1.)
            self.assertEqual(get_jacobian.call_count, baseline_calls + 1)
            self.assertEqual(get_jacobian.call_args.kwargs['transform'], 'euler')
            self.assertEqual(get_jacobian.call_args.kwargs['geometry'], 'square_offset')
            self.assertEqual(get_jacobian.call_args.kwargs['jacobian_mode'], 'absolute')
        self.assertIsNone(baseline['doi'])
        self.assertTrue(result['doi']['valid'])
        self.assertEqual(result['doi']['reason'], '')
        np.testing.assert_allclose(result['resistivities'], baseline['resistivities'])
        np.testing.assert_allclose(result['doi']['sensitivity'], [1.5, .75, .15])
        self.assertEqual(result['doi']['threshold'], .6)
        self.assertEqual(result['doi']['conservative_threshold'], 1.)

    def test_doi_uses_updated_final_model(self):
        observed = np.full(2, np.e)
        fit = [{'M': np.eye(2), 'obs': observed, 'noise': observed * .1}]
        with patch.object(inversion, 'fwd_circle_offset', side_effect=lambda thicknesses, rho, *args, **kwargs: -rho), \
                patch.object(inversion, 'getJ_ana', side_effect=lambda **kwargs: np.diag(np.exp(kwargs['log_resistivities']))) as get_jacobian, \
                patch.object(inversion, '_gn_solve', return_value=np.full(2, .5)):
            result = invert_joint(fit, np.array([10.]), np.ones(2), np.array([1e-5, 1e-4]),
                                  10., 'circle_offset', maxit=1, alpha_steps=1,
                                  verbose=False, calc_doi=True)
        self.assertTrue(np.all(result['resistivities'] > 1.))
        np.testing.assert_allclose(get_jacobian.call_args.kwargs['log_resistivities'],
                                   result['log_resistivities'])
        np.testing.assert_allclose(result['doi']['sensitivity'], [10., 10.])


if __name__ == '__main__':
    unittest.main()