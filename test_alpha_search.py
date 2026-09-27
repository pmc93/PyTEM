import unittest
from unittest.mock import Mock, patch

import numpy as np

import pytem.inversion as inversion


class TestAlphaSearch(unittest.TestCase):
    def search(self, forward, mode='auto', delta=3., alpha_steps=2):
        with patch.object(inversion, '_gn_solve', return_value=np.full(2, delta)):
            return inversion._alpha_search(
                10., alpha_steps, np.eye(2), np.ones(2), np.eye(2),
                np.zeros(2), np.array([10.]), forward,
                np.full(2, np.e), np.full(2, 2.), -10., 10.,
                rms_current=2., verbose=False, step_backtrack=mode)

    def test_no_extra_work_when_ladder_improves(self):
        baseline_forward = Mock(side_effect=np.exp)
        auto_forward = Mock(side_effect=np.exp)
        baseline = self.search(baseline_forward, mode=False, delta=.5)
        automatic = self.search(auto_forward, delta=.5)
        self.assertEqual(auto_forward.call_count, baseline_forward.call_count)
        for actual, expected in zip(automatic, baseline):
            np.testing.assert_allclose(actual, expected)

    def test_recovers_only_after_full_ladder_fails(self):
        forward = Mock(side_effect=np.exp)
        result = self.search(forward)
        self.assertEqual(forward.call_count, 3)
        np.testing.assert_allclose(result[1], [4., 4., 1.])
        np.testing.assert_allclose(result[2][-1], [1.5, 1.5])
        disabled = self.search(Mock(side_effect=np.exp), mode=False)
        self.assertGreater(min(disabled[1]), 2.)
        per_trial = self.search(Mock(side_effect=np.exp), mode=True)
        self.assertLess(min(per_trial[1]), 2.)

    def test_fallback_is_bounded_when_no_step_helps(self):
        forward = Mock(return_value=np.full(2, np.exp(-2.)))
        result = self.search(forward, alpha_steps=5)
        self.assertEqual(forward.call_count, 5 + 2 * 6)
        self.assertGreater(min(result[1]), 2.)

    def test_invalid_gate_cannot_be_dropped_to_win(self):
        for invalid in [0., -1., np.nan, np.inf]:
            with self.subTest(invalid=invalid):
                forward = Mock(return_value=np.array([np.e, invalid]))
                result = self.search(forward)
                self.assertTrue(np.all(np.isinf(result[1])))

    def test_invalid_full_step_can_recover(self):
        forward = Mock(side_effect=lambda model: np.exp(model) if np.max(model) < 2. else np.array([-1., 1.]))
        result = self.search(forward)
        self.assertEqual(forward.call_count, 3)
        self.assertLessEqual(result[1][-1], 1.)

    def test_reported_backtrack_step_includes_all_halvings(self):
        trial, step, prediction, rms = inversion._backtrack_rms(
            np.zeros(2), np.full(2, 6.), -2., 2., np.exp,
            np.full(2, np.e), np.full(2, 2.), 2.)
        np.testing.assert_allclose(trial, np.full(2, 6.) * step)
        self.assertEqual(step, .25)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.search(np.exp, mode='automatic')


class TestJointTermination(unittest.TestCase):
    def invert(self, maxit, delta, observed=np.e):
        fit = [{'M': np.eye(2), 'obs': np.full(2, observed),
                'noise': np.full(2, observed * .1)}]
        with patch.object(inversion, 'fwd_circle_offset',
                          side_effect=lambda thicknesses, rho, *args, **kwargs: -rho), \
                patch.object(inversion, 'getJ_ana', return_value=np.eye(2)), \
                patch.object(inversion, '_gn_solve', return_value=np.full(2, delta)):
            return inversion.invert_joint(
                fit, np.array([10.]), np.ones(2), np.array([1e-5, 1e-4]),
                3., 'circle_offset', maxit=maxit, alpha_steps=1, verbose=False)

    def test_iteration_limit_and_zero_budget(self):
        for budget in [0, 1, 3]:
            with self.subTest(budget=budget):
                result = self.invert(budget, .1)
                self.assertEqual(result['n_iter'], budget)
                self.assertEqual(result['termination_reason'], 'max_iterations')
                self.assertFalse(result['converged'])

    def test_stalled_and_converged(self):
        stalled = self.invert(5, 0.)
        self.assertEqual(stalled['n_iter'], 1)
        self.assertEqual(stalled['termination_reason'], 'stalled')
        converged = self.invert(1, 1.)
        self.assertTrue(converged['converged'])
        self.assertEqual(converged['termination_reason'], 'converged')
        initial = self.invert(5, .1, observed=1.)
        self.assertEqual(initial['n_iter'], 0)
        self.assertTrue(initial['converged'])


if __name__ == '__main__':
    unittest.main()