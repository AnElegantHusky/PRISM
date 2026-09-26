import unittest

import numpy as np

from prism import AugmentedCPIFeedback, PRISM, PureRMSESelector


class PRISMTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        latent = rng.normal(size=(3, 10))
        mixing = rng.normal(size=(12, 3))
        self.response = mixing @ latent

    def test_selector_is_deterministic_and_affine(self):
        first = PureRMSESelector(k=5).select(self.response)
        second = PureRMSESelector(k=5).select(self.response)
        self.assertEqual(first.selected_indices, second.selected_indices)
        np.testing.assert_allclose(first.reconstruction_weights.sum(axis=1), 1.0)
        np.testing.assert_allclose(
            first.reconstruction_weights[np.asarray(first.selected_indices)], np.eye(5)
        )

    def test_augmented_feedback_contains_identity_directions(self):
        feedback = AugmentedCPIFeedback(5, directions=300, seed=11)
        np.testing.assert_allclose(feedback.weights[:5], np.eye(5))
        np.testing.assert_allclose(feedback.weights.sum(axis=1), 1.0)
        values = np.arange(5, dtype=float)
        np.testing.assert_allclose(feedback.transform(values)[:5], values)

    def test_phase_transition_and_mean_feedback(self):
        prism = PRISM(12, k=5, target_algorithms=10, feedback="mean")
        for column in range(10):
            self.assertTrue(prism.observe_initial(column, self.response[:, column]))
        self.assertEqual(len(prism.selected_indices), 5)
        projected = prism.project_initial_population()
        self.assertEqual(set(projected), set(range(10)))
        self.assertTrue(all(np.isscalar(value) for value in projected.values()))


if __name__ == "__main__":
    unittest.main()
