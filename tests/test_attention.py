"""Check FeatLens integration, including attention paths through registers."""

import unittest

import numpy as np
import torch

from src.attention import attention_rollout, cosine_map
from src.models import TokenLayout


class FeatLensIntegrationTests(unittest.TestCase):
    """Protect feature semantics and probability mass at the package boundary."""

    def test_cosine_uses_selected_patch(self) -> None:
        """The seed is identical to itself, orthogonal to one patch and opposite another."""
        features = torch.tensor([[[1., 0.], [0., 1.]], [[-1., 0.], [0., 0.]]])
        actual = cosine_map(features, (0., 0.))
        np.testing.assert_array_equal(actual, [[1., 0.], [-1., 0.]])

    def test_rollout_retains_register_paths(self) -> None:
        """A CLS→register→patch path must survive until the final spatial slice."""
        layout = TokenLayout(registers=4, grid_size=2)
        count = layout.prefix + layout.grid_size**2
        first = torch.eye(count)
        first[1].zero_()
        first[1, layout.prefix] = 1
        last = torch.eye(count)
        last[0].zero_()
        last[0, 1] = 1
        layers = tuple(a[None, None].repeat(1, 2, 1, 1) for a in (first, last))
        heat, mass = attention_rollout(layers, layout)
        np.testing.assert_array_equal(heat, [[.25, 0.], [0., 0.]])
        np.testing.assert_array_equal(mass, [.5, .25, .25])

    def test_rollout_without_registers(self) -> None:
        """Residual CLS mass must not be renormalized away from the patch heatmap."""
        layout = TokenLayout(registers=0, grid_size=2)
        count = layout.prefix + layout.grid_size**2
        attention = torch.eye(count)
        attention[0].zero_()
        attention[0, layout.prefix] = 1
        heat, mass = attention_rollout((attention[None, None],), layout)
        np.testing.assert_array_equal(heat, [[.5, 0.], [0., 0.]])
        np.testing.assert_array_equal(mass, [.5, 0., .5])


if __name__ == "__main__":
    unittest.main()
