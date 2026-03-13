import unittest

import torch

from warp import warp_feature_map


class WarpTests(unittest.TestCase):
    def test_zero_flow_identity(self) -> None:
        feat = torch.randn(2, 4, 8, 8)
        flow = torch.zeros(2, 2, 64, 64)
        warped = warp_feature_map(feat, flow)
        self.assertTrue(torch.allclose(warped, feat, atol=1e-5))

    def test_positive_x_flow_shifts_content_right(self) -> None:
        feat = torch.zeros(1, 1, 8, 8)
        feat[0, 0, 3, 3] = 1.0
        flow = torch.zeros(1, 2, 64, 64)
        flow[:, 0, :, :] = 8.0  # +8px at 64x64 -> +1px at 8x8 after scale.
        warped = warp_feature_map(feat, flow)
        self.assertGreater(warped[0, 0, 3, 4].item(), 0.5)


if __name__ == "__main__":
    unittest.main()
