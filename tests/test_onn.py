import unittest

import torch

import onn


class NetSmokeTest(unittest.TestCase):
    def test_forward_returns_finite_class_scores(self):
        model = onn.Net(num_layers=1)
        model.eval()
        waves = torch.zeros(2, 200, 200, 2)
        waves[:, 86:114, 86:114, 0] = 1.0

        with torch.no_grad():
            output = model(waves)

        self.assertEqual(output.shape, (2, 10))
        self.assertTrue(torch.isfinite(output).all().item())


if __name__ == "__main__":
    unittest.main()
