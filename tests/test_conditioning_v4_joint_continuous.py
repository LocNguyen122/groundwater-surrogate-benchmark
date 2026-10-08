"""Unit tests for the v4_joint_continuous injection-matched comparator.

Review finding 3.2 asked for a non-factorized comparator that shares the FiLM injection site,
the shared modulation network and the parameter budget with the factorized variants, so that a
V3-minus-V4 contrast isolates factorization rather than the injection mechanism.

Finding 6.5 separately noted that V2 (joint lookup table) is a weak competitor by construction:
the row for a withheld pair receives no gradient. These tests assert that V4 does not share that
defect, which is what makes it a fair comparator.
"""
import unittest

import torch

from src.transport_surrogates.confirmatory_v5.conditioning import (
    CONDITIONING_VARIANTS,
    RegimeConditionedCTAUNet,
)

CENTERS = {"alpha_L": [-1.118, 0.0, 1.118], "alpha_T_ratio": [-1.0, 1.0]}
COND_CHILDREN = (
    "alpha_l_encoder",
    "alpha_t_ratio_encoder",
    "joint_encoder",
    "joint_continuous_encoder",
    "film",
)


def build(variant):
    return RegimeConditionedCTAUNet(
        conditioning_variant=variant,
        normalized_level_centers=CENTERS,
        out_ch=25,
        base=64,
        t_embed_dim=25,
        t_heads=5,
        pool_stride=4,
        film_embed_dim=8,
        film_hidden_dim=32,
    )


def conditioning_params(model):
    return sum(
        p.numel()
        for name, child in model.named_children()
        if name in COND_CHILDREN
        for p in child.parameters()
    )


def make_input(a, b, batch=2, size=64):
    x = torch.randn(batch, 3, size, size)
    x[:, 1, :, :] = a
    x[:, 2, :, :] = b
    return x


class TestV4JointContinuous(unittest.TestCase):
    def test_variant_is_registered(self):
        self.assertIn("v4_joint_continuous", CONDITIONING_VARIANTS)

    def test_forward_shape(self):
        model = build("v4_joint_continuous")
        with torch.no_grad():
            y = model(make_input(0.0, 1.0))
        self.assertEqual(tuple(y.shape), (2, 25, 64, 64))

    def test_parameter_budget_matches_v3(self):
        v3, v4 = conditioning_params(build("v3_factorized_continuous")), conditioning_params(
            build("v4_joint_continuous")
        )
        self.assertLess(abs(v3 - v4) / v3, 0.01, f"budget mismatch: v3={v3} v4={v4}")

    def test_unknown_variant_rejected(self):
        with self.assertRaises(ValueError):
            build("v9_not_a_variant")

    def test_code_depends_on_both_factors(self):
        """A joint encoder must respond to each code separately."""
        model = build("v4_joint_continuous")
        with torch.no_grad():
            base = model.encode_conditioning(torch.tensor([[0.0, 1.0]]))
            moved_a = model.encode_conditioning(torch.tensor([[1.118, 1.0]]))
            moved_b = model.encode_conditioning(torch.tensor([[0.0, -1.0]]))
        self.assertFalse(torch.allclose(base, moved_a))
        self.assertFalse(torch.allclose(base, moved_b))

    def test_withheld_pair_receives_gradient(self):
        """The property V2 lacks: every parameter used at a withheld pair is trained.

        Training on the five observed pairs must leave a gradient on the whole joint encoder,
        so evaluating the withheld pair uses trained parameters rather than initialization.
        """
        model = build("v4_joint_continuous")
        observed = [(-1.118, -1.0), (-1.118, 1.0), (0.0, -1.0), (1.118, -1.0), (1.118, 1.0)]
        withheld = (0.0, 1.0)
        self.assertNotIn(withheld, observed)

        model.zero_grad()
        loss = 0.0
        for a, b in observed:
            loss = loss + model.encode_conditioning(torch.tensor([[a, b]])).pow(2).sum()
        loss.backward()

        for name, param in model.joint_continuous_encoder.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} has no grad")
            self.assertGreater(
                float(param.grad.abs().sum()), 0.0, f"{name} received zero gradient"
            )

    def test_v2_withheld_row_gets_no_gradient(self):
        """Contrast case documenting why V2 alone is not a sufficient comparator."""
        model = build("v2_joint_embedding")
        observed = [(-1.118, -1.0), (-1.118, 1.0), (0.0, -1.0), (1.118, -1.0), (1.118, 1.0)]
        model.zero_grad()
        loss = 0.0
        for a, b in observed:
            loss = loss + model.encode_conditioning(torch.tensor([[a, b]])).pow(2).sum()
        loss.backward()
        withheld_row = 0 * 2 + 1  # alpha_L index 1 (interior), ratio index 1 -> row 3
        withheld_row = 1 * 2 + 1
        grad = model.joint_encoder.weight.grad
        self.assertEqual(float(grad[withheld_row].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
