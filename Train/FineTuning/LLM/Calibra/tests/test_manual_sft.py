import copy
import unittest
from types import SimpleNamespace

try:
    import torch
except ImportError:  # pragma: no cover - dependency-gated tests
    torch = None

from calibra.loss import (  # noqa: E402
    compute_loss,
    cross_entropy,
    cross_entropy_with_gradient,
    manual_sft_backward,
    manual_sft_forward,
)
if torch is not None:
    from calibra.optimizers.manual_adamw import ManualAdamW  # noqa: E402


if torch is not None:
    class TinyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(13, 5)
            self.output = torch.nn.Linear(5, 13)

        def forward(self, input_ids, attention_mask):
            del attention_mask
            return SimpleNamespace(logits=self.output(self.embedding(input_ids)))


class ManualSFTTest(unittest.TestCase):
    @unittest.skipUnless(torch is not None, "PyTorch is required")
    def test_explicit_cross_entropy_gradient_matches_autograd(self):
        torch.manual_seed(7)
        logits = torch.randn(6, 11, dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([2, -100, 7, 0, -100, 4])

        reference_loss = cross_entropy(logits, targets)
        reference_gradient, = torch.autograd.grad(reference_loss, logits)
        manual_loss, manual_gradient, count = cross_entropy_with_gradient(logits, targets)

        self.assertEqual(count, 4)
        torch.testing.assert_close(manual_loss, reference_loss.detach().float())
        torch.testing.assert_close(
            manual_gradient.double(), reference_gradient, rtol=1e-6, atol=1e-7
        )

    @unittest.skipUnless(torch is not None, "PyTorch is required")
    def test_manual_sft_backward_matches_loss_backward(self):
        torch.manual_seed(11)
        reference_model = TinyCausalLM()
        manual_model = copy.deepcopy(reference_model)
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4], [4, 5, 6, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
            "labels": torch.tensor([[-100, -100, 3, 4], [-100, 5, 6, -100]]),
        }

        reference_loss, _ = compute_loss(reference_model, batch)
        reference_loss.backward()

        manual_loss, shift_logits, logits_gradient, _ = manual_sft_forward(
            manual_model, batch
        )
        manual_sft_backward(shift_logits, logits_gradient)

        torch.testing.assert_close(manual_loss, reference_loss.detach())
        for reference_parameter, manual_parameter in zip(
            reference_model.parameters(), manual_model.parameters()
        ):
            torch.testing.assert_close(manual_parameter.grad, reference_parameter.grad)

    @unittest.skipUnless(torch is not None, "PyTorch is required")
    def test_manual_adamw_matches_torch_adamw(self):
        initial = torch.tensor([1.5, -0.5, 0.25], dtype=torch.float64)
        manual_parameter = torch.nn.Parameter(initial.clone())
        torch_parameter = torch.nn.Parameter(initial.clone())
        manual_optimizer = ManualAdamW(
            [manual_parameter],
            lr=2e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.1,
        )
        torch_optimizer = torch.optim.AdamW(
            [torch_parameter],
            lr=2e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.1,
        )

        for gradient in (
            torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64),
            torch.tensor([-0.4, 0.5, 0.2], dtype=torch.float64),
            torch.tensor([0.3, 0.1, -0.6], dtype=torch.float64),
        ):
            manual_parameter.grad = gradient.clone()
            torch_parameter.grad = gradient.clone()
            manual_optimizer.step()
            torch_optimizer.step()

        torch.testing.assert_close(
            manual_parameter, torch_parameter, rtol=1e-12, atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
