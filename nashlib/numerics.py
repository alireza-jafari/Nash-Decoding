"""Numerics for every run: float32 and TF32 off, asserted against a float64 reference.

TF32 silently rounds float32 matmul mantissas to 10 bits on Ampere and later. That
is invisible in a config dump and it changes which token wins an argmax when two
logits are close, which is exactly the situation a Nash gap of ~1e-3 lives in. So
the setting is asserted against a float64 reference instead of being trusted: a
2048x2048 product must agree to a relative error below 1e-5, which fails by about
three orders of magnitude when TF32 is active.
"""

from __future__ import annotations

import torch

TF32_RELATIVE_TOLERANCE = 1e-5


def configure(device: str | torch.device | None = None) -> None:
    """Put PyTorch into the numerical regime every experiment in the paper uses."""
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device is not None:
        assert_tf32_disabled(device)


def assert_tf32_disabled(device: str | torch.device) -> float | None:
    """Multiply two float32 matrices and compare against float64.

    Returns the relative error, or None on a device where TF32 does not exist and
    the check therefore did not run. A caller that prints the result must not
    present None as a passing error of zero.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return None
    left = torch.randn(2048, 2048, device=device)
    right = torch.randn(2048, 2048, device=device)
    reference = left.double() @ right.double()
    error = float(
        ((left @ right) - reference.float()).abs().max() / reference.abs().max()
    )
    if error >= TF32_RELATIVE_TOLERANCE:
        raise RuntimeError(
            f"TF32 appears to be enabled: float32 matmul differs from a float64 "
            f"reference by a relative {error:.2e}, above {TF32_RELATIVE_TOLERANCE:.0e}"
        )
    return error
