"""Configurable PyTorch MNIST classifier.

Architectures (arch param):
  linear  — Flatten → Linear(784, 10)                                ~92% acc
  small   — Conv(1,C,3,pad=1) → ReLU → MaxPool(2) → Linear(C×14×14, 10)  ~97% acc
  lenet   — Conv(1,C,3,pad=1) → ReLU → Pool(2)                       ~99% acc
            → Conv(C,2C,3,pad=1) → ReLU → Pool(2)
            → Flatten → Linear(2C×7×7, H) → ReLU → Linear(H, 10)

Active-bit tracking:
  Forward hooks on every nn.ReLU (and the output of the final hidden linear, if any)
  count  count_nonzero(output) × 16  (float32 = 16 expected-active-bits per non-zero).
  Hooks are registered/removed explicitly so they don't slow training.

Typical usage:
    model = ConventionalCNN(arch="lenet", n_filters=8, hidden_size=128)

    # Training (no hooks — fast)
    output = model(x)

    # Inference with cost tracking
    model.register_cost_hooks()
    model.eval()
    with torch.no_grad():
        model.reset_active_bits()
        output = model(x)
        bits = model.get_active_bits()
    model.remove_cost_hooks()
"""

import torch
import torch.nn as nn


class ConventionalCNN(nn.Module):

    def __init__(
        self,
        arch: str = "lenet",
        n_filters: int = 8,
        n_filters2: int | None = None,
        hidden_size: int = 128,
    ):
        """
        Parameters
        ----------
        arch        : "linear", "small", or "lenet"
        n_filters   : number of filters in first conv layer (ignored for "linear")
        n_filters2  : filters in second conv ("lenet" only); defaults to n_filters × 2
        hidden_size : hidden units in dense layer ("lenet" only)
        """
        super().__init__()
        self.arch = arch
        self.n_filters = n_filters
        self.n_filters2 = n_filters2 if n_filters2 is not None else n_filters * 2
        self.hidden_size = hidden_size

        self._active_bits = 0
        self._hooks: list = []

        if arch == "linear":
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 10),
            )
        elif arch == "small":
            flat = n_filters * 14 * 14
            self.net = nn.Sequential(
                nn.Conv2d(1, n_filters, 3, padding=1),  # (C, 28, 28)
                nn.ReLU(),
                nn.MaxPool2d(2),                         # (C, 14, 14)
                nn.Flatten(),
                nn.Linear(flat, 10),
            )
        elif arch == "lenet":
            nf2 = self.n_filters2
            flat = nf2 * 7 * 7
            self.net = nn.Sequential(
                nn.Conv2d(1, n_filters, 3, padding=1),  # (C, 28, 28)
                nn.ReLU(),
                nn.MaxPool2d(2),                         # (C, 14, 14)
                nn.Conv2d(n_filters, nf2, 3, padding=1), # (2C, 14, 14)
                nn.ReLU(),
                nn.MaxPool2d(2),                         # (2C, 7, 7)
                nn.Flatten(),
                nn.Linear(flat, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 10),
            )
        else:
            raise ValueError(f"Unknown arch: {arch!r}. Choose 'linear', 'small', or 'lenet'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # ------------------------------------------------------------------ #
    # Active-bit cost tracking
    # ------------------------------------------------------------------ #

    def _make_hook(self):
        """Return a forward hook that accumulates active bits from layer output."""
        def hook(module, input, output):
            # float32: 16 expected active bits per non-zero value
            self._active_bits += int(torch.count_nonzero(output).item()) * 16
        return hook

    def register_cost_hooks(self) -> None:
        """Attach hooks to all ReLU layers. Call once before eval inference."""
        self.remove_cost_hooks()
        self._active_bits = 0
        for module in self.net:
            if isinstance(module, nn.ReLU):
                h = module.register_forward_hook(self._make_hook())
                self._hooks.append(h)

    def remove_cost_hooks(self) -> None:
        """Detach all cost-tracking hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def reset_active_bits(self) -> None:
        """Zero the accumulated active-bit counter (call before each inference)."""
        self._active_bits = 0

    def get_active_bits(self) -> int:
        """Return total active bits accumulated since last reset_active_bits()."""
        return self._active_bits

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #

    def config_dict(self) -> dict:
        """Return a dict describing this model's configuration."""
        return {
            "arch": self.arch,
            "n_filters": self.n_filters,
            "n_filters2": self.n_filters2,
            "hidden_size": self.hidden_size,
        }
