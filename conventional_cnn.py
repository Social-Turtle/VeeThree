"""Configurable PyTorch MNIST classifier.

Architectures (arch param):
  linear  — Flatten → Linear(784, 10)                                ~92% acc
  small   — Conv(1,C,3,pad=1) → ReLU → MaxPool(2) → Linear(C×14×14, 10)  ~97% acc
  lenet   — Conv(1,C,3,pad=1) → ReLU → Pool(2)                       ~99% acc
            → Conv(C,2C,3,pad=1) → ReLU → Pool(2)
            → Flatten → Linear(2C×7×7, H) → ReLU → Linear(H, 10)

Cost tracking:
    Forward hooks on each Conv2d/Linear layer accumulate:
        1) active_signals: transmitted non-zero values, charged as 16 each
             (float32 expected active bits = bit_width/2)
        2) seq2s: always 0 for this model family
        3) adds: scalar additions, scaled by 32-bit arithmetic width
        4) multiplies: scalar multiplies, scaled by 32-bit arithmetic width

    Hooks are registered/removed explicitly so they don't slow training.

Typical usage:
    model = ConventionalCNN(arch="lenet", n_filters=8, hidden_size=128)

    # Training (no hooks — fast)
    output = model(x)

    # Inference with cost tracking
    model.register_cost_hooks()
    model.eval()
    with torch.no_grad():
        model.reset_costs()
        output = model(x)
        costs = model.get_costs()
    model.remove_cost_hooks()
"""

import torch
import torch.nn as nn


BIT_WIDTH = 32
EXPECTED_ACTIVE_SIGNAL_BITS = BIT_WIDTH // 2


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

        self._costs = {
            "active_signals": 0,
            "seq2s": 0,
            "adds": 0,
            "multiplies": 0,
        }
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
    # Cost tracking
    # ------------------------------------------------------------------ #

    def _make_io_hook(self, is_last_tracked_layer: bool):
        """Return a forward hook that tracks signals and arithmetic costs.

        For each Conv2d/Linear, charge:
          - active_signals on input and output tensors (output excluded on final layer)
          - additions and multiplications as 32-bit scalar arithmetic operations.
        """
        def hook(module, input, output):
            in_tensor = input[0]
            out_tensor = output

            self._costs["active_signals"] += int(torch.count_nonzero(in_tensor).item()) * EXPECTED_ACTIVE_SIGNAL_BITS
            if not is_last_tracked_layer:
                self._costs["active_signals"] += int(torch.count_nonzero(out_tensor).item()) * EXPECTED_ACTIVE_SIGNAL_BITS

            if isinstance(module, nn.Conv2d):
                batch, out_ch, out_h, out_w = out_tensor.shape
                out_elems = batch * out_ch * out_h * out_w
                k_h, k_w = module.kernel_size
                in_ch_per_group = module.in_channels // module.groups
                macs_per_output = in_ch_per_group * k_h * k_w

                multiplies = out_elems * macs_per_output
                adds = out_elems * max(macs_per_output - 1, 0)
                if module.bias is not None:
                    adds += out_elems
            elif isinstance(module, nn.Linear):
                if out_tensor.dim() == 1:
                    batch = 1
                    out_features = out_tensor.shape[0]
                else:
                    batch = out_tensor.shape[0]
                    out_features = out_tensor.shape[-1]
                out_elems = batch * out_features

                multiplies = out_elems * module.in_features
                adds = out_elems * max(module.in_features - 1, 0)
                if module.bias is not None:
                    adds += out_elems
            else:
                return

            self._costs["multiplies"] += int(multiplies) * BIT_WIDTH
            self._costs["adds"] += int(adds) * BIT_WIDTH
        return hook

    def register_cost_hooks(self) -> None:
        """Attach cost hooks to every Conv2d and Linear layer."""
        self.remove_cost_hooks()
        self.reset_costs()

        tracked_indices = [
            idx for idx, module in enumerate(self.net)
            if isinstance(module, (nn.Conv2d, nn.Linear))
        ]
        last_idx = tracked_indices[-1] if tracked_indices else None

        for idx, module in enumerate(self.net):
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                h = module.register_forward_hook(self._make_io_hook(is_last_tracked_layer=(idx == last_idx)))
                self._hooks.append(h)

    def remove_cost_hooks(self) -> None:
        """Detach all cost-tracking hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def reset_costs(self) -> None:
        """Zero all accumulated cost counters (call before each inference)."""
        self._costs = {
            "active_signals": 0,
            "seq2s": 0,
            "adds": 0,
            "multiplies": 0,
        }

    def get_costs(self) -> dict:
        """Return a copy of the current cost counters."""
        return dict(self._costs)

    # Backward-compatible aliases for older benchmarking code.
    def reset_active_bits(self) -> None:
        self.reset_costs()

    def get_active_bits(self) -> int:
        """Return active_signals counter (legacy name retained for compatibility)."""
        return self._costs["active_signals"]

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
