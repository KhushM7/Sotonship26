"""Window classifiers: (B, 30, 51) features -> (B, 4) logits."""

import torch
from torch import nn

from fallsense.features import N_FEATURES
from fallsense.windows import N_CLASSES


class FallGRU(nn.Module):
    def __init__(
        self,
        f_in: int = N_FEATURES,
        hidden: int = 64,
        n_cls: int = N_CLASSES,
        layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.inp = nn.Sequential(
            nn.Linear(f_in, hidden), nn.ReLU(), nn.LayerNorm(hidden)
        )
        self.gru = nn.GRU(
            hidden, hidden, num_layers=layers, batch_first=True, dropout=dropout
        )
        self.head = nn.Linear(hidden, n_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 30, 51)
        h, _ = self.gru(self.inp(x))
        return self.head(h[:, -1])  # logits from the last time step


class FallTCN(nn.Module):
    """3 dilated 1D conv blocks (dilation 1, 2, 4), kernel 5, 64 channels,
    BatchNorm + ReLU + Dropout(0.1), then concat(mean over time, last step) -> Linear."""

    def __init__(
        self,
        f_in: int = N_FEATURES,
        channels: int = 64,
        n_cls: int = N_CLASSES,
        kernel: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        c_in = f_in
        for dilation in (1, 2, 4):
            blocks += [
                # "same" padding keeps all 30 time steps
                nn.Conv1d(
                    c_in,
                    channels,
                    kernel,
                    dilation=dilation,
                    padding=dilation * (kernel - 1) // 2,
                ),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            c_in = channels
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(2 * channels, n_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 30, 51)
        h = self.body(x.transpose(1, 2))  # (B, C, 30)
        return self.head(torch.cat([h.mean(dim=2), h[:, :, -1]], dim=1))


MODELS = {"gru": FallGRU, "tcn": FallTCN}


def build_model(name: str) -> nn.Module:
    return MODELS[name]()
