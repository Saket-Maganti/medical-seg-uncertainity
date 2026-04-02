import torch
import torch.nn as nn


class TTAWrapper:
    def __init__(self, model: nn.Module, n_augmentations: int = 6):
        self.model = model
        self.n_augmentations = n_augmentations
        self.transforms = [
            ("identity", self._identity, self._identity),
            ("hflip", self._hflip, self._hflip),
            ("vflip", self._vflip, self._vflip),
            ("rot90", self._rot90, self._rot270),
            ("rot180", self._rot180, self._rot180),
            ("rot270", self._rot270, self._rot90),
        ][:n_augmentations]

    @staticmethod
    def _identity(x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def _hflip(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=(-1,))

    @staticmethod
    def _vflip(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=(-2,))

    @staticmethod
    def _rot90(x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, k=1, dims=(-2, -1))

    @staticmethod
    def _rot180(x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, k=2, dims=(-2, -1))

    @staticmethod
    def _rot270(x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, k=3, dims=(-2, -1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict:
        self.model.eval()
        all_probs = []

        for _, transform, inverse in self.transforms:
            logits = self.model(transform(x))
            probs = torch.sigmoid(logits)
            all_probs.append(inverse(probs))

        all_probs = torch.stack(all_probs, dim=0)
        mean_pred = all_probs.mean(dim=0)
        variance = all_probs.var(dim=0)

        eps = 1e-8
        entropy = -(mean_pred * torch.log(mean_pred + eps) +
                    (1 - mean_pred) * torch.log(1 - mean_pred + eps))

        return {
            "mean": mean_pred,
            "variance": variance,
            "entropy": entropy,
            "all_probs": all_probs,
            "method": "tta",
        }
