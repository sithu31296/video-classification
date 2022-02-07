import torch


class TemporalSubsample(torch.nn.Module):
    def __init__(self, num_samples: int, sample_rate: int, temporal_dim: int = -3) -> None:
        """
        Args:
            num_samples (int): The number of equispaced samples to be selected
            sample_rate (int): The sampling rate used to extract from the video
            temporal_dim (int): dimension of temporal to perform temporal subsample.
        """
        super().__init__()
        self._num_samples = num_samples
        self._sample_rate = sample_rate
        self._temporal_dim = temporal_dim
        self._sample_range = self._num_samples * self._sample_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        t = x.shape[self._temporal_dim]
        assert self._num_samples > 0 and t > 0
        sample_pos = max(1, 1 + t - self._sample_range)
        start_idx = 0 if sample_pos == 1 else sample_pos // 2
        offsets = torch.tensor([(idx * self._sample_rate + start_idx) % t for idx in range(self._num_samples)])
        indices = offsets + 1
        indices = torch.clamp(indices, 0, t-1).long()
        return torch.index_select(x, self._temporal_dim, indices)