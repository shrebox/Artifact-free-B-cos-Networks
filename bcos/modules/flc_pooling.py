"""FLC Pooling module
can be used and distributed under the MIT license
Reference:
[1] Grabinski, J., Jung, S., Keuper, J., & Keuper, M. (2022).
    "FrequencyLowCut Pooling--Plug & Play against Catastrophic Overfitting."
    European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022.
"""

import logging

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init

logging.basicConfig(
    filename="train_debug_flc.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class FLC_Pooling(nn.Module):
    """Frequency Low-Cut pooling.
    Device-safe (no hardcoded .cuda()), caches a 2D Hamming window per input size,
    and keeps buffers on the correct device for DDP.
    """

    def __init__(self, transpose: bool = True, odd: bool = True):
        super().__init__()
        self.transpose = transpose
        self.odd = odd
        # non-persistent buffer for the window so it follows .to(device) but isn't saved
        self.register_buffer("window2d", None, persistent=False)
        self._cached_hw = None  # cache (H, W)

    def _make_window(self, h: int, w: int, device, dtype):
        # create separable 2D Hamming window on the correct device/dtype
        wh = torch.from_numpy(np.abs(np.hamming(h))).to(device=device, dtype=dtype)
        ww = torch.from_numpy(np.abs(np.hamming(w))).to(device=device, dtype=dtype)
        w2d = torch.sqrt(torch.outer(wh, ww))
        return w2d.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device, dtype = x.device, x.dtype

        # pad once (keep device)
        x_padded = F.pad(
            x, [0, 1, 0, 1]
        )  # larger padding helps; mirror padding; [size(x)//2,size(x)//2 - 1,size(x)//2,size(x)//2 - 1]

        # optionally transpose (kept from original, though default path didn't use it)
        if self.transpose:
            # transpose spatial dims H<->W – matches prior commented behavior only if desired
            # If not needed, set transpose=False in the caller.
            x_fft_in = x_padded.transpose(2, 3)
        else:
            x_fft_in = x_padded

        low_part = torch.fft.fftshift(torch.fft.fft2(x_fft_in, norm="forward"))

        # (re)build cached window if size/device/dtype changed
        h, w = low_part.size(2), low_part.size(3)
        if (
            (self.window2d is None)
            or (self._cached_hw != (h, w))
            or (self.window2d.device != device)
            or (self.window2d.dtype != dtype)
        ):
            self.window2d = self._make_window(h, w, device, dtype)
            self._cached_hw = (h, w)

        low_part = low_part * self.window2d

        # crop central band
        h0 = int((x_fft_in.size(2) + 1) / 4)  # 3/8
        h1 = int((x_fft_in.size(2) + 1) / 4 * 3)  # 5/8 for stride 4
        w0 = int((x_fft_in.size(3) + 1) / 4)
        w1 = int((x_fft_in.size(3) + 1) / 4 * 3)
        low_part = low_part[:, :, h0:h1, w0:w1]

        out = torch.fft.ifft2(torch.fft.ifftshift(low_part), norm="forward").abs()

        # remove padding
        if self.odd:
            out = out[:, :, 1:, 1:]
        else:
            out = out[:, :, :-1, :-1]

        # undo optional transpose to keep shape consistent with caller expectations
        if self.transpose:
            out = out.transpose(2, 3)

        return out
