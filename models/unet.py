"""
Stage 2 — U-Net Coarse Breakpoint Localizer
============================================
Takes confirmed SV windows from Stage 1 and predicts a 2D breakpoint
probability heatmap over the multi-channel signal matrix.

Input  : (B, C, H, W)  — C signal channels (SR_RP, RD_LOW, RD_CLIPPED),
                          H=W=200 bins, log-normalized and [0,1] scaled.
Output : (B, 1, H, W)  — raw logits; spatial softmax gives probability map.

The argmax of the predicted heatmap gives the coarse bin-pair (i, j),
which is converted to genomic coordinates via:
    predA = intervalA.start + i * bin_bp
    predB = intervalB.start + j * bin_bp

Architecture
------------
Encoder:
    Block 1:  C  -> 32    (local features)
    Block 2:  32 -> 64    (spatial context)
    Block 3:  64 -> 128   (deep features)
    Bottleneck: 128 -> 256

Decoder (with skip connections):
    Up3: 256 -> 128  + skip e3 (128) -> dec3 block (256 -> 128)
    Up2: 128 -> 64   + skip e2 (64)  -> dec2 block (128 -> 64)
    Up1: 64  -> 32   + skip e1 (32)  -> dec1 block (64  -> 32)

Output head:
    1×1 Conv -> (B, 1, H, W) logits
    Training: spatial cross-entropy vs. 2D soft Gaussian target
              centered on true breakpoint bin from VCF coordinates.
"""

import torch
import torch.nn as nn


class UNetLocalizer(nn.Module):
    """
    U-Net coarse breakpoint localizer.

    Parameters
    ----------
    in_ch : int
        Number of input signal channels (default 3: SR_RP, RD_LOW, RD_CLIPPED).
    """

    def __init__(self, in_ch: int = 3):
        super().__init__()

        # --- Encoder ---
        self.enc1 = self._block(in_ch, 32)
        self.enc2 = self._block(32, 64)
        self.enc3 = self._block(64, 128)
        self.pool = nn.MaxPool2d(2)

        # --- Bottleneck ---
        self.bottleneck = self._block(128, 256)

        # --- Decoder (ConvTranspose + skip + conv block) ---
        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._block(256, 128)   # 128 up + 128 skip

        self.up2  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._block(128, 64)    # 64 up + 64 skip

        self.up1  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._block(64, 32)     # 32 up + 32 skip

        # --- Output head ---
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        """Two conv layers with BatchNorm and ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W)

        Returns
        -------
        logits : (B, 1, H, W)
            Raw logits; apply spatial softmax to get breakpoint probability map.
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)
