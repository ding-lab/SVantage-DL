"""
Stage 1 — Context-modulated Transformer Encoder
================================================
Classifies SV type (DEL / DUP / INS / INV) from tokenized long-read signal.

Architecture
------------
1. Linear projection + LayerNorm maps raw token features -> d_model.
2. A learnable CLS token is prepended to the token sequence.
3. N_LAYERS post-norm transformer blocks, each:
       MHA(x, x, x, attn_mask=attn_bias) -> x + out -> LayerNorm  [sub-block 1]
       MLP(x)                             -> x + out -> LayerNorm  [sub-block 2]
   The SR_RP read-pair bias is tiled across heads and added to attention logits.
4. CLS representation (position 0) feeds two heads:
       sv_head      -> SV type logits  (DEL / DUP / INS / INV)
       present_head -> SV presence confidence score  [0, 1]
"""

import torch
import torch.nn as nn


class BiasMHABlock(nn.Module):
    """
    Post-norm transformer block with injected attention bias.

    Parameters
    ----------
    d_model : int
    n_heads : int
    dim_ff  : int   hidden dimension of MLP
    dropout : float
    """

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.mha = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x         : (B, N, d_model)
        attn_bias : (B, N, N)  — tiled across heads inside this call

        Returns
        -------
        x : (B, N, d_model)
        """
        # tile bias across heads: (B, N, N) -> (B * n_heads, N, N)
        bias = attn_bias.repeat_interleave(self.n_heads, dim=0)
        out, _ = self.mha(x, x, x, attn_mask=bias, need_weights=False)
        x = self.norm1(x + out)
        x = self.norm2(x + self.mlp(x))
        return x


class Stage1Transformer(nn.Module):
    """
    Context-modulated Transformer Encoder for SV type classification.

    Parameters
    ----------
    d_in      : int   input token feature dimension (from Cue tokenization)
    n_classes : int   number of SV type classes
    d_model   : int   internal embedding dimension  (default 128)
    n_heads   : int   attention heads               (default 4)
    n_layers  : int   transformer blocks            (default 4)
    dropout   : float                               (default 0.1)
    """

    def __init__(
        self,
        d_in: int,
        n_classes: int,
        d_model: int  = 128,
        n_heads: int  = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # project raw token features -> d_model
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LayerNorm(d_model),
        )

        # learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # transformer blocks
        self.blocks = nn.ModuleList([
            BiasMHABlock(d_model=d_model, n_heads=n_heads,
                         dim_ff=4 * d_model, dropout=dropout)
            for _ in range(n_layers)
        ])

        # output heads
        self.sv_head      = nn.Linear(d_model, n_classes)  # SV type logits
        self.present_head = nn.Linear(d_model, 1)           # confidence score

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor):
        """
        Parameters
        ----------
        x         : (B, N, d_in)    token features
        attn_bias : (B, N, N)       log-normalized SR_RP matrix

        Returns
        -------
        sv_logits    : (B, n_classes)
        present_logit: (B,)
        """
        B, N, _ = x.shape

        h   = self.proj(x)                          # (B, N, d_model)
        cls = self.cls_token.expand(B, 1, -1)
        h   = torch.cat([cls, h], dim=1)            # (B, N+1, d_model)

        # pad attn_bias for the CLS token (CLS row/col = 0)
        bias = torch.zeros(B, N + 1, N + 1,
                           device=attn_bias.device, dtype=attn_bias.dtype)
        bias[:, 1:, 1:] = attn_bias

        for block in self.blocks:
            h = block(h, bias)

        h_cls = h[:, 0]                             # (B, d_model)
        return self.sv_head(h_cls), self.present_head(h_cls).squeeze(-1)
