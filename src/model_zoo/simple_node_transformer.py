"""Transformer operating on nodes to predict edges between nodes."""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt


class CrossAttentionBlock(nn.Module):
    """A single cross-attention block with MLP and residual connections."""

    def __init__(
        self,
        hidden_dim: int = 64,
        n_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, batch_first=True, dropout=dropout
        )

        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-attention with residual.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor, shape (B, N_q, D).
        kv : torch.Tensor
            Key/value tensor, shape (B, N_kv, D).
        kv_mask : torch.Tensor, optional
            Boolean mask for kv positions, shape (B, N_kv).
            True = real position, False = padding (will be ignored).
        """
        key_padding_mask = ~kv_mask if kv_mask is not None else None
        empty_rows = None
        if key_padding_mask is not None:
            # A sample with zero real kv positions has an all-True padding
            # mask; MultiheadAttention softmaxes over an all -inf row for
            # such rows and returns NaN. Temporarily unmask one dummy
            # position for those rows so the softmax stays finite, then
            # zero their output below (they're unused downstream anyway,
            # since detect_and_match already produced 0 detections there).
            empty_rows = key_padding_mask.all(dim=1)
            if empty_rows.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[empty_rows, 0] = False

        attn_out, _ = self.cross_attn(
            self.norm1(q),
            self.norm1(kv),
            self.norm1(kv),
            key_padding_mask=key_padding_mask,
        )

        if empty_rows is not None and empty_rows.any():
            attn_out = attn_out * (~empty_rows).view(-1, 1, 1)

        q = q + attn_out
        q = q + self.mlp(self.norm2(q))
        return q


class SimpleNodeTransformer(nn.Module):
    """Transformer for predicting edges between cell detections."""

    def __init__(
        self,
        feat_dim: int = 33,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_blocks: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.3,
        pair_chunk_size: int | None = 32,
        knn_k: int | None = None,
        use_geom_feats: bool = True,
    ):
        super().__init__()
        self.pair_chunk_size = pair_chunk_size
        self.knn_k = knn_k
        self.use_geom_feats = use_geom_feats
        # feat_dim == 0 is a supported configuration (unet_features=[]): every
        # node token is then just this layer's bias, and geometry alone drives
        # the pair scores. Torch warns that initialising the zero-element
        # weight is a no-op, which is precisely the intent.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Initializing zero-element tensors is a no-op"
            )
            self.proj = nn.Linear(feat_dim, hidden_dim)
        self.norm_in = nn.LayerNorm(hidden_dim)

        self.blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_blocks)
        ])

        self.norm_out = nn.LayerNorm(hidden_dim)

        # Pairwise scoring input: both node embeddings, the relative offset,
        # and (optionally) explicit geometry — see _geom_feats.
        n_geom = self.N_GEOM if use_geom_feats else 0
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3 + n_geom, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    N_GEOM = 4

    def _geom_feats(
        self,
        coords_t: torch.Tensor,   # (B, N_t, 3)   candidate parents
        coords_t1: torch.Tensor,  # (B, N_t1, 3)  children
        mask_t: torch.Tensor | None,
    ) -> torch.Tensor:            # (B, N_t, N_t1, N_GEOM)
        """Hand the pair scorer the nearest-neighbour answer explicitly.

        A geometry-only baseline already resolves ~90% of links on this data,
        but ``rel`` alone forces the MLP to rediscover that from a raw 3-vector.
        These features state it outright, so the head only has to learn where
        to *disagree* with proximity — which is what the sparse GT can teach.

        Channels, per (parent i, child j):
          0  distance, scaled
          1  is-nearest indicator (this is the NN answer)
          2  distance rank among candidate parents, normalised to [0, 1]
          3  log1p((distance - nearest) / nearest) — scale-free "how much
             worse than the best candidate", exactly 0 for the nearest itself
        """
        B, n_t = coords_t.shape[:2]
        d = torch.cdist(coords_t.float(), coords_t1.float())   # (B, N_t, N_t1)

        # Padded parents must not win the ranking or define the minimum.
        d_rank = d if mask_t is None else d.masked_fill(
            ~mask_t.unsqueeze(-1), float("inf")
        )
        min_d = d_rank.min(dim=1, keepdim=True).values          # (B, 1, N_t1)

        order = d_rank.argsort(dim=1)
        rank = torch.empty_like(order)
        ar = torch.arange(n_t, device=d.device).view(1, -1, 1).expand_as(order)
        rank.scatter_(1, order, ar)
        rank = rank.float()

        finite_min = torch.where(torch.isfinite(min_d), min_d, torch.zeros_like(min_d))
        return torch.stack(
            [
                d / 10.0,
                (rank == 0).float(),
                rank / max(n_t - 1, 1),
                # clamp: a padded parent can sit nearer than the nearest valid
                # one, which would make this negative and log1p undefined.
                torch.log1p(((d - finite_min) / (finite_min + 1e-3)).clamp(min=0.0)),
            ],
            dim=-1,
        )

    def _knn_keep(
        self,
        coords_t: torch.Tensor,   # (B, N_t, 3)   candidate parents
        coords_t1: torch.Tensor,  # (B, N_t1, 3)  children
        mask_t: torch.Tensor | None,
    ) -> torch.Tensor:            # (B, N_t, N_t1) bool
        """Keep only each child's K nearest candidate parents.

        Distances are Euclidean in detection-grid voxels. That grid is
        isotropic by construction for the configured UNet — ``pool_stride
        (1, 2, 2)`` over two stages contracts Y/X by 4, exactly cancelling the
        4x Z/XY voxel anisotropy — so voxel distance is proportional to
        physical distance. A pool_stride that breaks that property would need
        the physical voxel size threaded in here.
        """
        B, n_t = coords_t.shape[:2]
        n_t1 = coords_t1.shape[1]
        k = min(self.knn_k, n_t)

        d = torch.cdist(coords_t1.float(), coords_t.float())   # (B, N_t1, N_t)
        if mask_t is not None:
            # Padded slots sit at the origin; they must never win a topk.
            d = d.masked_fill(~mask_t.unsqueeze(1), float("inf"))

        idx = d.topk(k, dim=-1, largest=False).indices          # (B, N_t1, K)
        keep = torch.zeros_like(d, dtype=torch.bool).scatter_(-1, idx, True)
        if mask_t is not None:
            keep &= mask_t.unsqueeze(1)
            # A child left with no candidate (e.g. a fully padded frame) would
            # give an all -inf column and a NaN softmax; restore it instead.
            keep |= ~keep.any(dim=-1, keepdim=True)
        return keep.transpose(1, 2)                             # (B, N_t, N_t1)

    def forward(
        self,
        feat_t: torch.Tensor,
        feat_t1: torch.Tensor,
        coords_t: torch.Tensor,
        coords_t1: torch.Tensor,
        mask_t: torch.Tensor | None = None,
        mask_t1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predict edge logits between detections at consecutive frames.

        Accepts both unbatched (N, D) and batched (B, N, D) inputs.
        When unbatched, a batch dimension is added and removed automatically.

        Parameters
        ----------
        feat_t : torch.Tensor
            Features at time t, shape (N_t, D) or (B, N_t, D).
        feat_t1 : torch.Tensor
            Features at time t+1, shape (N_t1, D) or (B, N_t1, D).
        coords_t : torch.Tensor
            Coordinates (z, y, x) at time t, shape (N_t, 3) or (B, N_t, 3).
        coords_t1 : torch.Tensor
            Coordinates (z, y, x) at time t+1, shape (N_t1, 3) or (B, N_t1, 3).
        mask_t : torch.Tensor, optional
            Boolean mask for t nodes, shape (B, N_t). True = real, False = pad.
        mask_t1 : torch.Tensor, optional
            Boolean mask for t+1 nodes, shape (B, N_t1). True = real, False = pad.

        Returns
        -------
        torch.Tensor
            Edge logits, shape (N_t, N_t1) or (B, N_t, N_t1).
        """
        unbatched = feat_t.ndim == 2
        if unbatched:
            feat_t = feat_t.unsqueeze(0)
            feat_t1 = feat_t1.unsqueeze(0)
            coords_t = coords_t.unsqueeze(0)
            coords_t1 = coords_t1.unsqueeze(0)

        q = self.norm_in(self.proj(feat_t))   # (B, N_t, hidden)
        k = self.norm_in(self.proj(feat_t1))  # (B, N_t1, hidden)

        # Bi-directional cross-attention: t attends to t+1 and vice versa.
        for block in self.blocks:
            # def _q_fn(
            #     q: torch.Tensor,
            #     kv: torch.Tensor,
            #     mask: torch.Tensor | None,
            #     _b: CrossAttentionBlock = block,
            # ) -> torch.Tensor:
            #     return _b(q, kv, kv_mask=mask)

            # def _k_fn(
            #     k: torch.Tensor,
            #     kv: torch.Tensor,
            #     mask: torch.Tensor | None,
            #     _b: CrossAttentionBlock = block,
            # ) -> torch.Tensor:
            #     return _b(k, kv, kv_mask=mask)

            # if torch.is_grad_enabled():
            #     q = grad_ckpt(_q_fn, q, k, mask_t1, use_reentrant=False)
            #     k = grad_ckpt(_k_fn, k, q, mask_t, use_reentrant=False)
            # else:
            #     q = _q_fn(q, k, mask_t1)
            #     k = _k_fn(k, q, mask_t)

            q = block(q, k, kv_mask=mask_t1)
            k = block(k, q, kv_mask=mask_t)

        q = self.norm_out(q)  # (B, N_t, hidden)
        k = self.norm_out(k)  # (B, N_t1, hidden)

        # Build pairwise logits in chunks over N_t to avoid O(N²) peak allocation.
        # Full tensor (B, N_t, N_t1, 2*hidden+3) can be tens of GB for large N.
        # Each chunk is grad-checkpointed: forward peak = B×chunk×N_t1×(2H+3),
        # backward only re-stores tiny q_c / coords slice instead of all activations.
        N_t = q.shape[1]
        chunk = self.pair_chunk_size or N_t
        chunks = []
        pair_mlp = self.pair_mlp

        # Rank and nearest-distance are defined over *all* parents, so compute
        # the geometry once and slice it — never per chunk.
        geom_all = (
            self._geom_feats(coords_t, coords_t1, mask_t)
            if self.use_geom_feats
            else None
        )

        for i in range(0, N_t, chunk):
            q_c = q[:, i : i + chunk, :]
            coords_c = coords_t[:, i : i + chunk, :]
            geom_c = None if geom_all is None else geom_all[:, i : i + chunk]

            def _chunk_fn(
                qc: torch.Tensor,
                kk: torch.Tensor,
                cc: torch.Tensor,
                cc1: torch.Tensor,
                gg: torch.Tensor | None,
                _pm: nn.Module = pair_mlp,
            ) -> torch.Tensor:
                nc_i = qc.shape[1]
                n1 = kk.shape[1]
                qe = qc.unsqueeze(2).expand(-1, -1, n1, -1)
                ke = kk.unsqueeze(1).expand(-1, nc_i, -1, -1)
                rel = (cc.unsqueeze(2) - cc1.unsqueeze(1)) / 100.0
                parts = [qe, ke, rel]
                if gg is not None:
                    parts.append(gg.to(qe.dtype))
                return _pm(torch.cat(parts, dim=-1)).squeeze(-1)

            # if torch.is_grad_enabled():
            #     out = grad_ckpt(
            #         _chunk_fn, q_c, k, coords_c, coords_t1, use_reentrant=False
            #     )
            # else:
            out = _chunk_fn(q_c, k, coords_c, coords_t1, geom_c)

            chunks.append(out)

        logits = torch.cat(chunks, dim=1)  # (B, N_t, N_t1)

        # kNN gating. Softmax downstream runs over dim=0 (the parent axis), so
        # -inf here shrinks its denominator from N_t to K: a child is scored
        # only against parents it could plausibly have come from. Applied to
        # the logits themselves so every consumer (edge_loss, eval, predict)
        # inherits it without changes.
        if self.knn_k is not None and self.knn_k < logits.shape[1]:
            keep = self._knn_keep(coords_t, coords_t1, mask_t)
            logits = logits.masked_fill(~keep, float("-inf"))

        if unbatched:
            logits = logits.squeeze(0)

        return logits
