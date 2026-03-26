import torch
import numpy as np
import sys
sys.path.append('../')

import double_gmm as dogs
from astropy.io import fits
from astropy.table import Table

# starlet_cube: (E, L, X, Y) as torch
features = dogs.cube.mean(dim=0)        # (L, X, Y), average over energy
features = features.permute(1, 2, 0).contiguous()  # (X, Y, L) = (128, 128, L)
import torch

class StarletAttentionAggregator:
    def __init__(self,
                 lvl_start=1,
                 num_lvl=2,
                 include_raw=True,
                 sigma_feat=1.0,
                 window_radius=2,        # 5x5 neighborhood
                 device="cuda:0"):
        self.lvl_start = lvl_start
        self.num_lvl = num_lvl
        self.include_raw = include_raw
        self.sigma_feat = sigma_feat
        self.window_radius = window_radius
        self.device = torch.device(device)

    def _build_starlet_features(self, subset_df):
        # Use your existing starlet_cube() to get (E,L,X,Y)
        cube, e_lvls = dogs.starlet_cube(
            subset_df,
            lvl_start=self.lvl_start,
            num_lvl=self.num_lvl,
            include_raw=self.include_raw,
        )
        if not isinstance(cube, torch.Tensor):
            cube = torch.as_tensor(cube, device=self.device)
        else:
            cube = cube.to(self.device)

        # Reduce energy -> (L,X,Y) then (H,W,C)
        feat = cube.mean(dim=0)            # (L,X,Y)
        feat = feat.permute(1, 2, 0).contiguous()        # (H,W,C)
        return feat  # H=W=128 here

    @torch.no_grad()
    def extract_attention_from_df(self, subset_df) -> torch.Tensor:
        """
        Returns attention tensor of shape (bh,bw,h,w) with bh=bw=1,
        where internally we interpret it as a (H*W, H*W) Markov kernel.
        """
        feats = self._build_starlet_features(subset_df)  # (H,W,C), H=W=128
        H, W, C = feats.shape
        feats = feats.to(self.device)

        # Normalize features per-channel
        mu = feats.mean(dim=(0,1), keepdim=True)
        sigma = feats.std(dim=(0,1), keepdim=True) + 1e-8
        feats = (feats - mu) / sigma   # (H,W,C)

        R = self.window_radius
        sigma2 = 2.0 * (self.sigma_feat ** 2)

        # We'll build a (H,W,H,W) tensor in a sparse-like fashion
        # by only filling local neighborhoods.
        P = torch.zeros((H, W, H, W), device=self.device, dtype=torch.float32)

        for y in range(H):
            y0 = max(0, y - R)
            y1 = min(H, y + R + 1)
            for x in range(W):
                x0 = max(0, x - R)
                x1 = min(W, x + R + 1)

                f_center = feats[y, x]                 # (C,)
                f_neighbors = feats[y0:y1, x0:x1]      # (hy,hx,C)
                diff = f_neighbors - f_center          # (hy,hx,C)
                dist2 = (diff * diff).sum(dim=-1)      # (hy,hx)
                affinities = torch.exp(-dist2 / sigma2)

                # zero out center if you want no self-transition bias, or keep it
                # affinities[y - y0, x - x0] = 0.0

                # normalize over neighbors -> Markov row
                s = affinities.sum() + 1e-8
                P[y, x, y0:y1, x0:x1] = affinities / s

        # Now P[y,x,:,:] sums to 1 over (i,j); shape (H,W,H,W).
        # M2N2 expects (bh,bw,h,w); they then reshape to (bh*bw, h*w).
        # We'll treat (H*W,H*W) later by reshaping P appropriately in Markov code,
        # so here it's enough to return P with a dummy batch/block dimension.
        # But get_cached_attention_tensor assumes bh,bw,h,w from .shape.
        # We can just pick bh=H, bw=W and flatten carefully in create_semantic_markov_map_from_start_state.

        # Simpler: return shape (1,1,H,W) but store the full (H,W,H,W) internally.
        # However, your get_distance_map currently does:
        #   bh,bw,h,w = attn.shape
        #   A = attn.reshape(bh*bw, h*w)
        # which assumes attn is already (h,w,h,w) collapsed.
        #
        # So easiest: follow original M2N2 convention: attn is (h,w,h,w).
        # That means we can skip bh,bw entirely and patch get_cached_attention_tensor
        # to not expect them. If you want zero changes in M2N2, wrap P into (1,1,h*w,h*w):

        # flatten P to (H*W, H*W)
        H = W = 128
        r = H * W
        P_flat = P.view(r, r)  # (r,r)

        # Encode such that bh=1, bw=r, h=1, w=r:
        attn = P_flat.view(1, r, 1, r)  # shape (1, bw=r, h=1, w=r)
        return attn
    
class StarletAttentionAggregatorWrapper:
    def __init__(self, subset_df, **kwargs):
        self.inner = StarletAttentionAggregator(**kwargs)
        self.subset_df = subset_df

    def extract_attention(self, img: np.ndarray) -> torch.Tensor:
        # img is ignored; all semantics come from subset_df / starlet cube
        return self.inner.extract_attention_from_df(self.subset_df)
    
attn_agg = StarletAttentionAggregatorWrapper(subset_df=dogs.subset, sigma_feat=..., window_radius=...)