# pref_policy_cross.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x, mask, dim):
    # x: (..., L, D), mask: (..., L) with 1/0
    mask = mask.unsqueeze(-1).to(x.dtype)  # (..., L, 1)
    x = x * mask
    denom = mask.sum(dim=dim).clamp(min=1.0)
    return x.sum(dim=dim) / denom


def masked_softmax(logits, mask, dim=-1):
    # mask: 1=valid, 0=invalid
    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(mask == 0, neg_inf)
    return F.softmax(masked_logits, dim=dim), masked_logits


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers=2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hid_dim), nn.ReLU()]
            d = hid_dim
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SimpleCrossAttention(nn.Module):
    """
    Lightweight single cross-attention:
      queries: UAV tokens   (B, N, D)
      keys/values: sensing tokens (B, M, D)
      kv_mask: (B, M), 1=valid, 0=invalid

    Returns:
      ctx: (B, N, D)
      attn: (B, N, M)
    """
    def __init__(self, demb):
        super().__init__()
        self.q_proj = nn.Linear(demb, demb)
        self.k_proj = nn.Linear(demb, demb)
        self.v_proj = nn.Linear(demb, demb)
        self.out_proj = nn.Linear(demb, demb)
        self.scale = demb ** -0.5

    def forward(self, q_tokens, kv_tokens, kv_mask):
        # q_tokens: (B,N,D), kv_tokens: (B,M,D), kv_mask: (B,M)
        q = self.q_proj(q_tokens)                    # (B,N,D)
        k = self.k_proj(kv_tokens)                  # (B,M,D)
        v = self.v_proj(kv_tokens)                  # (B,M,D)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B,N,M)

        # Avoid all-invalid rows causing NaN
        all_invalid = (kv_mask.sum(dim=-1) == 0)    # (B,)
        safe_mask = kv_mask
        if all_invalid.any():
            safe_mask = kv_mask.clone()
            safe_mask[all_invalid, 0] = 1.0

        neg_inf = torch.finfo(scores.dtype).min
        masked_scores = scores.masked_fill(safe_mask.unsqueeze(1) == 0, neg_inf)
        attn = F.softmax(masked_scores, dim=-1)     # (B,N,M)

        if all_invalid.any():
            attn = torch.where(
                all_invalid[:, None, None],
                torch.zeros_like(attn),
                attn,
            )

        ctx = torch.matmul(attn, v)                 # (B,N,D)
        ctx = self.out_proj(ctx)                    # (B,N,D)
        return ctx, attn


class PreferenceActorCritic(nn.Module):
    """
    Factorized policy:
      pi(uav | obs) then pi(sens | uav, obs)
    Includes SKIP as sens_id=0 (virtual token).

    Compared with the original lightweight version:
      - keep MLP token encoders
      - add ONE cross-attention only: UAV -> Sensing
      - no UAV self-attention
      - no sensing self-attention
      - critic stays lightweight
    """
    def __init__(self, duav=9, dsens=8, dglobal=4, demb=128, hid=256, max_sens=128):
        super().__init__()
        self.max_sens = max_sens

        # Encoders
        self.uav_enc = MLP(duav, hid, demb)
        self.sens_enc = MLP(dsens, hid, demb)

        # Small global projection only for conditioning attention
        self.g_proj = nn.Linear(dglobal, demb)

        # Existing sensing-set summary pooling
        self.sens_query = nn.Parameter(torch.randn(demb))
        self.sens_attn = nn.Linear(demb, demb, bias=False)

        # NEW: one lightweight cross-attention (UAV queries sensing set)
        self.u2s_cross = SimpleCrossAttention(demb)

        # Stage-1 fusion and scoring (task-aware UAV scoring)
        # Keep interface and scale close to original version
        self.fus1 = MLP(demb + demb + dglobal, hid, demb)
        self.uav_head = nn.Linear(demb, 1)

        # Stage-2 fusion and scoring (UAV->Sensing)
        self.skip_token = nn.Parameter(torch.zeros(demb))
        self.fus2 = MLP(demb + demb + dglobal, hid, demb)
        self.sens_head = nn.Linear(demb, 1)

        # Critic: still lightweight
        self.v_mlp = MLP(demb + demb + dglobal, hid, 1)

    def encode_sensing_set(self, sens_e, sens_mask):
        """
        sens_e: (B, M, D), mask: (B, M)
        attention pooling -> z_sens: (B, D)
        """
        q = self.sens_query.view(1, 1, -1)          # (1,1,D)
        k = self.sens_attn(sens_e)                  # (B,M,D)
        score = (q * k).sum(-1)                     # (B,M)

        all_invalid = (sens_mask.sum(dim=-1) == 0)  # (B,)
        safe_mask = sens_mask
        if all_invalid.any():
            safe_mask = sens_mask.clone()
            safe_mask[all_invalid, 0] = 1.0

        attn, _ = masked_softmax(score, safe_mask, dim=-1)
        z = (attn.unsqueeze(-1) * sens_e).sum(dim=1)  # (B,D)

        if all_invalid.any():
            z = torch.where(
                all_invalid[:, None],
                torch.zeros_like(z),
                z,
            )
        return z

    def forward(self, obs):
        """
        obs: dict of tensors:
          uav_feats: (B, N, Duav), uav_mask: (B, N)
          sens_feats: (B, M, Dsens), sens_mask: (B, M)
          global_feats: (B, Dg)
        returns:
          uav_logits: (B,N)
          value: (B,)
          cache: embeddings to reuse
        """
        uav_x = obs["uav_feats"]       # (B,N,Duav)
        uav_mask = obs["uav_mask"]     # (B,N)
        sens_x = obs["sens_feats"]     # (B,M,Dsens)
        sens_mask = obs["sens_mask"]   # (B,M)
        g = obs["global_feats"]        # (B,Dg)

        # Base token encodings
        uav_e = self.uav_enc(uav_x)    # (B,N,D)
        sens_e = self.sens_enc(sens_x) # (B,M,D)

        # Global conditioning for attention only (cheap)
        g_e = self.g_proj(g).unsqueeze(1)  # (B,1,D)

        # Global-modulated tokens for cross-attention
        uav_q = uav_e + g_e
        sens_kv = sens_e + g_e

        # NEW: task-aware UAV context from sensing set
        uav_ctx, u2s_attn = self.u2s_cross(uav_q, sens_kv, sens_mask)  # (B,N,D), (B,N,M)

        # Residual fusion: contextualized UAV embedding
        # This is what stage-1 and stage-2 will use
        uav_actor_e = uav_e + uav_ctx

        # Existing global sensing summary
        z_sens = self.encode_sensing_set(sens_e, sens_mask)  # (B,D)

        # Stage-1 logits over UAV
        z_sens_rep = z_sens.unsqueeze(1).expand_as(uav_actor_e)  # (B,N,D)
        g_rep = g.unsqueeze(1).expand(uav_actor_e.size(0), uav_actor_e.size(1), g.size(-1))
        fus1_in = torch.cat([uav_actor_e, z_sens_rep, g_rep], dim=-1)
        uav_h = self.fus1(fus1_in)                              # (B,N,D)
        uav_logits = self.uav_head(uav_h).squeeze(-1)           # (B,N)

        # Critic stays lightweight:
        # use pooled base UAV embedding + pooled sensing summary + global
        z_uav = masked_mean(uav_e, uav_mask, dim=1)             # (B,D)
        v_in = torch.cat([z_uav, z_sens, g], dim=-1)
        value = self.v_mlp(v_in).squeeze(-1)                    # (B,)

        # IMPORTANT:
        # cache["uav_e"] intentionally stores contextualized UAV embedding
        # so your existing trainer can gather chosen_uav_e as before,
        # but stage-2 now benefits from the cross-attention automatically.
        cache = dict(
            uav_e=uav_actor_e,   # contextualized for actor / stage-2
            sens_e=sens_e,
            z_sens=z_sens,
            uav_base_e=uav_e,    # optional: raw UAV embedding for debugging
            uav_ctx=uav_ctx,     # optional: cross-attn context
            u2s_attn=u2s_attn,   # optional: attention map for analysis
        )
        return uav_logits, value, cache

    def conditional_sens_logits(self, chosen_uav_e, sens_e, sens_mask, global_feats):
        """
        chosen_uav_e: (B,D)
        sens_e: (B,M,D)
        Return sens_logits: (B, M+1) where index 0 is SKIP
        """
        B, M, D = sens_e.shape

        # prepend skip token
        skip = self.skip_token.view(1, 1, -1).expand(B, 1, D)       # (B,1,D)
        sens_all = torch.cat([skip, sens_e], dim=1)                 # (B,M+1,D)

        # build mask with skip always valid
        skip_mask = torch.ones((B, 1), device=sens_mask.device, dtype=sens_mask.dtype)
        sens_mask_all = torch.cat([skip_mask, sens_mask], dim=1)    # (B,M+1)

        u_rep = chosen_uav_e.unsqueeze(1).expand(B, M + 1, D)
        g_rep = global_feats.unsqueeze(1).expand(B, M + 1, global_feats.size(-1))

        fus2_in = torch.cat([u_rep, sens_all, g_rep], dim=-1)
        h = self.fus2(fus2_in)
        sens_logits = self.sens_head(h).squeeze(-1)                 # (B,M+1)

        return sens_logits, sens_mask_all