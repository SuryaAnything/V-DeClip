# model.py
#
# This file implements the MMCGD model, a multimodal video–text alignment
# architecture designed for retrieval, masked text prediction, and
# component-wise video decomposition. The model integrates several ideas:
#
# 1. A deep MLP adapter + temporal transformer for converting frame-level
#    video embeddings into structured temporal features.
#
# 2. Multi-component decomposition, where the video features are mapped into
#    several learnable component streams. These components are processed using
#    lightweight non-linear projectors, orthogonalization, and diversity
#    regularizers. This encourages the model to represent different semantic
#    factors separately.
#
# 3. A gated fusion mechanism where the global text embedding predicts a
#    distribution over components. These weights define how video components
#    are mixed to form the final video embedding.
#
# 4. Masked text prediction through cross-attention from video components
#    to masked token embeddings.
#
# 5. Contrastive alignment between final video and text embeddings using a
#    momentum encoder and a queue of negatives for stable optimization.
#
# 6. Additional regularizers including covariance, variance, uniformity,
#    and orthogonality losses to prevent feature collapse.
#
# This file contains the full model definition along with a diagnostic
# function to inspect how each block contributes during training.


import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class MMCGD(nn.Module):
    """
    Main multimodal model combining:
    - Multi-component decomposition of video features
    - Soft Gram-Schmidt orthogonalization
    - Diversity, variance, uniformity, and covariance regularizers
    - Text-conditioned gating over components
    - Masked text prediction via cross-attention
    - Momentum encoder with negative queue for contrastive learning
    """

    def __init__(self,
                 video_input_dim=1024,
                 text_dim=512,
                 field_dim=512,
                 num_heads=8,
                 transformer_layers=6,
                 num_components=5,
                 queue_size=8192,
                 momentum=0.998,
                 use_momentum=True):

        super().__init__()

        self.field_dim = field_dim
        self.text_dim = text_dim
        self.num_components = int(num_components)

        # Video Adapter
        # Converts raw precomputed video features into a common representation.
        # Multiple layers + dropout create a non-linear embedding field suitable
        # for temporal processing.
        self.video_proj = nn.Sequential(
            nn.Linear(video_input_dim, field_dim),
            nn.LayerNorm(field_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(field_dim, 2048),
            nn.Linear(2048, field_dim),
            nn.LayerNorm(field_dim),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(field_dim, field_dim),
        )

        # Learnable CLS token for temporal transformer pooling
        self.video_cls_token = nn.Parameter(torch.randn(1, 1, field_dim))

        # Temporal encoder for frame-level sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=field_dim,
            nhead=num_heads,
            batch_first=True,
            dropout=0.25
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )

        # Component Heads
        # Each head extracts a separate semantic component from each frame.
        self.component_heads = nn.ModuleList(
            [nn.Linear(field_dim, field_dim) for _ in range(self.num_components)]
        )

        # Optional non-linear projector for each component
        self.component_projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(field_dim),
                    nn.GELU(),
                    nn.Linear(field_dim, field_dim)
                )
                for _ in range(self.num_components)
            ]
        )

        # Masked text prediction via cross-attention
        # Query = masked text, Key/Value = concatenated video components
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=text_dim,
            kdim=field_dim * self.num_components,
            vdim=field_dim * self.num_components,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.25
        )

        # Predictor for masked token vectors
        self.text_predictor = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.LayerNorm(text_dim),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(text_dim, 768),
            nn.Linear(768, text_dim),
            nn.LayerNorm(text_dim),
            nn.Tanh(),
            nn.Dropout(0.6),
            nn.Linear(text_dim, text_dim),
        )

        # Gating Mechanism
        # Text-global embedding predicts weights over components.
        self.gate_projector = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_components)
        )

        # Trainable logit scale for contrastive loss
        self.logit_scale = nn.Parameter(torch.ones([]) * 3.5)

        # Negative queue and momentum encoder for contrastive learning
        self.queue_size = int(queue_size)
        self.momentum = float(momentum)
        self.use_momentum = bool(use_momentum)

        if self.queue_size > 0:
            self.register_buffer("queue_v", torch.zeros(self.queue_size, field_dim))
            self.register_buffer("queue_t", torch.zeros(self.queue_size, text_dim))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        else:
            self.register_buffer("queue_v", torch.zeros(0, field_dim))
            self.register_buffer("queue_t", torch.zeros(0, text_dim))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Build momentum encoder if enabled
        if self.use_momentum:
            self._create_momentum_encoder()
        else:
            self.momentum_encoder = None

        # Initialize all weights
        self.apply(self._init_weights)

    # Basic Xavier initialization for all linear projections
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    # Video component extraction
    # Runs the video adapter, adds CLS token, applies transformer, and
    # produces K learned video components with per-component projection.
    def get_video_fields(self, video_emb):
        if self.training:
            video_emb = video_emb + torch.randn_like(video_emb) * 0.02

        x = self.video_proj(video_emb)
        b = x.shape[0]

        cls_tokens = self.video_cls_token.expand(b, 1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.temporal_encoder(x)

        v_cls = x[:, 0:1]
        v_frames = x[:, 1:]

        components = []
        for head, proj in zip(self.component_heads, self.component_projectors):
            raw = head(v_frames)
            comp = proj(raw)
            comp = comp + v_cls
            components.append(comp)

        return components

    # Soft differentiable Gram-Schmidt to reduce component collapse
    def orthogonalize_components(self, components):
        K = len(components)
        B, T, D = components[0].shape

        comps_flat = [c.reshape(B * T, D) for c in components]
        u_list = []
        ortho = []

        for i, v in enumerate(comps_flat):
            if len(u_list) == 0:
                u = v
            else:
                proj_sum = torch.zeros_like(v)
                for u_prev in u_list:
                    denom = (u_prev * u_prev).sum(dim=1, keepdim=True) + 1e-6
                    coeff = (v * u_prev).sum(dim=1, keepdim=True) / denom
                    proj_sum += coeff * u_prev
                u = v - proj_sum

            u = u / (u.norm(dim=-1, keepdim=True) + 1e-6).pow(0.7)
            u_list.append(u)

        for u in u_list:
            ortho.append(u.view(B, T, D))

        return ortho

    # Penalizes components that are still correlated after orthogonalization
    def compute_diversity_loss(self, components):
        K = len(components)
        if K <= 1:
            return torch.tensor(0.0, device=components[0].device)

        means = [F.normalize(c.mean(dim=1), dim=-1) for c in components]

        vals = []
        for i in range(K):
            for j in range(i + 1, K):
                corr = (means[i] * means[j]).sum(dim=-1)
                vals.append((corr ** 2).mean())

        return sum(vals) / len(vals)

    # Create momentum encoder by copying the entire model
    def _create_momentum_encoder(self):
        self.momentum_encoder = copy.deepcopy(self)
        for p in self.momentum_encoder.parameters():
            p.requires_grad = False
        self.momentum_encoder.eval()

    # Exponential moving average update for stable targets
    @torch.no_grad()
    def momentum_update(self):
        if not self.use_momentum:
            return

        for param_q, param_k in zip(self.parameters(), self.momentum_encoder.parameters()):
            param_k.data.mul_(self.momentum).add_(param_q.data, alpha=1 - self.momentum)

    # Update negative queue with momentum-encoded embeddings
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_v, keys_t):
        if self.queue_size <= 0:
            return

        keys_v = keys_v.detach()
        keys_t = keys_t.detach()

        batch_size = keys_v.shape[0]
        ptr = int(self.queue_ptr.item())

        end = (ptr + batch_size) % self.queue_size

        if end > ptr:
            self.queue_v[ptr:end] = keys_v
            self.queue_t[ptr:end] = keys_t
        else:
            first = self.queue_size - ptr
            self.queue_v[ptr:] = keys_v[:first]
            self.queue_t[ptr:] = keys_t[:first]
            if batch_size > first:
                self.queue_v[: batch_size - first] = keys_v[first:]
                self.queue_t[: batch_size - first] = keys_t[first:]

        self.queue_ptr[0] = end

    # Uniformity loss encourages embeddings to spread out on the hypersphere
    def compute_uniformity_loss(self, x, t=2.0):
        if x.shape[0] < 2:
            return torch.tensor(0.0, device=x.device)
        sq_dist = torch.pdist(x, p=2).pow(2)
        return ((-(t * sq_dist)).exp().mean().clamp(min=1e-12)).log()

    # Variance loss prevents dimension collapse
    def compute_variance_loss(self, x, eps=1e-4):
        std = torch.sqrt(x.var(dim=0) + eps)
        return torch.mean(F.relu(1.0 - std))

    # Covariance loss encourages decorrelated embedding dimensions
    def compute_cov_loss(self, x):
        x = x - x.mean(dim=0)
        cov = (x.T @ x) / (x.shape[0] - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        return (off_diag ** 2).mean()


    def forward(self, batch, update_queue=True):

        vid_emb = batch["video_emb"]
        txt_masked = batch["text_input"]
        txt_target = batch["text_target"]
        txt_global = batch["text_global"]
        mask_bool = batch["text_mask_bool"]

        # Component extraction
        components = self.get_video_fields(vid_emb)

        # Orthogonalized components
        ortho_components = self.orthogonalize_components(components)
        diversity_loss = self.compute_diversity_loss(ortho_components)

        # Concatenate components for cross-attention
        v_full_ctx = torch.cat(ortho_components, dim=-1)

        # Masked text prediction
        attn_out, _ = self.cross_attn(query=txt_masked, key=v_full_ctx, value=v_full_ctx)
        pred_text_vecs = self.text_predictor(attn_out)

        if mask_bool.sum() > 0:
            loss_l1 = F.mse_loss(
                F.normalize(pred_text_vecs[mask_bool], dim=-1),
                F.normalize(txt_target[mask_bool], dim=-1)
            ) * 100.0
        else:
            loss_l1 = torch.tensor(0.0, device=vid_emb.device)

        # Component means for gating fusion
        comp_means = [F.normalize(c.mean(dim=1), dim=-1) for c in ortho_components]

        gate_logits = self.gate_projector(txt_global)
        gate_weights = F.softmax(gate_logits, dim=-1)

        stacked = torch.stack(comp_means, dim=1)
        v_final = (gate_weights.unsqueeze(-1) * stacked).sum(dim=1)

        v_final_norm = F.normalize(v_final, dim=-1)
        t_final_norm = F.normalize(txt_global, dim=-1)

        cov_loss = self.compute_cov_loss(v_final_norm)

        # Momentum encoder forward pass
        with torch.no_grad():
            if self.use_momentum:
                m_components = self.momentum_encoder.get_video_fields(vid_emb)
                m_ortho = self.momentum_encoder.orthogonalize_components(m_components)
                m_means = [c.mean(dim=1) for c in m_ortho]
                m_stacked = torch.stack(m_means, dim=1)
                m_weights = F.softmax(
                    self.momentum_encoder.gate_projector(txt_global),
                    dim=-1
                ).unsqueeze(-1)
                m_v_final = (m_weights * m_stacked).sum(dim=1)
                m_v_final_norm = F.normalize(m_v_final, dim=-1)
                m_t_final_norm = F.normalize(txt_global, dim=-1)
            else:
                m_v_final_norm = v_final_norm.detach()
                m_t_final_norm = t_final_norm.detach()

        # Contrastive alignment loss
        logit_scale = self.logit_scale.exp()
        logits_inbatch = logit_scale * (v_final_norm @ t_final_norm.t())

        if self.queue_size > 0 and self.queue_t.shape[0] > 0:
            logits_queue = logit_scale * (v_final_norm @ self.queue_t.t())
            logits = torch.cat([logits_inbatch, logits_queue], dim=1)
            labels = torch.arange(logits_inbatch.size(0), device=logits.device)
            loss_v2t = F.cross_entropy(logits, labels)
        else:
            labels = torch.arange(logits_inbatch.size(0), device=logits_inbatch.device)
            loss_v2t = F.cross_entropy(logits_inbatch, labels)

        logits_inbatch_t = logit_scale * (t_final_norm @ v_final_norm.t())

        if self.queue_size > 0 and self.queue_v.shape[0] > 0:
            logits_queue_t = logit_scale * (t_final_norm @ self.queue_v.t())
            logits_t = torch.cat([logits_inbatch_t, logits_queue_t], dim=1)
            labels_t = torch.arange(logits_inbatch_t.size(0), device=logits_t.device)
            loss_t2v = F.cross_entropy(logits_t, labels_t)
        else:
            labels_t = torch.arange(logits_inbatch_t.size(0), device=logits_inbatch_t.device)
            loss_t2v = F.cross_entropy(logits_inbatch_t, labels_t)

        loss_l2 = 0.5 * (loss_v2t + loss_t2v)

        # Regularizers
        loss_uniformity = self.compute_uniformity_loss(v_final_norm)
        loss_variance = self.compute_variance_loss(v_final_norm)

        # Combined loss
        total_loss = (
            loss_l2 +
            0.5 * loss_l1 +
            0.05 * loss_uniformity +
            0.2 * loss_variance +
            0.07 * diversity_loss +
            0.02 * cov_loss
        )

        logs = {
            "loss_total": total_loss.item(),
            "loss_contrastive": loss_l2.item(),
            "loss_mask": loss_l1.item(),
            "loss_uniformity": loss_uniformity.item(),
            "loss_variance": loss_variance.item(),
            "loss_diversity": diversity_loss.item(),
            "gate_avg": gate_weights.mean(dim=0).detach()
        }

        if self.training and update_queue and self.queue_size > 0:
            self._dequeue_and_enqueue(m_v_final_norm, m_t_final_norm)

        return total_loss, logs


    @torch.no_grad()
    def debug_stats(self, batch):
        vid_emb = batch["video_emb"]
        txt_global = batch["text_global"]

        # Variance after video projection
        vp = self.video_proj(vid_emb)
        var_video_proj = vp.var().item()

        cls = self.video_cls_token.expand(vp.shape[0], 1, -1)
        tx = torch.cat((cls, vp), dim=1)
        before = tx.var().item()
        after = self.temporal_encoder(tx).var().item()
        transformer_gain = after - before

        # Component behavior
        components = self.get_video_fields(vid_emb)
        comp_stds = [c.std().item() for c in components]

        # Orthogonalization stats
        ortho = self.orthogonalize_components(components)

        corr_before = []
        corr_after = []
        for i in range(self.num_components):
            for j in range(i+1, self.num_components):
                c1 = F.normalize(components[i].mean(dim=1), dim=-1)
                c2 = F.normalize(components[j].mean(dim=1), dim=-1)
                o1 = F.normalize(ortho[i].mean(dim=1), dim=-1)
                o2 = F.normalize(ortho[j].mean(dim=1), dim=-1)
                corr_before.append((c1 * c2).sum(dim=-1).mean().item())
                corr_after.append((o1 * o2).sum(dim=-1).mean().item())

        avg_corr_before = sum(corr_before) / len(corr_before)
        avg_corr_after = sum(corr_after) / len(corr_after)
        ortho_reduction = avg_corr_before - avg_corr_after

        gate_logits = self.gate_projector(txt_global)
        gate_weights = F.softmax(gate_logits, dim=-1)
        gate_entropy = (-gate_weights * gate_weights.log()).sum(dim=-1).mean().item()
        gate_std = gate_weights.std(dim=1).mean().item()

        stacked = torch.stack([c.mean(dim=1) for c in ortho], dim=1)
        v_final = (gate_weights.unsqueeze(-1) * stacked).sum(dim=1)
        contrib = gate_weights.mean(dim=0).cpu().tolist()

        uniform = self.compute_uniformity_loss(F.normalize(v_final, dim=-1)).item()
        variance = self.compute_variance_loss(F.normalize(v_final, dim=-1)).item()

        if self.use_momentum:
            m_comp = self.momentum_encoder.get_video_fields(vid_emb)
            m_ortho = self.momentum_encoder.orthogonalize_components(m_comp)
            m_stack = torch.stack([c.mean(dim=1) for c in m_ortho], dim=1)
            m_weights = F.softmax(self.momentum_encoder.gate_projector(txt_global), dim=-1).unsqueeze(-1)
            m_v = (m_weights * m_stack).sum(dim=1)
            diff_momentum = (m_v - v_final).abs().mean().item()
        else:
            diff_momentum = 0.0

        return {
            "video_proj_variance": var_video_proj,
            "transformer_variance_gain": transformer_gain,
            "component_stds": comp_stds,
            "avg_corr_before_ortho": avg_corr_before,
            "avg_corr_after_ortho": avg_corr_after,
            "ortho_reduction": ortho_reduction,
            "gate_entropy": gate_entropy,
            "gate_std": gate_std,
            "component_contributions": contrib,
            "uniformity": uniform,
            "variance_loss": variance,
            "momentum_difference": diff_momentum,
        }
