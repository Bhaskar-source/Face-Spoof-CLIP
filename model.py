"""
UniAttackDetection Model
Based on: "Unified Physical-Digital Face Attack Detection"

Modules:
  - Teacher-Student Prompt (TSP)
  - Unified Knowledge Mining (UKM)
  - Sample-Level Prompt Interaction (SLPI)
  - Live-Center One-Class Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


# ------------------------------------------------------------------
# Teacher templates
# ------------------------------------------------------------------

TEACHER_TEMPLATES = [
    "This photo contains {}.",
    "There is a {} in this photo.",
    "{} is in this photo.",
    "A photo of a {}.",
    "This is an example of a {}.",
    "This is how a {} looks like.",
    "This is an image of {}.",
    "The picture is a {}.",
]

UNIFIED_CLASSES = ["real face", "spoof face"]

SPECIFIC_CLASSES = [
    "real face",
    "physical attack",
    "adversarial attack",
    "digital attack",
]


# ------------------------------------------------------------------
# Fusion Block
# ------------------------------------------------------------------

class FusionBlock(nn.Module):

    def __init__(self, d_model, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        hidden = int(d_model * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):

        attn_out, _ = self.attn(x, x, x)

        x = self.norm1(x + attn_out)

        x = self.norm2(x + self.mlp(x))

        return x


# ------------------------------------------------------------------
# Lightweight head
# ------------------------------------------------------------------

class LightweightHead(nn.Module):

    def __init__(self, d_model, cs=4, cu=2):
        super().__init__()

        self.fc = nn.Linear(cs * d_model, cu * d_model)

        self.d = d_model
        self.cu = cu

    def forward(self, fsc):

        x = fsc.flatten()

        x = self.fc(x)

        return x.view(self.cu, self.d)


# ------------------------------------------------------------------
# Main model
# ------------------------------------------------------------------

class UniAttackDetection(nn.Module):

    def __init__(
        self,
        clip_model_name="ViT-B/16",
        num_student_tokens=16,
        num_teacher_templates=6,
        lam=1.0,
        device="cuda",
    ):
        super().__init__()

        self.device = device
        self.lam = lam
        self.num_teacher_templates = num_teacher_templates

        # ------------------------------------------------
        # Load CLIP
        # ------------------------------------------------

        self.clip, _ = clip.load(clip_model_name, device=device)

        for p in self.clip.parameters():
            p.requires_grad_(False)

        d = self.clip.text_projection.shape[1]

        self.d = d

        cu = len(UNIFIED_CLASSES)
        cs = len(SPECIFIC_CLASSES)

        # ------------------------------------------------
        # Student tokens
        # ------------------------------------------------

        self.student_tokens = nn.Parameter(
            torch.randn(num_student_tokens, d) * 0.02
        )

        # ------------------------------------------------
        # Teacher features
        # ------------------------------------------------

        self._build_teacher_features()

        # ------------------------------------------------
        # Fusion
        # ------------------------------------------------

        fusion_seq_len = cu * num_teacher_templates + cs

        self.fusion_block = FusionBlock(d_model=d)

        self.fusion_proj = nn.Linear(fusion_seq_len * d, cu * d)

        # ------------------------------------------------
        # Lightweight head
        # ------------------------------------------------

        self.lightweight_head = LightweightHead(d, cs=cs, cu=cu)

        # ------------------------------------------------
        # Visual prompt projection
        # ------------------------------------------------

        dv = self.clip.visual.conv1.out_channels

        self.interaction_projector = nn.Linear(d, dv)

        self.cu = cu
        self.cs = cs
        self.dv = dv

        # ------------------------------------------------
        # Live center
        # ------------------------------------------------

        self.live_center = nn.Parameter(torch.randn(d))


    # ------------------------------------------------
    # Build teacher features
    # ------------------------------------------------

    @torch.no_grad()
    def _build_teacher_features(self):

        templates = TEACHER_TEMPLATES[:self.num_teacher_templates]

        all_feats = []

        for tmpl in templates:

            texts = clip.tokenize(
                [tmpl.format(c) for c in UNIFIED_CLASSES]
            ).to(self.device)

            feats = self.clip.encode_text(texts)

            feats = feats / feats.norm(dim=-1, keepdim=True)

            all_feats.append(feats)

        teacher = torch.stack(all_feats, dim=0).permute(1, 0, 2)

        self.register_buffer("teacher_feats", teacher)


    # ------------------------------------------------
    # Encode student prompts
    # ------------------------------------------------

    def _encode_student(self):

        class_tokens = clip.tokenize(SPECIFIC_CLASSES).to(self.device)

        x = self.clip.token_embedding(class_tokens).type(self.clip.dtype)

        N = self.student_tokens.shape[0]

        x[:, 1:1 + N, :] = self.student_tokens.unsqueeze(0).expand(
            self.cs, -1, -1
        ).type(self.clip.dtype)

        x = x + self.clip.positional_embedding.type(self.clip.dtype)

        x = x.permute(1, 0, 2)

        x = self.clip.transformer(x)

        x = x.permute(1, 0, 2)

        x = self.clip.ln_final(x).type(self.clip.dtype)

        eot = class_tokens.argmax(dim=-1)

        fsc = x[torch.arange(self.cs), eot] @ self.clip.text_projection

        fsc = fsc / fsc.norm(dim=-1, keepdim=True)

        return fsc.float()


    # ------------------------------------------------
    # UFM loss
    # ------------------------------------------------

    def _ufm_loss(self, ffusion):

        G = self.num_teacher_templates

        loss = 0.0

        for g in range(G):

            ftc_g = self.teacher_feats[:, g, :]

            cos = F.cosine_similarity(ffusion, ftc_g, dim=-1)

            loss += (1 - cos).mean()

        return loss / G


    # ------------------------------------------------
    # Live center loss
    # ------------------------------------------------

    def live_center_loss(self, features, labels, margin=1.0):

        center = F.normalize(self.live_center, dim=0)

        live_mask = labels == 1
        attack_mask = labels == 0

        loss = 0.0

        if live_mask.sum() > 0:

            live_feat = features[live_mask]

            loss_live = ((live_feat - center) ** 2).sum(dim=1).mean()

            loss += loss_live

        if attack_mask.sum() > 0:

            attack_feat = features[attack_mask]

            dist = torch.norm(attack_feat - center, dim=1)

            loss_attack = torch.relu(margin - dist).mean()

            loss += loss_attack

        return loss


    # ------------------------------------------------
    # Visual encoder with prompts
    # ------------------------------------------------

    def _encode_image_with_prompt(self, images, vp):

        vit = self.clip.visual

        x = vit.conv1(images.type(self.clip.dtype))

        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = x.permute(0, 2, 1)

        cls = vit.class_embedding.to(x.dtype).unsqueeze(0).unsqueeze(0)

        cls = cls.expand(x.shape[0], -1, -1)

        x = torch.cat([cls, x], dim=1)

        x = x + vit.positional_embedding.to(x.dtype)

        vp_expanded = vp.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.dtype)

        x = torch.cat([x, vp_expanded], dim=1)

        x = vit.ln_pre(x)

        x = x.permute(1, 0, 2)

        x = vit.transformer(x)

        x = x.permute(1, 0, 2)

        x = vit.ln_post(x[:, 0, :])

        if vit.proj is not None:
            x = x @ vit.proj

        fv = x.float() / x.norm(dim=-1, keepdim=True)

        return fv


    # ------------------------------------------------
    # Forward
    # ------------------------------------------------

    def forward(self, images):

        fsc = self._encode_student()

        vp = self.interaction_projector(self.student_tokens.float())

        fv = self._encode_image_with_prompt(images, vp)

        tf = self.teacher_feats.reshape(-1, self.d)

        combined = torch.cat([tf, fsc], dim=0)

        combined = combined.unsqueeze(0)

        fused_seq = self.fusion_block(combined)

        fused_flat = fused_seq.squeeze(0).flatten()

        ffusion = self.fusion_proj(fused_flat).view(self.cu, self.d)

        ffusion = ffusion / ffusion.norm(dim=-1, keepdim=True)

        ufm = self._ufm_loss(ffusion)

        logits = fv @ ffusion.T * self.clip.logit_scale.exp()

        return logits, ufm, fv