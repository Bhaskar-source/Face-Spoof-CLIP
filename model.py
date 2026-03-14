"""
UniAttackDetection Model
Based on: "Unified Physical-Digital Face Attack Detection" (arXiv:2401.17699)

Modules:
  - Teacher-Student Prompt (TSP)
  - Unified Knowledge Mining (UKM)
  - Sample-Level Prompt Interaction (SLPI)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


# ──────────────────────────────────────────────────────────────────────────────
# Teacher prompt templates (Table 6 in paper)
# ──────────────────────────────────────────────────────────────────────────────
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

# Unified classes for teacher prompts (binary)
UNIFIED_CLASSES = ["real face", "spoof face"]

# Specific classes for student prompts (4-way)
SPECIFIC_CLASSES = [    "real face",
                        "physical attack",
                        "adversarial attack",
                        "digital attack"]


# ──────────────────────────────────────────────────────────────────────────────
# Fusion Block  (Self-Attention + MLP)
# ──────────────────────────────────────────────────────────────────────────────
class FusionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
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
        # x: (B, seq_len, d) or (seq_len, d) for text features
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.mlp(x))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight head: maps specific-class space → unified (binary) space
# ──────────────────────────────────────────────────────────────────────────────
class LightweightHead(nn.Module):
    """Map (cs, d) → (cu, d)  where cs=3 specific, cu=2 unified."""
    def __init__(self, d_model: int, cs: int = 3, cu: int = 2):
        super().__init__()
        self.fc = nn.Linear(cs * d_model, cu * d_model)
        self.d = d_model
        self.cu = cu

    def forward(self, fsc):
        # fsc: (cs, d)  →  flatten  →  (cu, d)
        x = fsc.flatten()            # (cs*d,)
        x = self.fc(x)               # (cu*d,)
        return x.view(self.cu, self.d)


# ──────────────────────────────────────────────────────────────────────────────
# Main UniAttackDetection model
# ──────────────────────────────────────────────────────────────────────────────
class UniAttackDetection(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "ViT-B/16",
        num_student_tokens: int = 16,
        num_teacher_templates: int = 6,
        lam: float = 1.0,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.lam = lam
        self.num_teacher_templates = num_teacher_templates

        # ── Load CLIP ──────────────────────────────────────────────────────────
        self.clip, _ = clip.load(clip_model_name, device=device)
        # Freeze CLIP backbone
        for p in self.clip.parameters():
            p.requires_grad_(False)

        d = self.clip.text_projection.shape[1]   # CLIP embedding dim (512 for ViT-B/16)
        self.d = d
        cu = len(UNIFIED_CLASSES)    # 2
        cs = len(SPECIFIC_CLASSES)   # 3

        # ── TSP: Student Prompt tokens ─────────────────────────────────────────
        # Learnable prefix vectors  [p1 ... pN]  prepended to specific class names
        self.student_tokens = nn.Parameter(
            torch.randn(num_student_tokens, d) * 0.02
        )

        # ── TSP: Pre-encode teacher prompts (fixed, no grad) ──────────────────
        # Shape after encode: (cu, G, d)  where G = num_teacher_templates
        self._build_teacher_features()

        # ── UKM: Fusion Block ─────────────────────────────────────────────────
        # Teacher feats flattened: (cu*G, d); concat student (cs, d) → seq len = cu*G+cs
        fusion_seq_len = cu * num_teacher_templates + cs
        self.fusion_seq_len = fusion_seq_len
        self.fusion_block = FusionBlock(d_model=d)
        # Project fused sequence → (cu, d)
        self.fusion_proj = nn.Linear(fusion_seq_len * d, cu * d)

        # ── TSP: Lightweight head (specific → unified) ─────────────────────────
        self.lightweight_head = LightweightHead(d, cs=cs, cu=cu)

        # ── SLPI: Interaction projector  dp → dv ──────────────────────────────
        # dp = d (student token dim),  dv = visual patch dim
        # For ViT-B/16 the visual transformer width is 768
        dv = self.clip.visual.conv1.out_channels  # 768
        self.interaction_projector = nn.Linear(d, dv)

        # Store dims
        self.cu = cu
        self.cs = cs
        self.dv = dv

    # ── Build and cache teacher text features (no grad) ───────────────────────
    @torch.no_grad()
    def _build_teacher_features(self):
        templates = TEACHER_TEMPLATES[: self.num_teacher_templates]
        all_feats = []
        for tmpl in templates:
            texts = clip.tokenize([tmpl.format(c) for c in UNIFIED_CLASSES]).to(self.device)
            feats = self.clip.encode_text(texts)   # (cu, d)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats)
        # (G, cu, d)  →  (cu, G, d)
        teacher = torch.stack(all_feats, dim=0).permute(1, 0, 2)
        self.register_buffer("teacher_feats", teacher)

    # ── Encode student prompts using CLIP text encoder ────────────────────────
    def _encode_student(self):
        """Returns fsc of shape (cs, d)."""
        # Build token sequences: [p1..pN][class_token]
        # We use CLIP's tokenizer for the class names and inject learned tokens
        class_tokens = clip.tokenize(SPECIFIC_CLASSES).to(self.device)  # (cs, 77)

        # Get CLIP text embeddings at the word-embedding level
        x = self.clip.token_embedding(class_tokens).type(self.clip.dtype)  # (cs, 77, d)

        # Replace the first N positions (after SOT) with learned student tokens
        N = self.student_tokens.shape[0]
        x[:, 1: 1 + N, :] = self.student_tokens.unsqueeze(0).expand(self.cs, -1, -1).type(self.clip.dtype)

        # Add positional embeddings and pass through CLIP text transformer
        x = x + self.clip.positional_embedding.type(self.clip.dtype)
        x = x.permute(1, 0, 2)      # (77, cs, d)
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)      # (cs, 77, d)
        x = self.clip.ln_final(x).type(self.clip.dtype)

        # EOS token position
        eot = class_tokens.argmax(dim=-1)
        fsc = x[torch.arange(self.cs), eot] @ self.clip.text_projection   # (cs, d)
        fsc = fsc / fsc.norm(dim=-1, keepdim=True)
        return fsc.float()

    # ── UFM loss (Eq. 1) ──────────────────────────────────────────────────────
    def _ufm_loss(self, ffusion):
        """
        ffusion: (cu, d)
        teacher_feats: (cu, G, d)
        """
        G = self.num_teacher_templates
        loss = 0.0
        for g in range(G):
            ftc_g = self.teacher_feats[:, g, :]   # (cu, d)
            # cosine similarity for each class
            cos = F.cosine_similarity(ffusion, ftc_g, dim=-1)  # (cu,)
            loss += (1 - cos).mean()
        return loss / G
    # def _ufm_loss(self, ffusion):
    #     G = self.num_teacher_templates
    #     loss = 0.0
    #     eps = 1e-8  # numerical stability

    #     for g in range(G):
    #         ftc_g = self.teacher_feats[:, g, :]   # (cu, d)

    #         # --- Soft Jaccard similarity ---
    #         intersection = (ffusion * ftc_g).sum(dim=-1)  # (cu,)
    #         union = (
    #             (ffusion * ffusion).sum(dim=-1)
    #             + (ftc_g * ftc_g).sum(dim=-1)
    #             - intersection
    #         )

    #         jaccard = intersection / (union + eps)  # (cu,)

    #         loss += (1 - jaccard).mean()

    #     return loss / G
    # def _ufm_loss(self, ffusion):
    #     """
    #     Supervised Contrastive Loss between:
    #     - ffusion (student fused features)
    #     - teacher_feats (teacher text features)

    #     ffusion: (cu, d)
    #     teacher_feats: (cu, G, d)
    #     """
    #     temperature = 0.07
    #     G = self.num_teacher_templates
    #     total_loss = 0.0

    #     for g in range(G):
    #         ftc_g = self.teacher_feats[:, g, :]  # (cu, d)

    #         # Stack student + teacher features
    #         features = torch.cat([ffusion, ftc_g], dim=0)  # (2*cu, d)

    #         # Labels: same class index repeated twice
    #         labels = torch.arange(self.cu, device=ffusion.device)
    #         labels = torch.cat([labels, labels], dim=0)  # (2*cu,)

    #         # Similarity matrix
    #         sim_matrix = torch.matmul(features, features.T) / temperature

    #         # Remove self-similarity
    #         mask = torch.eye(sim_matrix.shape[0], device=sim_matrix.device).bool()
    #         sim_matrix = sim_matrix.masked_fill(mask, -1e9)

    #         # Positive mask
    #         labels = labels.unsqueeze(0)
    #         pos_mask = (labels == labels.T).float()

    #         # Log-softmax
    #         log_prob = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)

    #         # SupCon loss
    #         loss = -(pos_mask * log_prob).sum(dim=1) / pos_mask.sum(dim=1)
    #         total_loss += loss.mean()

    #     return total_loss / G
    # ── SLPI: encode image with visual prompt injected ────────────────────────
    def _encode_image_with_prompt(self, images, vp):
        """
        images: (B, 3, H, W)
        vp: (N, dv)  – visual prompts derived from student tokens
        """
        # Get patch embeddings from CLIP's visual backbone
        vit = self.clip.visual
        x = vit.conv1(images.type(self.clip.dtype))       # (B, dv, H', W')
        x = x.reshape(x.shape[0], x.shape[1], -1)         # (B, dv, n_patches)
        x = x.permute(0, 2, 1)                            # (B, n_patches, dv)

        # Prepend CLS token
        cls = vit.class_embedding.to(x.dtype).unsqueeze(0).unsqueeze(0)  # (1,1,dv)
        cls = cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)                    # (B, n_patches+1, dv)

        # Add positional embedding
        x = x + vit.positional_embedding.to(x.dtype)

        # Concatenate visual prompts [VP]
        vp_expanded = vp.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.dtype)
        x = torch.cat([x, vp_expanded], dim=1)            # (B, n_patches+1+N, dv)

        x = vit.ln_pre(x)
        x = x.permute(1, 0, 2)                            # (seq, B, dv)
        x = vit.transformer(x)
        x = x.permute(1, 0, 2)                            # (B, seq, dv)
        x = vit.ln_post(x[:, 0, :])                       # CLS token

        if vit.proj is not None:
            x = x @ vit.proj
        fv = x.float() / x.norm(dim=-1, keepdim=True).float()
        return fv   # (B, d)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, images):
        """
        images: (B, 3, 224, 224)
        Returns:
            logits  (B, cu)   – binary live/spoof
            ufm_loss  scalar
        """
        B = images.shape[0]

        # ── 1. Student features  (cs, d) ──────────────────────────────────────
        fsc = self._encode_student()   # (cs, d)

        # ── 2. SLPI: build visual prompts from student tokens ─────────────────
        vp = self.interaction_projector(self.student_tokens.float())   # (N, dv)

        # ── 3. Encode image with visual prompts ───────────────────────────────
        fv = self._encode_image_with_prompt(images, vp)   # (B, d)

        # ── 4. UKM: fuse teacher + student features ───────────────────────────
        # teacher_feats: (cu, G, d)
        # Flatten teacher across groups: (cu*G, d)
        tf = self.teacher_feats.reshape(-1, self.d)         # (cu*G, d)
        # Concat with student: (cu*G + cs, d)
        combined = torch.cat([tf, fsc], dim=0)              # (cu*G+cs, d)
        combined = combined.unsqueeze(0)                    # (1, seq, d)
        fused_seq = self.fusion_block(combined)             # (1, seq, d)
        fused_flat = fused_seq.squeeze(0).flatten()         # (seq*d,)
        ffusion = self.fusion_proj(fused_flat).view(self.cu, self.d)  # (cu, d)
        ffusion = ffusion / ffusion.norm(dim=-1, keepdim=True)

        # ── 5. UFM loss ───────────────────────────────────────────────────────
        ufm = self._ufm_loss(ffusion)

        # ── 6. Map student → unified space via lightweight head ───────────────
        fsc_unified = self.lightweight_head(fsc)            # (cu, d)
        fsc_unified = fsc_unified / fsc_unified.norm(dim=-1, keepdim=True)

        # ── 7. Fuse text features with image features for classification ──────
        # Use ffusion for classification labels (text side)
        # Compute cosine similarity: image (B, d) vs text (cu, d)
        logits = fv @ ffusion.T * self.clip.logit_scale.exp()   # (B, cu)

        return logits, ufm
