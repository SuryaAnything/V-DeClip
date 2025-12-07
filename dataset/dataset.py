import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random


class MSRVTT_Alignment_Dataset(Dataset):
    """
    Dataset for loading aligned video–text pairs from MSR-VTT embeddings.

    This class provides:
        - Framewise video embeddings truncated or padded to a fixed length.
        - Text embeddings for captions with dynamic caption sampling.
        - Token-level masking for masked prediction loss (L1).
        - Global text representation used for contrastive learning and gating.
        - Optional video-only supervision for stability.

    The dataset operates directly on pre-extracted embeddings to speed up
    multimodal training and remove dependencies on raw videos or text models.
    """

    def __init__(
        self,
        video_embed_path,
        text_embed_path,
        max_frames=32,
        mask_ratio=0.2,
        video_only_prob=0.15
    ):
        """
        Args:
            video_embed_path: Path to .pt file mapping video_id → [T, 1024] tensor.
            text_embed_path: Path to .pt file mapping video_id → list of [L, 512] tensors.
            max_frames: Maximum number of frames for video (truncated or padded).
            mask_ratio: Fraction of text tokens masked for reconstruction.
            video_only_prob: Probability of masking all text tokens for a sample.
        """
        super().__init__()

        # Load pre-computed video and text embeddings
        self.video_data = torch.load(video_embed_path, map_location="cpu")
        self.text_data = torch.load(text_embed_path, map_location="cpu")

        self.max_frames = max_frames
        self.mask_ratio = mask_ratio
        self.video_only_prob = video_only_prob

        # Zero vector used as mask token for text
        self.mask_token_vec = torch.zeros(512)

        # Build index of valid videos and determine maximum caption length
        self.video_ids = []
        self.max_text_len_found = 0

        for vid in self.video_data.keys():
            if vid not in self.text_data:
                continue

            captions = self.text_data[vid]
            if not isinstance(captions, list) or len(captions) == 0:
                continue

            self.video_ids.append(vid)

            # Track maximum length across all captions in dataset
            for cap in captions:
                self.max_text_len_found = max(self.max_text_len_found, cap.shape[0])

        # Final fixed maximum number of text tokens
        self.max_tokens = self.max_text_len_found

    def _pad_tensor(self, tensor, target_len, dim=0):
        """
        Pads or truncates a tensor to a fixed length along a given dimension.

        Args:
            tensor: Input tensor.
            target_len: Desired length.
            dim: Dimension along which padding/truncation occurs.

        Returns:
            padded_or_truncated_tensor, original_length
        """
        current_len = tensor.shape[dim]

        if current_len >= target_len:
            return tensor[:target_len], current_len

        pad_amount = target_len - current_len

        if dim == 0:
            # Pad rows (temporal dimension)
            return F.pad(tensor, (0, 0, 0, pad_amount)), current_len

        return tensor, current_len

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        """
        Loads a single aligned video–text example.

        Returns:
            video_emb:         [max_frames, 1024]
            video_mask:        [max_frames]
            text_input:        [max_tokens, 512] (masked)
            text_target:       [max_tokens, 512] (unmasked)
            text_mask_bool:    [max_tokens]
            text_global:       [512]
        """
        
        vid = self.video_ids[idx]
        captions = self.text_data[vid]
        cap_tensor = random.choice(captions)

        v_emb = self.video_data[vid]

        # Ensure 2D shape [T, D]
        if v_emb.dim() == 1:
            v_emb = v_emb.unsqueeze(0)

        # Pad or truncate to max_frames
        v_padded, v_real_len = self._pad_tensor(v_emb, self.max_frames, dim=0)

        # Temporal mask indicating valid frames
        v_attn_mask = torch.zeros(self.max_frames)
        v_attn_mask[:v_real_len] = 1.0

        t_padded, t_real_len = self._pad_tensor(cap_tensor, self.max_tokens, dim=0)

        t_masked = t_padded.clone()
        mask_bool = torch.zeros(self.max_tokens, dtype=torch.bool)

        if t_real_len > 0:
            # Optionally mask the entire caption for video-only learning
            if random.random() < self.video_only_prob:
                mask_bool[:t_real_len] = True
            else:
                num_to_mask = max(1, int(t_real_len * self.mask_ratio))
                indices = torch.randperm(t_real_len)[:num_to_mask]
                mask_bool[indices] = True

            t_masked[mask_bool] = self.mask_token_vec

        if t_real_len > 0:
            t_global = t_padded[:t_real_len].mean(dim=0)
        else:
            t_global = torch.zeros(512)

        return {
            "video_emb": v_padded,
            "video_mask": v_attn_mask,

            "text_input": t_masked,
            "text_target": t_padded,
            "text_mask_bool": mask_bool,

            "text_global": t_global,
        }


if __name__ == "__main__":
    # Simple check on dataset loader
    import os

    EMBED_BASE = "/home/csebtp/Surya-VXTLabs/Embeddings"
    train_video_path = os.path.join(EMBED_BASE, "msrvtt_video_train_framewise.pth")
    train_text_path = os.path.join(EMBED_BASE, "msrvtt_text_train_embeds.pth")

    ds = MSRVTT_Alignment_Dataset(train_video_path, train_text_path, max_frames=16)
    dl = DataLoader(ds, batch_size=2, shuffle=True)

    batch = next(iter(dl))
    for k, v in batch.items():
        print(f"{k:<15} {v.shape}")
