# train.py
#
# This script trains the MMCGD multimodal model on MSR-VTT video–text
# alignment data. It covers the complete training pipeline:
#
# 1. Dataset loading and batching for both training and evaluation.
# 2. Model initialization, optimizer, scheduler, and mixed precision utilities.
# 3. Full training loop with loss accumulation, momentum encoder updates,
#    gradient clipping, and checkpointing based on validation metrics.
# 4. Evaluation routine that reconstructs per-component video features,
#    applies text-conditioned gating, and computes retrieval metrics
#    (R@1, R@5, R@10, MedR).
# 5. Debug statistics printed every few epochs for analyzing the internal
#    behavior of the model.
#
# The script is structured for clarity and extensibility, and mirrors the
# conventions of common vision–language training frameworks.


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import numpy as np
import os
import random

from dataset.dataset import MSRVTT_Alignment_Dataset
from model.model import MMCGD


CONFIG = {
    "BASE_DIR": "....",
    "TRAIN_VID": "msrvtt_video_train_framewise.pth",
    "TRAIN_TXT": "msrvtt_text_train_embeds.pth",
    "TEST_VID": "msrvtt_video_test_framewise.pth",
    "TEST_TXT": "msrvtt_text_test_embeds.pth",
    "SAVE_DIR": "./checkpoints",

    "BATCH_SIZE": 64,
    "EPOCHS": 1000,
    "LR": 1e-4,
    "WD": 0.01,
    "NUM_WORKERS": 4,
    "MAX_FRAMES": 16,
    "EVAL_FREQ": 1,
    "TEST_SUBSET_SIZE": 500,

    "USE_AMP": True,
}


def compute_metrics(sim_matrix):
    sx = np.argsort(-sim_matrix, axis=1)
    ranks = []
    for i in range(sim_matrix.shape[0]):
        rank_pos = np.where(sx[i] == i)[0][0]
        ranks.append(rank_pos)

    ranks = np.array(ranks)
    r1 = 100.0 * (ranks < 1).sum() / len(ranks)
    r5 = 100.0 * (ranks < 5).sum() / len(ranks)
    r10 = 100.0 * (ranks < 10).sum() / len(ranks)
    medr = np.floor(np.median(ranks)) + 1

    return r1, r5, r10, medr


# Computes component embeddings, applies gating, builds ranking
# matrix, and computes retrieval metrics.
def run_evaluation(model, test_loader, device):
    model.eval()
    total_test_loss = 0.0
    test_logs = {}
    num_batches = 0

    all_video_components = []
    all_text_emb = []

    print(f"\nEvaluating on {len(test_loader.sampler)} samples...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Forward Pass"):
            # Move batch to GPU
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Forward pass without queue updates
            loss, logs = model(batch, update_queue=False)
            total_test_loss += loss.item()

            # Accumulate losses
            for k, v in logs.items():
                test_logs[k] = test_logs.get(k, 0) + v

            num_batches += 1

            # Extract video components for retrieval gating
            components = model.get_video_fields(batch["video_emb"])
            ortho_components = model.orthogonalize_components(components)

            comp_means = [c.mean(dim=1).cpu() for c in ortho_components]
            all_video_components.append(comp_means)
            all_text_emb.append(batch["text_global"].cpu())

    # Normalize logs
    avg_test_loss = total_test_loss / max(1, num_batches)
    for k in test_logs:
        test_logs[k] /= max(1, num_batches)

    print(f"\nTest Loss: {avg_test_loss:.4f}")
    print(f"   L1 (Mask): {test_logs.get('loss_mask', 0):.4f}")
    print(f"   Diversity: {test_logs.get('loss_diversity', 0):.4f}")


    all_text = torch.cat(all_text_emb, dim=0).to(device)
    N = all_text.shape[0]
    K = model.num_components
    D = model.field_dim

    # Convert list-of-lists to a [N, K, D] tensor
    video_db = torch.zeros(N, K, D)
    idx = 0
    for comp_group in all_video_components:
        B = comp_group[0].shape[0]
        for b in range(B):
            for k in range(K):
                video_db[idx, k] = comp_group[k][b]
            idx += 1

    video_db = video_db.to(device)


    sim_matrix = np.zeros((N, N))
    chunk = 50

    print("\nComputing Ranking Matrix...")
    with torch.no_grad():
        for i in range(0, N, chunk):
            t_chunk = all_text[i : i + chunk]
            C = t_chunk.shape[0]

            # Compute gating weights from text
            gate_logits = model.gate_projector(t_chunk)
            gate_weights = F.softmax(gate_logits, dim=-1).unsqueeze(-1)

            # Expand video DB to match chunk dimension
            V = video_db.unsqueeze(0).expand(C, -1, -1, -1)

            # Compute gated video embedding
            V_dynamic = (V * gate_weights.unsqueeze(1)).sum(dim=2)
            V_dynamic = F.normalize(V_dynamic, dim=-1)

            # Normalize text embeddings
            t_norm = F.normalize(t_chunk, dim=-1).unsqueeze(-1)

            scores = torch.bmm(V_dynamic, t_norm).squeeze(-1)
            sim_matrix[i : i + C] = scores.cpu().numpy()

    r1, r5, r10, medr = compute_metrics(sim_matrix)
    return avg_test_loss, r1, r5, r10, medr



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)

    print("Loading Training Data...")
    train_ds = MSRVTT_Alignment_Dataset(
        video_embed_path=os.path.join(CONFIG["BASE_DIR"], CONFIG["TRAIN_VID"]),
        text_embed_path=os.path.join(CONFIG["BASE_DIR"], CONFIG["TRAIN_TXT"]),
        max_frames=CONFIG["MAX_FRAMES"],
    )

    print("Loading Test Data...")
    test_ds = MSRVTT_Alignment_Dataset(
        video_embed_path=os.path.join(CONFIG["BASE_DIR"], CONFIG["TEST_VID"]),
        text_embed_path=os.path.join(CONFIG["BASE_DIR"], CONFIG["TEST_TXT"]),
        max_frames=CONFIG["MAX_FRAMES"],
    )

    # Random sampling for evaluation subset
    test_indices = list(range(len(test_ds)))
    subset_size = min(CONFIG["TEST_SUBSET_SIZE"], len(test_ds))
    random.seed(42)
    subset_indices = random.sample(test_indices, subset_size)
    test_sampler = SubsetRandomSampler(subset_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=True,
        num_workers=CONFIG["NUM_WORKERS"]
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=CONFIG["BATCH_SIZE"],
        sampler=test_sampler,
        num_workers=CONFIG["NUM_WORKERS"]
    )

    print("Initializing model...")
    model = MMCGD().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LR"], weight_decay=CONFIG["WD"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["EPOCHS"])

    # Mixed precision support
    use_amp = CONFIG.get("USE_AMP", True) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_r1 = 0.0

    print("Starting training...")
    for epoch in range(CONFIG["EPOCHS"]):

        model.train()
        total_loss = 0.0
        logs_accum = {}

        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{CONFIG['EPOCHS']}")
        for batch in pbar:
            # Move batch to GPU
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            optimizer.zero_grad()

            # Forward with mixed precision
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, logs = model(batch, update_queue=True)

            # Backprop
            scaler.scale(loss).backward()

            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            # Update momentum encoder state
            if hasattr(model, "momentum_update"):
                model.momentum_update()

            total_loss += loss.item()
            for k, v in logs.items():
                logs_accum[k] = logs_accum.get(k, 0) + v

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(1, len(train_loader))
        for k in logs_accum:
            logs_accum[k] /= max(1, len(train_loader))

        scheduler.step()

        # Print training statistics
        print(f"\nEpoch {epoch+1} Stats:")
        print(f"   Train Loss: {avg_loss:.4f}")
        print(f"   L1: {logs_accum.get('loss_mask', 0):.4f} | L3: {logs_accum.get('loss_helmholtz', 0):.4f}")

        # Debug information
        if (epoch+1) % 10 == 0:
            stats = model.debug_stats(batch)
            print("\nDebug Stats:", stats)

        # Periodic evaluation
        if (epoch + 1) % CONFIG["EVAL_FREQ"] == 0:
            test_loss, r1, r5, r10, medr = run_evaluation(model, test_loader, device)
            print(f"   R@1: {r1:.2f}% | R@5: {r5:.2f}% | R@10: {r10:.2f}% | MedR: {medr}")
            print(f"   Test Loss: {test_loss:.4f}")

            # Save best checkpoint
            if r1 > best_r1:
                best_r1 = r1
                with open("best_results.txt", "w") as f:
                    f.write(f"R@1: {r1:.2f}% | R@5: {r5:.2f}% | R@10: {r10:.2f}% | MedR: {medr}")
                torch.save(model.state_dict(), f"{CONFIG['SAVE_DIR']}/best_model.pth")
                print("New best model saved!")

        print("-" * 60)


if __name__ == "__main__":
    main()
