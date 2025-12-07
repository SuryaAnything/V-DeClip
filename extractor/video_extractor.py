# video_extractor_framewise.py
#
# Extracts frame-wise video embeddings from the V-JEPA2 model
# for the MSR-VTT dataset. Each video is sampled uniformly in
# time, preprocessed with the model's video processor, and
# encoded into one embedding per frame.
#
# Output format:
#     A dictionary saved to disk:
#         video_id -> tensor [NUM_FRAMES, hidden_dim]
#
# These embeddings serve as inputs to the alignment model

import os
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoVideoProcessor
from decord import VideoReader, cpu


# Base directory configuration
BASE_DIR = "...."
VIDEO_DIR_TRAIN = os.path.join(BASE_DIR, "MSRVTT/MSR-VTT/TrainValVideo")
VIDEO_DIR_TEST = os.path.join(BASE_DIR, "MSRVTT/MSR-VTT/TestVideo")

# Annotation files for video indexing
ANNOT_TRAIN = os.path.join(BASE_DIR, "MSRVTT/MSR-VTT/train_val_videodatainfo.json")
ANNOT_TEST = os.path.join(BASE_DIR, "MSRVTT/MSR-VTT/test_videodatainfo.json")

# Directory to store extracted embeddings
EMBED_DIR = os.path.join(BASE_DIR, "embeddings")
os.makedirs(EMBED_DIR, exist_ok=True)

SAVE_TRAIN = os.path.join(EMBED_DIR, "msrvtt_video_train_framewise.pth")
SAVE_TEST = os.path.join(EMBED_DIR, "msrvtt_video_test_framewise.pth")

# V-JEPA2 model configuration
MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Number of uniformly sampled frames per video
NUM_FRAMES = 32


def process_video_frames(video_path, processor, num_frames):
    """
    Extracts uniformly spaced frames from a raw video file and applies
    preprocessing expected by the V-JEPA video processor.

    Returns:
        A tensor of shape [1, num_frames, C, H, W] suitable for the model.
    """
    height = processor.crop_size["height"]
    width = processor.crop_size["width"]

    try:
        vr = VideoReader(video_path, ctx=cpu(0), width=width, height=height)
        total_frames = len(vr)
        if total_frames <= 0:
            raise ValueError("Video contains zero frames.")

        # Uniform temporal sampling
        indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()

    except Exception as e:
        # If a video cannot be decoded, return blank frames to avoid training interruption
        print(f"[Warning] Could not process {video_path}. Reason: {e}")
        frames = [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(num_frames)]

    # Preprocess frames for V-JEPA2
    return processor(videos=list(frames), return_tensors="pt")["pixel_values_videos"]


def extract_framewise_embeddings(annotation_path, video_dir, save_path):
    """
    Loads video metadata, extracts frame-wise features using V-JEPA2,
    and saves them to a file for downstream training.

    Output:
        save_path contains a dictionary:
            video_id -> tensor [NUM_FRAMES, hidden_dim]
    """

    print(f"\nLoading annotation file: {annotation_path}")
    print(f"Reading videos from: {video_dir}")

    # Load MSR-VTT annotation file
    with open(annotation_path, "r") as f:
        data = json.load(f)

    # Extract IDs of videos listed in the annotation file
    video_ids = [v["video_id"] for v in data["videos"]]

    # Filter out IDs where mp4 file is missing
    valid_ids = [
        vid for vid in video_ids
        if os.path.exists(os.path.join(video_dir, f"{vid}.mp4"))
    ]

    print(f"Found {len(valid_ids)} valid videos for this split.")

    # Load encoder and processor
    print("Loading V-JEPA2 model...")
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    processor = AutoVideoProcessor.from_pretrained(MODEL_NAME)

    cache = {}

    with torch.no_grad():
        for vid in tqdm(valid_ids, desc="Extracting frame-wise embeddings"):
            video_path = os.path.join(video_dir, f"{vid}.mp4")

            try:
                # Preprocess frames for the model
                pixel_values = process_video_frames(video_path, processor, NUM_FRAMES).to(DEVICE)

                # Forward pass to obtain hidden states
                outputs = model(pixel_values_videos=pixel_values)

                # outputs.last_hidden_state: [1, seq_len, hidden_dim]
                hidden = outputs.last_hidden_state.squeeze(0)

                seq_len, hidden_dim = hidden.shape

                # If the sequence is longer than num_frames, partition evenly
                chunk_size = seq_len // NUM_FRAMES
                trimmed = hidden[:chunk_size * NUM_FRAMES]

                # Reshape into [NUM_FRAMES, chunk_size, hidden_dim]
                grouped = trimmed.view(NUM_FRAMES, chunk_size, hidden_dim)

                # Temporal average within each segment
                frame_embs = grouped.mean(dim=1).cpu()  # [NUM_FRAMES, hidden_dim]

                cache[vid] = frame_embs

            except Exception as e:
                print(f"[Warning] Failed to process {vid}: {e}")
                continue

    # Save the extracted embeddings
    torch.save(cache, save_path)
    print(f"Saved {len(cache)} extracted embeddings to: {save_path}")

    # Sanity check on one random entry
    try:
        example_vid = random.choice(list(cache.keys()))
        ex = cache[example_vid]
        print(f"Example video: {example_vid}")
        print(f"Frame embedding shape: {ex.shape}")
        print(f"First frame sample: {ex[0][:5].tolist()}")
    except Exception as e:
        print(f"Sanity check error: {e}")


if __name__ == "__main__":
    print("Extracting frame-wise V-JEPA2 embeddings for MSR-VTT...\n")

    extract_framewise_embeddings(ANNOT_TRAIN, VIDEO_DIR_TRAIN, SAVE_TRAIN)
    extract_framewise_embeddings(ANNOT_TEST, VIDEO_DIR_TEST, SAVE_TEST)

    print("Extraction complete.")
