import os
import json
import torch
import random
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel
import torch.nn.functional as F


# Base paths for MSR-VTT raw data and output embedding storage
BASE_DIR = "...."
VIDEO_DIR_TRAIN = os.path.join(BASE_DIR, "MSRVTT/MSR-VTT/TrainValVideo")
VIDEO_DIR_TEST = os.path.join(BASE_DIR, "MSRVTT/MSR-VTT/TestVideo")

# Annotation files for captions and indexing
ANNOT_TRAIN = os.path.join(BASE_DIR, "MSRVTT/MSR-VTT/train_val_videodatainfo.json")
ANNOT_TEST = os.path.join(BASE_DIR, "MSRVTT/MSR-VTT/test_videodatainfo.json")

# Directory where final text embeddings will be saved
EMBED_DIR = os.path.join(BASE_DIR, "embeddings")
os.makedirs(EMBED_DIR, exist_ok=True)

SAVE_TRAIN = os.path.join(EMBED_DIR, "msrvtt_text_train_embeds.pth")
SAVE_TEST = os.path.join(EMBED_DIR, "msrvtt_text_test_embeds.pth")

# CLIP model backbone configuration
TEXT_ENCODER_MODEL = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_embeddings(annotation_path, video_dir, save_path):
    """
    Extracts token-level CLIP text embeddings for all captions in MSR-VTT.

    This script:
        - loads caption annotations,
        - filters out missing videos,
        - runs the CLIP text encoder,
        - retrieves final hidden layer token embeddings,
        - normalizes them for stability,
        - stores per-video lists of variable-length token matrices.

    Result:
        save_path stores a dictionary:
            video_id → [list of tensors shape [L_i, 512]]
    """

    print(f"\nLoading annotation file: {annotation_path}")
    print(f"Expecting corresponding videos under: {video_dir}")

    with open(annotation_path, "r") as f:
        data = json.load(f)

    video_caption_map = {}
    missing = 0

    # The JSON contains a list of sentence entries with {video_id, caption}
    for item in data["sentences"]:
        vid = item["video_id"]
        caption = item["caption"].strip()
        video_path = os.path.join(video_dir, f"{vid}.mp4")

        # Ignore captions for which the expected video file does not exist
        if not os.path.exists(video_path):
            missing += 1
            continue

        video_caption_map.setdefault(vid, []).append(caption)

    print(f"Found {len(video_caption_map)} usable videos. Skipped: {missing}")

    tokenizer = CLIPTokenizer.from_pretrained(TEXT_ENCODER_MODEL)
    model = CLIPTextModel.from_pretrained(
        TEXT_ENCODER_MODEL,
        output_hidden_states=True
    ).to(DEVICE).eval()

    cache = {}
    with torch.no_grad():
        for vid, captions in tqdm(video_caption_map.items(), desc="Extracting CLIP text embeddings"):
            token_list = []

            for cap in captions:
                # Tokenization without padding; each caption length varies
                inputs = tokenizer(
                    cap,
                    return_tensors="pt",
                    truncation=True,
                    padding=False
                ).to(DEVICE)

                # Forward pass to retrieve hidden states
                outputs = model(**inputs)

                # Last hidden layer: [1, seq_len, 512]
                token_embs = outputs.hidden_states[-1].squeeze(0).cpu()

                # Normalize embeddings for downstream stability
                token_embs = F.normalize(token_embs, dim=-1)

                token_list.append(token_embs)

            cache[vid] = token_list

    # Save all embeddings for the split
    torch.save(cache, save_path)
    print(f"Saved {len(cache)} text embedding entries to: {save_path}")

    try:
        example_vid = random.choice(list(cache.keys()))
        ex = cache[example_vid][0]
        print(f"Example video: {example_vid}")
        print(f"Token count: {ex.shape[0]}, Embedding dimension: {ex.shape[1]}")
    except Exception as e:
        print(f"Sanity check error: {e}")


if __name__ == "__main__":
    print("Extracting MSR-VTT CLIP text embeddings...")
    print(f"Output directory: {EMBED_DIR}")

    # Training split embeddings
    extract_embeddings(ANNOT_TRAIN, VIDEO_DIR_TRAIN, SAVE_TRAIN)

    # Test split embeddings
    extract_embeddings(ANNOT_TEST, VIDEO_DIR_TEST, SAVE_TEST)

    print("Completed extraction for training and test sets.")
