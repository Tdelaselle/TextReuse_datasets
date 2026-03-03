import csv
import json
import logging
import torch
from sentence_transformers import SentenceTransformer, util, models
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def load_model(model_name_or_path, device, from_huggingface=False):
    """Loads a SentenceTransformer-compatible model.

    If `from_huggingface=True`, the model is fetched from the HuggingFace Hub
    and wrapped with a mean-pooling layer to make it usable for sentence
    embeddings (useful for raw BERT models such as
    'ashleygong03/bamman-burns-latin-bert' that are not natively packaged as
    SentenceTransformer models).

    If `from_huggingface=False` (default), the model is loaded directly as a
    SentenceTransformer (works for local paths and HF Hub repos that are
    already SentenceTransformer-compatible).
    """
    if from_huggingface:
        logger.info(f"Loading HuggingFace BERT model '{model_name_or_path}' and wrapping with mean pooling...")
        transformer = models.Transformer(model_name_or_path)
        pooling = models.Pooling(
            transformer.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False
        )
        model = SentenceTransformer(modules=[transformer, pooling], device=device)
    else:
        logger.info(f"Loading SentenceTransformer model from '{model_name_or_path}'...")
        model = SentenceTransformer(model_name_or_path, device=device)
    return model


def load_doublets_from_tsv(filepath):
    """Loads [Anchor, Positive] doublets from the TSV file.
    
    Anchors = 'sentence' column
    Positives = 'biblical_text' column
    Only rows where 'biblical_text' is non-empty are kept.
    """
    anchors = []
    positives = []
    logger.info(f"Loading doublets from TSV: {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            sentence = row.get('sentence', '').strip()
            biblical_text = row.get('biblical_text', '').strip()
            if sentence and biblical_text:
                anchors.append(sentence.replace('j', 'i').replace('v', 'u'))  # Normalize 'j'->'i' and 'v'->'u'
                positives.append(biblical_text.replace('j', 'i').replace('v', 'u'))  # Normalize 'j'->'i' and 'v'->'u'
    logger.info(f"Loaded {len(anchors)} doublets (sentence / biblical_text pairs).")
    return anchors, positives

def load_candidates_from_tsv(filepath):
    """Loads the list of [Negative] candidates from VG.tsv.
    
    Candidates = 'text' column from the Vulgate Bible TSV file.
    Only rows where 'text' is non-empty are kept.
    """
    candidates = []
    logger.info(f"Loading negative candidates from TSV: {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            text = row.get('text', '').strip()
            if text:
                candidates.append(text.replace('j', 'i').replace('v', 'u'))  # Normalize 'j'->'i' and 'v'->'u'
    logger.info(f"Loaded {len(candidates)} candidates.")
    return candidates

def build_triplet_dataset(
    model_path,
    doublets_path,
    candidates_path,
    output_json_path,
    batch_size=256,
    from_huggingface=False
):
    # 1. Check for GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        logger.warning("GPU not found! Computations will run on CPU and may be slow.")
    else:
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # 2. Load the model (local SentenceTransformer or HuggingFace BERT)
    model = load_model(model_path, device=device, from_huggingface=from_huggingface)

    # 3. Load data
    anchors, positives = load_doublets_from_tsv(doublets_path)
    neg_candidates = load_candidates_from_tsv(candidates_path)

    # 4. Pre-compute embeddings for all negative candidates
    # This fits easily in GPU memory (~100-200MB for 37k embeddings)
    logger.info("Computing embeddings for all negative candidates...")
    neg_embeddings = model.encode(
        neg_candidates, 
        batch_size=batch_size, 
        convert_to_tensor=True, 
        device=device,
        show_progress_bar=True
    )

    # 5. Mine Hard Negatives
    logger.info("Mining optimal negative candidates for each anchor...")
    triplets = []
    overlap_rejections = 0  # Counts candidates rejected due to positive-negative overlap
    
    # Process anchors in batches to prevent Out-Of-Memory errors on the GPU 
    # when computing the similarity matrix
    for i in tqdm(range(0, len(anchors), batch_size), desc="Processing Anchors"):
        batch_anchors = anchors[i : i + batch_size]
        batch_positives = positives[i : i + batch_size]
        
        # Embed the current batch of anchors
        anchor_embeddings = model.encode(
            batch_anchors, 
            convert_to_tensor=True, 
            device=device,
            show_progress_bar=False
        )

        # Compute cosine similarity between the anchor batch and ALL negative candidates
        # Result shape: (batch_size, num_candidates)
        cos_scores = util.cos_sim(anchor_embeddings, neg_embeddings)

        # Get the top K most similar candidates for each anchor.
        # We fetch top 10 to ensure we have backups in case the top candidate == positive
        top_k = min(10, len(neg_candidates))
        top_results = torch.topk(cos_scores, k=top_k, dim=1)

        # Iterate through the batch to build the final triplets
        for j in range(len(batch_anchors)):
            anchor_text = batch_anchors[j]
            positive_text = batch_positives[j]
            
            # Find the best negative that is strictly NOT the positive
            best_negative = None
            best_neg_score = None
            positive_norm = positive_text.lower().strip()
            for k in range(top_k):
                candidate_idx = top_results.indices[j][k].item()
                candidate_text = neg_candidates[candidate_idx]
                candidate_norm = candidate_text.lower().strip()
                
                # VERIFICATION: Negative candidate must not be equal to, contained in,
                # or contain the positive (handles positives that are combinations of verses)
                if candidate_norm not in positive_norm and positive_norm not in candidate_norm:
                    best_negative = candidate_text
                    best_neg_score = top_results.values[j][k].item()
                    break
                else:
                    overlap_rejections += 1
            
            # Fallback if somehow all top-k overlapped with the positive (highly unlikely)
            if best_negative is None:
                # Just pick a random candidate that doesn't overlap
                for idx_fb, cand in enumerate(neg_candidates):
                    cand_norm = cand.lower().strip()
                    if cand_norm not in positive_norm and positive_norm not in cand_norm:
                        best_negative = cand
                        best_neg_score = util.cos_sim(
                            anchor_embeddings[j].unsqueeze(0),
                            neg_embeddings[idx_fb].unsqueeze(0)
                        ).item()
                        break
                    else:
                        overlap_rejections += 1

            triplets.append({
                "anchor": anchor_text,
                "positive": positive_text,
                "negative": best_negative,
                "neg_cos_sim": best_neg_score
            })

    # 6. Compute mean cosine similarity between anchors and their negatives
    scores = [t["neg_cos_sim"] for t in triplets if t["neg_cos_sim"] is not None]
    mean_neg_sim = sum(scores) / len(scores) if scores else float('nan')

    # 7. Save to JSON
    logger.info(f"Overlap rejections (candidate == positive / candidate in positive / positive in candidate): {overlap_rejections}")
    logger.info(f"Mean cosine similarity (anchor / negative): {mean_neg_sim:.4f}")
    logger.info(f"Saving {len(triplets)} triplets to {output_json_path}...")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(triplets, f, ensure_ascii=False, indent=2)
        
    logger.info("Done!")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Option A: local SentenceTransformer model
    LOCAL_MODEL_PATH = "/home/tdelaselle/Documents/BibliReuse/PatriBERT/models/PatriBERT_e6_lr5e-05_bs64_acc4/checkpoint-9000"
    FROM_HUGGINGFACE = False

    # Option B: raw HuggingFace BERT model (uncomment to use)
    # LOCAL_MODEL_PATH = "ashleygong03/bamman-burns-latin-bert"
    # FROM_HUGGINGFACE = True

    DOUBLETS_TSV_PATH = "Latin_reuses/latin_reuses_sat-seg.tsv"  # TSV with sentence & biblical_text
    CANDIDATES_TSV_PATH = "Biblical_data/VG.tsv"   # TSV with Vulgate Bible verses (text column)
    OUTPUT_JSON_PATH = "Latin_reuses/X_optimal_triplets_v2.json"

    # Run the mining process
    build_triplet_dataset(
        model_path=LOCAL_MODEL_PATH,
        doublets_path=DOUBLETS_TSV_PATH,
        candidates_path=CANDIDATES_TSV_PATH,
        output_json_path=OUTPUT_JSON_PATH,
        batch_size=256,  # Adjust based on your GPU VRAM (e.g., 128, 512)
        from_huggingface=FROM_HUGGINGFACE
    )