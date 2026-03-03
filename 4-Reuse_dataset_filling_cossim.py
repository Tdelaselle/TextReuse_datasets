import csv
import logging
import argparse
import torch
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util, models
from tqdm import tqdm

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reuse Dataset Cosine-Similarity Biblical Text Filler

This script loads the latin_reuses_sentences.tsv file and fills empty 'biblical_text'
cells with the most similar biblical verse from VG.tsv (by cosine similarity),
but only for rows with 'NaR' labels.

Usage:
    python 4-Reuse_dataset_filling_cossim.py -i <input_tsv> -b <biblical_tsv> -o <output_tsv> -m <model_path>

"""

"""
By T. de la Selle, feb. 2026
For BibliReuse project, BiblIndex team
From Institut des Sources Chrétiennes
License: MIT
"""

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


def load_datasets(input_tsv: str, biblical_tsv: str) -> tuple:
    """
    Load the sentence dataset and biblical texts dataset.
    
    Args:
        input_tsv: Path to latin_reuses_sentences.tsv
        biblical_tsv: Path to VG.tsv
        
    Returns:
        Tuple of (sentence_df, biblical_df)
    """
    try:
        sentence_df = pd.read_csv(input_tsv, sep='\t', encoding='utf-8')
        logger.info(f"Loaded sentence dataset: {len(sentence_df)} rows")
    except Exception as e:
        logger.error(f"Error loading {input_tsv}: {e}")
        return None, None
    
    try:
        biblical_df = pd.read_csv(biblical_tsv, sep='\t', encoding='utf-8')
        logger.info(f"Loaded biblical texts dataset: {len(biblical_df)} rows")
    except Exception as e:
        logger.error(f"Error loading {biblical_tsv}: {e}")
        return sentence_df, None
    
    return sentence_df, biblical_df


def load_candidates(biblical_df: pd.DataFrame) -> tuple:
    """
    Extract candidate texts and their references from the biblical DataFrame.
    
    Returns:
        Tuple of (candidate_texts, candidate_references)
    """
    valid = biblical_df.dropna(subset=['text'])
    valid = valid[valid['text'].str.strip() != '']
    
    candidate_texts = valid['text'].tolist()
    candidate_refs = valid['reference'].tolist()
    
    # Normalize 'j'->'i' and 'v'->'u' for Latin consistency
    candidate_texts_norm = [t.replace('j', 'i').replace('v', 'u') for t in candidate_texts]
    
    logger.info(f"Loaded {len(candidate_texts)} biblical candidate texts.")
    return candidate_texts, candidate_texts_norm, candidate_refs


def fill_biblical_texts_cossim(
    sentence_df: pd.DataFrame,
    candidate_texts: list,
    candidate_texts_norm: list,
    candidate_refs: list,
    model,
    device: str,
    batch_size: int = 256
) -> pd.DataFrame:
    """
    Fill empty 'biblical_text' cells with the most cosine-similar biblical verse
    for rows with 'NaR' labels.
    
    Args:
        sentence_df: DataFrame with sentence-level data
        candidate_texts: Original candidate texts from VG.tsv
        candidate_texts_norm: Normalized candidate texts (j->i, v->u)
        candidate_refs: References corresponding to candidate texts
        model: SentenceTransformer model
        device: 'cuda' or 'cpu'
        batch_size: Batch size for encoding
        
    Returns:
        Modified DataFrame with filled biblical_text column
    """
    df = sentence_df.copy()
    
    # Find rows with 'NaR' label and empty biblical_text
    nar_mask = (df['label'] == 'NaR') | (df['label'].astype(str).str.contains('NaR', na=False))
    empty_text_mask = (df['biblical_text'].isna()) | (df['biblical_text'] == '')
    rows_to_fill = df[nar_mask & empty_text_mask].index.tolist()
    
    logger.info(f"Found {len(rows_to_fill)} rows with 'NaR' label and empty biblical_text to fill.")
    
    if not rows_to_fill:
        logger.info("Nothing to fill.")
        return df
    
    # Collect the sentences to embed (normalized)
    sentences_to_fill = [
        df.at[idx, 'sentence'].replace('j', 'i').replace('v', 'u')
        for idx in rows_to_fill
    ]
    
    # Pre-compute embeddings for all biblical candidates
    logger.info("Computing embeddings for all biblical candidates...")
    candidate_embeddings = model.encode(
        candidate_texts_norm,
        batch_size=batch_size,
        convert_to_tensor=True,
        device=device,
        show_progress_bar=True
    )
    
    # Process NaR sentences in batches
    logger.info("Finding most similar biblical text for each NaR sentence...")
    best_texts = []
    best_scores = []
    
    for i in tqdm(range(0, len(sentences_to_fill), batch_size), desc="Processing NaR sentences"):
        batch_sentences = sentences_to_fill[i : i + batch_size]
        
        # Embed the current batch of sentences
        sentence_embeddings = model.encode(
            batch_sentences,
            convert_to_tensor=True,
            device=device,
            show_progress_bar=False
        )
        
        # Compute cosine similarity: (batch_size, num_candidates)
        cos_scores = util.cos_sim(sentence_embeddings, candidate_embeddings)
        
        # Get the top-1 most similar candidate for each sentence
        top_results = torch.topk(cos_scores, k=1, dim=1)
        
        for j in range(len(batch_sentences)):
            best_idx = top_results.indices[j][0].item()
            best_score = top_results.values[j][0].item()
            best_texts.append(candidate_texts[best_idx])
            best_scores.append(best_score)
    
    # Fill the DataFrame
    for i, idx in enumerate(rows_to_fill):
        df.at[idx, 'biblical_text'] = best_texts[i]
    
    # Stats
    mean_score = sum(best_scores) / len(best_scores) if best_scores else float('nan')
    logger.info(f"Filled {len(rows_to_fill)} rows with most similar biblical texts.")
    logger.info(f"Mean cosine similarity (sentence / best biblical match): {mean_score:.4f}")
    
    return df


def main(input_tsv=None, biblical_tsv=None, output_tsv=None,
         model_path=None, from_huggingface=False, batch_size=256):
    script_dir = Path(__file__).parent
    
    input_path = Path(input_tsv) if input_tsv else script_dir / 'Latin_reuses' / 'latin_reuses_sentences.tsv'
    biblical_path = Path(biblical_tsv) if biblical_tsv else script_dir / 'Biblical_data' / 'VG.tsv'
    output_path = Path(output_tsv) if output_tsv else script_dir / 'Latin_reuses' / 'latin_reuses_sentences_CosSimFilled.tsv'
    
    if model_path is None:
        model_path = "/home/delaselt/Documents/BibliReuse/PatriSBERT/models/Patrisbert-lat2-2class-NLI_e2/"
    
    print("=" * 60)
    print("REUSE DATASET COSINE-SIMILARITY BIBLICAL TEXT FILLER")
    print("=" * 60)
    
    # 1. Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        logger.warning("GPU not found! Computations will run on CPU and may be slow.")
    else:
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # 2. Load model
    print("\n[Step 1] Loading model...")
    model = load_model(model_path, device=device, from_huggingface=from_huggingface)
    
    # 3. Load datasets
    print("\n[Step 2] Loading datasets...")
    sentence_df, biblical_df = load_datasets(str(input_path), str(biblical_path))
    
    if sentence_df is None or biblical_df is None:
        logger.error("Could not load required datasets. Exiting.")
        return
    
    # 4. Prepare candidates
    print("\n[Step 3] Preparing biblical candidates...")
    candidate_texts, candidate_texts_norm, candidate_refs = load_candidates(biblical_df)
    
    if not candidate_texts:
        logger.error("No biblical candidate texts found. Exiting.")
        return
    
    # 5. Fill empty biblical texts with cosine similarity
    print("\n[Step 4] Filling empty biblical_text cells for 'NaR' labels by cosine similarity...")
    filled_df = fill_biblical_texts_cossim(
        sentence_df,
        candidate_texts,
        candidate_texts_norm,
        candidate_refs,
        model,
        device,
        batch_size=batch_size
    )
    
    # 6. Save result
    print("\n[Step 5] Saving filled dataset...")
    filled_df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
    logger.info(f"Saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fill empty biblical_text cells with the most cosine-similar biblical verse for NaR labels.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 4-Reuse_dataset_filling_cossim.py -i Latin_reuses/latin_reuses_sat-seg.tsv -b Biblical_data/VG.tsv -o Latin_reuses/latin_reuses_sat-seg_CosSim.tsv -m /home/tdelaselle/Documents/BibliReuse/PatriBERT/models/PatriBERT_e6_lr5e-05_bs64_acc4

  # Using a raw HuggingFace BERT model:
  python 4-Reuse_dataset_filling_cossim.py -i Latin_reuses/latin_reuses_sentences.tsv -b Biblical_data/VG.tsv -o Latin_reuses/latin_reuses_sentences_CosSimFilled.tsv -m ashleygong03/bamman-burns-latin-bert --from-huggingface
        """
    )
    parser.add_argument(
        '-i', '--input',
        required=False,
        default=None,
        help='Path to the latin_reuses_sentences.tsv file (default: Latin_reuses/latin_reuses_sentences.tsv)'
    )
    parser.add_argument(
        '-b', '--biblical',
        required=False,
        default=None,
        help='Path to the VG.tsv biblical texts file (default: Biblical_data/VG.tsv)'
    )
    parser.add_argument(
        '-o', '--output',
        required=False,
        default=None,
        help='Path to the output TSV file (default: Latin_reuses/latin_reuses_sentences_CosSimFilled.tsv)'
    )
    parser.add_argument(
        '-m', '--model',
        required=False,
        default=None,
        help='Path to SentenceTransformer model or HuggingFace model name'
    )
    parser.add_argument(
        '--from-huggingface',
        action='store_true',
        default=False,
        help='If set, wraps the HuggingFace model with mean pooling for sentence embeddings'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size for encoding (default: 256, adjust based on GPU VRAM)'
    )
    
    args = parser.parse_args()
    
    main(
        input_tsv=args.input,
        biblical_tsv=args.biblical,
        output_tsv=args.output,
        model_path=args.model,
        from_huggingface=args.from_huggingface,
        batch_size=args.batch_size
    )
