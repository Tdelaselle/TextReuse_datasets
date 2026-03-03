import pandas as pd
import numpy as np
from pathlib import Path
import argparse

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reuse Dataset Random Biblical Text Filler

This script loads the latin_reuses_sentences.tsv file and fills empty 'biblical_text'
cells with random biblical verses from VG.tsv, but only for rows with 'NaR' labels.

Usage:
    python 4-Reuse_dataset_filling_random.py -i <input_tsv> -b <biblical_tsv> -o <output_tsv>

"""

"""
By T. de la Selle, feb. 2026
For BibliReuse project, BiblIndex team
From Institut des Sources Chrétiennes
License: MIT
"""


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
        print(f"Loaded sentence dataset: {len(sentence_df)} rows")
    except Exception as e:
        print(f"Error loading {input_tsv}: {e}")
        return None, None
    
    try:
        biblical_df = pd.read_csv(biblical_tsv, sep='\t', encoding='utf-8')
        print(f"Loaded biblical texts dataset: {len(biblical_df)} rows")
    except Exception as e:
        print(f"Error loading {biblical_tsv}: {e}")
        return sentence_df, None
    
    return sentence_df, biblical_df


def fill_empty_biblical_texts(sentence_df: pd.DataFrame, 
                               biblical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill empty 'biblical_text' cells with random biblical verses for 'NaR' labels.
    
    Args:
        sentence_df: DataFrame with sentence-level data
        biblical_df: DataFrame with biblical texts
        
    Returns:
        Modified sentence_df with filled biblical_text column
    """
    # Get list of all biblical texts
    biblical_texts = biblical_df['text'].dropna().tolist()
    
    if not biblical_texts:
        print("Warning: No biblical texts found to use for filling.")
        return sentence_df
    
    # Create a copy to avoid modifying original
    df = sentence_df.copy()
    
    # Find rows with 'NaR' label and empty biblical_text
    nar_mask = (df['label'] == 'NaR') | (df['label'].astype(str).str.contains('NaR', na=False))
    empty_text_mask = (df['biblical_text'].isna()) | (df['biblical_text'] == '')
    
    rows_to_fill = df[nar_mask & empty_text_mask].index
    
    print(f"Found {len(rows_to_fill)} rows with 'NaR' label and empty biblical_text")
    
    # Fill with random biblical texts
    for idx in rows_to_fill:
        random_text = np.random.choice(biblical_texts)
        df.at[idx, 'biblical_text'] = random_text
    
    print(f"Filled {len(rows_to_fill)} rows with random biblical texts")
    
    return df


def main(input_tsv=None, biblical_tsv=None, output_tsv=None):
    script_dir = Path(__file__).parent
    
    input_path = Path(input_tsv) if input_tsv else script_dir / 'latin_reuses_sentences.tsv'
    biblical_path = Path(biblical_tsv) if biblical_tsv else script_dir / 'Biblical_data' / 'VG.tsv'
    output_path = Path(output_tsv) if output_tsv else script_dir / 'latin_reuses_sentences_Rfilled.tsv'
    
    print("=" * 60)
    print("REUSE DATASET RANDOM BIBLICAL TEXT FILLER")
    print("=" * 60)
    
    # Load datasets
    print("\n[Step 1] Loading datasets...")
    sentence_df, biblical_df = load_datasets(str(input_path), str(biblical_path))
    
    if sentence_df is None or biblical_df is None:
        print("Error: Could not load required datasets. Exiting.")
        return
    
    # Fill empty biblical texts
    print("\n[Step 2] Filling empty biblical_text cells for 'NaR' labels...")
    filled_df = fill_empty_biblical_texts(sentence_df, biblical_df)
    
    # Save result
    print("\n[Step 3] Saving filled dataset...")
    filled_df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
    print(f"Saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fill empty biblical_text cells with random verses for NaR labels.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 4-Reuse_dataset_filling_random.py -i latin_reuses_sentences.tsv -b Biblical_data/VG.tsv -o latin_reuses_sentences_Rfilled.tsv
        """
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Path to the latin_reuses_sentences.tsv file'
    )
    parser.add_argument(
        '-b', '--biblical',
        required=True,
        help='Path to the VG.tsv biblical texts file'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Path to the output TSV file'
    )
    
    args = parser.parse_args()
    
    main(
        input_tsv=args.input,
        biblical_tsv=args.biblical,
        output_tsv=args.output
    )