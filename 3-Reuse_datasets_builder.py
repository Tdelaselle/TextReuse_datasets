#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reuse Inference Dataset Builder

This script reads and stacks all TSV files containing biblical reuse annotations
from the 'Latin_reuses' folder, then builds a sentence-level dataset.

Usage:
    python 3-Reuse_datasets_builder.py -i <input_folder> -o <output_folder> [-b <biblical_texts_tsv>]

Arguments:
    -i, --input     Path to the input folder containing *_reuses.tsv files
    -o, --output    Path to the output folder for generated datasets
    -b, --biblical  (Optional) Path to biblical texts TSV for text lookup

Example:
    python 3-Reuse_datasets_builder.py -i Latin_reuses -o Reuses_datasets -b Biblical_data/VG.tsv

"""
    
"""
By T. de la Selle, feb. 2026
For BibliReuse project, BiblIndex team
From Institut des Sources Chrétiennes
License: MIT
"""

import os
import pandas as pd
from pathlib import Path
import re
import argparse
import torch
import numpy as np
from wtpsplit import SaT


def find_tsv_files(base_folder: str) -> list:
    """
    Recursively find all TSV files ending with '_reuses.tsv' in the given folder.
    
    Args:
        base_folder: Path to the base folder to search
        
    Returns:
        List of paths to TSV files
    """
    tsv_files = []
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file.endswith('_reuses.tsv'):
                tsv_files.append(os.path.join(root, file))
    return tsv_files


def load_and_stack_tsv_files(tsv_files: list) -> pd.DataFrame:
    """
    Load and stack all TSV files into a single DataFrame.
    
    Args:
        tsv_files: List of paths to TSV files
        
    Returns:
        Stacked DataFrame with all data and source file information
    """
    all_dataframes = []
    
    for tsv_file in tsv_files:
        try:
            # Read TSV with tab separator
            df = pd.read_csv(tsv_file, sep='\t', encoding='utf-8')
            
            # Add source file column
            df['source_file'] = os.path.basename(tsv_file)
            
            # Add word index within file
            df['word_index'] = range(len(df))
            
            all_dataframes.append(df)
            print(f"Loaded: {os.path.basename(tsv_file)} ({len(df)} words)")
            
        except Exception as e:
            print(f"Error loading {tsv_file}: {e}")
    
    if all_dataframes:
        stacked_df = pd.concat(all_dataframes, ignore_index=True)
        return stacked_df
    else:
        return pd.DataFrame()


def is_sentence_boundary(word: str) -> bool:
    """
    Check if a word marks the end of a sentence.
    
    Args:
        word: The word to check
        
    Returns:
        True if the word is a sentence-ending punctuation
    """
    sentence_endings = {'.', '?', '!', ';'}
    return str(word).strip() in sentence_endings


def build_sentence_dataset(word_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a sentence-level dataset from word-level data.
    
    Each sentence will have:
    - sentence: The concatenated words forming the sentence
    - reference: Aggregated references (unique, non-O values)
    - label: Aggregated labels (unique, non-O values)
    - source_file: The source file
    - sentence_index: Index of the sentence within the file
    
    Args:
        word_df: DataFrame with word-level data
        
    Returns:
        DataFrame with sentence-level data
    """
    sentences_data = []
    
    # Group by source file to process each file separately
    for source_file, file_group in word_df.groupby('source_file', sort=False):
        current_sentence_words = []
        current_references = []
        current_initial_refs = []
        current_labels = []
        current_lines = []
        sentence_index = 0
        
        for _, row in file_group.iterrows():
            word = str(row['word'])
            reference = str(row['reference normalized']) if pd.notna(row['reference normalized']) else 'o'
            initial_ref = str(row['reference']) if pd.notna(row['reference']) else 'o'
            label = str(row['label']) if pd.notna(row['label']) else 'NaR'
            line = str(row['line']).replace('.0', '') if pd.notna(row['line']) else 'o'
            
            current_sentence_words.append(word)
            
            # Collect non-O references and labels
            if reference != 'o':
                current_references.append(reference)
            if initial_ref != 'o':
                current_initial_refs.append(initial_ref)
            if label != 'NaR' and label != 'o':
                current_labels.append(label)
            if line != 'o':
                current_lines.append(line)
            
            # Check for sentence boundary
            if is_sentence_boundary(word):
                # Build sentence text
                sentence_text = ' '.join(current_sentence_words)
                # Clean up spacing around punctuation
                sentence_text = re.sub(r'\s+([.,;:?!])', r'\1', sentence_text)

                # Aggregate references (unique values, joined)
                unique_refs = list(dict.fromkeys(current_references))  # Preserve order
                aggregated_refs = ' ; '.join(unique_refs) if unique_refs else 'o'
                
                # Aggregate initial references (unique values, joined)
                unique_initial_refs = list(dict.fromkeys(current_initial_refs))
                aggregated_initial_refs = ' ; '.join(unique_initial_refs) if unique_initial_refs else 'o'
                
                # Aggregate labels (unique values, joined)
                unique_labels = list(dict.fromkeys(current_labels))
                aggregated_labels = ' ; '.join(unique_labels) if unique_labels else 'NaR'
                
                # Aggregate lines (first and last line of sentence)
                if current_lines:
                    unique_lines = list(dict.fromkeys(current_lines))
                    if len(unique_lines) == 1:
                        aggregated_lines = unique_lines[0]
                    else:
                        aggregated_lines = f"{unique_lines[0]}-{unique_lines[-1]}"
                else:
                    aggregated_lines = 'o'
                
                sentences_data.append({
                    'source_file': source_file,
                    'sentence_index': sentence_index,
                    'line': aggregated_lines,
                    'sentence': sentence_text,
                    'label': aggregated_labels,
                    'initial_reference': aggregated_initial_refs,
                    'reference': aggregated_refs.strip(),
                })
                
                # Reset for next sentence
                current_sentence_words = []
                current_references = []
                current_initial_refs = []
                current_labels = []
                current_lines = []
                sentence_index += 1
        
        # Handle remaining words (incomplete sentence at end of file)
        if current_sentence_words:
            sentence_text = ' '.join(current_sentence_words)
            sentence_text = re.sub(r'\s+([.,;:?!])', r'\1', sentence_text)
            
            unique_refs = list(dict.fromkeys(current_references))
            aggregated_refs = ' ; '.join(unique_refs) if unique_refs else 'o'
            
            unique_initial_refs = list(dict.fromkeys(current_initial_refs))
            aggregated_initial_refs = ' ; '.join(unique_initial_refs) if unique_initial_refs else 'o'
            
            unique_labels = list(dict.fromkeys(current_labels))
            aggregated_labels = ' ; '.join(unique_labels) if unique_labels else 'NaR'
            
            # Aggregate lines (first and last line of sentence)
            if current_lines:
                unique_lines = list(dict.fromkeys(current_lines))
                if len(unique_lines) == 1:
                    aggregated_lines = unique_lines[0]
                else:
                    aggregated_lines = f"{unique_lines[0]}-{unique_lines[-1]}"
            else:
                aggregated_lines = 'o'
            
            sentences_data.append({
                'source_file': source_file,
                'sentence_index': sentence_index,
                'line': aggregated_lines,
                'sentence': sentence_text,
                'label': aggregated_labels,
                'initial_reference': aggregated_initial_refs,
                'reference': aggregated_refs.strip()
            })
    
    # deagglomerate references and labels if needed
    for entry in sentences_data:
        references = entry['reference'].split(' ; ')
        labels = entry['label'].split(' ; ')
        # duplicate the row for each reference and guess that labels and references are aligned
        if len(references) > 1 or len(labels) > 1:
            entry['reference'] = references[0].replace('; ', '').strip()
            entry['label'] = labels[0] if labels else 'NaR'
            for ref, lab in zip(references[1:], labels[1:] + [labels[-1]] * (len(references) - len(labels))):
                new_entry = entry.copy()
                new_entry['reference'] = ref.replace('; ', '').strip()
                new_entry['label'] = lab
                sentences_data.append(new_entry)

    for entry in sentences_data:
        references = entry['reference'].split('.')
        # duplicate the row for each reference and guess that labels and references are aligned
        if len(references) > 1:
            entry['reference'] = references[0]
            for ref in references[1:]:
                new_entry = entry.copy()
                new_entry['reference'] = entry['reference'].split(':')[0] + ':' + ref.strip()
                sentences_data.append(new_entry)

    return pd.DataFrame(sentences_data)


def build_reuse_set_dataset(word_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a reuse-set dataset from word-level data.
    
    Each entry will have:
    - reuse_tag_set: Only the words that are part of the biblical reuse
    - reference: The reference for this reuse
    - label: The label for this reuse
    - source_file: The source file
    - sentence_index: Index of the sentence within the file (for context)
    
    Args:
        word_df: DataFrame with word-level data
        
    Returns:
        DataFrame with reuse-set data (only reuse words, not full sentences)
    """
    reuses_data = []
    
    # Group by source file to process each file separately
    for source_file, file_group in word_df.groupby('source_file', sort=False):
        current_reuse_words = []
        current_reference = None
        current_initial_ref = None
        current_label = None
        current_lines = []
        sentence_index = 0
        
        for _, row in file_group.iterrows():
            word = str(row['word'])
            reference = str(row['reference normalized']) if pd.notna(row['reference normalized']) else 'o'
            initial_ref = str(row['reference']) if pd.notna(row['reference']) else 'o'
            label = str(row['label']) if pd.notna(row['label']) else 'NaR'
            line = str(row['line']).replace('.0', '') if pd.notna(row['line']) else 'o'
            
            # Check if this word is part of a reuse
            if reference != 'o':
                # Check if we're continuing the same reuse or starting a new one
                if current_reference is None:
                    # Starting a new reuse
                    current_reference = reference
                    current_initial_ref = initial_ref
                    current_label = label
                    current_reuse_words = [word]
                    current_lines = [line] if line != 'o' else []
                elif current_reference == reference:
                    # Continuing the same reuse
                    current_reuse_words.append(word)
                    if line != 'o':
                        current_lines.append(line)
                else:
                    # Different reference - save current reuse and start new one
                    if current_reuse_words:
                        reuse_tag_set = ' '.join(current_reuse_words)
                        reuse_tag_set = re.sub(r'\s+([.,;:?!])', r'\1', reuse_tag_set)
                        
                        # Aggregate lines
                        if current_lines:
                            unique_lines = list(dict.fromkeys(current_lines))
                            if len(unique_lines) == 1:
                                aggregated_lines = unique_lines[0]
                            else:
                                aggregated_lines = f"{unique_lines[0]}-{unique_lines[-1]}"
                        else:
                            aggregated_lines = 'o'
                        
                        reuses_data.append({
                            'source_file': source_file,
                            'sentence_index': sentence_index,
                            'line': aggregated_lines,
                            'reuse_tag_set': reuse_tag_set,
                            'initial_reference': current_initial_ref,
                            'reference': current_reference.strip(),
                            'label': current_label
                        })
                    
                    # Start new reuse
                    current_reference = reference
                    current_initial_ref = initial_ref
                    current_label = label
                    current_reuse_words = [word]
                    current_lines = [line] if line != 'o' else []
            else:
                # Word is not part of a reuse - save current reuse if any
                if current_reuse_words:
                    reuse_tag_set = ' '.join(current_reuse_words)
                    reuse_tag_set = re.sub(r'\s+([.,;:?!])', r'\1', reuse_tag_set)
                    
                    # Aggregate lines
                    if current_lines:
                        unique_lines = list(dict.fromkeys(current_lines))
                        if len(unique_lines) == 1:
                            aggregated_lines = unique_lines[0]
                        else:
                            aggregated_lines = f"{unique_lines[0]}-{unique_lines[-1]}"
                    else:
                        aggregated_lines = 'o'
                    
                    reuses_data.append({
                        'source_file': source_file,
                        'sentence_index': sentence_index,
                        'line': aggregated_lines,
                        'reuse_tag_set': reuse_tag_set,
                        'label': current_label,
                        'initial_reference': current_initial_ref,
                        'reference': current_reference.strip()
                    })
                    
                    current_reuse_words = []
                    current_reference = None
                    current_initial_ref = None
                    current_label = None
                    current_lines = []
            
            # Track sentence boundaries for context
            if is_sentence_boundary(word):
                sentence_index += 1
        
        # Handle remaining reuse at end of file
        if current_reuse_words:
            reuse_tag_set = ' '.join(current_reuse_words)
            reuse_tag_set = re.sub(r'\s+([.,;:?!])', r'\1', reuse_tag_set)
            
            # Aggregate lines
            if current_lines:
                unique_lines = list(dict.fromkeys(current_lines))
                if len(unique_lines) == 1:
                    aggregated_lines = unique_lines[0]
                else:
                    aggregated_lines = f"{unique_lines[0]}-{unique_lines[-1]}"
            else:
                aggregated_lines = 'o'
            
            reuses_data.append({
                'source_file': source_file,
                'sentence_index': sentence_index,
                'line': aggregated_lines,
                'reuse_tag_set': reuse_tag_set,
                'label': current_label,
                'initial_reference': current_initial_ref,
                'reference': current_reference.strip()
            })

    # deagglomerate references and labels if needed
    for entry in reuses_data:
        references = entry['reference'].split(' ; ')
        labels = entry['label'].split(' ; ')
        # duplicate the row for each reference and guess that labels and references are aligned
        if len(references) > 1 or len(labels) > 1:
            entry['reference'] = references[0].replace('; ', '').strip()
            entry['label'] = labels[0] if labels else 'o'
            for ref, lab in zip(references[1:], labels[1:] + [labels[-1]] * (len(references) - len(labels))):
                new_entry = entry.copy()
                new_entry['reference'] = ref.replace('; ', '').strip()
                new_entry['label'] = lab
                reuses_data.append(new_entry)
                
    for entry in reuses_data:
        references = entry['reference'].split('.')
        # duplicate the row for each reference and guess that labels and references are aligned
        if len(references) > 1:
            entry['reference'] = references[0]
            for ref in references[1:]:
                new_entry = entry.copy()
                new_entry['reference'] = entry['reference'].split(':')[0] + ':' + ref.strip()
                reuses_data.append(new_entry)

    
    return pd.DataFrame(reuses_data)


def load_sat_model(model_name: str) -> tuple:
    """
    Loads the SaT model on the available device.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SaT model: '{model_name}' on {device}...")
    
    sat = SaT(model_name)
    
    # Be careful not to unwrap the SaT object if .to() returns the internal model
    if hasattr(sat, "half") and device == "cuda":
        sat.half()
    if hasattr(sat, "to"):
        sat.to(device)
        
    return sat, device


def build_sat_segment_dataset(word_df: pd.DataFrame, model_name="sat-12l-sm", threshold=0.5) -> pd.DataFrame:
    """
    Build a segment-level dataset using SaT (Split and Tokenize) model.
    """
    segments_data = []
    
    # Load model once
    sat, device = load_sat_model(model_name)
    kwargs = {}
    if threshold is not None:
        kwargs["threshold"] = threshold
    batch_size = 64 if device == "cuda" else 8

    # Group by source file to process each file separately
    for source_file, file_group in word_df.groupby('source_file', sort=False):
        print(f"Processing file {source_file} with SaT...")
        
        # Extract lists from DataFrame
        words = [str(x) for x in file_group['word'].tolist()]
        references = [str(x) if pd.notna(x) else 'o' for x in file_group['reference normalized'].tolist()]
        initial_refs = [str(x) if pd.notna(x) else 'o' for x in file_group['reference'].tolist()]
        labels = [str(x) if pd.notna(x) else 'NaR' for x in file_group['label'].tolist()]
        # Handle lines: replace .0 and convert to string
        lines = [str(x).replace('.0', '') if pd.notna(x) else 'o' for x in file_group['line'].tolist()]
        
        # Chunking (size 5000)
        chunk_size = 5000
        chunks_text = []
        chunks_indices = [] # Store start index for mapping back
        
        for i in range(0, len(words), chunk_size):
            chunk_w = words[i : i + chunk_size]
            chunks_text.append(" ".join(chunk_w))
            chunks_indices.append(i)
            
        # Process all chunks for this file
        chunk_iterator = sat.split(chunks_text, batch_size=batch_size, **kwargs)
        
        sentence_index = 0
        
        for i, segments in enumerate(chunk_iterator):
            chunk_start_idx = chunks_indices[i]
            current_local_idx = 0 # Index relative to start of chunk
            
            for seg in segments:
                seg_clean = seg.strip()
                if not seg_clean:
                    continue
                
                # Split segment into words to determine length
                # Note: This assumes SaT tokenization aligns with simple split()
                # If SaT produces different tokenization (e.g. splitting punctuation), we might desynchronize.
                # However, since input to sat.split was created with " ".join(words), 
                # simple split() should recover the original words if they didn't contain spaces.
                # The 'word' column typically contains single tokens.
                seg_words = seg_clean.split()
                n_words = len(seg_words)
                
                # Get indices in full file lists
                start_abs_idx = chunk_start_idx + current_local_idx
                end_abs_idx = start_abs_idx + n_words
                
                # Extract slice of metadata
                # Safe slicing even if index goes out of bounds (though it shouldn't structurally)
                seg_refs = references[start_abs_idx : end_abs_idx]
                seg_initial_refs = initial_refs[start_abs_idx : end_abs_idx]
                seg_labels = labels[start_abs_idx : end_abs_idx]
                seg_lines = lines[start_abs_idx : end_abs_idx]
                
                # Build segment data
                # Clean up punctuation spacing
                segment_text = re.sub(r'\s+([.,;:?!])', r'\1', seg_clean)
                
                # Aggregate references (unique values, joined)
                current_references = [r for r in seg_refs if r != 'o']
                unique_refs = list(dict.fromkeys(current_references))
                aggregated_refs = ' ; '.join(unique_refs) if unique_refs else 'o'
                
                # Aggregate initial references
                current_init_refs = [r for r in seg_initial_refs if r != 'o']
                unique_init_refs = list(dict.fromkeys(current_init_refs))
                aggregated_init_refs = ' ; '.join(unique_init_refs) if unique_init_refs else 'o'
                
                # Aggregate labels
                current_labels = [l for l in seg_labels if l != 'NaR' and l != 'o']
                unique_labels = list(dict.fromkeys(current_labels))
                aggregated_labels = ' ; '.join(unique_labels) if unique_labels else 'NaR'
                
                # Aggregate lines
                current_lines = [l for l in seg_lines if l != 'o']
                if current_lines:
                    unique_lines = list(dict.fromkeys(current_lines))
                    if len(unique_lines) == 1:
                        aggregated_lines = unique_lines[0]
                    else:
                        aggregated_lines = f"{unique_lines[0]}-{unique_lines[-1]}"
                else:
                    aggregated_lines = 'o'
                
                segments_data.append({
                    'source_file': source_file,
                    'sentence_index': sentence_index,
                    'line': aggregated_lines,
                    'sentence': segment_text,
                    'label': aggregated_labels,
                    'initial_reference': aggregated_init_refs,
                    'reference': aggregated_refs.strip(),
                })
                
                sentence_index += 1
                current_local_idx += n_words
                
    # Post-processing (same as other datasets)
    # deagglomerate references and labels
    processed_data = []
    for entry in segments_data:
        processed_data.append(entry)
        
    # Re-using the deagglomeration logic (copied from build_sentence_dataset to ensure consistency)
    final_data = []
    for entry in processed_data:
        references = entry['reference'].split(' ; ')
        labels = entry['label'].split(' ; ')
        
        should_split = len(references) > 1 or len(labels) > 1
        
        if should_split:
            # Add the first one
            first_entry = entry.copy()
            first_entry['reference'] = references[0].replace('; ', '').strip()
            first_entry['label'] = labels[0] if labels else 'NaR'
            final_data.append(first_entry)
            
            # Add subsequent ones
            for i in range(1, max(len(references), len(labels))):
                ref = references[i] if i < len(references) else references[-1]
                lab = labels[i] if i < len(labels) else (labels[-1] if labels else 'NaR')
                
                new_entry = entry.copy()
                new_entry['reference'] = ref.replace('; ', '').strip()
                new_entry['label'] = lab
                final_data.append(new_entry)
        else:
            final_data.append(entry)

    # Secondary split on '.'
    final_data_2 = []
    for entry in final_data:
        references = entry['reference'].split('.')
        if len(references) > 1:
            base_ref = references[0]
            # Add first part
            entry['reference'] = base_ref
            final_data_2.append(entry)
            
            # Add subsequent parts with prefix
            prefix = base_ref.split(':')[0] if ':' in base_ref else base_ref
            for ref in references[1:]:
                new_entry = entry.copy()
                new_entry['reference'] = prefix + ':' + ref.strip()
                final_data_2.append(new_entry)
        else:
            final_data_2.append(entry)
            
    return pd.DataFrame(final_data_2)


def main(input_folder=None, output_folder=None, biblical_texts_path=None, sat_model="sat-12l-sm", threshold=0.5):
    # Define paths
    script_dir = Path(__file__).parent
    latin_reuses_folder = Path(input_folder) if input_folder else script_dir
    output_folder = Path(output_folder) if output_folder else script_dir
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("REUSE INFERENCE DATASET BUILDER")
    print("=" * 60)
    
    # Step 1: Find all TSV files
    print("\n[Step 1] Searching for TSV files in 'Latin_reuses' folder...")
    tsv_files = find_tsv_files(latin_reuses_folder)
    print(f"Found {len(tsv_files)} TSV files")
    
    if not tsv_files:
        print("No TSV files found. Exiting.")
        return
    
    # Step 2: Load and stack all TSV files
    print("\n[Step 2] Loading and stacking TSV files...")
    word_level_df = load_and_stack_tsv_files(tsv_files)
    print(f"\nTotal words in stacked dataset: {len(word_level_df)}")

    word_level_df = pd.DataFrame(word_level_df, columns=['source_file','word_index','line','word','label','reference','reference normalized'])
    
    # Step 3: Save word-level dataset
    word_output_path = output_folder / 'latin_reuses_tags.tsv'
    word_level_df.to_csv(word_output_path, sep='\t', index=False, encoding='utf-8')
    print(f"\n[STEP 3] Word-level dataset saved to: {word_output_path}")
    
    # Step 4: Build sentence-level dataset
    print("\n[STEP 4] Building sentence-level dataset...")
    sentence_level_df = build_sentence_dataset(word_level_df)
    print(f"Total sentences in dataset: {len(sentence_level_df)}")
    
    # Step 5: Build reuse-set dataset (only words involved in reuses)
    print("\n[STEP 5] Building reuse-set dataset...")
    reuse_set_df = build_reuse_set_dataset(word_level_df)
    print(f"Total reuse sets in dataset: {len(reuse_set_df)}")

    # Step 5b: Build SaT segment dataset
    print("\n[STEP 5b] Building SaT segment dataset...")
    sat_segment_df = build_sat_segment_dataset(word_level_df, model_name=sat_model, threshold=threshold)
    print(f"Total SaT segments in dataset: {len(sat_segment_df)}")
    
    # Step 6: load and merge biblical texts if provided
    if biblical_texts_path:
        print("\n[STEP 6] Merging biblical texts...")
        try:
            biblical_texts_df = pd.read_csv(biblical_texts_path, sep='\t', encoding='utf-8')
            # VG_extracted.csv has columns: Reference, Book, Chapter, Verse, Part, Text
            # Create a lookup dictionary: Reference -> Text
            biblical_lookup = dict(zip(
                biblical_texts_df['reference'],
                biblical_texts_df['text']
            ))
            print(f"Loaded {len(biblical_lookup)} biblical verses for lookup.")
            
            def lookup_biblical_texts(reference_str) -> str:
                reference_str = str(reference_str).strip().replace(';','')
                """Look up biblical texts for potentially multiple references."""
                if reference_str == 'o' or pd.isna(reference_str):
                    return ''
                texts = []
                if reference_str in biblical_lookup:
                    val = biblical_lookup[reference_str]
                    if pd.notna(val):
                        texts.append(str(val))
                elif '-' in reference_str:
                    texts_concatenated = []
                    # Handle ranges (e.g., "Jn 1:1-5")
                    start_ref, end_ref = reference_str.split('-')
                    for v in range(int(start_ref.split(':')[-1]), int(end_ref) + 1):
                        single_ref = f"{start_ref.split(':')[0]}:{v}"
                        if single_ref in biblical_lookup:
                            val = biblical_lookup[single_ref]
                            if pd.notna(val):
                                texts_concatenated.append(str(val))
                    texts.append(' '.join(texts_concatenated))

                return ' ; '.join(texts) if texts else ''
            
            # Apply lookup to create biblical_text column
            sentence_level_df['biblical_text'] = sentence_level_df['reference'].apply(lookup_biblical_texts)
            reuse_set_df['biblical_text'] = reuse_set_df['reference'].apply(lookup_biblical_texts)
            sat_segment_df['biblical_text'] = sat_segment_df['reference'].apply(lookup_biblical_texts)
            
            # Count successful matches
            matched = (sentence_level_df['biblical_text'] != '').sum()
            with_refs = (sentence_level_df['reference'] != 'o').sum()

            print(f"Biblical texts merged for sentences: {matched}/{with_refs} sentences with references matched.")
            
            matched_sets = (reuse_set_df['biblical_text'] != '').sum()
            print(f"Biblical texts merged for reuse sets: {matched_sets}/{len(reuse_set_df)} sets matched.")

            matched_sat = (sat_segment_df['biblical_text'] != '').sum()
            with_refs_sat= (sat_segment_df['reference'] != 'o').sum()            
            print(f"Biblical texts merged for SaT segments: {matched_sat}/{with_refs_sat} segments matched.\n")
        except Exception as e:
            print(f"Error loading biblical texts during matching file : {e}")

    # Step 7: Save sentence-level dataset
    sentence_output_path = output_folder / 'latin_reuses_sentences.tsv'
    sentence_level_df.to_csv(sentence_output_path, sep='\t', index=False, encoding='utf-8')
    print(f"[STEP 7] Sentence-level dataset saved to: {sentence_output_path}\n")
    
    # Step 8: Save reuse-set dataset
    reuse_set_output_path = output_folder / 'latin_reuses_sets.tsv'
    reuse_set_df.to_csv(reuse_set_output_path, sep='\t', index=False, encoding='utf-8')
    print(f"[STEP 8] Reuse-set dataset saved to: {reuse_set_output_path}\n")

    # Step 9: Save SaT segment dataset
    sat_output_path = output_folder / 'latin_reuses_sat_segments.tsv'
    sat_segment_df.to_csv(sat_output_path, sep='\t', index=False, encoding='utf-8')
    print(f"[STEP 9] SaT segment dataset saved to: {sat_output_path}\n")
    print(f"[STEP 8] Reuse-set dataset saved to: {reuse_set_output_path}\n")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Source files processed: {len(tsv_files)}")
    print(f"Total words: {len(word_level_df)}")
    print(f"Total sentences: {len(sentence_level_df)}")
    print(f"Total reuse sets: {len(reuse_set_df)}")
    print(f"Total SaT segments: {len(sat_segment_df)}")
    
    # Count sentences with reuses
    sentences_with_reuses = sentence_level_df[sentence_level_df['reference'] != 'o']
    print(f"Sentences with biblical references: {len(sentences_with_reuses)}")
    
    # Label distribution in sentence dataset
    if 'label' in sentence_level_df.columns:
        print("\nLabel distribution (sentence-level):")
        # Count non-O labels
        labeled_sentences = sentence_level_df[sentence_level_df['label'] != 'o']
        print(f"  Sentences with labels: {len(labeled_sentences)}")
    
    print("\n" + "=" * 60)
    print("Dataset building complete!")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build reuse inference datasets from annotated TSV files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 3-Reuse_datasets_builder.py -i Latin_reuses -o Reuses_datasets
  python 3-Reuse_datasets_builder.py -i Latin_reuses -o Reuses_datasets -b Biblical_data/VG.tsv
        """
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Path to the input folder containing *_reuses.tsv files'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Path to the output folder for generated datasets'
    )
    parser.add_argument(
        '-b', '--biblical',
        default=None,
        help='(Optional) Path to biblical texts TSV for text lookup'
    )
    parser.add_argument(
        '--sat_model',
        default="sat-12l-sm",
        help='(Optional) SaT model name. Default: sat-12l-sm'
    )
    parser.add_argument(
        '--threshold',
        default=0.6,
        type=float,
        help='(Optional) Threshold for SaT matching. Default: 0.5'
    )

    args = parser.parse_args()
    
    main(
        input_folder=args.input,
        output_folder=args.output,
        biblical_texts_path=args.biblical,
        sat_model=args.sat_model,
        threshold=args.threshold
    )
