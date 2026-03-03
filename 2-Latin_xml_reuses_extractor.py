#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latin XML Reuses Extractor

This script extracts biblical reuse annotations from TEI/XML files and exports
them to TSV format with word-level labels and references.

Usage:
    python 2-Latin_xml_reuses_extractor.py -i <input_folder> -o <output_folder>

Arguments:
    -i, --input     Path to the input folder containing TEI/XML files
    -o, --output    Path to the output folder for generated TSV files

Example:
    python 2-Latin_xml_reuses_extractor.py -i Latin_reuses/Ref_normalized_files -o Latin_reuses/Extracted

Author: BibliReuse Project
"""

"""
By T. de la Selle, feb. 2026
For BibliReuse project, BiblIndex team
From Institut des Sources Chrétiennes
"""

import xml.etree.ElementTree as ET
import re
import os
import csv
import argparse
from processing_files import Latin_preprocessor as lp


# ==================== XML/TEI PARSING FUNCTIONS ====================

def get_clean_tag(element):
    """
    Removes namespace from tag name.
    e.g., '{http://www.tei-c.org/ns/1.0}quote' -> 'quote'
    """
    # Handle comment elements (callable tag)
    if callable(element.tag):
        return None
    if '}' in element.tag:
        return element.tag.split('}', 1)[1]
    return element.tag

def tokenize(text):
    """
    Splits text into a list of words.
    Uses simple whitespace splitting to preserve punctuation attached to words.
    """
    if not text:
        return []
    text = re.sub(r'([\.,;:!?])', r' \1 ', text)  # Ensure spacing around punctuation
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces into one
    return text.split()

def extract_comment_from_element(element):
    """
    Extracts XML comment content (<!--...-->) from inside an element.
    Returns the comment text or None if no comment found.
    Example: <seg><!--[Vg]Lk1:78-->Lk 1,78</seg> -> returns "[Vg]Lk1:78"
    """
    # Check direct children for Comment elements
    for child in element:
        # Comments have tag as a function (ET.Comment)
        if callable(child.tag):
            return child.text.strip() if child.text else None
    return None

def get_element_text_with_comment(element):
    """
    Gets the text content of an element that may have a comment as first child.
    When comment exists: <seg><!--comment-->Text</seg> -> text is in comment's tail
    When no comment: <seg>Text</seg> -> text is in element.text
    Returns (text_content, comment_content)
    """
    comment_text = None
    text_content = None
    
    # Check for comment child first
    for child in element:
        if callable(child.tag):  # This is a comment
            comment_text = child.text.strip() if child.text else None
            # The actual text follows the comment (in .tail)
            if child.tail:
                text_content = child.tail.strip()
            break
    
    # If no comment found, use element.text
    if comment_text is None and element.text:
        text_content = element.text.strip()
    
    return text_content, comment_text

def get_citation_info(element, next_sibling=None):
    """
    Helper to extract citation reference AND typelysis (type) based on known TEI patterns.
    Returns a tuple: (reference_string, type_string, initial_reference_string)
    Returns (None, None, None) if no citation context is found.
    """
    tag = get_clean_tag(element)
    
    # Skip if element is a comment (tag is None)
    if tag is None:
        return None, None, None
    
    # Pattern 1 <seg type="biblicalQuotation"><note><seg type="biblicalRef" ana="..."><!--initial-->Ref</seg></note></seg>
    if tag == 'seg' and element.get('type') == 'biblicalQuotation':
        found_refs = []
        found_types = []
        found_initial_refs = []
        for child in element:
            child_tag = get_clean_tag(child)
            if child_tag == 'note':
                # Iterate grandchildren to find the biblicalRef or refScripturaire segs
                for grand in child:
                    grand_tag = get_clean_tag(grand)
                    if grand_tag == 'seg' and (grand.get('type') == 'biblicalRef' or grand.get('type') == 'refScripturaire'):
                        # Extract text and comment using helper
                        ref_text, initial_ref = get_element_text_with_comment(grand)
                        # Always append to found_refs to keep lists aligned (even if empty)
                        found_refs.append(ref_text if ref_text else "")
                        found_initial_refs.append(initial_ref if initial_ref else "")
                        # Extract 'type' attribute
                        type_val = grand.get('ana')
                        if type_val:
                            found_types.append(type_val)
                        else:
                            found_types.append('Occurrence')  # Default type if none specified

        # Join multiple refs/types if present (e.g. joined by semicolon)
        # Return if we have any refs OR any initial refs (comment-only case)
        if any(found_refs) or any(found_initial_refs):
            ref_str = " ; ".join(found_refs) if any(found_refs) else None
            type_str = " ; ".join(found_types) if found_types else None
            initial_str = " ; ".join([r for r in found_initial_refs if r]) if any(found_initial_refs) else None
            return ref_str, type_str, initial_str
        
        # Pattern 2: <seg type="biblicalQuotation">[text]</seg><note><seg type="biblicalRef" ana="[Type]"><!--initial-->Ref</seg></note>
        # Check next sibling for the note element
        if next_sibling is not None and get_clean_tag(next_sibling) == 'note':
            for child in next_sibling:
                child_tag = get_clean_tag(child)
                if child_tag == 'seg' and child.get('type') == 'biblicalRef':
                    # Extract text and comment using helper
                    ref_text, initial_ref = get_element_text_with_comment(child)
                    # Always append to found_refs to keep lists aligned (even if empty)
                    found_refs.append(ref_text if ref_text else "")
                    found_initial_refs.append(initial_ref if initial_ref else "")
                    # Extract 'type' attribute
                    type_val = child.get('ana')
                    if type_val:
                        found_types.append(type_val)
                    else:
                        found_types.append('Occurrence')  # Default type if none specified
            
            # Return if we have any refs OR any initial refs (comment-only case)
            if any(found_refs) or any(found_initial_refs):
                ref_str = " ; ".join(found_refs) if any(found_refs) else None
                type_str = " ; ".join(found_types) if found_types else None
                initial_str = " ; ".join([r for r in found_initial_refs if r]) if any(found_initial_refs) else None
                return ref_str, type_str, initial_str
        
        # Pattern 3: <seg type="biblicalQuotation">[Text]</seg><note place="foot" xml:id="..." n="..."><p rend="footnote text"><!--[initial ref]-->[Ref]</p></note>
        # Check next sibling for a note element containing <p rend="footnote text">
        if next_sibling is not None and get_clean_tag(next_sibling) == 'note':
            for child in next_sibling:
                child_tag = get_clean_tag(child)
                if child_tag == 'p' and child.get('rend') == 'footnote text':
                    # Extract text and comment using helper (same as Patterns 1/2)
                    ref_text, initial_ref = get_element_text_with_comment(child)
                    if ref_text or initial_ref:
                        return ref_text if ref_text else None, 'Occurrence', initial_ref if initial_ref else None
            
    return None, None, None

def parse_element(element, current_ref=None, current_type=None, current_initial_ref=None, current_line=None):
    """
    Recursively parses the element and its children to extract words, refs, types, initial refs, and line numbers.
    Excludes content from metadata tags like <note> or <ref>.
    
    Args:
        element: The current XML element.
        current_ref: The active citation reference (if any).
        current_type: The active typelysis string (if any).
        current_initial_ref: The active initial reference from comment (if any).
        current_line: The current line number from <lb n="X"/> elements (if any).
    
    Returns:
        (words, refs, types, initial_refs, lines): Five aligned lists of strings.
    """
    words = []
    refs = []
    types = []
    initial_refs = []
    lines = []
    
    tag = get_clean_tag(element)
    
    # Skip comment elements
    if tag is None:
        return [], [], [], [], []
    
    # --- Filter: Skip metadata tags for word extraction ---
    if tag in ['note','ref','rdg','teiHeader','standOff']:
        return [], [], [], [], []
    
    # Check if this element is a line break <lb n="X"/>
    if tag == 'lb':
        line_num = element.get('n')
        if line_num:
            current_line = line_num

    # 1. Determine Context
    # Check if this element defines a new citation reference/type
    # Note: next_sibling is computed in the parent's loop for Pattern 2 support
    new_ref, new_type, new_initial_ref = get_citation_info(element)
    
    # If any citation info is found (ref, type, or initial_ref), it overrides the current context.
    # This handles cases where only an initial ref comment exists without normalized ref.
    if new_ref is not None or new_type is not None or new_initial_ref is not None:
        active_ref = new_ref
        active_type = new_type
        active_initial_ref = new_initial_ref
    else:
        active_ref = current_ref
        active_type = current_type
        active_initial_ref = current_initial_ref
    
    # Track active line number
    active_line = current_line

    # 2. Process Text (Content before the first child)
    if element.text:
        tokens = tokenize(element.text)
        words.extend(tokens)
        refs.extend([active_ref] * len(tokens))
        types.extend([active_type] * len(tokens))
        initial_refs.extend([active_initial_ref] * len(tokens))
        lines.extend([active_line] * len(tokens))
        
    # 3. Process Children
    children = list(element)
    for i, child in enumerate(children):
        # Skip comment elements (they have a callable tag)
        if callable(child.tag):
            continue
            
        # Get next sibling for Pattern 2 detection in get_citation_info()
        next_child = children[i + 1] if i + 1 < len(children) else None
        # Skip comments when looking for next sibling
        while next_child is not None and callable(next_child.tag):
            i += 1
            next_child = children[i + 1] if i + 1 < len(children) else None
        
        # Check if child is a line break element and update active_line
        child_tag = get_clean_tag(child)
        if child_tag == 'lb':
            line_num = child.get('n')
            if line_num:
                active_line = line_num
        
        # Check if this child defines a new citation context (with sibling lookahead for Pattern 2)
        child_ref, child_type, child_initial_ref = get_citation_info(child, next_child)
        if child_ref is not None or child_type is not None or child_initial_ref is not None:
            # Override context for this child's subtree
            child_words, child_refs, child_types, child_initial_refs, child_lines = parse_element(child, child_ref, child_type, child_initial_ref, active_line)
        else:
            # Recurse with inherited context
            child_words, child_refs, child_types, child_initial_refs, child_lines = parse_element(child, active_ref, active_type, active_initial_ref, active_line)
        words.extend(child_words)
        refs.extend(child_refs)
        types.extend(child_types)
        initial_refs.extend(child_initial_refs)
        lines.extend(child_lines)
        
        # 4. Process Tail (Content after the child, but inside the current element)
        # The tail belongs to *this* element's scope (active_ref), 
        # NOT the child's internal scope.
        if child.tail:
            tokens = tokenize(child.tail)
            words.extend(tokens)
            refs.extend([active_ref] * len(tokens))
            types.extend([active_type] * len(tokens))
            initial_refs.extend([active_initial_ref] * len(tokens))
            lines.extend([active_line] * len(tokens))
            
    return words, refs, types, initial_refs, lines

def process_tei_file(filepath):
    """
    Main entry point to parse a TEI file.
    """ 

    try:
        # Read file content for preprocessing
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Remove <seg type="modified">, <seg type="insertion">, <seg type="insertion/modified"> wrappers
        # while keeping their inner text content
        content = re.sub(r'<seg type="modified">(.+?)</seg>', r'\1', content, flags=re.DOTALL)
        content = re.sub(r'<seg type="insertion">(.+?)</seg>', r'\1', content, flags=re.DOTALL)
        content = re.sub(r'<seg type="insertion/modified">(.+?)</seg>', r'\1', content, flags=re.DOTALL)
        content = re.sub(r'<c>[^</c>]+</c>', '', content)  # remove <c>...</c> tags entirely
        content = re.sub(r'&.+?;', '', content)  # remove any &...; entities entirely
        content = re.sub(r'<seg type="biblicalQuotation">[.,;:!?]</seg>', '', content)  # fix isolated punctuation in biblicalQuotation segs
        content = re.sub(r'<seg type="biblicalQuotation">[.,;:!?]</seg>', '', content)  # fix isolated punctuation in biblicalQuotation segs
        content = re.sub(r'<pb\s*break="no"\s*edRef=([\s\S]*?)\s*n=([\s\S]*?)\s*>\s*</pb\s*>', '', content)  # remove specific pb and lb tags
        content = re.sub(r'<lb\s*break="no"\s*n=([\s\S]*?)\s*>\s*</lb\s*>', '', content)  # remove specific pb and lb tags

        # Use TreeBuilder with insert_comments=True to preserve XML comments
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        root = ET.fromstring(content, parser=parser)
        words, refs, types, initial_refs, lines = parse_element(root)
        return words, refs, types, initial_refs, lines
        
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return [], [], [], [], []

def postprocess_text(latin_words_list, keep_punctuation=False):
    """
    Postprocesses text by removing unwanted characters.
    Currently a placeholder for any additional processing needed.
    """
    # 1. Strict Cleaning (Remove numbers/Latin) BEFORE normalization
    latin_only_word_list = [processor.clean_multiple_punct(word) for word in latin_words_list] # word_list already cleaned by linguists
    latin_only_word_list = [processor.aggregate_splitted_words(word) for word in latin_only_word_list]

    # 2. Normalize (NFC, Lowercase, Sigma)
    normalized_word_list = [processor.normalize(word) for word in latin_only_word_list]

    # 3. Remove stop words if applicable
    if processor.stop_words:
        normalized_word_list = [processor.remove_stop_words(wd) for wd in normalized_word_list]

    # 4. Optionally drop punctuation
    if not keep_punctuation:
        normalized_word_list = [processor.drop_punctuation(word) for word in normalized_word_list]
    else:
        for i in range(1,len(normalized_word_list)):
            if normalized_word_list[i] in ['.', ',', ';', ':', '!', '?'] and normalized_word_list[i-1] in ['.', ',', ';', ':', '!', '?']:
                normalized_word_list[i] = ''  # remove consecutive punctuation

    # 5. Remove any combination of one letter followed by one dot
    normalized_word_list = [re.sub(r'^[a-zA-Z]\.$', '', word) for word in normalized_word_list]

    return normalized_word_list

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description='Extract biblical reuse annotations from TEI/XML files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 2-Latin_xml_reuses_extractor.py -i Latin_reuses/Ref_normalized_files -o Latin_reuses/Extracted
        """
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Path to the input folder containing TEI/XML files'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Path to the output folder for generated TSV files'
    )
    
    args = parser.parse_args()
    
    input_folder = args.input
    output_folder = args.output
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    total_words = 0
    total_reuses = 0
    reuses_markers_list = []
    reuses_markers_files = []

    for root, _, files in os.walk(input_folder):
        for fname in sorted(files):
            
            # Determine file type and process accordingly
            fname_lower = fname.lower()
            if fname_lower.endswith(".xml"):
                filename = os.path.join(root, fname)

                w_list, r_list, a_list, r_list_initial, l_list = process_tei_file(filename)

                # Initialize Latin Preprocessor
                processor = lp.LatinPreprocessor(stop_words_path=None, filter_words_path=None)

                # Postprocess lists: filter non greek characters or punctuation
                w_list = postprocess_text(w_list, keep_punctuation=True) # /!\ if stop words are filtered, punctuation may need to be dropped too to avoid consecutive punctuations

                # Postprocess lists: normalize biblical references
                r_list = [text if text else '' for text in r_list] # replace None values by '' for normalization
                r_list_initial = [text.replace('\t', '') if text else '' for text in r_list_initial] # replace None values by '' for normalization
                a_list = [text if text else 'NaR' for text in a_list] # replace None values by 'NaR' (Not a Reuse) for label
                l_list = [text if text else '' for text in l_list] # replace None values by '' for line numbers

                # Drop all lines where the word is empty after filtering
                filtered_data = [(w, r, a, r_i, l) for w, r, a, r_i, l in zip(w_list, r_list, a_list, r_list_initial, l_list) if w.strip()]
                w_list, r_list, a_list, r_list_initial, l_list = zip(*filtered_data) if filtered_data else ([], [], [], [], [])

                # export to csv file in output folder
                output_tsv = os.path.join(output_folder, fname.replace(".xml", "_reuses.tsv"))
                with open(output_tsv, mode='w', newline='', encoding='utf-8') as tsvfile:
                    writer = csv.writer(tsvfile, delimiter='\t')
                    writer.writerow(['line','word', 'label', 'reference', 'reference normalized'])
                    for l, w, a, r_i, r in zip(l_list, w_list, a_list, r_list_initial, r_list):
                        writer.writerow([l, w, a if a else "o", r_i if r_i != '' else "o", r if r != '' else "o"])
                        # writer.writerow([l, w, a, r_i, r])

                total_words += len(w_list)
                total_reuses += len([r for r in r_list_initial if r])
                reuses_markers_list += [r for r in r_list_initial if r]
                reuses_markers_files += [fname] * len([r for r in r_list_initial if r])

                print(f"{filename}: {len(w_list)} words, including {len([r for r in r_list_initial if r])} reuse markers")

            else:
                continue

    print(f"Total words & labels extracted: {total_words} with {total_reuses} reuse markers\n")

    output_unique_reuses_markers = os.path.join(output_folder, "unique_reuses_markers.txt")
    reuses_lists_dataframe = list(zip(reuses_markers_list, reuses_markers_files))
    reuses_lists_dataframe = list(dict.fromkeys(reuses_lists_dataframe))  # remove duplicates
    reuses_lists_dataframe.sort(key=lambda x: x[0])  # sort by reuse marker
    with open(output_unique_reuses_markers, mode='w', encoding='utf-8') as outfile:
        for reuse_marker, source_file in reuses_lists_dataframe:
            outfile.write(f"{reuse_marker}\t{source_file}\n")