#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEI Biblical Reference Normalizer

This script normalizes biblical references within TEI XML files found in an input directory 
and saves the processed files to a specified output directory.

Usage:
    python3 TEI_biblicalRef_normalizer.py -i <input_folder> -o <output_folder>

Arguments:
    -i, --input : Input directory containing XML files.
    -o, --output: Output directory for normalized files.
"""

"""
By T. de la Selle, feb. 2026
For BibliReuse project, BiblIndex team
From Institut des Sources Chrétiennes
"""
    
'''
/!/ Warning: commentary e.g. <!--revoir cette note-->, are note handled in this script and may cause some issues in the normalization process.
Review the references by automatic research of "biblicalRef" 

List of unregular patterns to be handled:
- verify for each file that the total number of matches are equal to the number of biblicalRef segs (from terminal print)
- 'corresp="1610983 1610984"' in biblicalRef segs (wrong place ?)
- 'part="N"' in biblicalRef segs (wrong place ?)
- 'xml:id="ftn933"' in biblicalRef segs (wrong place ?)
- look for commentaries next to biblicalRef segs (e.g. <!--revoir cette note-->) and check if they contain biblical references that should be normalized and included in the segs.
'''

import re
import os
import argparse

# ==================== XML/TEI PARSING FUNCTIONS ====================

def normalize_biblical_reference(match: re.Match) -> str:
    """Normalize a single biblical reference string."""
    
    Bible_book_table="processing_files/book_abbreviations_normalisation.csv"
    Corrections_table="processing_files/biblical_ref_corrections.csv"
    
    initial_ref = match.group(1)

    # Normalize punctuation and spaces
    ref = initial_ref.replace('et', ';')
    ref = ref.replace('_', ' ')
    ref = ref.replace(',', ':')
    ref = ref.replace('--', '-')
    ref = re.sub(r'\s*([\.:])', r'\1', ref)
    ref = re.sub(r'([\.:])\s*', r'\1', ref)

    # filter "." not between 2 digits
    ref = re.sub(r'(?<!\d)\.(?!\d)', ' ', ref)
    # filter ":" preceded by a letter
    ref = re.sub(r'(?<=[A-Za-z]):', ' ', ref)
    ref = re.sub(r'\s+', ' ', ref)
    ref = re.sub(r'\s$', '', ref)
       
    # Biblical books abbreviations normalization
    ref = re.sub(r'([A-Za-z])(\d)', r'\1 \2', ref)  # maintain a space between letter and numbers
    ref = ref.replace('Ép', 'Ep')  # specific case for Ép
    ref = ref.replace('Éz', 'Ez')  # specific case for Éz
    ref = ref.replace('Co ', 'Col ')  # specific case for Co (Colossiens)
    with open(Bible_book_table, 'r', encoding='utf-8') as f:
        book_abbrs = f.readlines()
    book_abbrs = [line.strip().split(',') for line in book_abbrs]
    book_abbrs = {abbr: full for abbr, full in book_abbrs}

    for abbr in book_abbrs.keys():
        ref = ref.replace(abbr, book_abbrs[abbr]) # add a space after the full book name to maintain separation with chapter numbers
    ref = re.sub(r'([A-Za-z])(\d)', r'\1 \2', ref)  # maintain a space between letter and numbers

    # Corrections of mismatched verses from corrections table (e.g. Jn 3:16-17 -> Jn 3:16-18)
    with open(Corrections_table, 'r', encoding='utf-8') as f:
        corrections = f.readlines()
    corrections = [line.strip().split('\t') for line in corrections]
    corrections = {wrong: correct for wrong, correct in corrections}

    for wrong, correct in corrections.items():
        ref = ref.replace(wrong, correct)

    # Handle 's' after numbers by adding two verses to range (e.g. Ps 2:3s -> Ps 2:3-5)
    if ref.endswith(('s', 'sq', 'ss','etc')):
        ref = re.sub(r'(s|sq|ss|etc)$', r'', ref)
        verse = ref.split(':')[-1]
        try:
            ref = ref.split('-')[0] + '-' + str(int(verse.split('-')[-1]) + 2)
        except Exception as e:
            print(f"Error handling 's' suffix in reference: {ref}. Error: {e}")
            # If there's an error, we can choose to keep the original reference or set it to an empty string
            # ref = '<!--' + initial_ref + '-->' + ref  # keep original reference in case of error
            ref = '<!--' + initial_ref + '-->'  # or set to empty string with comment
            pass

    # Final filtering: drop all characters that are not part of a valid biblical reference format
    ref_pattern = r'[1-3]?\s?[A-Za-z]+\s+\d{1,3}:\d{1,3}(?:\.\d{1,3})*(?:-\d{1,3})?'
    match = re.findall(ref_pattern, ref)

    initial_ref = initial_ref.replace('\t', '').replace('\n', '')

    if match.__len__() > 0:
        ref = '<!--' + initial_ref + '-->' + ' ; '.join(match)
    else:
        ref = '<!--' + initial_ref + '-->' 

    # print(f"Initial reference: {initial_ref} | Normalized reference: {ref}")

    return ref

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize biblical references in TEI XML files.")
    parser.add_argument("-i", "--input", help="Input directory containing XML files", required=True)
    parser.add_argument("-o", "--output", help="Output directory for normalized files", required=True)
    args = parser.parse_args()
    
    input_folder = args.input
    output_folder = args.output

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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

                # Read the content of the file
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()

                # ana attribute normalization and ponctual patterns normalization
                content = content.replace('citationExacte', 'Quotation')
                content = content.replace('citationScripturaire', 'Quotation')
                content = content.replace('citationInexacte', 'inexactQuotation')
                content = content.replace('inextQuotation', 'inexactQuotation')
                content = content.replace('inexctQuotation', 'inexactQuotation')
                content = content.replace('inexactQuotattion', 'inexactQuotation')
                content = content.replace('refScripturaire', 'biblicalRef')
                content = content.replace('inexactQuotation - lemme', 'inexactQuotation')
                content = content.replace('lemma', 'Quotation')
                content = content.replace('quotation', 'Quotation')
                content = content.replace('part="N"', '')
                # content = content.replace('<seg type="biblicalRef" part="N">', '<seg type="biblicalRef">')

                # add a default 'reuse type to seg without ana attribute
                content = re.sub(r'<seg\s*type="biblicalRef"\s*>','<seg type="biblicalRef" ana="Quotation">', content)
                content = re.sub(r'<hi rend="bold">([\s\S]*?)</hi>','', content)
  
                patterns = [[r'<seg\s*ana="allusion"\s*type="biblicalRef"\s*>',r'<'], 
                            [r'<seg\s*type="biblicalRef"\s*ana="allusion"\s*>',r'<'],
                            [r'<seg\s*ana="Quotation"\s*type="biblicalRef"\s*>',r'<'], 
                            [r'<seg\s*type="biblicalRef"\s*ana="Quotation"\s*>',r'<'],
                            [r'<seg\s*ana="inexactQuotation"\s*type="biblicalRef"\s*>',r'<'], 
                            [r'<seg\s*type="biblicalRef"\s*ana="inexactQuotation"\s*>',r'<'],
                            [r'<seg\s*ana="Occurrence"\s*type="biblicalRef"\s*>',r'<'], # for files converted in TEI from docx (no reuse type specified)
                            [r'<p\s*rend="footnote\s*text"\s*>',r'<'],
                            ]

                num_matches = 0
                for pat in patterns:

                    # count the number of matches for the current pattern
                    matches = re.findall(pat[0]+r'([\s\S]*?)'+pat[1], content)
                    num_matches += len(matches)

                    content = re.sub(pat[0]+r'([\s\S]*?)'+pat[1],lambda m: pat[0].replace('\s*>','>').replace('\s*',' ') + normalize_biblical_reference(m) + pat[1], content)  # find and normalize

                # write the content back to the file
                with open(os.path.join(output_folder, fname), 'w', encoding='utf-8') as f:
                    f.write(content)

                print(f"{filename} : Found {num_matches} matches for {re.findall('biblicalRef', content).__len__()} biblicalRef.")

