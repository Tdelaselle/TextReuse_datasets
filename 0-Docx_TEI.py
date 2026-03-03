"""
Docx to TEI XML Converter

This script converts Microsoft Word (.docx) documents into TEI XML format.
It handles text, footnotes, and specific styling to markup biblical quotations
and insertions.

Usage:
    python3 "Docx->TEI.py" <input_folder> [--output_folder OUTPUT_FOLDER]

Arguments:
    input_folder:   Path to the folder containing .docx files to convert.
    output_folder:  (Optional) Path to save the generated XML files.
"""

"""
By T. de la Selle, feb. 2026
For BibliReuse project, BiblIndex team
From Institut des Sources Chrétiennes
"""

import zipfile
import xml.etree.ElementTree as ET
import re
import os
import argparse
from xml.dom import minidom

# Microsoft OpenXML Namespace
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
TEI_NS = "http://www.tei-c.org/ns/1.0"

def get_style_signature(r_element):
    """
    Extracts a signature tuple for color/highlight/shading/style to identify grouped text.
    """
    rPr = r_element.find(f"{{{NS['w']}}}rPr")
    if rPr is None:
        return None
    
    props = []
    # A. Font Color
    color = rPr.find(f"{{{NS['w']}}}color")
    if color is not None:
        val = color.get(f"{{{NS['w']}}}val")
        if val and val != "auto": props.append(f"c:{val}")
    
    # B. Highlight
    highlight = rPr.find(f"{{{NS['w']}}}highlight")
    if highlight is not None:
        val = highlight.get(f"{{{NS['w']}}}val")
        if val and val != "none": props.append(f"h:{val}")
    
    # C. Shading
    shd = rPr.find(f"{{{NS['w']}}}shd")
    if shd is not None:
        val = shd.get(f"{{{NS['w']}}}fill")
        if val and val not in ["auto", "clear"]: props.append(f"s:{val}")

    # D. Character Style
    style = rPr.find(f"{{{NS['w']}}}rStyle")
    if style is not None:
        val = style.get(f"{{{NS['w']}}}val")
        if val: props.append(f"st:{val}")

    if not props:
        return None
    return tuple(sorted(props))

def extract_footnotes_map(zip_ref):
    """
    Extracts footnote content map {id: text}.
    """
    footnotes = {}
    try:
        xml_content = zip_ref.read('word/footnotes.xml')
        root = ET.fromstring(xml_content)
        for fn in root.findall('.//w:footnote', NS):
            fn_id = fn.get(f"{{{NS['w']}}}id")
            fn_type = fn.get(f"{{{NS['w']}}}type")
            if fn_type in ['separator', 'continuationSeparator']: continue
            
            text_parts = []
            for t_node in fn.findall('.//w:t', NS):
                if t_node.text: text_parts.append(t_node.text)
            footnotes[fn_id] = "".join(text_parts).strip()
    except KeyError:
        pass
    return footnotes

def parse_docx_to_tokens(zip_ref):
    """
    Parses document.xml into a flat list of tokens for processing.
    """
    doc_xml = zip_ref.read('word/document.xml')
    root = ET.fromstring(doc_xml)
    
    tokens = []
    p_counter = 0
    
    for p in root.findall('.//w:p', NS):
        p_counter += 1
        tokens.append({'type': 'p_start', 'p_id': p_counter})
        
        for child in p:
            if child.tag == f"{{{NS['w']}}}r":
                style = get_style_signature(child)
                
                for item in child:
                    tag = item.tag
                    
                    # 1. Page Break
                    if tag == f"{{{NS['w']}}}br" and item.get(f"{{{NS['w']}}}type") == 'page':
                        tokens.append({'type': 'pb'})
                    elif tag == f"{{{NS['w']}}}lastRenderedPageBreak":
                        tokens.append({'type': 'pb'})
                        
                    # 2. Text
                    elif tag == f"{{{NS['w']}}}t":
                        text = item.text or ""
                        # Split by line number pattern (digits)
                        parts = re.split(r'(\(\d+\))', text)
                        for part in parts:
                            if not part: continue
                            if re.match(r'^\(\d+\)$', part):
                                n = part.strip('()')
                                tokens.append({'type': 'lb', 'n': n})
                            else:
                                # Split by spaces but preserve them as separate tokens 
                                # to maintain spacing fidelity
                                words = re.split(r'(\s+)', part)
                                for w in words:
                                    if not w: continue
                                    tokens.append({'type': 'text', 'content': w, 'style': style})
                                    
                    # 3. Footnote Reference
                    elif tag == f"{{{NS['w']}}}footnoteReference":
                        fn_id = item.get(f"{{{NS['w']}}}id")
                        tokens.append({'type': 'fn_ref', 'id': fn_id, 'style': style})
                        
        tokens.append({'type': 'p_end'})
    return tokens

def apply_citations(tokens, fn_map):
    """
    Modifies the token list in-place to add 'citation' and 'is_insertion' fields to text tokens.
    Includes logic to stop at previous footnotes and handle long insertions.
    """
    for i, token in enumerate(tokens):
        if token['type'] == 'fn_ref':
            fn_id = token['id']
            ref_content = fn_map.get(fn_id, "")
            
            # --- Backtracking Logic ---
            target_style = None
            
            # 1. Peek backwards to find the style of the text being referenced
            scan_idx = i - 1
            while scan_idx >= 0:
                t = tokens[scan_idx]
                if t['type'] == 'p_start': break
                # Stop looking if we hit a boundary that separates citations
                if t['type'] == 'fn_ref': break 
                if t['type'] == 'text' and t.get('citation'): break

                if t['type'] == 'text' and t.get('style'):
                    target_style = t['style']
                    break
                scan_idx -= 1
            
            # 2. Apply citation to matching tokens
            if target_style:
                curr = i - 1
                while curr >= 0:
                    t = tokens[curr]
                    
                    # --- STOP CONDITIONS (Break Procedure) ---
                    if t['type'] == 'p_start': break
                    if t['type'] == 'fn_ref': break # Stop immediately if we hit another footnote
                    if t['type'] == 'text' and t.get('citation'): break # Stop if overlap with existing citation
                    
                    # --- PROCESS TOKEN ---
                    if t['type'] == 'text':
                        if t.get('style') == target_style:
                            t['citation'] = ref_content
                            t['is_insertion'] = False
                        else:
                            # Not a match? Check if we can bridge a gap (spaces/line breaks/uncolored text)
                            # Look deeper back to see if the block continues
                            found_more = False
                            search = curr - 1
                            lookback_limit = 500 # Limit increased to support long insertions
                            
                            while search >= 0 and lookback_limit > 0:
                                st = tokens[search]
                                
                                # Safety stops during look-ahead
                                if st['type'] == 'p_start': break
                                if st['type'] == 'fn_ref': break
                                if st['type'] == 'text':
                                    lookback_limit -= 1
                                    if st.get('citation'): break # Hit another citation
                                    
                                    if st.get('style') == target_style:
                                        found_more = True
                                        break
                                search -= 1
                            
                            if found_more:
                                # It's a gap inside the quote -> tag it as insertion
                                t['citation'] = ref_content
                                t['is_insertion'] = True
                            else:
                                # End of quote block
                                break
                    
                    elif t['type'] in ['lb', 'pb']:
                        # Neutral elements inherit citation context if we continue looping
                        pass
                        
                    curr -= 1
            else:
                # Fallback: Tag only the immediately preceding word if no style found
                # Enforce boundary checks here too
                prev = i - 1
                while prev >= 0:
                    pt = tokens[prev]
                    if pt['type'] == 'p_start': break
                    if pt['type'] == 'fn_ref': break
                    if pt['type'] == 'text':
                         if pt.get('citation'): break
                         pt['citation'] = ref_content
                         pt['is_insertion'] = False
                         break # Only tag one word
                    prev -= 1

def generate_tei_xml(tokens):
    """
    Generates the XML tree from tokens.
    """
    # Root Structure
    root = ET.Element("TEI", xmlns="http://www.tei-c.org/ns/1.0")
    
    # Header
    header = ET.SubElement(root, "teiHeader")
    fileDesc = ET.SubElement(header, "fileDesc")
    titleStmt = ET.SubElement(fileDesc, "titleStmt")
    ET.SubElement(titleStmt, "title").text = "Converted Document"
    ET.SubElement(fileDesc, "publicationStmt").append(ET.Element("p"))
    ET.SubElement(fileDesc, "sourceDesc").append(ET.Element("p"))
    
    # Body
    text_node = ET.SubElement(root, "text")
    body = ET.SubElement(text_node, "body")
    div = ET.SubElement(body, "div", type="textpart", subtype="section", n="1")
    
    current_p = None
    active_citation = None
    seg_node = None       # <seg type="biblicalQuotation">
    insertion_node = None # <seg type="insertion">
    
    def append_text(node, text):
        if len(node) > 0:
            last = node[-1]
            last.tail = (last.tail or "") + text
        else:
            node.text = (node.text or "") + text

    for token in tokens:
        if token['type'] == 'p_start':
            # Start new paragraph
            current_p = ET.SubElement(div, "p")
            active_citation = None 
            seg_node = None
            insertion_node = None
            
        elif token['type'] == 'p_end':
            # Close stray segments
            if active_citation and seg_node is not None:
                note = ET.SubElement(seg_node, "note", type="biblicalNote")
                ref = ET.SubElement(note, "seg", ana="Occurrence", type="biblicalRef")
                ref.text = active_citation
            current_p = None
            active_citation = None
            seg_node = None
            insertion_node = None
            
        elif token['type'] == 'lb':
            # Determine parent: insertion -> citation -> paragraph
            parent = current_p
            if active_citation is not None:
                if insertion_node is not None:
                    parent = insertion_node
                elif seg_node is not None:
                    parent = seg_node
            
            if parent is not None:
                ET.SubElement(parent, "lb", n=token['n'])
                
        elif token['type'] == 'pb':
            parent = current_p
            if active_citation is not None:
                if insertion_node is not None:
                    parent = insertion_node
                elif seg_node is not None:
                    parent = seg_node
                    
            if parent is not None:
                ET.SubElement(parent, "pb")
                
        elif token['type'] == 'text':
            citation = token.get('citation')
            is_insertion = token.get('is_insertion', False)
            content = token['content']
            
            if current_p is None: continue 
            
            # --- State Management ---
            
            # 1. Check if Citation Scope Changed (Start/End of Quote)
            if citation != active_citation:
                # Close everything old
                insertion_node = None
                if active_citation is not None and seg_node is not None:
                    note = ET.SubElement(seg_node, "note", type="biblicalNote")
                    ref = ET.SubElement(note, "seg", ana="Occurrence", type="biblicalRef")
                    ref.text = active_citation
                    seg_node = None
                
                # Open New Citation
                if citation is not None:
                    seg_node = ET.SubElement(current_p, "seg", type="biblicalQuotation")
                
                active_citation = citation

            # 2. Inside a Citation: Check Insertion Scope
            target = current_p
            if active_citation is not None:
                if is_insertion:
                    if insertion_node is None:
                        insertion_node = ET.SubElement(seg_node, "seg", type="insertion/modified")
                    target = insertion_node
                else:
                    # Colored text
                    if insertion_node is not None:
                        insertion_node = None # Close insertion
                    target = seg_node
            
            append_text(target, content)

    return root

def prettify_xml(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def drop_empty_insertions(root):
    """
    Removes empty <seg type="insertion/modified"> elements (containing only whitespace).
    Preserves any tail text by attaching it to the previous sibling or parent.
    """
    # Find all seg elements with type="insertion/modified"
    for parent in root.iter():
        children = list(parent)
        for i, child in enumerate(children):
            if child.tag == "seg" or child.tag == f"{{{TEI_NS}}}seg":
                seg_type = child.get("type")
                if seg_type == "insertion/modified":
                    # Check if empty (no child elements and text is empty/whitespace)
                    text_content = (child.text or "").strip()
                    has_children = len(child) > 0
                    
                    if not has_children and not text_content:
                        # Preserve tail text
                        tail = child.tail
                        if tail:
                            if i > 0:
                                # Append to previous sibling's tail
                                prev = children[i - 1]
                                prev.tail = (prev.tail or "") + tail
                            else:
                                # Append to parent's text
                                parent.text = (parent.text or "") + tail
                        parent.remove(child)

def main(input_folder, output_folder=None):
    """Process all .docx files in the input folder and convert them to TEI XML."""
    if not os.path.isdir(input_folder):
        print(f"Error: '{input_folder}' is not a valid directory.")
        return
    
    docx_count = 0
    for dirpath, _, files in os.walk(input_folder):
        for file in sorted(files):
            if file.endswith(".docx"):
                docx_count += 1
                print(f"Processing {file}...")
            
                # 1. Parse Zip
                with zipfile.ZipFile(os.path.join(dirpath, file), 'r') as z:
                    # 2. Extract Footnotes
                    fn_map = extract_footnotes_map(z)
                    # 3. Parse Document Tokens
                    tokens = parse_docx_to_tokens(z)
                
                # 4. Link Footnotes to Text
                apply_citations(tokens, fn_map)
                
                # 5. Build XML
                tei_root = generate_tei_xml(tokens)
                
                # 6. Clean up empty insertions
                drop_empty_insertions(tei_root)

                # 7. Save
                if output_folder:
                    # Maintain subdirectory structure
                    rel_path = os.path.relpath(dirpath, input_folder)
                    target_dir = os.path.join(output_folder, rel_path)
                    os.makedirs(target_dir, exist_ok=True)
                    output_path = os.path.join(target_dir, file[:-5] + ".xml")
                else:
                    output_path = os.path.join(dirpath, file[:-5] + ".xml")

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(prettify_xml(tei_root))
                    
                print(f"Done! Saved to {output_path}")
    
    if docx_count == 0:
        print(f"No .docx files found in '{input_folder}'.")
    else:
        print(f"\nProcessed {docx_count} file(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Word (.docx) files to TEI XML format."
    )
    parser.add_argument(
        "input_folder",
        help="Path to the folder containing .docx files to convert"
    )
    parser.add_argument(
        "--output_folder",
        help="Optional path to helper output folder. Defaults to input folder.",
        default=None
    )
    
    args = parser.parse_args()
    main(args.input_folder, args.output_folder)