"""
tsv_to_rawtext.py
-----------------
Reads every tsv file in a given directory, concatenates the tokens from the
'word' column into a single line of raw text, and writes the result to a .txt
file (same base name) in an output directory.

Punctuation characters are attached to the preceding word (no leading space).

Usage
-----
    python tsv_to_rawtext.py <input_directory> [output_directory]

If no output directory is given, a folder called "raw_text" is created inside
the input directory.
"""

import csv
import os
import sys
import string

# Characters that should be glued to the preceding word (no space before them)
PUNCT = set(string.punctuation)          # includes , . ; : ! ? ' " etc.
PUNCT.update({"»", "›", ")", "]", "}"}) # common closing marks
OPENING = {"«", "‹", "(", "[", "{"}      # opening marks: no space after them


def tokens_to_text(tokens: list[str]) -> str:
    """Join a list of tokens into natural-looking text on a single line.

    - Punctuation like , . ; : is attached to the preceding word.
    - Opening brackets / quotes suppress the space before the next token.
    """
    if not tokens:
        return ""

    parts: list[str] = [tokens[0]]
    glue_next = tokens[0] in OPENING  # after an opening mark, glue the next word

    for tok in tokens[1:]:
        if tok in PUNCT:
            # Attach punctuation directly to the previous word
            parts.append(tok)
            glue_next = False
        elif glue_next:
            parts.append(tok)
            glue_next = False
        else:
            parts.append(" " + tok)

        if tok in OPENING:
            glue_next = True

    return "".join(parts)


def tsv_to_rawtext(tsv_path: str, txt_path: str) -> None:
    """Read a single tsv and write its word column as one-line raw text."""
    tokens: list[str] = []

    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            word = row.get("word", "").strip()
            if word:
                tokens.append(word)

    text = tokens_to_text(tokens)

    with open(txt_path, "w", encoding="utf-8") as out:
        out.write(text + "\n")


def process_directory(input_dir: str, output_dir: str | None = None) -> None:
    """Process every .tsv file found in *input_dir*."""
    if output_dir is None:
        output_dir = os.path.join(input_dir, "raw_text")
    os.makedirs(output_dir, exist_ok=True)

    tsv_files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".tsv"))

    if not tsv_files:
        print(f"No tsv files found in {input_dir}")
        return

    for fname in tsv_files:
        tsv_path = os.path.join(input_dir, fname)
        txt_name = os.path.splitext(fname)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_name)

        tsv_to_rawtext(tsv_path, txt_path)
        print(f"  ✓ {fname}  →  {txt_name}")

    print(f"\nDone – {len(tsv_files)} file(s) written to {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tsv_to_rawtext.py <input_directory> [output_directory]")
        sys.exit(1)

    in_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    process_directory(in_dir, out_dir)
