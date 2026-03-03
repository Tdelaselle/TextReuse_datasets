"""
Parse Ps-NT_reuses.tsv and VG.tsv to produce a TSV of reuse samples
with Vulgate (Latin) text for both Psalm and NT references.

Output columns:
  ps_ref | ps_text | reuse_label | nt_ref | nt_text
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

REUSES_FILE   = BASE_DIR / "Biblical_reuses" / "Ps-NT_reuses.tsv"
VG_FILE       = BASE_DIR / "Biblical_data" / "VG.tsv"
ABBREV_FILE   = BASE_DIR / "Biblical_reuses" / "book_abbreviations.csv"
OUTPUT_FILE   = BASE_DIR / "Biblical_reuses" / "Ps-NT_reuses_VG.tsv"

# ── 1.  Load book abbreviation mappings ──────────────────────────────

def load_abbreviations(path: Path) -> dict:
    """Return two dicts:
       full_to_abbr  : 'Psalms' -> 'Ps'
       abbr_to_full  : 'Ps'     -> 'Psalms'
    Also handle normalised short forms found in the reuses file.
    """
    full_to_abbr: dict[str, str] = {}
    abbr_to_full: dict[str, str] = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) != 2:
                continue
            full_name, abbr = parts[0].strip(), parts[1].strip()
            full_to_abbr[full_name] = abbr
            abbr_to_full[abbr] = full_name

    return full_to_abbr, abbr_to_full


# ── 2.  Load VG.tsv into a lookup  (book, chapter, verse) → text ─────

def load_vulgate(path: Path) -> dict:
    """Return dict keyed by (book_full_name, chapter_int, verse_int) -> text"""
    vg: dict[tuple, str] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if len(row) < 6:
                continue
            book = row[1].strip()
            try:
                chapter = int(row[2].strip())
                verse   = int(row[3].strip())
            except ValueError:
                continue
            text = row[5].strip()
            vg[(book, chapter, verse)] = text
    return vg


# ── 3.  Normalise a book name coming from the reuses file ────────────

# Some rows in the reuses file use compact abbreviations that differ
# from book_abbreviations.csv (e.g. "1Co" instead of "1 Co", "He" instead
# of "Heb").  We build a secondary look-up from those to the canonical
# full name used in VG.tsv.

# Manually-curated aliases for forms found in the reuses file that are
# NOT already in book_abbreviations.csv.
_EXTRA_ALIASES: dict[str, str] = {
    # abbreviated forms without spaces
    "1Co":  "1 Corinthians",
    "2Co":  "2 Corinthians",
    "1Jn":  "1 John",
    "2Jn":  "2 John",
    "1P":   "1 Peter",
    "2P":   "2 Peter",  # in case it appears
    "1Th":  "1 Thessalonians",
    "2Th":  "2 Thessalonians",  # in case it appears
    "2Tm":  "2 Timothy",
    "He":   "Hebrews",
    "Ja":   "James",
    "Jn":   "John",
    "Lk":   "Luke",
    "Mk":   "Mark",
    "Mt":   "Matthew",
    "Ac":   "Acts",
    "Ep":   "Ephesians",
    "Ga":   "Galatians",
    "Re":   "Revelation",
    "Ro":   "Romans",
    # Spelling variants
    "1 Thessalonicians": "1 Thessalonians",
    "2 Thessalonicians": "2 Thessalonians",
    "Tite":              "Titus",
}


def build_name_resolver(full_to_abbr, abbr_to_full):
    """Return a function  resolve(name) -> (full_name, abbreviation)."""
    # Build a combined lookup: any known string → full name
    known: dict[str, str] = {}
    for full, abbr in full_to_abbr.items():
        known[full]  = full
        known[abbr]  = full
    for abbr, full in abbr_to_full.items():
        known[abbr]  = full
        known[full]  = full
    for alias, full in _EXTRA_ALIASES.items():
        known[alias] = full

    def resolve(name: str):
        name = name.strip()
        if name in known:
            full = known[name]
            abbr = full_to_abbr.get(full, name)
            return full, abbr
        # Fallback: try stripping spaces in number-prefix forms
        collapsed = re.sub(r'^(\d+)\s+', r'\1', name)
        if collapsed in known:
            full = known[collapsed]
            abbr = full_to_abbr.get(full, name)
            return full, abbr
        print(f"  [WARN] Could not resolve book name: {name!r}", file=sys.stderr)
        return name, name

    return resolve


# ── 4.  Retrieve text from VG for a verse or verse range ─────────────

def get_vg_text(vg: dict, book_full: str, chapter: int, verse_str: str) -> str:
    """
    verse_str may be a single verse ('6') or a range ('6-8').
    Concatenate the Vulgate text for each verse in the range.
    """
    verse_str = verse_str.strip()
    if not verse_str:
        return ""

    m = re.match(r'^(\d+)\s*-\s*(\d+)$', verse_str)
    if m:
        v_start, v_end = int(m.group(1)), int(m.group(2))
        parts = []
        for v in range(v_start, v_end + 1):
            t = vg.get((book_full, chapter, v), "")
            if t:
                parts.append(t)
        return " ".join(parts)
    else:
        # May contain something like "15 16" (space-separated in Ps field)
        # Try as single int first
        try:
            v = int(verse_str)
            return vg.get((book_full, chapter, v), "")
        except ValueError:
            # Try space-separated
            nums = re.findall(r'\d+', verse_str)
            if nums:
                parts = []
                for n in nums:
                    t = vg.get((book_full, chapter, int(n)), "")
                    if t:
                        parts.append(t)
                return " ".join(parts)
            return ""


# ── 5.  Build reference string ────────────────────────────────────────

def build_ref(abbr: str, chapter: str, verse: str) -> str:
    """Build e.g. 'Ps 98:6' or '1 Co 10:3-4'."""
    verse = verse.strip()
    chapter = chapter.strip()
    return f"{abbr} {chapter}:{verse}"


# ── 6.  Main ──────────────────────────────────────────────────────────

def main():
    full_to_abbr, abbr_to_full = load_abbreviations(ABBREV_FILE)
    resolve = build_name_resolver(full_to_abbr, abbr_to_full)
    vg = load_vulgate(VG_FILE)

    print(f"Loaded {len(vg)} Vulgate verses.", file=sys.stderr)

    rows_out = []
    missing_ps = 0
    missing_nt = 0

    with open(REUSES_FILE, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)

        for i, row in enumerate(reader, start=2):  # line 2 onwards
            if len(row) < 11:
                continue  # malformed row

            nt_book_raw = row[0].strip()
            nt_chapter  = row[1].strip()
            nt_verse    = row[2].strip()
            reuse_label = row[4].strip()

            ps_chapter  = row[9].strip()
            ps_verse    = row[10].strip()

            if not nt_book_raw or not nt_chapter or not nt_verse:
                continue
            if not ps_chapter or not ps_verse:
                continue
            # Skip rows where the verse field is clearly not a number/range
            if not re.search(r'\d', nt_verse) or not re.search(r'\d', ps_verse):
                continue

            # Resolve book names
            nt_full, nt_abbr = resolve(nt_book_raw)
            ps_full = "Psalms"
            ps_abbr = "Ps"

            # Parse chapters as int
            try:
                nt_ch = int(nt_chapter)
                ps_ch = int(ps_chapter)
            except ValueError:
                continue

            # Get Vulgate texts
            ps_text = get_vg_text(vg, ps_full, ps_ch, ps_verse)
            nt_text = get_vg_text(vg, nt_full, nt_ch, nt_verse)

            if not ps_text:
                missing_ps += 1
            if not nt_text:
                missing_nt += 1

            ps_ref = build_ref(ps_abbr, ps_chapter, ps_verse)
            nt_ref = build_ref(nt_abbr, nt_chapter, nt_verse)

            rows_out.append((nt_ref, nt_text, reuse_label, ps_ref, ps_text))

    # Write output
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["NT_ref", "NT_text", "reuse_label", "Ps_ref", "Ps_text"])
        for row in rows_out:
            writer.writerow(row)

    print(f"Wrote {len(rows_out)} reuse samples to {OUTPUT_FILE}", file=sys.stderr)
    if missing_ps:
        print(f"  {missing_ps} rows with missing Psalm Vulgate text", file=sys.stderr)
    if missing_nt:
        print(f"  {missing_nt} rows with missing NT Vulgate text", file=sys.stderr)


if __name__ == "__main__":
    main()
