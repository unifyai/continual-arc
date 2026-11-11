"""Compare the figures and tables the scripts wrote under build/ with the files the paper sets, figures/, sections/ and
sections/tables/. A section or a table matches when its text is the same as arXiv's copy of the source has it: its
comments taken out, and each conditional on \\arxivpreprint read as the preprint reads it. A figure matches when its
PDF holds the same objects, each stream decompressed, but for the date matplotlib writes in it and the numbers of its
objects: matplotlib writes a figure's fonts in the order of their files' paths, so the same figure drawn from another
checkout numbers its fonts' objects, and orders its font dictionary, otherwise.

    uv run python scripts/compare.py [--out DIR]

It prints a line for each file the paper sets that the scripts made, same or differs, and exits 1 if any differs.
"""

from __future__ import annotations

import argparse
import re
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def uncommented(text: str) -> str:
    """``text`` as TeX reads it, its comments gone: a line that is only a comment goes whole, its newline with it, and a
    trailing comment keeps its %, which joins the line to the next as before."""
    out = []
    for line in text.split("\n"):
        comment = re.search(r"(?<!\\)%", line)
        if comment is None:
            out.append(line)
        elif line[: comment.start()].strip():
            out.append(line[: comment.start() + 1])
    return "\n".join(out)


CONDITIONAL = re.compile(r"\\(if[a-zA-Z]*|else|fi)(?![a-zA-Z])")
PREPRINT = "\\ifdefined\\arxivpreprint"


def preprint(text: str) -> str:
    """``text`` with every \\ifdefined\\arxivpreprint X\\else Y\\fi as X, as TeX reads it with \\arxivpreprint
    defined."""
    while (start := text.find(PREPRINT)) != -1:
        depth, middle, end = 0, None, None
        for found in CONDITIONAL.finditer(text, start + len(PREPRINT)):
            word = found.group(1)
            if word.startswith("if"):
                depth += 1
            elif word == "else" and depth == 0:
                middle = found
            elif word == "fi":
                if depth == 0:
                    end = found
                    break
                depth -= 1
        kept = text[start + len(PREPRINT): (middle or end).start()]
        lead = kept[: len(kept) - len(kept.lstrip())]
        kept = ("\n\n" if lead.count("\n") >= 2 else "") + kept.lstrip()
        text = text[:start] + kept + text[end.end():]
    return text


def pdf_objects(data: bytes) -> list[bytes]:
    """A PDF's objects, sorted, each stream decompressed, every reference to another object and every stream's length
    written alike, a dictionary of references alone (as a page's fonts are) in sorted order, and its creation date
    taken out."""
    data = re.sub(rb"/CreationDate \(D:[^)]*\)", b"/CreationDate ()", data)
    out = []
    for m in re.finditer(rb"\n\d+ 0 obj\n(.*?)\nendobj", data, re.S):
        body = m.group(1)
        start = body.find(b"stream\n")
        if start != -1:
            head, raw = body[:start], body[start + len(b"stream\n"):]
            raw = raw[: raw.rindex(b"endstream")].rstrip(b"\n")
            if b"/FlateDecode" in head:
                raw = zlib.decompress(raw)
            body = re.sub(rb"/Length \d+( 0 R)?", b"/Length", head) + b"stream\n" + raw
        body = re.sub(rb"\d+ 0 R", b"R", body)
        refs = re.fullmatch(rb"<<((?:\s*/\S+ R)+)\s*>>", body)
        if refs:
            body = b"<< " + b" ".join(sorted(re.findall(rb"/\S+ R", refs.group(1)))) + b" >>"
        out.append(body)
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "build", help="where the scripts wrote (default build/)")
    out = ap.parse_args().out
    differ = 0
    for made in sorted((out / "figures").glob("*.pdf")) + sorted((out / "sections").rglob("*.tex")):
        paper = ROOT / made.relative_to(out)
        if not paper.exists():
            continue
        if made.suffix == ".pdf":
            same = pdf_objects(made.read_bytes()) == pdf_objects(paper.read_bytes())
        else:
            same = preprint(uncommented(made.read_text())) == uncommented(paper.read_text())
        differ += not same
        print(f"{'same   ' if same else 'DIFFERS'}  {paper.relative_to(ROOT)}")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
