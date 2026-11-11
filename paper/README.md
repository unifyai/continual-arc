# The paper

The LaTeX source of [*Continual-ARC: Measuring Skill Acquisition over a Working
Lifetime*](https://arxiv.org/abs/2611.14505) (Daniel Lenton, 2026), arXiv:2611.14505v1, as it was posted to arXiv on
10 November 2026. Every file here is the file arXiv serves for that version at
`https://arxiv.org/e-print/2611.14505v1`, byte for byte, but for three: this README, and `references.bib` and
`preprint.bst`, the bibliography's database and its style, which the upload replaced with the `main.bbl` they make.

The paper is under CC BY 4.0, the license it carries on arXiv. `preprint.bst`, `natbib.sty` and `fancyhdr.sty` are
under the LaTeX Project Public License.

## Layout

```
main.tex            the paper: its packages, title and author, and its sections in reading order
sections/           a file per section, 00 to 09 for the main text and the statements after it, A1 to A12 for the
                    appendix; main.tex sets their order
sections/tables/    the tables, and the numbers the prose takes from them (solved-numbers.tex, chance-numbers.tex)
figures/            the figures as the paper sets them
notes/              the author block and the acknowledgments, which main.tex reads
math_commands.tex   the notation's macros
preprint.sty        the page style, with fancyhdr.sty and natbib.sty
main.bbl            the bibliography, as the build reads it
references.bib      the BibTeX database main.bbl is made from
preprint.bst        the natbib author-year style main.bbl is made with
00README.json       arXiv's record of how it compiles the source: pdflatex under TeX Live 2025
```

## Building the PDF

With [tectonic](https://tectonic-typesetting.github.io), which fetches the packages it needs:

```bash
cd paper
tectonic --outdir build main.tex    # build/main.pdf, 51 pages
```

tectonic runs BibTeX on `references.bib` with `preprint.bst`, which writes this folder's `main.bbl` byte for byte.
With TeX Live, as arXiv compiles it, pdflatex reads `main.bbl` and needs no BibTeX run:

```bash
cd paper
pdflatex main.tex && pdflatex main.tex && pdflatex main.tex    # main.pdf
```

## The figures and tables

The figures, but for the three that need no runs (`protocol.pdf`, `modes.pdf` and `taskid.pdf`), and the tables are
computed from the runs' records, which are in `runs/` at the repository's root: `runs/runs.json` lists the GPT-6
Luna runs, `runs/open-models.json` the open-model runs with the runs each is read against, and `runs/machines.json`
the machines whose every run the text reports, and each run's records are in `runs/<track>/<run>/`.

The code that computes them is not in this folder yet. It reads the open-model runs in the layout they had on the
machines that played them, which `runs/` does not keep, so it cannot yet be run from `runs/`. Some of its inputs are
not in `runs/` at all:

- Table 13 (Appendix J), from the calibration reports of the private set, whose tasks are not public;
- Figures 7 and 8 (Appendix H), from the training records of DAgger's runs, which `runs/` holds by their plays alone;
- Tables 17 and 18 and Figure 13 (Appendix L), from the providers' prices and the costs recorded for each family of
  experiments.

## What is not here

- The code that computes the figures and tables, above.
- The working notes, drafts and reviews behind the paper, which are not part of it.
