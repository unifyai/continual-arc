# The paper

The LaTeX source of [*Continual-ARC: Measuring Skill Acquisition over a Working
Lifetime*](https://arxiv.org/abs/2611.14505) (Daniel Lenton, Yusha Arif and Haris Mahmood, 2026), arXiv:2611.14505v2, as
it was posted to arXiv on 11 November 2026. Every file here is the file arXiv serves for that version at
`https://arxiv.org/e-print/2611.14505v2`, byte for byte, but for this README, the code that computes the figures and
tables with what it reads besides the runs (`Makefile`, `scripts/`, `data/` and `fonts/`, below), and `references.bib`
and `preprint.bst`, the bibliography's database and its style, which the upload replaced with the `main.bbl` they make.

The paper is under CC BY 4.0, the license it carries on arXiv. `preprint.bst`, `natbib.sty` and `fancyhdr.sty` are
under the LaTeX Project Public License, the fonts in `fonts/` under the SIL Open Font License, each with its license
beside it, and the code in `scripts/` under the repository's Apache 2.0 license.

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
Makefile            the figures and tables computed again from the runs, and compared with these (below)
scripts/            the code that computes them
data/               what that code reads besides the runs: providers' prices, recorded costs, the instances an outage
                    played through
fonts/              the fonts the figures are set in
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

The code in `scripts/` computes them again from those records, with the benchmark installed from the same checkout,
and writes them to `build/` in this folder's own layout, `build/figures/`, `build/sections/` and
`build/sections/tables/`, beside the files the paper sets rather than over them. With
[uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/unifyai/continual-arc && cd continual-arc/paper
make            # the data Figure 6 reads, every figure and table below, and each compared with the paper's own
```

| In the paper | File | Made by |
| --- | --- | --- |
| Figure 1 | `figures/protocol.pdf` | `make_figures.py protocol` |
| Figure 2 | `figures/modes.pdf` | `make_figures.py modes` |
| Figure 3 | `figures/tests.pdf` | `make_figures.py tests` |
| Figure 4 | `figures/notes-weights.pdf` | `make_figures.py notes-weights` |
| Figure 5 | `figures/plane-luna.pdf` | `make_figures.py plane-luna` |
| Figure 6 | `figures/taskid.pdf` | `make_figures.py taskid`, after `make data` |
| Figure 7 | `figures/weight-updates.pdf` | `weight_updates.py` |
| Figure 8 | `figures/weight-losses.pdf` | `make_figures.py weight-losses` |
| Figure 9 | `figures/forgetting.pdf` | `forgetting.py`, with the numbers Appendix H quotes of it |
| Figure 10 | `figures/luna-cost.pdf` | `make_figures.py luna-cost` |
| Figures 11 and 12 | `figures/tests-arc2.pdf`, `figures/tests-arc3.pdf` | `make_figures.py tests-arc2 tests-arc3` |
| Figure 13 and Table 17 | `figures/inference-cost.pdf`, `sections/tables/inference-pricing.tex` | `make_figures.py inference-cost` |
| Tables 1, 6 and 11 | `sections/tables/tests.tex`, `tests-notes.tex` and `tests-glm.tex`, `tests-users.tex` | `make_figures.py tests-table` |
| Table 4 and Appendix C's listings | `sections/A2-generated.tex` | `make appendix`, from the releases `v0.3.0` and `v0.5.0` |
| Table 7 | `sections/tables/ablations-arc1.tex` | `make_figures.py ablations` |
| Table 9 | `sections/tables/solved-budget.tex` | `make_figures.py solved-budget-table` |
| Table 10 | `sections/tables/tests-absolute.tex` | `make_figures.py tests-absolute` |
| Table 12 | `sections/tables/tests-luna.tex` | `make_figures.py tests-luna` |
| Tables 15 and 16, set by hand | their numbers, printed | `tracks_tables.py` (`make tracks`) |
| Table 18 | `sections/tables/experiment-costs.tex` | `experiment_costs.py`, with its manifest of sources |
| The numbers the text takes from the tables | `sections/tables/chance-numbers.tex`, `solved-numbers.tex`, `status.tex` | `make_figures.py tests-table`, `solved-budget-table`, `status` |

Each script runs as `uv run --extra plots python scripts/<script> [names]` from this folder, `make_figures.py` making
every name in its `GENERATORS` when given none, and `make compare` (`scripts/compare.py`) checks each file in `build/`
against the paper's own: a table as arXiv's copy of the source has it, its comments taken out, and a figure by the
objects of its PDF, since matplotlib orders a figure's fonts by the paths of their files, so that the same figure
drawn from another checkout numbers them otherwise. `build/figures/SOURCES.json` lists the runs each one read.

Every one of them is the paper's, arXiv's v1 and v2 alike, but Table 18. Its ARC-AGI-2 and ARC-AGI-3 rows were summed
on 7 November 2026 from the rows those tracks had that day, the two floors and a first K and K0 played through
OpenRouter, before the rows Appendix K reports played again with every call pinned to Moonshot AI's own endpoint. The
script sums the rows the paper reports: \$1,549.09 and \$683.04 where the paper prints \$898.03 and \$698.62, and a
subtotal of \$13,042.22 for \$12,406.75. The first K and K0 are not in `runs/`.

What the code reads besides the runs' records:

- the GPT-6 Luna runs' later plays of the frozen tests, `frozen/<test>-K/play/` beside each first play, and the
  training records of DAgger's runs, `train/` and `rounds/` in `runs/arc1/QDE-qwen3.5-9b-ws0-u{1,2,4}/`, which are in
  `runs/` as [`runs/README.md`](../runs/README.md) lays them out;
- `data/inference-pricing.json`, the providers' prices of Table 17 and Figure 13, as retrieved on 7 November 2026;
- `data/experiment-costs.json`, what Table 18 sums that `runs/` does not hold, each part said there: the plays the
  measures leave out, the private set's calibration, the charges of DAgger's teacher labels, and the corrections and
  the price behind its training estimate;
- `data/outages.json`, the instances an outage played through, which every measure leaves out.

Not computed here:

- Table 13 (Appendix J), from the calibration reports of the private set, whose tasks must stay unseen for a verified
  run to mean anything;
- Table 14 (Appendix K), the first-sight answers, which are kept with the private set's calibration;
- Tables 2, 3, 5 and 8, which describe the benchmark, its configurations, what each harness is built to persist and
  the related work, and read no run.

## What is not here

- The code of Table 13 and the answers behind Table 14, above.
- The scripts the text cites for the numbers it quotes beside the figures and tables, `scripts/read_blocks.py`,
  `scripts/ablations_open.py` and the others it names, all but `scripts/forgetting.py`, which is here.
- The working notes, drafts and reviews behind the paper, which are not part of it.

## Citation

To cite the paper, use the BibTeX in the [README](../README.md#citation).
