# The paper's generated figures and tables, computed again from the runs in ../runs/ and written under $(OUT) in the
# paper's own layout (figures/, sections/ and sections/tables/), then compared with the files the paper sets. uv runs
# every script in the repository's own environment, with the benchmark installed from this checkout.
OUT ?= build
RUN = uv run --extra plots python

.PHONY: all data figures appendix tracks compare

all: data figures appendix compare

# the pinned Re-ARC and ARC-AGI data that Figure 6 reads
data:
	uv run continual-arc data fetch --datasets arc1

# every generated figure and table of the paper (scripts/make_figures.py), Figure 7, Figure 9 and Table 18
figures:
	$(RUN) scripts/make_figures.py --out $(OUT)
	$(RUN) scripts/weight_updates.py --out $(OUT)
	$(RUN) scripts/forgetting.py --out $(OUT)
	$(RUN) scripts/experiment_costs.py --out $(OUT)

# Appendix C's generated part and Table 4, rendered from this repository's releases v0.3.0 and v0.5.0
appendix:
	uv sync --all-packages --extra plots
	uv run --no-sync python scripts/make_prompts_appendix.py --out $(OUT)

# the numbers of Tables 15 and 16, which the paper sets by hand
tracks:
	$(RUN) scripts/tracks_tables.py

compare:
	uv run python scripts/compare.py --out $(OUT)
