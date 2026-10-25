# Working in training/

- Read [README.md](README.md) first, then [docs/how-a-run-works.md](docs/how-a-run-works.md) and
  [docs/code.md](docs/code.md). Every script's header comment or module docstring explains it in full.
- The benchmark is the repository root (`../src/continual_arc`), the baselines are `../baselines`; scripts find both
  from where they sit. Node folders are parameters: `PRIME_RL`, `VLLM_VENV`, `RUNS`, `BASELINES`.
- `ask_sd`, the trainer package, is a dependency that is not in this repository: tests that need it skip without it.
- Run the tests from this folder:
  `PYTHONPATH=.:../src:../baselines/src python -m pytest`, then `bash test_frozen.sh` and
  `bash test_paper_backup.sh`. None needs a GPU or a network.
- Lint with `ruff check` (this folder's [ruff.toml](ruff.toml): ruff's default rules, line length 120).
- Nothing here rents or sets up a machine, and no script holds an address: a node is always an argument or a
  variable.
