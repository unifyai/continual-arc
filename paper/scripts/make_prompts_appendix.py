"""sections/A2-generated.tex, Appendix C's generated part with Table 4, rendered from the benchmark's repository: the
grid protocol's rules, which every learner on the first two tracks reads once per session, and an instance's first
message, both verbatim, and a table of the files that hold every other text a learner reads, so that the paper prints
what a reader needs to judge the protocol and says where the code keeps the rest. Run it from paper/ in a clone of the
repository that holds its releases' tags, its workspace synced whole:

    uv sync --all-packages && uv run --no-sync python scripts/make_prompts_appendix.py [--out DIR]

The packages of each release, the benchmark at the repository's root, the learners in baselines/ and DAgger's
training in training/, are taken from its tag into a folder of their own, and this script renders that release's
texts there in a process of its own (--release), so that two releases' modules never meet. The script stops if the two
releases give a printed text differently, or if a file the table names is missing from a release whose rows read it.
It writes DIR/sections/A2-generated.tex, DIR being build/ unless given, which sections/A2-prompts.tex inputs.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # paper/
WIDTH = 94  # characters of scriptsize Courier that fit the text width: 94 x 4.2pt of 397.5pt

# The release each row reads. GPT-6 Luna's grid was launched from commits that v0.3.0 carries unchanged in what a
# learner is sent. The rows on the open models were launched from the commit that rewrote the expert guidance and gave
# the teacher a page of its own, and the light rows of the second and third tracks from v0.4.0, the first release to
# carry both; v0.5.0, the release the paper cites for those rows, carries every text they read unchanged.
LUNA, REST = "v0.3.0", "v0.5.0"
RELEASES = (LUNA, REST)

TASK = "task-3f2a9c1e"   # a task's name as a learner sees it (continual_arc.aliases)


# -- the printed texts, each rendered by the code of the release that this process imported (--release) --------------

def protocol():
    from continual_arc_baselines.systems import protocol
    return protocol


def first_message() -> str:
    """The message that shows an instance of ``TASK`` on the first track after the rules, a 2 by 2 test input."""
    from continual_arc.configs import get_config
    from continual_arc.env import LearnerObservation
    from continual_arc.grids import Grid
    env = get_config("arc1-small").env
    seen = LearnerObservation(step=0, task_id=TASK, input=Grid([[2, 0], [0, 2]]), attempts_used=0, demo_requests_used=0,
                              attempt_cap=env.attempt_cap, demo_cap=env.demo_cap)
    return protocol().render_observation(seen, pending=[], demo_cost=1, max_cost=16, first_in_session=False,
                                         finish=True)


TEXTS = {
    "rules": lambda: protocol().RULES_FINISH.replace("{demo_cost}", "1"),
    "first message": first_message,
}
PRINTED = [("Grid protocol: rules shown once per session (demo cost 1)", "rules"),
           ("Grid protocol: an instance's first message, after the rules (an example)", "first message")]

# Every other text a learner reads, by the file that holds it, with the releases whose rows read that file
BASELINES, TRAINING = "baselines/src/continual_arc_baselines/", "training/src/continual_arc_training/"
FILES = [
    (BASELINES + "systems/protocol.py",
     "the grid protocol's other messages: the replies to an answer or a request, the verdicts, the re-prompts and "
     "the budget notice; the ask after every verdict of methods QA, GA and \\mbox{H-A}; the expert guidance of methods QE, GE, \\mbox{H-E}, "
     "\\mbox{H-E\\textsubscript{0}} and QETE and its asks", RELEASES),
    (BASELINES + "systems/prime_agent.py", "prime-agent's one addition to the rules, its fact about its store",
     RELEASES),
    (BASELINES + "systems/bare.py",
     "the demo-cache floor's cached pairs, and the program library's addition to the rules, its kept program, its "
     "checks and its re-prompt", RELEASES),
    (BASELINES + "systems/arc3_protocol.py", "the third track's protocol: its rules, messages, feedback and re-prompt",
     (REST,)),
    (TRAINING + "references.py", "c1, the page of playing rules that DAgger's teacher reads", (REST,)),
    (TRAINING + "oracle.py", "the rules and the note with which DAgger's teacher labels the student's calls", (REST,)),
]


def wrap(text: str) -> str:
    """Wrap lines longer than the page at spaces, keeping each line's indentation, its continuations indented two more;
    the words are untouched, so the listing still reproduces the text and only its line breaks are ours. A word longer
    than the page stays whole on a line of its own."""
    out = []
    for line in text.split("\n"):
        body = line.lstrip(" ")
        indent = line[: len(line) - len(body)]
        if len(line) <= WIDTH:
            out.append(line)
        else:
            out.extend(textwrap.wrap(body, WIDTH, initial_indent=indent, subsequent_indent=indent + "  ",
                                     break_long_words=False, break_on_hyphens=False))
    return "\n".join(out)


def verbatim(title: str, text: str) -> str:
    text = wrap(text.rstrip("\n"))
    return f"\\paragraph{{{title}}}\n\\begin{{scriptsize}}\\begin{{verbatim}}\n{text}\n\\end{{verbatim}}\\end{{scriptsize}}\n"


def files_table() -> str:
    """Table 4: each file that holds a text the appendix does not print, relative to its package, the learners' files
    first, and what it holds."""
    def tex(path: str) -> str:
        return path.replace("_", "\\_")
    if FILES != sorted(FILES, key=lambda f: not f[0].startswith(BASELINES)):
        sys.exit("FILES lists the learners' files first, as the table's caption counts them")
    rows = "\n".join("\\texttt{%s} & %s \\\\" % (tex(path.removeprefix(BASELINES).removeprefix(TRAINING)), what)
                     for path, what, _ in FILES)
    words = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
    caption = ("Where the benchmark's code keeps every other text a learner is shown: the first %s files under "
               "\\texttt{%s} and the rest under \\texttt{%s}, at tag \\texttt{%s} for GPT-6 Luna's methods and "
               "\\texttt{%s} for the rest, which alone has DAgger's teacher and the third track."
               % (words[sum(f[0].startswith(BASELINES) for f in FILES) - 1], tex(BASELINES), tex(TRAINING), LUNA, REST))
    return ("\\begin{table}[htbp]\n\\centering\n\\footnotesize\n\\begin{tabular}{@{}lp{0.6\\linewidth}@{}}\n"
            "\\toprule\nfile & the texts it holds \\\\\n\\midrule\n" + rows + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{" + caption + "}\n\\label{tab:prompts}\n\\end{table}\n")


def release_tree(clone: Path, tag: str, root: Path) -> Path:
    """The packages of release ``tag``, the benchmark, the learners and DAgger's training, from the clone into
    ``root``."""
    archive = subprocess.run(["git", "-C", str(clone), "archive", tag, "src", "baselines/src", "training/src"],
                             stdout=subprocess.PIPE, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(root / tag, filter="data")
    return root / tag


def rendered(tree: Path, names: list[str]) -> dict[str, str]:
    """``names``' texts from the release in ``tree``, rendered by this script in a process that imports that release
    alone (-I: neither this folder nor PYTHONPATH on its path)."""
    out = subprocess.run([sys.executable, "-I", __file__, "--release", str(tree), *names], stdout=subprocess.PIPE,
                         text=True, check=True).stdout
    return json.loads(out)


def main(clone: Path, out: Path) -> None:
    commit = {tag: subprocess.run(["git", "-C", str(clone), "rev-parse", "--short=7", f"{tag}^{{commit}}"],
                                  stdout=subprocess.PIPE, text=True, check=True).stdout.strip() for tag in RELEASES}
    for path, _, tags in FILES:
        for tag in tags:
            if subprocess.run(["git", "-C", str(clone), "cat-file", "-e", f"{tag}:{path}"]).returncode:
                sys.exit(f"{tag} has no {path}, which Table 4 sends its rows' readers to")
    with tempfile.TemporaryDirectory() as root:
        texts = {tag: rendered(release_tree(clone, tag, Path(root)), list(TEXTS)) for tag in RELEASES}
    for name in TEXTS:
        if texts[LUNA][name] != texts[REST][name]:
            sys.exit(f"{name!r} differs between {LUNA} and {REST}, whose rows both read it")
    releases = " and ".join(f"{tag} ({commit[tag]})" for tag in RELEASES)
    parts = [f"% Generated by scripts/make_prompts_appendix.py from the benchmark's repository at {releases}. "
             "Do not edit.\n",
             "Every learner on the first two tracks is shown the grid protocol's rules below once per session, and "
             "each instance opens with a message like the one after them. Both are reproduced from the benchmark's "
             f"repository, the same at tag \\texttt{{{LUNA}}}\\ifdefined\\arxivpreprint, commit \\texttt{{{commit[LUNA]}}}\\fi, which GPT-6 Luna's "
             f"methods read, and at tag \\texttt{{{REST}}}\\ifdefined\\arxivpreprint, commit \\texttt{{{commit[REST]}}}\\fi, which the rest read, with "
             "lines longer than the page wrapped at spaces; placeholders in braces are filled at run time with the "
             "values named in Section~\\ref{sec:benchmark}. Every other text a learner is shown, from the replies to "
             "its actions to what each method adds to the rules and the third track's protocol, is in the code"
             "\\ifdefined\\arxivpreprint\\ released with the paper\\else\\ in the supplementary material\\fi, in the "
             "files of Table~\\ref{tab:prompts}.\n"]
    parts += [verbatim(title, texts[REST][name]) for title, name in PRINTED]
    parts.append(files_table())
    path = out / "sections" / "A2-generated.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts))
    print(f"wrote {path} from {releases}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--release"]:
        sys.path[:0] = [str(Path(sys.argv[2]) / folder) for folder in ("src", "baselines/src", "training/src")]
        print(json.dumps({name: TEXTS[name]() for name in sys.argv[3:]}))
    else:
        ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        ap.add_argument("--out", type=Path, default=ROOT / "build", help="write sections/ here (default build/)")
        main(ROOT.parent, ap.parse_args().out.resolve())
