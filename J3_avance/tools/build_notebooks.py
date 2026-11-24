"""Build the student and solution notebooks from the sources in notebooks/src/*.nbsrc.

Source format (one file per notebook):

    %% md
    Markdown text ...
    %% code
    python code ...
        # BEGIN SOLUTION
        solution lines (removed from the student version)
        # END SOLUTION

* Student version  -> notebooks/<name>.ipynb : every BEGIN/END SOLUTION block is replaced
  by "# TODO" (same indentation). A block may give its own placeholder with
  "# BEGIN SOLUTION <placeholder text>" (e.g. "# BEGIN SOLUTION result = ...").
  "# BEGIN SOLUTION !" removes the block silently (no TODO).
* "# BEGIN BUG" ... "# END BUG": lines kept ONLY in the student version (buggy code to fix);
  combine with a silent solution block holding the corrected code.
* Solution version -> notebooks/solutions/<name>_solution.ipynb : markers removed.
* "%% md solution" cells only appear in the solution version (answers to questions).

Usage:  python build_notebooks.py [--execute] [name ...]
        --execute runs the solutions with nbclient; names (prefixes, e.g. J2_07) restrict the build.
"""
import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent / "notebooks"
SRC = ROOT / "src"
BEGIN = re.compile(r"^(\s*)# BEGIN SOLUTION ?(.*)$")
END = re.compile(r"^\s*# END SOLUTION\s*$")
BUG_BEGIN = re.compile(r"^\s*# BEGIN BUG\s*$")
BUG_END = re.compile(r"^\s*# END BUG\s*$")


def parse(path):
    cells, kind, buf = [], None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("%% "):
            if kind:
                cells.append((kind, "\n".join(buf).strip("\n")))
            kind, buf = line[3:].strip(), []
        else:
            buf.append(line)
    if kind:
        cells.append((kind, "\n".join(buf).strip("\n")))
    return cells


def strip_solutions(code, keep):
    out, inside, in_bug = [], False, False
    for line in code.splitlines():
        if BUG_BEGIN.match(line):
            in_bug = True
            continue
        if BUG_END.match(line):
            in_bug = False
            continue
        if in_bug:
            if not keep:
                out.append(line)
            continue
        m = BEGIN.match(line)
        if m:
            inside = True
            if not keep and m.group(2).strip() != "!":
                placeholder = m.group(2).strip() or "..."
                out.append(f"{m.group(1)}# TODO")
                out.append(f"{m.group(1)}{placeholder}")
            continue
        if END.match(line):
            inside = False
            continue
        if keep or not inside:
            out.append(line)
    return "\n".join(out)


def build(path, solution):
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for kind, text in parse(path):
        if kind == "md":
            nb.cells.append(nbformat.v4.new_markdown_cell(text))
        elif kind == "md solution":
            if solution:
                nb.cells.append(nbformat.v4.new_markdown_cell("**Solution.** " + text))
        elif kind == "code":
            nb.cells.append(nbformat.v4.new_code_cell(strip_solutions(text, solution)))
        else:
            raise ValueError(f"{path.name}: unknown cell kind {kind!r}")
    return nb


def main():
    execute = "--execute" in sys.argv
    wanted = [a for a in sys.argv[1:] if not a.startswith("--")]
    (ROOT / "solutions").mkdir(exist_ok=True)
    for src in sorted(SRC.glob("*.nbsrc")):
        if wanted and not any(src.stem.startswith(w) for w in wanted):
            continue
        student = build(src, solution=False)
        sol = build(src, solution=True)
        nbformat.write(student, ROOT / f"{src.stem}.ipynb")
        sol_path = ROOT / "solutions" / f"{src.stem}_solution.ipynb"
        nbformat.write(sol, sol_path)
        print("built", src.stem)
        if execute:
            from nbclient import NotebookClient
            client = NotebookClient(sol, timeout=1800, kernel_name="python3",
                                    resources={"metadata": {"path": str(sol_path.parent)}})
            client.execute()
            nbformat.write(sol, sol_path)  # keep outputs in the solution notebooks
            print("  executed OK")


if __name__ == "__main__":
    main()
