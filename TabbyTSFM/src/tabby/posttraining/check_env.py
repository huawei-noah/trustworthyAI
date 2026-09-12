"""Resolve GIFT_EVAL exactly the way the data loader does, and report it.

    python -m tabby.posttraining.check_env

Both this checker and the data loader search upward from the current working
directory. Running a release recipe from the repository root therefore finds
the root `.env`; an explicitly exported environment variable still wins.

Exit code 0 = resolved and the directory exists; 1 = not usable.
"""
import os
import sys

from dotenv import find_dotenv, load_dotenv


def resolve(var: str = "GIFT_EVAL"):
    """Return (root, source) as the loader would see them; root is None if unset."""
    before = os.environ.get(var)
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    after = os.environ.get(var)
    if before:
        return before, "environment"
    if after:
        return after, ".env"
    return None, None


def main() -> int:
    root, source = resolve()
    if not root:
        print("[ABORT] GIFT_EVAL is not set, and no .env on the search path defines it.\n"
              "        Either:  export GIFT_EVAL=/path/to/gift-eval/data\n"
              "        or:      cp .env.example .env   and fill it in", file=sys.stderr)
        return 1
    if not os.path.isdir(root):
        print(f"[ABORT] GIFT_EVAL={root} (from {source}) is not a directory", file=sys.stderr)
        return 1
    n = len([e for e in os.scandir(root) if e.is_dir()])
    print(f"[env] GIFT_EVAL={root}  (from {source}, {n} dataset directories)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
