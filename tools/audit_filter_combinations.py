from __future__ import annotations

import argparse
from pathlib import Path

from cinelingus.filter_lab.combination import compile_compatibility_matrix
from cinelingus.util import write_json
from cinelingus.validation import validate_artifact


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile and validate every ordered Cinelingus filter combination."
    )
    parser.add_argument(
        "output_path",
        type=Path,
        nargs="?",
        default=Path("output/filter_combination_compatibility_matrix.json"),
    )
    args = parser.parse_args()
    output_path = args.output_path.expanduser().resolve()
    matrix = compile_compatibility_matrix()
    write_json(output_path, matrix)
    validate_artifact("filter_combination_compatibility_matrix", output_path, Path.cwd() / "schemas")
    print(
        f"PASS {matrix['ordered_pair_count']} ordered pairs; "
        f"{len(matrix['executable_pair_ids'])} certified executable; {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
