import json

import pandas as pd

from config import RESULTS_DIR


def find_problems() -> list[str]:
    problems = []
    for path in RESULTS_DIR.rglob("*"):
        if not path.is_file():
            continue
        name = path.relative_to(RESULTS_DIR).as_posix()
        if "_FAST" in path.name or "_DRYRUN" in path.name:
            problems.append(f"{name}: FAST/DRYRUN file name")
        elif path.suffix == ".csv":
            table = pd.read_csv(path)
            if "mode" not in table:
                problems.append(f"{name}: no mode column")
            elif (table["mode"] != "FULL").any():
                problems.append(f"{name}: rows with mode != FULL")
        elif path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("mode") != "FULL":
                problems.append(f"{name}: mode != FULL")
    return problems


if __name__ == "__main__":
    problems = find_problems()
    for problem in problems[:20]:
        print(problem)
    print(f"check_results: {'FAIL' if problems else 'PASS'} ({len(problems)} problems)")
    raise SystemExit(1 if problems else 0)
