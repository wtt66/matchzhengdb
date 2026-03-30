from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    ROOT / "第一部分" / "第一部分分析.py",
    ROOT / "第二部分" / "第二部分分析.py",
    ROOT / "第三部分" / "第三部分分析.py",
    ROOT / "第四部分" / "第四部分分析.py",
    ROOT / "第五部分" / "第五部分分析.py",
    ROOT / "第六部分" / "第六部分分析.py",
    ROOT / "generate_overall_report.py",
    ROOT / "generate_extended_analysis_report.py",
]


def main() -> None:
    for script in SCRIPTS:
        print(f"Running {script.name} ...")
        runpy.run_path(str(script), run_name="__main__")
    print("All questionnaire analyses finished.")


if __name__ == "__main__":
    main()
