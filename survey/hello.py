"""Survey Module - SafePath.

Responsible for human safety rating collection, survey analysis, and
ground truth targets for ANFIS/GA optimization. (Owner C)
"""

from pathlib import Path
import csv


def hello_survey() -> str:
    """Print status and read sample survey ratings."""
    module_name = "Survey Module (Owner C)"
    sample_file = Path(__file__).resolve().parent.parent / "data" / "sample" / "survey_ratings.csv"
    count = 0
    if sample_file.exists():
        with open(sample_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            count = sum(1 for _ in reader)
    return f"[{module_name}] Ready. Found {count} survey ratings in {sample_file.name}."


def main():
    print(hello_survey())


if __name__ == "__main__":
    main()

