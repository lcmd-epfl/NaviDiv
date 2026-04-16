import argparse
import os

import pandas as pd

from navidiv.utils import (
    add_mean_of_numeric_columns,
    groupby_results,
    initialize_scorer,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test groupby_results on a scorer output CSV."
    )
    parser.add_argument(
        "csv_file",
        help="Path to a scorer output CSV (e.g. Fragments_Match_with_score.csv)",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help=(
            "Directory for grouped output CSV "
            "(default: same directory as csv_file)"
        ),
    )
    parser.add_argument(
        "--name",
        default="Fragments_Match",
        help="Scorer name used for the output filename (default: Fragments_Match)",
    )
    args = parser.parse_args()

    csv_file = args.csv_file
    output_path = args.output_path or os.path.dirname(os.path.abspath(csv_file))
    name = args.name

    df = pd.read_csv(csv_file)
    print("columns", df.columns)

    grouped_by_df = groupby_results(df)
    grouped_by_df.to_csv(
        os.path.join(output_path, name, f"groupby_results_{name}.csv"),
        index=False,
    )
