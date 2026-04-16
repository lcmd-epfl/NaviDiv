import time  

import pandas as pd
from pathlib import Path
from navidiv import (
    cluster_similarity_scorer,
)
from navidiv.utils import get_smiles_column


def run_cluster_scorer(df, output_path):
    """Test the cluster similarity scorer."""
    cluster_score = cluster_similarity_scorer.ClusterSimScorer(
        output_path=output_path,
        threshold=0.25,
    )
    start = time.time()
    scores_cluster = cluster_score.aggregate_df(df)
    cluster_score._fragments_df.to_csv(
        f"{cluster_score._output_path}/grouped_by_aggregate_clusters_with_score.csv",
        index=False,
    )
    end = time.time()
    print(
        f"ClusterSimilarityScorer.get_score time: {end - start:.3f} seconds"
    )
    print("Cluster scores")
    print("scores", scores_cluster)
    return scores_cluster


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test cluster scorer aggregate on a groupby CSV."
    )
    parser.add_argument(
        "csv_file",
        help="Path to a groupby clusters CSV (e.g. groupby_results_clusters.csv)",
    )
    args = parser.parse_args()
    csv_file = args.csv_file
    output_path = Path(csv_file).parent
    df = pd.read_csv(csv_file).sample(frac=1).reset_index(drop=True)
    # df = df[df["step"] == 1000]
    print("columns", df.columns)
    smiles_col = get_smiles_column(df)
    df = df.dropna(subset=[smiles_col])

    run_cluster_scorer(df, output_path)

    print("done")
