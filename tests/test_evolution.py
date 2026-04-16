import argparse
import os
import time

import pandas as pd

from navidiv import (
    cluster_similarity_scorer,
    orginal_similarity_scorer,
)
from navidiv.scaffold import Scaffold_scorer
from navidiv.stringbased import Ngram_scorer

_DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "tmp", "results")


def run_orginal_scorer(df, steps=[], output_path=_DEFAULT_OUTPUT, reference_csv=None):
    """Test the original similarity scorer."""
    if reference_csv is None:
        raise ValueError("reference_csv must be provided for the Original scorer")
    df_original = pd.read_csv(reference_csv)
    smiles_list_to_compare_to = df_original["smiles"].tolist()
    orginal_score = orginal_similarity_scorer.OriginalSimScorer(
        output_path=output_path,
        threshold=0.3,
        smiles_list_to_compare_to=smiles_list_to_compare_to,
    )
    scores_list = []
    for step in steps:
        if hasattr(orginal_score, "_mol_smiles"):
            delattr(orginal_score, "_mol_smiles")
        df_copy = df[df["step"] == step]
        smiles_list = df_copy["SMILES"].tolist()
        scores = df_copy["Score"].tolist()

        start = time.time()
        scores_orginal = orginal_score.get_score(
            smiles_list=smiles_list,
            scores=scores,
            additional_columns_df={"step": step},
        )
        scores_list.append(scores_orginal)
        end = time.time()
        print(
            f"OriginalSimilarityScorer.get_score time: {end - start:.3f} seconds"
        )
        print("Original scores")
        print("scores", scores_orginal)
    df_scores = pd.DataFrame(scores_list)
    df_scores["step"] = steps
    df_scores.to_csv(
        os.path.join(output_path, "step_scores_original.csv"),
        index=False,
    )
    return scores_orginal

def run_scaffold_scorer(df, steps=[], output_path=_DEFAULT_OUTPUT):
    """Test the scaffold scorer."""
    scaffold_score = Scaffold_scorer.Scaffold_scorer(
        output_path=output_path,
        scaffold_type="csk_bm",
    )
    scores_list = []
    for step in steps:
        df_copy = df[df["step"] == step]
        smiles_list = df_copy["SMILES"].tolist()
        scores = df_copy["Score"].tolist()

        start = time.time()
        scores_scaffold = scaffold_score.get_score(
            smiles_list=smiles_list,
            scores=scores,
            additional_columns_df={"step": step},
        )
        scores_list.append(scores_scaffold)
        end = time.time()
        print(
            f"ScaffoldScorer.get_score time: {end - start:.3f} seconds"
        )
        print("Scaffold scores")
        print("scores", scores_scaffold)
    df_scores = pd.DataFrame(scores_list)
    df_scores["step"] = steps
    df_scores.to_csv(
        os.path.join(output_path, "step_scores_scaffolds.csv"),
        index=False,
    )
    scaffold_score.overrepresented_fragments.to_csv(
        os.path.join(output_path, "overrepresented_scaffolds.csv"),
        index=False,
    )
    return scores_scaffold


def run_cluster_scorer(df, steps=[], output_path=_DEFAULT_OUTPUT):
    """Test the cluster similarity scorer."""
    cluster_score = cluster_similarity_scorer.ClusterSimScorer(
        output_path=output_path,
        threshold=0.25,
    )

    scores_list = []
    for step in steps:
        if hasattr(cluster_score, "_mol_smiles"):
            delattr(cluster_score, "_mol_smiles")
        df_copy = df[df["step"] == step]
        smiles_list = df_copy["SMILES"].tolist()
        scores = df_copy["Score"].tolist()

        start = time.time()
        scores_ngram = cluster_score.get_score(
            smiles_list=smiles_list,
            scores=scores,
            additional_columns_df={"step": step},
        )
        scores_list.append(scores_ngram)
        end = time.time()
        print(
            f"NgramScorer.get_score time: {end - start:.3f} seconds"
        )
        print("clusters scores")
        print("scores", scores_ngram)
    df_scores = pd.DataFrame(scores_list)
    df_scores["step"] = steps
    df_scores.to_csv(
        os.path.join(output_path, "step_scores_clusters.csv"),
        index=False,
    )
    cluster_score.overrepresented_fragments.to_csv(
        os.path.join(output_path, "overrepresented_clusters.csv"),
        index=False,
    )

    return scores_ngram


def run_ngram_scorer(df, steps=[], output_path=_DEFAULT_OUTPUT):
    """Test the ngram scorer."""
    ngram_score = Ngram_scorer.NgramScorer(
        ngram_size=10,
        output_path=output_path,
    )
    scores_list = []
    for step in steps:
        df_copy = df[df["step"] == step]
        smiles_list = df_copy["SMILES"].tolist()
        scores = df_copy["Score"].tolist()

        start = time.time()
        scores_ngram = ngram_score.get_score(
            smiles_list=smiles_list,
            scores=scores,
            additional_columns_df={"step": step},
        )
        scores_list.append(scores_ngram)
        end = time.time()
        print(
            f"NgramScorer.get_score time: {end - start:.3f} seconds"
        )
        print("Ngram scores")
        print("scores", scores_ngram)
    df_scores = pd.DataFrame(scores_list)
    df_scores["step"] = steps
    df_scores.to_csv(
        os.path.join(output_path, "step_scores_ngrams.csv"),
        index=False,
    )
    ngram_score.overrepresented_fragments.to_csv(
        os.path.join(output_path, "overrepresented_ngrams.csv"),
        index=False,
    )

    return scores_ngram


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test evolution of diversity scorers over RL steps."
    )
    parser.add_argument(
        "csv_file",
        help="Path to input CSV file with SMILES and step columns",
    )
    parser.add_argument(
        "--output_path",
        default=_DEFAULT_OUTPUT,
        help="Directory for output files (default: tests/tmp/results)",
    )
    parser.add_argument(
        "--reference_csv",
        default=None,
        help="Path to reference CSV for the Original scorer (requires 'smiles' column)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    df = pd.read_csv(args.csv_file)
    df = df.dropna(subset=["SMILES"])

    # run_ngram_scorer(df, steps=range(5, 1000, 50), output_path=args.output_path)
    run_cluster_scorer(df, steps=range(50, 1000, 50), output_path=args.output_path)
    # run_scaffold_scorer(df, steps=range(5, 1000, 50), output_path=args.output_path)
    if args.reference_csv:
        run_orginal_scorer(
            df,
            steps=range(5, 1000, 50),
            output_path=args.output_path,
            reference_csv=args.reference_csv,
        )
    print("done")
