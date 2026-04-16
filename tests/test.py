import argparse
import os
import time

import pandas as pd

from navidiv.fragment import (
    fg_scorer,
    fragment_scorer,
    fragment_scorer_matching,
    ring_scorer,
)
from navidiv.scaffold import Scaffold_GNN, Scaffold_scorer
from navidiv.simlarity import (
    cluster_similarity_scorer,
    orginal_similarity_scorer,
)
from navidiv.stringbased import Ngram_scorer
from navidiv.utils import get_smiles_column

_DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "tmp", "resultstests")


def test_scaffold_gnn_scorer(smiles_list, scores, output_path=_DEFAULT_OUTPUT):
    """Test the Scaffold GNN scorer."""
    scaffold_gnn_score = Scaffold_GNN.ScaffoldGNNScorer(
        output_path=output_path,
    )
    scaffold_gnn_score._min_count_fragments = 0
    start = time.time()
    scores_scaffold_gnn = scaffold_gnn_score.get_score(
        smiles_list=smiles_list,
        scores=scores,
    )
    end = time.time()
    print(
        f"ScaffoldGNNScorer.get_score time: {end - start:.3f} seconds"
    )
    print("Scaffold GNN scores")
    print("scores", scores_scaffold_gnn)
    return scores_scaffold_gnn


def test_fragment_match_scorer(smiles_list, scores, output_path=_DEFAULT_OUTPUT):
    """Test the fragment match scorer."""
    frag_match_score = fragment_scorer_matching.FragmentMatchScorer(
        output_path=output_path,
        min_count_fragments=1,
    )
    start = time.time()
    scores_frag_match = frag_match_score.get_score(
        smiles_list=smiles_list,
        scores=scores,
    )
    end = time.time()
    print(
        f"FragmentMatchScorer.get_score time: {end - start:.3f} seconds"
    )
    print("Fragment Match scores")
    print("scores", scores_frag_match)
    return scores_frag_match


def test_orginal_scorer(smiles_list, scores, output_path=_DEFAULT_OUTPUT, reference_csv=None):
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
    start = time.time()
    scores_orginal = orginal_score.get_score(
        smiles_list=smiles_list,
        scores=scores,
    )
    end = time.time()
    print(
        f"OriginalSimilarityScorer.get_score time: {end - start:.3f} seconds"
    )
    print("Original scores")
    print("scores", scores_orginal)
    return scores_orginal


def test_scaffold_scorer(smiles_list, scores, output_path=_DEFAULT_OUTPUT):
    """Test the scaffold scorer."""
    scaffold_score = Scaffold_scorer.Scaffold_scorer(
        output_path=output_path,
        scaffold_type="csk_bm",
    )
    start = time.time()
    scores_scaffold = scaffold_score.get_score(
        smiles_list=smiles_list,
        scores=scores,
    )
    end = time.time()
    print(
        f"ScaffoldScorer.get_score time: {end - start:.3f} seconds"
    )
    print("Scaffold scores")
    print("scores", scores_scaffold)
    return scores_scaffold


def test_cluster_scorer(smiles_list, scores, output_path=_DEFAULT_OUTPUT):
    """Test the cluster similarity scorer."""
    cluster_score = cluster_similarity_scorer.ClusterSimScorer(
        output_path=output_path,
        threshold=0.25,
    )
    start = time.time()
    scores_cluster = cluster_score.get_score(
        smiles_list=smiles_list,
        scores=scores,
    )
    end = time.time()
    print(
        f"ClusterSimilarityScorer.get_score time: {end - start:.3f} seconds"
    )
    print("Cluster scores")
    print("scores", scores_cluster)
    return scores_cluster


def test_ring_scorer(smiles_list, scores, output_path=_DEFAULT_OUTPUT):
    """Test the ring scorer."""
    ring_score = ring_scorer.RingScorer(
        output_path=output_path,
    )
    start = time.time()
    scores_ring = ring_score.get_score(
        smiles_list=smiles_list,
        scores=scores,
    )
    end = time.time()
    print(
        f"RingScorer.get_score time: {end - start:.3f} seconds"
    )
    print("Ring scores")
    print("scores", scores_ring)
    return scores_ring


def test_fragment_scorer(smiles_list, scores, output_path=_DEFAULT_OUTPUT):
    """Test the fragment scorer."""
    frag_score = fragment_scorer.FragmentScorer(
        output_path=output_path,
    )
    frag_score.update_transformation_mode("basic_framework")
    start = time.time()
    scores = frag_score.get_score(
        smiles_list=smiles_list,
        scores=scores,
    )

    end = time.time()
    print(
        f"FragmentScorer.get_score time: {end - start:.3f} seconds"
    )
    print("Fragment scores")
    scores.pop("Unique Fragments")
    print("scores", scores)
    return scores


def test_ngram_scorer(smiles_list, scores, output_path=_DEFAULT_OUTPUT):
    """Test the ngram scorer."""
    ngram_score = Ngram_scorer.NgramScorer(
        ngram_size=10,
        output_path=output_path,
    )
    start = time.time()
    scores_ngram = ngram_score.get_score(
        smiles_list=smiles_list,
        scores=scores,
    )
    end = time.time()
    print(
        f"NgramScorer.get_score time: {end - start:.3f} seconds"
    )
    print("Ngram scores")
    print("scores", scores_ngram)
    return scores_ngram


def test_fg_scorer(smiles_list, scores, output_path=_DEFAULT_OUTPUT):
    """Test the FG scorer."""
    fg_score = fg_scorer.FGScorer(
        output_path=output_path,
    )
    start = time.time()
    scores_fg = fg_score.get_score(
        smiles_list=smiles_list,
        scores=scores,
    )
    end = time.time()
    print(
        f"FGScorer.get_score time: {end - start:.3f} seconds"
    )
    print("FG scores")
    return scores_fg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run individual scorer tests on a SMILES CSV."
    )
    parser.add_argument(
        "csv_file",
        help="Path to input CSV file with SMILES, Score, and step columns",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=920,
        help="Which RL step to filter on (default: 920)",
    )
    parser.add_argument(
        "--output_path",
        default=_DEFAULT_OUTPUT,
        help="Directory for output files (default: tests/tmp/resultstests)",
    )
    parser.add_argument(
        "--reference_csv",
        default=None,
        help="Path to reference CSV for the Original scorer (requires 'smiles' column)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    df = pd.read_csv(args.csv_file).sample(frac=1).reset_index(drop=True)
    df = df[df["step"] == args.step]
    print("columns", df.columns)
    smiles_col = get_smiles_column(df)
    df = df.dropna(subset=[smiles_col])
    smiles_list = df[smiles_col].tolist()
    scores = df["Score"].tolist()
    print("will process", len(smiles_list), "smiles")
    print("will process", len(scores), "scores")
    # test_fragment_match_scorer(smiles_list, scores, args.output_path)
    # test_scaffold_gnn_scorer(smiles_list, scores, args.output_path)
    test_fg_scorer(smiles_list, scores, args.output_path)
    # test_ngram_scorer(smiles_list, scores, args.output_path)
    test_ring_scorer(smiles_list, scores, args.output_path)
    # test_cluster_scorer(smiles_list, scores, args.output_path)
    # test_scaffold_scorer(smiles_list, scores, args.output_path)
    test_fragment_scorer(smiles_list, scores, args.output_path)
    print("done")
