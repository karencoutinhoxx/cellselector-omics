import pytest
import pandas as pd
from src.models.classical.ranker import rank, explain

class TestRank:
    """Test the main ranking function."""
    
    def test_returns_dataframe(self):
        result = rank("EGFR", top_n=5)
        assert isinstance(result, pd.DataFrame)
    
    def test_respects_top_n(self):
        result = rank("EGFR", top_n=3)
        assert len(result) <= 3
    
    def test_results_sorted_descending(self):
        result = rank("EGFR", top_n=10)
        scores = result["final_score"].tolist()
        assert scores == sorted(scores, reverse=True)
    
    def test_final_score_between_0_and_1(self):
        result = rank("EGFR", top_n=10)
        assert result["final_score"].min() >= 0.0
        assert result["final_score"].max() <= 1.0
    
    def test_has_required_columns(self):
        result = rank("EGFR", top_n=5)
        required = ["cellosaurus_id", "final_score", "rna_score",
                     "protein_score", "quality_score", "context_score"]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"
    
    def test_disease_filter_only_returns_matching(self):
        result = rank("EGFR", disease_filter="lung", top_n=10)
        if len(result) > 0:
            # All results should have context_score > 0
            assert all(result["context_score"] > 0)
    
    def test_nonexistent_gene_returns_empty(self):
        result = rank("TOTALLYFAKEGENE999", top_n=5)
        assert len(result) == 0
    
    def test_exclude_genes_reduces_scores(self):
        without_exclude = rank("EGFR", disease_filter="lung", top_n=1)
        with_exclude = rank("EGFR", disease_filter="lung",
                           exclude_genes=["TP53"], top_n=1)
        if len(without_exclude) > 0 and len(with_exclude) > 0:
            # Exclusion should reduce or maintain score, not increase
            assert (with_exclude.iloc[0]["final_score"] <= 
                    without_exclude.iloc[0]["final_score"])
    
    def test_same_gene_search_and_exclude_prevented(self):
        """Searching for a gene and excluding it should fail."""
        # This should be caught at the API level, but test the logic
        result = rank("EGFR", exclude_genes=["EGFR"], top_n=5)
        # Either returns empty or the API should reject this

    class TestExplain:
    """Test the explanation generator."""
    
    def test_returns_string(self):
        result = rank("EGFR", disease_filter="lung", top_n=1)
        if len(result) > 0:
            explanation = explain(result.iloc[0])
            assert isinstance(explanation, str)
            assert len(explanation) > 0
    
    def test_includes_cell_line_name(self):
        result = rank("EGFR", disease_filter="lung", top_n=1)
        if len(result) > 0:
            explanation = explain(result.iloc[0])
            name = result.iloc[0]["official_name"]
            assert name in explanation


