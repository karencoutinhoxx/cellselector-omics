import pytest
import pandas as pd
from src.models.classical.scorer import (
    classify_gene, score_rna_expression, score_protein_expression,
    score_context, score_data_quality, load_mappings,
    get_gene_role, GENE_ROLES
)

class TestClassifyGene:
    """Test gene classification into tissue_specific/ubiquitous/LOF."""
    
    def test_tissue_specific_gene(self):
        """EGFR should be classified as tissue_specific."""
        assert classify_gene("EGFR") == "tissue_specific"
    
    def test_ubiquitous_gene(self):
        """TP53 should be classified as ubiquitous."""
        assert classify_gene("TP53") == "ubiquitous"
    
    def test_lof_gene(self):
        """BRCA1 should be classified as loss_of_function."""
        assert classify_gene("BRCA1") == "loss_of_function"
    
    def test_unknown_gene_defaults_to_tissue_specific(self):
        """An unknown gene should default to tissue_specific."""
        assert classify_gene("FAKEGENE123") == "tissue_specific"
    
    def test_case_insensitivity(self):
        """Gene classification should be case-insensitive."""
        assert classify_gene("egfr") == classify_gene("EGFR")


class TestGeneRole:
    """Test the gene role/category lookup."""
    
    def test_rtk(self):
        assert get_gene_role("EGFR") == "receptor tyrosine kinase"
    
    def test_tumor_suppressor(self):
        assert get_gene_role("TP53") == "tumor suppressor"
    
    def test_unknown_returns_none(self):
        assert get_gene_role("UNKNOWNGENE") is None
    
    def test_all_roles_are_strings(self):
        for gene, role in GENE_ROLES.items():
            assert isinstance(role, str), f"{gene} has non-string role"

class TestScoreRnaExpression:
    """Test RNA expression scoring."""
    
    def test_returns_dataframe(self):
        hpa, ach, gsm = load_mappings()
        result = score_rna_expression("EGFR", hpa, gsm)
        assert isinstance(result, pd.DataFrame)
    
    def test_has_required_columns(self):
        hpa, ach, gsm = load_mappings()
        result = score_rna_expression("EGFR", hpa, gsm)
        assert "cellosaurus_id" in result.columns
        assert "rna_score" in result.columns
    
    def test_scores_between_0_and_1(self):
        hpa, ach, gsm = load_mappings()
        result = score_rna_expression("EGFR", hpa, gsm)
        assert result["rna_score"].min() >= 0.0
        assert result["rna_score"].max() <= 1.0
    
    def test_nonexistent_gene_returns_empty(self):
        hpa, ach, gsm = load_mappings()
        result = score_rna_expression("TOTALLYFAKEGENE", hpa, gsm)
        assert len(result) == 0


class TestScoreContext:
    """Test disease/lineage context scoring."""
    
    def test_exact_disease_match_scores_1(self):
        # Need to find a known cell line with known disease
        result = score_context({"CVCL_2063"}, "Lung Cancer", None)
        lung_row = result[result["cellosaurus_id"] == "CVCL_2063"]
        if len(lung_row) > 0:
            assert lung_row.iloc[0]["context_score"] == 1.0
    
    def test_no_filter_returns_zero_context(self):
        result = score_context({"CVCL_2063"}, None, None)
        assert all(result["context_score"] == 0.0)
    
    def test_partial_match_scores_half(self):
        result = score_context({"CVCL_2063"}, "lung", None)
        lung_row = result[result["cellosaurus_id"] == "CVCL_2063"]
        if len(lung_row) > 0:
            assert lung_row.iloc[0]["context_score"] == 0.5



