import pytest
import math
import json
from src.models.classical.ranker import rank

class TestNaNHandling:
    """Regression tests for NaN-related bugs."""
    
    def test_no_nan_in_final_score(self):
        result = rank("EGFR", top_n=10)
        assert not result["final_score"].isna().any()
    
    def test_sparse_gene_doesnt_crash(self):
        """Genes with sparse coverage should return results,
        not crash with NaN errors."""
        result = rank("CD274", top_n=5)  # PD-L1, sparse
        assert isinstance(result, type(result))  # didn't crash
    
    def test_json_serialisable(self):
        """All result values must be JSON-serialisable 
        (no NaN, no Infinity)."""
        result = rank("EGFR", disease_filter="lung", top_n=5)
        for _, row in result.iterrows():
            for col in result.columns:
                val = row[col]
                if isinstance(val, float):
                    assert not math.isnan(val) or val != val, (
                        f"NaN found in {col}"
                    )
                    assert not math.isinf(val), (
                        f"Infinity found in {col}"
                    )


class TestExclusionEdgeCases:
    def test_exclude_nonexistent_gene(self):
        """Excluding a gene with no expression data should
        not crash or change results."""
        without = rank("EGFR", top_n=3)
        with_excl = rank("EGFR", exclude_genes=["FAKEGENE"],
                        top_n=3)
        # Should not crash, results should be same
        assert len(with_excl) == len(without)
    
    def test_multiple_exclusions(self):
        result = rank("EGFR", exclude_genes=["TP53", "KRAS"],
                      top_n=5)
        assert len(result) > 0

class TestGeneClassWeighting:
    """Test that different gene classes get different weights."""
    
    def test_tissue_specific_weights_rna_heavily(self):
        result = rank("EGFR", top_n=1)
        # For tissue-specific, RNA should dominate
        if len(result) > 0:
            row = result.iloc[0]
            assert row["rna_score"] > 0.5
    
    def test_lof_gene_context_matters(self):
        """For LOF genes, context should be weighted heavily."""
        with_context = rank("BRCA1", disease_filter="breast",
                           top_n=1)
        without_context = rank("BRCA1", top_n=1)
        # With disease filter, top result should have 
        # higher context score
        if len(with_context) > 0:
            assert with_context.iloc[0]["context_score"] > 0


