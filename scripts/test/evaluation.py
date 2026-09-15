import pytest
from src.models.classical.ranker import rank
from src.models.classical.scorer import classify_gene
from src.models.classical.weights_learned import VALIDATION_SET

class TestValidationSet:
    """Test that the system handles all 25 validation genes."""
    
    def test_all_validation_genes_return_results(self):
        for gene in VALIDATION_SET:
            result = rank(gene, top_n=5)
            assert len(result) > 0, f"No results for {gene}"
    
    def test_all_validation_genes_classifiable(self):
        for gene in VALIDATION_SET:
            cls = classify_gene(gene)
            assert cls in ["tissue_specific", "ubiquitous",
                          "loss_of_function"]
    
    def test_known_associations_present_in_data(self):
        """Check that at least some known cell lines appear
        somewhere in the ranked results (not necessarily top-N)."""
        found = 0
        total = 0
        for gene, known_lines in VALIDATION_SET.items():
            result = rank(gene, top_n=None)
            cvcls = set(result["cellosaurus_id"])
            names = set(result["official_name"].dropna())
            for line in known_lines:
                total += 1
                if line in names or line in cvcls:
                    found += 1
        # At least 80% of known associations should be findable
        assert found / total > 0.8, (
            f"Only {found}/{total} known associations found"
        )

