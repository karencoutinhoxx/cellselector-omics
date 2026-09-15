import pytest
import pandas as pd
from src.models.classical.similarity import find_alternatives
from src.models.classical.ranker import rank
from config.config import MASTER_MERGED

class TestFindAlternatives:
    """Test similarity-based alternative recommendations."""
    
    def test_returns_dict(self):
        ranked = rank("EGFR", disease_filter="lung", top_n=3)
        result = find_alternatives("EGFR", ranked, MASTER_MERGED, top_k=3)
        assert isinstance(result, dict)
    
    def test_alternatives_have_required_fields(self):
        ranked = rank("EGFR", disease_filter="lung", top_n=3)
        result = find_alternatives("EGFR", ranked, MASTER_MERGED, top_k=3)
        for cvcl, alts in result.items():
            for alt in alts:
                assert "cellosaurus_id" in alt
                assert "similarity_score" in alt
    
    def test_similarity_scores_between_0_and_1(self):
        ranked = rank("EGFR", disease_filter="lung", top_n=3)
        result = find_alternatives("EGFR", ranked, MASTER_MERGED, top_k=3)
        for cvcl, alts in result.items():
            for alt in alts:
                assert 0.0 <= alt["similarity_score"] <= 1.0
    
    def test_alternatives_not_same_as_original(self):
        ranked = rank("EGFR", disease_filter="lung", top_n=3)
        result = find_alternatives("EGFR", ranked, MASTER_MERGED, top_k=3)
        for cvcl, alts in result.items():
            for alt in alts:
                assert alt["cellosaurus_id"] != cvcl

