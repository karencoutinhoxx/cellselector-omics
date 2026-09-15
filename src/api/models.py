from typing import Optional
from pydantic import BaseModel, Field


class ClassicalRequest(BaseModel):
    gene: str
    disease_filter: Optional[str] = None
    lineage_filter: Optional[str] = None
    exclude_genes: list[str] = Field(default_factory=list)
    top_n: int = Field(default=10, ge=1, le=50)
    use_learned_weights: bool = True


class AgenticRequest(ClassicalRequest):
    ollama_model: str = "llama3.1:8b"
    target_cellosaurus_id: Optional[str] = None

