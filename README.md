<div align="center">

# CellSelector Omics

### A Multi-Omics Recommendation System for Cell Line Selection

*A decision-support framework that reconciles fragmented public omics resources into a single, evidence-weighted, and interpretable recommendation for experimental cell line selection.*

**University of Bristol · MSc Data Science Group Project · Team 26**
**Conducted in partnership with AstraZeneca**

            [Live site](https://cellselectoromics.com) · [Repository](https://github.com/karencoutinhoxx/cellselector-omics)

</div>

---

## Abstract

Selecting an appropriate immortalised human cell line is a foundational determinant of experimental validity in biopharmaceutical research, yet in practice it remains a manual, expertise-dependent task. The evidence required is distributed across independent public repositories that differ in nomenclature, measurement modality, and unit conventions, and no single resource supports gene-driven, cross-source model selection. **CellSelector Omics** addresses this gap. It integrates five public multi-omics datasets, the Human Protein Atlas, DepMap transcriptomics, DepMap CRISPR gene dependency, the Gene Expression Omnibus, and CCLE proteomics, onto a single identity layer spanning 2,076 human cell lines and around 222 million expression records, and returns a ranked, confidence-weighted, and evidence-cited set of candidate cell lines for any queried gene.

The main methodological contribution is a **gene-class-aware scoring model**. A single global weighting of the evidence does not hold across gene classes: tissue-specific, ubiquitously-expressed, and loss-of-function genes each require different evidence weightings. Notably, an optimiser given only ranking performance, with no biological guidance, independently arrived at biologically sensible weightings for each class, weighting expression heavily for tissue-specific genes and disease and tissue context for loss-of-function genes. The classical ranker is complemented by a retrieval-augmented generative model that produces structured, source-attributed justifications, constraining the language model to synthesis over verified evidence rather than open-ended generation.

---

## Motivation and Problem Statement

The scientific value of a cell-line experiment is bounded by how well the chosen line reflects the biological context of interest. Despite this, model selection is typically performed by manually cross-referencing resources such as DepMap, HPA, and GEO, a process that is slow, irreproducible, and sensitive to the individual expertise of the researcher. The underlying difficulty is one of **heterogeneous data integration under uncertainty**:

1. **Identity fragmentation.** The same physical cell line is recorded under divergent identifiers across sources (for example `HCC827`, `Hcc-827`, and the DepMap accession `ACH-000164` all denote a single line). Absent reconciliation, evidence for one line is silently partitioned across apparent duplicates, systematically distorting any downstream ranking.
2. **Incompatible evidence.** Transcript-level (normalised or log TPM), protein-level (mass-spectrometry intensity), and functional (CRISPR dependency) signals are measured on scales that are not directly comparable and cannot simply be averaged together.
3. **Class-dependent semantics.** The notion of a "suitable" model is gene-dependent: high specific expression identifies a good model for a receptor tyrosine kinase, but is uninformative for a tumour suppressor studied through loss-of-function.

This work treats each of these as a first-class design constraint rather than an implementation detail.

---

## System Architecture

```
   Public omics sources            Canonical data layer           Inference and delivery
 ┌────────────────────┐        ┌──────────────────────┐       ┌────────────────────┐
 │ HPA · DepMap RNA   │        │ Nomenclature         │       │ Classical ranker   │
 │ CRISPR dependency  │  ───►  │ resolution           │  ───► │ Agentic RAG model  │
 │ GEO · CCLE prot.   │        │ → 2,076 cell lines   │       │ Neo4j knowledge    │
 │                    │        │   (Cellosaurus IDs)  │       │   graph            │
 └────────────────────┘        └──────────────────────┘       │ FastAPI + React UI │
                                                               └────────────────────┘
```

The **nomenclature-resolution layer** is the foundation the whole system rests on. It maps every source-specific identifier onto a stable Cellosaurus accession, achieving 92 to 100 percent coverage across the expression sources, so that evidence for a cell line accumulates against one consistent identity. Without this, the same line would be split across duplicate records and the cross-source agreement the recommender relies on could not be measured.

---

## Methodology

### Data pipeline
Raw sources (up to approximately 1.1 GB per file) are processed via chunked streaming into a columnar Parquet store, bounding peak memory below 500 MB and enabling gene-scoped reads rather than full-matrix scans. Identity resolution draws on the Cellosaurus catalogue together with dataset-specific cross-reference metadata.

### Classical scoring model
For a queried gene, each cell line is assigned a score combining percentile-normalised RNA, protein, quality (cross-source consistency), and disease and tissue-context terms, with GEO incorporated as an additive confirmatory signal owing to its inconsistent unit documentation. Weights are optimised per gene class by maximising Mean Reciprocal Rank over a literature-curated validation set (`scipy.optimize`). CRISPR gene-dependency is surfaced as a distinct, non-blended evidence dimension, reflecting its conceptual independence from expression. Complementary scorers (pathway proximity, mutation status, copy number, random-walk-with-restart, ssGSEA, and a LambdaMART learning-to-rank model) are implemented alongside the primary scorer to allow evidence fusion to be extended and benchmarked.

### Agentic explanation model
A retrieval-augmented pipeline assembles the underlying evidence, KEGG pathway memberships, Cellosaurus growth properties, and targeted PubMed abstracts, and prompts a language model to produce a structured, source-cited justification. The ranking is decided by the classical model; the generative component is confined to explanation, so that any model failure degrades interpretability alone and never the recommendation itself.

### Knowledge graph
Gene-pathway-cell-line relationships are materialised in a Neo4j graph, enabling multi-hop queries (for example, cell lines linked to a target via shared pathway membership) that are inexpressible as relational joins. Pre-materialisation reduced query latency from tens of seconds to sub-second.

---

## Evaluation

The classical model was assessed against a literature-curated validation set using Mean Reciprocal Rank and precision@k, and independently against a partner-defined panel of test genes from AstraZeneca. Results confirm that per-class weighting outperforms a global scheme, and that the low aggregate performance for ubiquitous and loss-of-function classes reflects a genuine biological property, the non-informativeness of expression for those classes, rather than a modelling deficiency. On the industry-defined panel the system reproduced expected behaviour across both routine and adversarial cases, including a housekeeping gene (correctly yielding a flat, winner-free distribution) and a gene with strong literature support but sparse coverage (correctly recovering the relevant lineage).

*Full metrics: `src/models/classical/evaluate.py` and `outputs/model_evaluation.json`.*

---

## Technology Stack

| Layer | Tools |
| --- | --- |
| Data engineering | Python, pandas, PyArrow (Parquet), chunked-streaming ingestion |
| Modelling | scipy (weight optimisation), scikit-learn |
| Generative AI | Groq API (`gpt-oss-20b`) with local Ollama fallback; Entrez/PubMed, KEGG, Cellosaurus retrieval |
| Knowledge graph | Neo4j (self-hosted, Cypher) |
| Backend service | FastAPI, Uvicorn |
| Frontend | React, TypeScript, Vite, Tailwind CSS |
| Deployment | Docker |
| Project management | GitHub Projects (Kanban board for sprint and task tracking) |

---

## Project Management

Work was tracked and coordinated through a **Kanban board on GitHub Projects**, linked directly to repository issues and pull requests. Tasks moved through backlog, in-progress, review, and done columns, giving the team continuous visibility into workload distribution, blockers, and sprint velocity across the data pipeline, modelling, and frontend workstreams.

---

## Live Deployment

A hosted instance of the system is available at **[cellselectoromics.com](https://cellselectoromics.com)**, providing gene search, per-dataset evidence breakdowns, and interactive graph exploration through the React frontend described below.

---

## Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+ (frontend)
- Preprocessed data artefacts in `outputs/parquet/` (produced by the ingestion pipeline)
- A Neo4j instance (knowledge-graph features)
- A Groq API key, or a local Ollama installation (agentic justifications)

### Backend
```bash
git clone https://github.com/karencoutinhoxx/cellselector-omics.git
cd cellselector-omics
pip install -r requirements.txt

# Launch the API (loads data at startup)
uvicorn src.api.main:app --port 8000
```

### Frontend
```bash
cd src/frontend
npm install
npm run dev
```

### Containerised deployment
```bash
docker build -t cellselector-omics .
docker run -p 8000:8000 cellselector-omics
```

---

## API Reference

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Service status and dataset summary |
| `GET` | `/genes/search?q={gene}` | Gene lookup |
| `POST` | `/recommend/classical` | Ranked recommendations (statistical model) |
| `POST` | `/recommend/agentic` | Recommendations with generated justifications |
| `GET` | `/recommend/export/{format}` | Export (JSON / CSV / PDF) |
| `GET` | `/cell-lines/{cellosaurus_id}` | Single cell-line record |
| `GET` | `/stats` | Model and dataset statistics |

```bash
curl -X POST http://localhost:8000/recommend/classical \
  -H "Content-Type: application/json" \
  -d '{"gene": "EGFR", "top_n": 10}'
```

---

## Repository Structure

```
cellselector-omics/
├── config/         Configuration files
├── data/           Nomenclature and cross-reference mappings
├── docs/           Project documentation
├── figures/        Generated figures and plots
├── outputs/        Unified data store and evaluation artefacts
├── scripts/        Ingestion and utility scripts
├── src/
│   ├── api/            FastAPI service (routes, request/response models)
│   ├── frontend/       React + TypeScript + Vite web interface
│   ├── models/
│   │   ├── classical/  Scoring, ranking, weight optimisation, evaluation
│   │   ├── agentic/    Retrieval-augmented generation pipeline
│   │   └── graph/      Neo4j ingestion and Cypher queries
│   └── services/       Recommendation orchestration, confidence scoring, validation
└── Dockerfile
```

---

## Limitations and Future Work

The validation set, while spanning the three gene classes, is modest and is used both to fit and to report per-class weights; a larger, expert-curated, held-out set would yield a less optimistic and more robust estimate. Coverage is uneven across modalities (proteomics covers 375 of 2,076 lines), a sparsity the confidence measure surfaces rather than resolves. Justifications are generated on demand, trading first-request latency for storage economy and evidential currency. Full public deployment remains constrained by the in-memory footprint of the complete dataset. Priorities for further work include expanded and held-out validation, broadened yet still-grounded explanation coverage, and further-provisioned deployment of the API-backed interface.

---

## Contributors

**Team 26 · University of Bristol MSc Data Science**
Ishaan Bhalla · Karen Coutinho · Aman Raj · Dixit Kaloorani Malarmannan

Academic supervision: **Dr Daniel D'Andrea** · Industry partner: **AstraZeneca**

---

## Acknowledgements

This work is built upon data from the Human Protein Atlas, the Cancer Dependency Map (DepMap), the Gene Expression Omnibus (GEO), the Cancer Cell Line Encyclopedia (CCLE), and the Cellosaurus knowledge resource. We gratefully acknowledge the guidance of our academic supervisor and our AstraZeneca partner.
