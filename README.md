# 🏗️ DDR-AI-Builder

**AI-powered pipeline that converts Inspection Report PDFs and Thermal Report PDFs into professional Detailed Diagnostic Reports (DDR).**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Overview

DDR-AI-Builder automates the creation of comprehensive building diagnostic reports by:

1. **Parsing** both inspection and thermal PDFs (text, headings, tables, images)
2. **Extracting** structured observations using LLM-powered analysis
3. **Merging** findings using semantic similarity matching (SentenceTransformers)
4. **Detecting** conflicts between reports and identifying missing data
5. **Generating** a professional DDR with source traceability and embedded images
6. **Exporting** to HTML, PDF, and Markdown formats

---

## 🏛️ Architecture

```
DDR-AI-Builder/
│
├── app.py                          # Streamlit web interface
├── pipeline.py                     # End-to-end pipeline orchestrator
├── config.py                       # Central configuration
├── llm_client.py                   # Unified LLM client (OpenAI / Claude)
├── .env.example                    # Environment template
├── requirements.txt                # Python dependencies
│
├── parser/                         # PDF parsing layer
│   ├── pdf_parser.py               # Text/heading/table extraction (PyMuPDF)
│   └── image_extractor.py          # Image extraction with quality filters
│
├── extraction/                     # LLM-based observation extraction
│   ├── observation_extractor.py    # Inspection report observations
│   └── thermal_extractor.py        # Thermal report observations
│
├── processing/                     # Data processing & merging
│   ├── merger.py                   # Semantic similarity merge
│   ├── conflict_detector.py        # Cross-report conflict detection
│   └── missing_data_handler.py     # Missing/unclear data handling
│
├── generation/                     # Report generation
│   ├── ddr_generator.py            # DDR synthesis & export
│   └── templates/
│       └── ddr_template.html       # Professional HTML template
│
├── outputs/                        # Generated reports
├── sample_inputs/                  # Sample PDFs for testing
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/DDR-AI-Builder.git
cd DDR-AI-Builder

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

```bash
# Copy the example environment file
copy .env.example .env         # Windows
# cp .env.example .env         # macOS/Linux

# Edit .env with your API key
notepad .env
```

Set your LLM API key:
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
```

Or for Anthropic Claude:
```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 3. Run — Command Line

```bash
python pipeline.py path/to/inspection.pdf path/to/thermal.pdf
```

Options:
```bash
python pipeline.py inspection.pdf thermal.pdf \
    --title "123 Main St - DDR Report" \
    --formats html markdown pdf
```

### 4. Run — Streamlit Web UI

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 📊 DDR Output Structure

The generated report contains these sections:

| # | Section | Description |
|---|---------|-------------|
| 1 | **Property Issue Summary** | Executive overview of all findings |
| 2 | **Area-wise Observations** | Detailed per-area analysis with images |
| 3 | **Probable Root Cause** | Root cause analysis with evidence |
| 4 | **Severity Assessment** | Severity ratings with reasoning |
| 5 | **Recommended Actions** | Prioritized action plan |
| 6 | **Additional Notes** | Methodology and corroboration notes |
| 7 | **Missing or Unclear Information** | Data gaps and completeness score |

---

## 🔧 Pipeline Phases

### Phase 1: PDF Parsing
- Extracts page text, headings (font-size aware), and tables
- Detects section structure using bold/size heuristics
- Extracts and saves embedded images with quality filtering

### Phase 2: Observation Extraction
- LLM converts raw text into structured JSON observations
- Each observation includes: area, finding, severity, recommendation
- Separate specialized extractors for inspection and thermal reports
- Thermal extractor captures temperature data and IR image references

### Phase 3: Observation Merging
- Computes semantic embeddings via SentenceTransformers
- Greedy bipartite matching above cosine similarity threshold
- Area name overlap scoring for validation
- Confidence scores boosted for corroborated findings

### Phase 4: Conflict Detection & Data Quality
- LLM-based analysis of corroborated observations for contradictions
- Distinguishes genuine conflicts from complementary information
- Scans all fields for missing/unclear data
- Fills absent values with "Not Available"

### Phase 5: DDR Generation & Export
- LLM synthesizes each DDR section from merged observations
- Images mapped to areas by page proximity and text matching
- Exports: HTML (Jinja2 template), PDF (WeasyPrint), Markdown

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `OPENAI_API_KEY` | — | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model to use |
| `ANTHROPIC_API_KEY` | — | Your Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Anthropic model to use |
| `LLM_TEMPERATURE` | `0.1` | Lower = more deterministic |
| `LLM_MAX_TOKENS` | `4096` | Max tokens per LLM call |
| `SIMILARITY_THRESHOLD` | `0.75` | Cosine similarity for merging |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformers model |
| `HEADING_FONT_SIZE_THRESHOLD` | `12.0` | Min font size for headings |
| `MIN_IMAGE_WIDTH` | `100` | Min image width (px) |
| `MIN_IMAGE_HEIGHT` | `100` | Min image height (px) |

---

## 🎯 Key Features

- **Dual LLM Support**: OpenAI GPT-4o or Anthropic Claude with automatic retry
- **Semantic Merging**: SentenceTransformers-based observation matching across reports
- **Conflict Detection**: LLM identifies genuine contradictions vs complementary info
- **Source Traceability**: Every finding tracks back to document name and page number
- **Confidence Scoring**: Corroborated findings get higher confidence scores
- **Image Mapping**: Extracted images automatically placed in relevant DDR sections
- **Data Completeness**: Missing fields flagged and filled with "Not Available"
- **Multi-format Export**: HTML (professional template), PDF, and Markdown
- **Streamlit UI**: Upload-and-generate web interface with progress tracking

---

## 🛡️ Quality & Safety

- All LLM prompts include strict anti-hallucination instructions
- Source references preserved throughout the pipeline
- Conflict detection prevents contradictory information from being silently merged
- Every generated report includes a disclaimer about AI-generated content
- Missing data is explicitly flagged rather than silently omitted

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `PyMuPDF` | PDF text, image, and table extraction |
| `openai` | OpenAI API client |
| `anthropic` | Anthropic API client |
| `sentence-transformers` | Semantic similarity embeddings |
| `Jinja2` | HTML template rendering |
| `weasyprint` | HTML → PDF conversion |
| `streamlit` | Web interface |
| `loguru` | Structured logging |
| `tenacity` | Retry logic for API calls |
| `pydantic` | Data validation |
| `python-dotenv` | Environment variable management |

---

## 📝 License

MIT License — feel free to use, modify, and distribute.