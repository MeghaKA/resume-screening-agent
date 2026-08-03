# Resume Screening Agent

An AI agent that screens resumes against a job description and returns a ranked,
scored shortlist with plain-English reasoning for every candidate — the kind of
first-pass review a recruiter would otherwise do by hand.

Built for the Rooman AI Challenge (Junior AI Research Associate — 24-Hour AI Agent Challenge).

---

## Overview

Given a job description and a folder of resumes, the agent:

1. Reads every resume (`.txt`, `.pdf`, `.docx`).
2. Scores each one against the job description using **TF-IDF + cosine similarity** — a fast, deterministic NLP similarity method.
3. Sends each resume, the job description, and its score to an **LLM (Llama 3.3 70B, served free via Groq)**, which extracts structured fields (skills, experience, education) and writes a short reason for the ranking.
4. Saves a ranked shortlist to `outputs/ranked_candidates.csv` and `.json`.

It's designed to be **explainable end to end** — every score traces back to either a deterministic calculation or a short piece of reasoning tied to the actual resume text. Nothing is a black box.

## Architecture

```
Job Description ──┐
                   ├──► TF-IDF + Cosine Similarity ──► similarity_score (0-1)
Resumes (.txt/.pdf/.docx) ──┘                                │
                                                              ▼
                              Groq LLM (Llama 3.3 70B) per resume
                              in: resume text + JD + similarity_score
                              out: skills, experience, education,
                                   reasoning, recommendation label
                                                              │
                                                              ▼
                          Ranked table ──► outputs/ranked_candidates.csv / .json
```

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/resume-screening-agent.git
cd resume-screening-agent
```

**2. Create a virtual environment and install dependencies**
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Get a free Groq API key**
Go to [console.groq.com](https://console.groq.com) → sign up (free, no card needed) → **API Keys** → **Create API Key** → copy it (starts with `gsk_...`).

**4. Add your key**
```bash
cp .env.example .env
```
Open `.env` and replace the placeholder:
```
GROQ_API_KEY=gsk_your-real-key-here
```

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Free key from console.groq.com, used for extraction and reasoning |

## Usage

Run on the included sample data:
```bash
python main.py --jd sample_data/job_description.txt --resumes sample_data/resumes
```

Run on your own data:
```bash
python main.py --jd path/to/job_description.txt --resumes path/to/resumes_folder
```

| Flag | Default | Description |
|---|---|---|
| `--jd` | `sample_data/job_description.txt` | Path to the job description |
| `--resumes` | `sample_data/resumes` | Folder of resumes to screen |
| `--output` | `outputs` | Folder to save ranked results |

Expected console output:
```
Loaded 4 resumes. Scoring against job description...
Analyzing candidate_1_arjun.txt...
Analyzing candidate_2_priya.txt...
Analyzing candidate_3_sneha.txt...
Analyzing candidate_4_rahul.txt...

Done. Results saved to:
  outputs/ranked_candidates.csv
  outputs/ranked_candidates.json

 rank               filename  similarity_score recommendation
    1  candidate_1_arjun.txt             0.50   Strong Match
    2  candidate_3_sneha.txt             0.40   Strong Match
    3  candidate_2_priya.txt             0.22 Possible Match
    4  candidate_4_rahul.txt             0.06    Weak Match
```

## Testing

Covers the parsing and scoring logic that doesn't need a live API call:
```bash
pytest test_agent.py -v
```
Checks: resume text extracts correctly, every resume gets a score between 0 and 1, and a relevant candidate scores higher than an irrelevant one — verifying the ranking logic itself, not just that it runs.

## Sample Output

| rank | filename | similarity_score | recommendation |
|---|---|---|---|
| 1 | candidate_1_arjun.txt | 0.50 | Strong Match |
| 2 | candidate_3_sneha.txt | 0.40 | Strong Match |
| 3 | candidate_2_priya.txt | 0.22 | Possible Match |
| 4 | candidate_4_rahul.txt | 0.06 | Weak Match |

Matches expectations: Arjun and Sneha both have direct ML/Python/Scikit-learn backgrounds; Priya is a data analyst with partial overlap; Rahul is a front-end developer with almost no overlap with the role.

## Design Choices & Tradeoffs

- **TF-IDF over embeddings for scoring** — fast, free, deterministic, and easy to explain. Tradeoff: it matches on word overlap, not deeper meaning, so different phrasing for the same skill can score lower than it should. A future version could swap in a sentence-embedding model (e.g. `all-MiniLM-L6-v2`) for semantic similarity.
- **LLM used only for extraction and reasoning, never for the score itself** — keeps the ranking deterministic and reproducible, and avoids the LLM hallucinating a number.
- **Groq (free Llama 3.3 70B) instead of a paid API** — free with generous rate limits, so reviewers can run this without setting up billing.
- **Graceful JSON fallback** — if the LLM returns malformed JSON for one candidate, that candidate is marked `"Unknown"` instead of crashing the whole batch.

## Limitations

- Only reads `.txt`, `.pdf`, and `.docx` — no OCR for scanned/image-only PDFs.
- TF-IDF similarity is wording-sensitive (see tradeoffs above).
- Resumes are processed sequentially; a large batch (100+) would benefit from batching or async LLM calls.
- No deduplication if the same resume is submitted twice.

## Project Structure

```
resume-screening-agent/
├── main.py                  # Core agent: scoring, LLM calls, ranking
├── parsing.py                # File readers for .txt/.pdf/.docx
├── test_agent.py              # Tests for parsing + scoring logic
├── requirements.txt
├── .env.example
├── .gitignore
├── outputs/                  # Generated after running main.py
│   ├── ranked_candidates.csv
│   └── ranked_candidates.json
└── sample_data/
    ├── job_description.txt
    └── resumes/
        ├── candidate_1_arjun.txt
        ├── candidate_2_priya.txt
        ├── candidate_3_sneha.txt
        └── candidate_4_rahul.txt
```

## Future Improvements

- Semantic (embedding-based) similarity instead of TF-IDF
- Batched/parallel LLM calls for large resume sets
- OCR support for scanned PDF resumes
- A lightweight Streamlit UI for non-technical reviewers
