# Resume Screening Agent

An AI agent that takes a job description and a folder of resumes, and
returns a ranked, scored shortlist with reasoning for each candidate.

## What it does

1. Reads the job description and every resume in a folder (`.txt`, `.pdf`, `.docx`).
2. Scores each resume against the job description using **TF-IDF + cosine
   similarity** (a classic NLP similarity method).
3. Sends each resume, the job description, and its similarity score to
   an **LLM (Llama 3.3 70B, served free via Groq)**, which extracts
   structured fields (skills, experience, education) and writes a short
   reason for the ranking.
4. Saves a ranked list to `outputs/ranked_candidates.csv` and `.json`.

## Setup

### 1. Clone this repo and enter the folder
```bash
git clone https://github.com/YOUR_USERNAME/resume-screening-agent.git
cd resume-screening-agent
```

### 2. Create a virtual environment and install dependencies
```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Get a free Groq API key
1. Go to https://console.groq.com and sign up / log in (free, no credit card required).
2. Go to **API Keys** and click **Create API Key**.
3. Copy the key (it starts with `gsk_...`).

### 4. Add your API key
```bash
cp .env.example .env
```
Open `.env` and paste your key in:
```
GROQ_API_KEY=gsk_your-real-key-here
```

## Usage

Run it on the included sample data:
```bash
python main.py --jd sample_data/job_description.txt --resumes sample_data/resumes
```

Run it on your own data:
```bash
python main.py --jd path/to/job_description.txt --resumes path/to/resumes_folder
```

Output is saved to `outputs/ranked_candidates.csv` and `outputs/ranked_candidates.json`.

## Sample Output

| rank | filename | similarity_score | recommendation |
|------|----------|-------------------|-----------------|
| 1 | candidate_1_arjun.txt | 0.50 | Strong Match |
| 2 | candidate_3_sneha.txt | 0.40 | Strong Match |
| 3 | candidate_2_priya.txt | 0.22 | Possible Match |
| 4 | candidate_4_rahul.txt | 0.06 | Weak Match |

## Design Choices & Tradeoffs

- **TF-IDF instead of embeddings for similarity scoring.** TF-IDF is fast,
  free, deterministic, and easy to explain — good for a 24-hour build. Its
  limitation is that it matches on word overlap, not deeper meaning, so a
  resume that describes the same skill with different wording could score
  lower than it should. A future version could swap this for a sentence
  embedding model (e.g. `all-MiniLM-L6-v2`) for semantic similarity.
- **LLM used for extraction and reasoning, not for scoring.** The ranking
  score comes from TF-IDF (deterministic, reproducible), while the LLM
  is only used to extract structured fields and explain the ranking in
  plain English. This avoids the LLM "hallucinating" a numeric score.
- **Groq (free Llama 3.3 70B) instead of a paid API.** Groq offers this
  model for free with generous rate limits, which keeps the whole project
  reproducible for reviewers without needing billing set up.
- **JSON parsing has a fallback.** If Claude ever returns malformed JSON
  for one candidate, that candidate is marked `"Unknown"` instead of
  crashing the whole batch — reliability over a single bad row.

## Limitations

- Currently only reads `.txt`, `.pdf`, and `.docx` files.
- TF-IDF similarity is wording-sensitive (see tradeoffs above).
- Processes resumes sequentially; a large batch (100+) would benefit
  from batching or async calls to the LLM.

## Project Structure
```
resume-screening-agent/
├── main.py                 # Core agent: scoring, LLM calls, ranking
├── parsing.py               # File readers for .txt/.pdf/.docx
├── requirements.txt
├── .env.example
└── sample_data/
    ├── job_description.txt
    └── resumes/
        ├── candidate_1_arjun.txt
        ├── candidate_2_priya.txt
        ├── candidate_3_sneha.txt
        └── candidate_4_rahul.txt
```
