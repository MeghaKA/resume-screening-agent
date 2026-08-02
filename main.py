"""
Resume Screening Agent
-----------------------
Takes a job description and a folder of resumes, and produces a ranked,
scored shortlist with reasoning for each candidate.

Pipeline:
1. Read the job description and every resume (.txt, .pdf, .docx supported).
2. Score each resume against the job description using TF-IDF + cosine
   similarity (classic NLP similarity — fast, deterministic, explainable).
3. Send each resume + the JD + its similarity score to an LLM (Claude) to:
   - extract structured fields (skills, experience, education)
   - write a short human-readable reason for the ranking
4. Combine everything into a ranked list and save it as CSV + JSON.

Run:
    python main.py --jd sample_data/job_description.txt --resumes sample_data/resumes
"""

import os
import json
import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from anthropic import Anthropic

from parsing import extract_text  # local helper module

load_dotenv()

MODEL_NAME = "claude-sonnet-4-6"  # any current Claude model works here


def load_resumes(resume_dir: str) -> dict:
    """Read every resume file in a folder and return {filename: raw_text}."""
    texts = {}
    for path in sorted(Path(resume_dir).iterdir()):
        if path.suffix.lower() in (".txt", ".pdf", ".docx"):
            texts[path.name] = extract_text(path)
    return texts


def score_similarity(jd_text: str, resume_texts: dict) -> dict:
    """
    Compute a 0-1 relevance score for each resume against the job
    description using TF-IDF vectors + cosine similarity.

    Why TF-IDF instead of a fancier embedding model?
    - It's fast, free, deterministic, and easy to explain to a reviewer.
    - For short documents like resumes vs. a JD, it works well in practice.
    - Trade-off: it matches on wording, not deep meaning, so a resume that
      uses different terms for the same skill may score lower. See README
      for how this could be swapped for embeddings later.
    """
    filenames = list(resume_texts.keys())
    documents = [jd_text] + [resume_texts[f] for f in filenames]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)

    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(jd_vector, resume_vectors)[0]
    return {filenames[i]: round(float(similarities[i]), 4) for i in range(len(filenames))}


def analyze_with_llm(client: Anthropic, jd_text: str, resume_text: str, score: float) -> dict:
    """
    Ask Claude to extract structured info from the resume and explain,
    in plain English, why it does or doesn't fit the role.
    Returns a dict so it can be merged straight into the output table.
    """
    prompt = f"""You are helping screen a candidate for this role.

JOB DESCRIPTION:
{jd_text}

CANDIDATE RESUME:
{resume_text}

The candidate's TF-IDF similarity score against the job description is {score}.

Respond with ONLY a JSON object (no markdown, no extra text) in this exact shape:
{{
  "skills": ["skill1", "skill2"],
  "years_experience_estimate": "e.g. 2 years",
  "education": "highest relevant qualification",
  "reasoning": "1-2 sentence explanation of fit for this specific role",
  "recommendation": "Strong Match" | "Possible Match" | "Weak Match"
}}"""

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.content[0].text.strip()
    # Defensive cleanup in case the model wraps the JSON in ```json fences
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # If the model ever returns something unparsable, fail safely
        # instead of crashing the whole batch run.
        return {
            "skills": [],
            "years_experience_estimate": "unknown",
            "education": "unknown",
            "reasoning": "Could not parse LLM response for this candidate.",
            "recommendation": "Unknown",
        }


def run(jd_path: str, resume_dir: str, output_dir: str):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not found. Copy .env.example to .env and add your key."
        )
    client = Anthropic(api_key=api_key)

    jd_text = Path(jd_path).read_text(encoding="utf-8")
    resume_texts = load_resumes(resume_dir)

    if not resume_texts:
        raise RuntimeError(f"No resumes found in {resume_dir}")

    print(f"Loaded {len(resume_texts)} resumes. Scoring against job description...")
    scores = score_similarity(jd_text, resume_texts)

    rows = []
    for filename, text in resume_texts.items():
        print(f"Analyzing {filename}...")
        details = analyze_with_llm(client, jd_text, text, scores[filename])
        rows.append({
            "filename": filename,
            "similarity_score": scores[filename],
            **details,
        })

    df = pd.DataFrame(rows).sort_values("similarity_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    os.makedirs(output_dir, exist_ok=True)
    csv_path = Path(output_dir) / "ranked_candidates.csv"
    json_path = Path(output_dir) / "ranked_candidates.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    print(f"\nDone. Results saved to:\n  {csv_path}\n  {json_path}\n")
    print(df[["rank", "filename", "similarity_score", "recommendation"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank resumes against a job description.")
    parser.add_argument("--jd", default="sample_data/job_description.txt", help="Path to job description text file")
    parser.add_argument("--resumes", default="sample_data/resumes", help="Path to folder of resumes")
    parser.add_argument("--output", default="outputs", help="Folder to save ranked results")
    args = parser.parse_args()

    run(args.jd, args.resumes, args.output)
