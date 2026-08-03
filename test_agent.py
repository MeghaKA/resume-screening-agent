"""
Basic tests for the resume screening agent.

Run with:
    pytest test_agent.py -v

These check the two pieces of logic that don't need an API call:
file parsing and TF-IDF similarity scoring. The LLM-based extraction
(analyze_with_llm) isn't tested here since it needs a live API key —
that part is verified manually by running main.py end-to-end instead.
"""

from pathlib import Path
from parsing import extract_text
from main import score_similarity


def test_extract_text_reads_txt_file():
    """A .txt resume should be read back as non-empty plain text."""
    path = Path("sample_data/resumes/candidate_1_arjun.txt")
    text = extract_text(path)
    assert isinstance(text, str)
    assert len(text) > 0
    assert "Arjun" in text


def test_score_similarity_returns_value_for_every_resume():
    """Every resume in the folder should get a numeric score back."""
    jd_text = Path("sample_data/job_description.txt").read_text()
    resume_texts = {
        "candidate_1_arjun.txt": Path("sample_data/resumes/candidate_1_arjun.txt").read_text(),
        "candidate_4_rahul.txt": Path("sample_data/resumes/candidate_4_rahul.txt").read_text(),
    }

    scores = score_similarity(jd_text, resume_texts)

    assert set(scores.keys()) == set(resume_texts.keys())
    for score in scores.values():
        assert 0.0 <= score <= 1.0


def test_relevant_candidate_scores_higher_than_irrelevant_one():
    """
    Sanity check on the ranking logic itself: a machine-learning-focused
    resume (Arjun) should score higher against an ML job description than
    an unrelated front-end developer resume (Rahul).
    """
    jd_text = Path("sample_data/job_description.txt").read_text()
    resume_texts = {
        "candidate_1_arjun.txt": Path("sample_data/resumes/candidate_1_arjun.txt").read_text(),
        "candidate_4_rahul.txt": Path("sample_data/resumes/candidate_4_rahul.txt").read_text(),
    }

    scores = score_similarity(jd_text, resume_texts)

    assert scores["candidate_1_arjun.txt"] > scores["candidate_4_rahul.txt"]
