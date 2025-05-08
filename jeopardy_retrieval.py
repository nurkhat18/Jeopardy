#!/usr/bin/env python3
"""
jeopardy_retrieval.py

Jeopardy Retrieval Script
--------------------------

This script implements a Jeopardy question-answer retrieval system using:

1. BM25 baseline ranking (Precision@1, MRR@K).
2. Hybrid BM25 + GPT reranking approach.

Functionality:
- Parse a local Wikipedia corpus (plain text or gzipped XML).
- Build a BM25 index on article titles and snippets.
- Load Jeopardy clues and expected answers from a text file.
- Evaluate both the BM25 baseline and the hybrid BM25+GPT methods.
- Print Precision@1 and MRR metrics for each method.

Usage:
    export OPENAI_API_KEY=<your_api_key>
    python jeopardy_retrieval.py

Dependencies:
- Python 3.7+
- openai library
- Local Wikipedia data directory or file
- `questions.txt` file containing Jeopardy clues and answers
"""
import os
from dotenv import load_dotenv
import re
import math
import gzip
import json
import hashlib
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from typing import List, Tuple, Iterable
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("Missing OPENAI_API_KEY in environment.")

# ─────────────── Config ───────────────────────────────────────────
DUMP         = "wiki-subset-20140602"  # Path to Wikipedia dump folder or file
QUEST        = "questions.txt"         # Jeopardy clues file
BASELINE_K   = 50                        # BM25 depth for baseline evaluation
RERANK_BM25  = 20                        # BM25 depth feeding GPT reranker
RERANK_TOP   = 10                        # Number of final answers to keep after reranking
SNIPPET_LEN  = 100                       # Number of words in snippet from each article
CACHE_FILE   = "gpt_cache.json"        # File to cache GPT API responses
# ────────────────────────────────────────────────────────────────────

# ─────────────── GPT Cache Utilities ───────────────────────────────
try:
    with open(CACHE_FILE) as f:
        GPT_CACHE = json.load(f)
except FileNotFoundError:
    GPT_CACHE = {}

def save_cache():
    """
    Save the in-memory GPT_CACHE dictionary to disk as formatted JSON.
    """
    with open(CACHE_FILE, "w") as f:
        json.dump(GPT_CACHE, f, indent=2)

# ─────────────── Environment Setup ──────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable.")

# ─────────────── Text Processing Helpers ────────────────────────────
STOP = set(
    """
    the of and to a in for on with at by an be is are was were as that this from it or into
    its their his her has have had not but about over after before more most some such no nor so
    than then them they you your we our will would can could should may might up out if when while
    where which who whom what
    """.split()
)

def tokenize(text: str) -> List[str]:
    """
    Convert text to lowercase tokens, remove punctuation and stopwords.

    Args:
        text: Raw input string.
    Returns:
        List of cleaned word tokens.
    """
    return [w for w in re.findall(r"\b\w+\b", text.lower()) if w not in STOP]


def canon(s: str) -> str:
    """
    Canonicalize a Wikipedia title by lowercasing, stripping, removing leading 'the ' and brackets.

    Args:
        s: Original title string.
    Returns:
        Normalized title string.
    """
    s = s.strip().lower()
    if s.startswith("the "): s = s[4:]
    return re.sub(r'[\[\]"]', '', s)

# ─────────────── Wikipedia Parser ───────────────────────────────────
class WikiParser:
    """
    Iterator over Wikipedia data files (plain .txt or gzipped XML), yielding
    (title, snippet, full text) tuples for each page.
    """

    TITLE_RE = re.compile(r"^\s*\[\[(.+?)]]")

    def __init__(self, folder: str):
        """
        Initialize parser with a directory or single file path.

        Args:
            folder: Path to folder containing .txt/.gz XML files or a single file.
        """
        if os.path.isdir(folder):
            self.files = [os.path.join(folder, f) for f in sorted(os.listdir(folder))]
        else:
            self.files = [folder]

    def __iter__(self) -> Iterable[Tuple[str, str, str]]:
        """
        Yield (title, snippet, full_text) for each page in all files.
        """
        for fp in self.files:
            if fp.endswith(".txt"):
                yield from self._plain(fp)
            else:
                yield from self._xml(fp)

    def _plain(self, fp: str) -> Iterable[Tuple[str, str, str]]:
        """
        Parse a plain-text Wikipedia dump. Titles marked by [[Title]].
        """
        title, buf = None, []
        with open(fp, encoding="utf8", errors="ignore") as f:
            for line in f:
                m = self.TITLE_RE.match(line)
                if m:
                    if title:
                        text = "".join(buf)
                        snippet = " ".join(text.split()[:SNIPPET_LEN])
                        yield title, snippet, text
                    title, buf = m.group(1), []
                else:
                    buf.append(line)
            if title:
                text = "".join(buf)
                snippet = " ".join(text.split()[:SNIPPET_LEN])
                yield title, snippet, text

    @staticmethod
    def _xml(fp: str) -> Iterable[Tuple[str, str, str]]:
        """
        Parse a gzipped XML Wikipedia dump using ElementTree iterparse.
        """
        opener = gzip.open if fp.endswith(".gz") else open
        with opener(fp, "rb") as f:
            for _, el in ET.iterparse(f, events=("end",)):
                if el.tag.endswith("page"):
                    t = (el.find('./{*}title').text or "").strip()
                    x = (el.find('./{*}revision}/{*}text').text or "")
                    snippet = " ".join(x.split()[:SNIPPET_LEN])
                    yield t, snippet, x
                    el.clear()

# ─────────────── BM25 Index & Scoring ──────────────────────────────
class BM25:
    """
    BM25 ranking implementation for a collection of documents.

    Attributes:
        inv: Inverted index mapping term -> list of (doc_id, term_freq).
        len: Mapping doc_id -> document length (# tokens).
        titles: List of document titles.
        snips: List of document snippets.
        idf: Term -> inverse document frequency.
    """

    def __init__(self, docs: Iterable[Tuple[str, str, str]]):
        """
        Build inverted index, store document lengths, and compute IDF values.

        Args:
            docs: Iterable of (title, snippet, full_text).
        """
        self.inv, self.len, self.titles, self.snips = defaultdict(list), {}, [], []
        for did, (t, sn, text) in enumerate(docs):
            self.titles.append(t)
            self.snips.append(sn)
            toks = tokenize(t) * 3 + tokenize(" ".join(text.split()[:500]))
            self.len[did] = len(toks)
            for term, tf in Counter(toks).items():
                self.inv[term].append((did, tf))
        self.N = len(self.titles)
        self.avgdl = sum(self.len.values()) / self.N
        self.df = {term: len(postings) for term, postings in self.inv.items()}
        self.idf = {
            term: math.log((self.N - self.df[term] + 0.5) / (self.df[term] + 0.5) + 1)
            for term in self.df
        }

    def score(self, query: str, k: int) -> List[Tuple[int, float]]:
        """
        Score documents given a query and return top-k by BM25.

        Args:
            query: Query string.
            k: Number of top documents to return.
        Returns:
            List of (doc_id, score) tuples sorted descending.
        """
        sc = defaultdict(float)
        for term in Counter(tokenize(query)):
            for did, tf in self.inv.get(term, []):
                dl = self.len[did]
                denom = tf + 0.9 * (1 - 0.25 + 0.25 * dl / self.avgdl)
                sc[did] += self.idf.get(term, 0) * tf * 1.9 / denom
        return sorted(sc.items(), key=lambda x: -x[1])[:k]

# ─────────────── Jeopardy Clue Loader ──────────────────────────────
def load_clues(path: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Load Jeopardy clues file, splitting into categories, instructions, questions, and answers.

    Args:
        path: Path to clues file where every 3 lines represent one clue.
    Returns:
        Tuple of lists: (categories, instructions, questions, answers).
    """
    lines = [l.rstrip() for l in open(path, encoding="utf8") if l.strip()]
    cats, instrs, qs, ans = [], [], [], []
    for i in range(0, len(lines), 3):
        first = lines[i]
        m = re.match(r'^(.*?)\s*\((Alex:.*)\)\s*$', first)
        if m:
            cats.append(m.group(1)); instrs.append(m.group(2))
        else:
            cats.append(first); instrs.append("")
        qs.append(lines[i+1])
        ans.append(lines[i+2])
    return cats, instrs, qs, ans

# ─────────────── GPT Helper ────────────────────────────────────────
def _ckey(p: str) -> str:
    """
    Compute a SHA-256 hash key for a prompt string.
    """
    return hashlib.sha256(p.encode()).hexdigest()


def ask_gpt(prompt: str) -> str:
    """
    Query the OpenAI API (gpt-4-turbo) with a given prompt, using a cache to avoid duplicate calls.

    Args:
        prompt: Text prompt for the model.
    Returns:
        Model's text response.
    """
    key = _ckey(prompt)
    if key in GPT_CACHE:
        return GPT_CACHE[key]
    resp = openai.chat.completions.create(
        model="gpt-4-turbo",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    print("→ OpenAI used model:", resp.model)
    out = resp.choices[0].message.content
    GPT_CACHE[key] = out
    save_cache()
    return out

# ─────────────── GPT-based Reranking ──────────────────────────────
def gpt_rerank(full_q: str,
               cands: List[Tuple[int, float]],
               idx: BM25) -> List[str]:
    """
    Use GPT to rerank top BM25 candidates for a query.

    Args:
        full_q: Full query string (category + question).
        cands: List of (doc_id, score) from BM25.
        idx: BM25 index instance for title/snippet lookup.
    Returns:
        List of document titles in ranked order.
    """
    opts = [f"{i+1}. {idx.titles[d]} — {idx.snips[d]}"
            for i, (d, _) in enumerate(cands)]
    prompt = (
        f"Reorder these for: \u201c{full_q}\u201d best→worst\n"
        + "\n".join(opts)
        + f"\n\nReturn the numbers 1–{len(opts)} in ranked order."
    )
    out = ask_gpt(prompt)
    picks = []
    for n in re.findall(r"\b(\d+)\b", out):
        i = int(n) - 1
        if 0 <= i < len(cands) and i not in picks:
            picks.append(i)
        if len(picks) >= RERANK_TOP:
            break
    for i in range(len(cands)):
        if len(picks) >= RERANK_TOP:
            break
        if i not in picks:
            picks.append(i)
    return [idx.titles[cands[i][0]] for i in picks]

# ─────────────── Main Script Entry ───────────────────────────────
if __name__ == "__main__":
    # Index Wikipedia dump
    print("Indexing Wikipedia… (≈1 min)")
    docs = list(WikiParser(DUMP))
    idx = BM25(docs)

    # Load Jeopardy clues and answers
    cats, instrs, qs, ans = load_clues(QUEST)

    # Evaluate BM25 baseline
    hits, mrr_sum = 0.0, 0.0
    for cat, _, q, g in zip(cats, instrs, qs, ans):
        qry = f"{cat} {q}".strip()
        cands = idx.score(qry, BASELINE_K)
        if cands and canon(idx.titles[cands[0][0]]) == canon(g):
            hits += 1
        for rank, (did, _) in enumerate(cands, 1):
            if canon(idx.titles[did]) == canon(g):
                mrr_sum += 1 / rank
                break
    p1, mrr = hits / len(qs), mrr_sum / len(qs)
    print(f"\nBM25 baseline  ► P@1 {p1:.3f}   MRR@{BASELINE_K} {mrr:.3f}")

    # Evaluate Hybrid BM25+GPT
    hits, mrr_sum = 0.0, 0.0
    for cat, instr, q, g in zip(cats, instrs, qs, ans):
        qry = f"{cat} {q}".strip()
        if instr.startswith("Alex:"):
            prompt = f"Instruction: {instr}\nClue: {q}\nAnswer with the EXACT Wikipedia title."
            preds = [ask_gpt(prompt).splitlines()[0].strip()]
        else:
            cands = idx.score(qry, RERANK_BM25)
            preds = gpt_rerank(qry, cands, idx)
        if preds and canon(preds[0]) == canon(g):
            hits += 1
        for rank, title in enumerate(preds, 1):
            if canon(title) == canon(g):
                mrr_sum += 1 / rank
                break
    p1h, mrrh = hits / len(qs), mrr_sum / len(qs)
    print(f"Hybrid BM25+GPT ► P@1 {p1h:.3f}   MRR@{RERANK_TOP} {mrrh:.3f}\n")
        # ─────────────── Hybrid BM25+GPT Evaluation ──────────────────────────
    hits, mrr_sum = 0.0, 0.0
    expected, predicted = [], []

    for cat, instr, q, g in zip(cats, instrs, qs, ans):
        qry = f"{cat} {q}".strip()

        if instr.startswith("Alex:"):
            prompt = f"Instruction: {instr}\nClue: {q}\nAnswer with the EXACT Wikipedia title."
            preds = [ask_gpt(prompt).splitlines()[0].strip()]
        else:
            cands = idx.score(qry, RERANK_BM25)
            preds = gpt_rerank(qry, cands, idx)

        # record expected & predicted top-1
        expected.append(g)
        top1 = preds[0] if preds else ""
        predicted.append(top1)

        # update P@1 hit
        if canon(top1) == canon(g):
            hits += 1

        # update MRR
        for rank, title in enumerate(preds, 1):
            if canon(title) == canon(g):
                mrr_sum += 1 / rank
                break

    p1h, mrrh = hits / len(qs), mrr_sum / len(qs)
    print(f"\nHybrid BM25+GPT ► P@1 {p1h:.3f}   MRR@{RERANK_TOP} {mrrh:.3f}\n")

    # ─────────────── Print Expected vs Predicted ────────────────────────
    print("Expected vs Predicted (Hybrid top-1):")
    for exp, pred in zip(expected, predicted):
        status = "✔" if canon(pred) == canon(exp) else "✘"
        print(f"{status} Expected: {exp!r}   Predicted: {pred!r}")

    print(f"\nTotal correct (P@1): {int(hits)} / {len(qs)}")
