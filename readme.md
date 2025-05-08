# Jeopardy Retrieval Script Documentation

## Overview

The **Jeopardy Retrieval Script** implements a two-stage retrieval system for Jeopardy clues:

1. **BM25 Baseline**

   * Indexes a local Wikipedia corpus.
   * Ranks articles by BM25 score.
   * Computes Precision\@1 and MRR metrics.

2. **Hybrid BM25 + GPT Reranking**

   * Uses BM25 to retrieve the top candidates.
   * Reranks those candidates with a GPT model for improved accuracy.

This documentation explains how to install, configure, and use the script, and details its internal components.

---

## Table of Contents

* [Dependencies](#dependencies)

* [Architecture](#architecture)

  * [`WikiParser` Class](#wikiparser-class)
  * [`BM25` Class](#bm25-class)
  * `load_clues` Function
  * `ask_gpt` & Caching
  * `gpt_rerank` Function
* [Performance](#performance)


---

## Dependencies

* **Python 3.7+**
* `openai` library for API access
* `python-dotenv` to load environment variables
* Standard libraries: `os`, `re`, `math`, `gzip`, `json`, `hashlib`, `xml.etree.ElementTree`, `collections`, `typing`
* Add the wiki-subset folder to the project near the .py file. it's too big to upload it here

Install with:

```bash
pip install openai python-dotenv
export OPENAI_API_KEY="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"
Output
```


## Output

   * Prints BM25 baseline metrics: `P@1` and `MRR@K`.
   * Prints Hybrid metrics: `P@1` and `MRR@N` after GPT reranking.

---

## Architecture

### `WikiParser` Class

* **Purpose**: Iterates over a Wikipedia dump (plain-text or gzipped XML).
* **Key Methods**:

  * `__iter__`: Yields `(title, snippet, full_text)` tuples.
  * `_plain`: Parses `.txt` dumps (titles marked `[[Title]]`).
  * `_xml`: Parses gzipped XML using `ElementTree.iterparse`.

### `BM25` Class

* **Purpose**: Builds an inverted index and scores documents with BM25.
* **Attributes**:

  * `inv`: Term → list of `(doc_id, term_freq)`.
  * `len`: Document lengths (# tokens).
  * `titles`: List of page titles.
  * `snips`: List of page snippets.
  * `idf`: Inverse document frequencies.
* **Key Methods**:

  * `__init__`: Indexes documents, computes `idf`.
  * `score(query, k)`: Returns top-`k` `(doc_id, score)` by BM25.

#### Tokenization

* Uses a simple regex-based tokenizer.
* Stops common English stopwords.
* Titles are weighted more heavily (×3).

### `load_clues` Function

* **Purpose**: Reads the clues file (`questions.txt`).
* **Format**: Every 3 non-empty lines → one clue:

  1. Category (with optional `(Alex:...)` instruction)
  2. Clue text
  3. Expected Wikipedia title (answer)

Returns four parallel lists:

* `categories`, `instructions`, `questions`, `answers`.

### `ask_gpt` & Caching

* **Function**: Sends prompts to the OpenAI API (`gpt-4-turbo`).
* **Cache**: Saves responses in `gpt_cache.json` to avoid duplicate calls.
* **Key steps**:

  1. Compute a SHA-256 key of the prompt.
  2. Return cached response if available.
  3. Otherwise, call OpenAI, store result, and save cache.

### `gpt_rerank` Function

* **Purpose**: Rerank BM25 candidates using GPT.
* **Input**: Full query (`category + question`) and BM25 top candidates.
* **Process**:

  1. Formats a prompt listing each candidate with its snippet.
  2. Asks GPT to reorder them best→worst.
  3. Parses the returned ranking numbers.
  4. Returns the ordered list of titles (up to `RERANK_TOP`).

---

## Performance 

* **Indexing Time**: approximately ~1 minute for 280k pages.
* **First Run**: It will take 5-10 minutes to get the result for the first run if you clear the cache. if you work with cache results should come out in 2 minutes.


