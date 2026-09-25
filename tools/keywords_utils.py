# tools/keywords_utils.py

import json
import sqlite3
import hashlib
from functools import partial
from datetime import datetime, timezone
from pyvi import ViTokenizer, ViPosTagger

FALLBACK_POS_TAGS = {"N", "Np", "Ny", "A"}
GEN_MODEL_LABEL = "rule_based_v1"

import model_state

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

def _normalize_query(query: str) -> str:
    """Lowercase + collect whitespace, used as cache key. Do not remove diacritics/stopwords in this function, 
    see get_or_generate_query_keywords() docstring for details."""
    return " ".join(query.strip().lower().split())


def get_or_generate_query_keywords(query: str, 
                                   generate_fn=None,
                                   gen_model: str = None) -> list:
    """Get keywords for a query from SQLite cache, or generate them if not cached. 
    This is the only point of change when moving from rule-based to LLM in the future 
    (change the generate_fn value, see make_query_expander() below).
    : param query: original query, unprocessed
    : param generate_fn: callback (query: str) -> List[str]. None = only read cache, do not generate new.
    : param gen_model: record in gen_model column for audit/comparison of future methods (e.g., 'rule_based_v1', 'claude-haiku-4-5')
    : return: List[str] of expanded keywords, may be empty
    """
    normalized = _normalize_query(query)
    query_id = hashlib.md5(normalized.encode("utf-8")).hexdigest()

    conn = sqlite3.connect(model_state.KEYWORDS_RESOURCES["query_cache_db_path"])
    row = conn.execute(
        "SELECT generated_keywords FROM query_keywords WHERE query_id = ?", (query_id,)
    ).fetchone()

    if row is not None:
        conn.execute(
            "UPDATE query_keywords SET hit_count = hit_count + 1 WHERE query_id = ?",
            (query_id,),
        )
        conn.commit()
        conn.close()
        return json.loads(row[0])

    keywords = generate_fn(query) if generate_fn is not None else []
    conn.execute(
        """INSERT INTO query_keywords
           (query_id, raw_query, normalized_query, generated_keywords, gen_method, gen_model, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (query_id, query, normalized, json.dumps(keywords, ensure_ascii=False),
         "none" if generate_fn is None else "rule_based", gen_model,
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    return keywords


def rule_based_keyword_extractor(query: str, 
                                 vocab: set, 
                                 max_ngram: int = 4) -> list:
    """Generate n-grams (decreasing length) from a tokenized query (ViTokenizer), check against a preloaded vocabulary.
    Prioritize matching longer phrases first to avoid splitting meaningful phrases. 
    Only finds phrases that EXACTLY match entries in the vocab - does not infer synonyms (hence the need for LLM later).
    : param query: original query
    : param vocab: set of keywords from load_keyword_vocabulary()
    : param max_ngram: maximum phrase length (number of "words" after ViTokenizer merges compound words)
    : return: List[str] of found keywords, sorted alphabetically, may be empty
    """
    normalized = query.strip().lower()
    tokenized = ViTokenizer.tokenize(normalized)
    tokens = tokenized.split()
    n = len(tokens)

    found = set()
    for window in range(min(max_ngram, n), 0, -1):
        for i in range(n - window + 1):
            phrase = " ".join(tokens[i:i + window]).replace("_", " ")
            if phrase in vocab:
                found.add(phrase)
    return sorted(found)


def fallback_phrase_extractor(query: str, 
                              pos_tags: set = FALLBACK_POS_TAGS,
                              min_len: int = 3) -> list:
    """
    Trich cum tu (danh tu/ten rieng/tinh tu lien tiep) truc tiep tu query,
    KHONG can vocab. Dung lam FALLBACK khi rule_based_keyword_extractor
    tra ve rong.
    """
    tokenized = ViTokenizer.tokenize(query.strip())
    tokens, tags = ViPosTagger.postagging(tokenized)

    phrases, current = [], []
    for tok, tag in zip(tokens, tags):
        if tag in pos_tags:
            current.append(tok.replace("_", " "))
        else:
            if current:
                phrases.append(" ".join(current))
                current = []
    if current:
        phrases.append(" ".join(current))

    phrases = [p.lower().strip() for p in phrases if len(p.strip()) >= min_len]
    phrases = list(set(phrases))
    final = [p for p in phrases if not any(p != q and p in q for q in phrases)]
    return sorted(final)


def rule_based_with_fallback_extractor(query: str, 
                                       vocab: set,
                                       max_ngram: int = 4,
                                       pos_tags: set = FALLBACK_POS_TAGS,
                                       min_len: int = 3) -> list:
    """Priority: first try rule_based_keyword_extractor (vocab), if no matches, fallback to fallback_phrase_extractor (POS tags)."""
    vocab_matches = rule_based_keyword_extractor(query, vocab, max_ngram=max_ngram)
    if vocab_matches:
        return vocab_matches
    return fallback_phrase_extractor(query, pos_tags=pos_tags, min_len=min_len)


def get_keywords(query: str) -> list:
    """Process a query through the query expander, based on preloaded vocab/DB resources.
    : param query: original query, unprocessed
    : return: list of keywords corresponding to the query (in the same order)"""
    query_expander = partial(
        rule_based_with_fallback_extractor,
        vocab=model_state.KEYWORDS_RESOURCES["keyword_vocab"],
        max_ngram=4,
        pos_tags=FALLBACK_POS_TAGS,
        min_len=3,
    )
    extra_keywords = get_or_generate_query_keywords(
        query,
        generate_fn=query_expander,
        gen_model=GEN_MODEL_LABEL,
    )
    logger.info(f"{query} => {extra_keywords}")
    return extra_keywords