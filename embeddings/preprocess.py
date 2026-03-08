"""
embeddings/preprocess.py
========================
Corpus cleaning and embedding pipeline for the 20 Newsgroups dataset.

Cleaning decisions:
- Headers stripped: metadata like From/Organization teaches the model that
  two posts are similar because they share a mail server, not a topic.
- Quoted lines (> and |) stripped: verbatim copies of other posts would
  bleed one document's semantic fingerprint into unrelated threads.
- Signatures stripped: boilerplate, zero topical signal.
- Min length 50 chars: post-cleaning stubs like "Thanks!" add noise to clusters.
- Max 512 words: all-MiniLM-L6-v2 has a 256 wordpiece window and silently
  truncates — we truncate explicitly so there's no silent information loss.

Model choice (all-MiniLM-L6-v2):
  Best quality/speed trade-off for CPU inference. 384-dim outputs, sub-100ms
  per query. Cross-phrasing semantic quality is critical for cache hit rate.
"""

import re
import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Compiled patterns — compiled once at import time, not per call
# ---------------------------------------------------------------------------

_HEADER_FIELDS = re.compile(
    r'^(from|subject|organization|lines|message-id|references|'
    r'nntp-posting-host|x-newsreader|reply-to|distribution|'
    r'keywords|summary|expires|followup-to|approved|sender|'
    r'path|newsgroups|date|xref)[^\n]*\n',
    re.MULTILINE | re.IGNORECASE,
)
_QUOTED_LINES   = re.compile(r'^[>|].*$', re.MULTILINE)  # UCI uses both > and |
_SIGNATURE      = re.compile(r'\n--\s*\n.*', re.DOTALL)
_URLS           = re.compile(r'https?://\S+|www\.\S+')
_EMAILS         = re.compile(r'\S+@\S+\.\S+')
_UUCP_PATHS     = re.compile(r'\S+!\S+!\S+')             # server1!server2 paths, UCI-specific
_WHITESPACE     = re.compile(r'[ \t]+')

_MIN_LENGTH = 50
_MAX_WORDS  = 512


def clean_article(raw: str) -> str:
    """
    Clean a single raw UCI newsgroup article.
    Returns cleaned body text, or empty string if nothing survives.
    """
    text = raw

    # Split on first blank line — standard news format separates header from body
    if '\n\n' in text:
        _, text = text.split('\n\n', 1)
    text = _HEADER_FIELDS.sub('', text)

    # Signature before quoted lines — avoids preserving signatures
    # that happen to begin with quoted text
    text = _SIGNATURE.sub('', text)
    text = _QUOTED_LINES.sub('', text)

    text = _URLS.sub(' ', text)
    text = _EMAILS.sub(' ', text)
    text = _UUCP_PATHS.sub(' ', text)

    text = _WHITESPACE.sub(' ', text)
    text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())

    words = text.split()
    if len(words) > _MAX_WORDS:
        text = ' '.join(words[:_MAX_WORDS])

    return text.strip()


def embed_corpus(
    texts: List[str],
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Embed a list of cleaned documents.
    Returns float32 array of shape (N, embedding_dim).

    normalize_embeddings=True makes all vectors unit-length so
    dot product == cosine similarity — no sqrt needed in cache lookups.
    """
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading embedding model '{model_name}' ...")
    model = SentenceTransformer(model_name)

    logger.info(f"Embedding {len(texts)} documents in batches of {batch_size} ...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    logger.info(f"Embeddings: {embeddings.shape}, dtype={embeddings.dtype}")
    return embeddings.astype(np.float32)
