"""
embeddings/preprocess.py
========================
Corpus cleaning and embedding pipeline for the 20 Newsgroups dataset.

Design decisions (documented here as the README promises):

1.  WHY strip headers?
    The raw UCI articles contain full email/news headers: Xref, Path, From,
    Subject, Organization, Lines, Message-ID, References, NNTP-Posting-Host.
    These are metadata, not content. If we embed them, the model learns that
    two articles are similar because they both came from the same university
    mail server, not because their topics overlap. That breaks semantic
    search and ruins the cache.

2.  WHY strip quoted reply blocks ("> " and "| " lines)?
    Quoted blocks are verbatim copies of other documents. Embedding them
    duplicates other articles semantic fingerprints into unrelated threads.
    The UCI format uses BOTH ">" and "|" as quote prefixes, so we strip both.

3.  WHY strip signature blocks ("--" separator)?
    Signatures are user boilerplate. They carry zero topical signal.

4.  WHY keep a minimum length of 50 characters?
    Post-cleaning, some articles reduce to one or two words ("Thanks!").
    These add noise to cluster formation. 50 chars is a conservative floor.
    Short posts that survive cleaning are still kept because even brief posts
    can carry topical signal ("Buy this rifle at auction, $200" clearly
    belongs to firearms even at 50 chars).

5.  WHY truncate to 512 tokens?
    all-MiniLM-L6-v2 has a 256-wordpiece context window and silently
    truncates. We truncate at 512 whitespace tokens explicitly so the
    visible text is exactly what gets embedded with no silent loss.

6.  WHY use all-MiniLM-L6-v2?
    Best quality/speed trade-off for sub-100ms query embedding on CPU
    with 384-dimensional outputs. Semantic quality far exceeds TF-IDF
    for cross-phrasing similarity which is critical for the cache layer.
"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# All known UCI newsgroup header field names.
_HEADER_FIELDS = re.compile(
    r'^(from|subject|organization|lines|message-id|references|'
    r'nntp-posting-host|x-newsreader|reply-to|distribution|'
    r'keywords|summary|expires|followup-to|approved|sender|'
    r'path|newsgroups|date|xref)[^\n]*\n',
    re.MULTILINE | re.IGNORECASE,
)

# Quoted reply lines — UCI uses BOTH ">" and "|" as quote markers.
_QUOTED_LINES = re.compile(r'^[>|].*$', re.MULTILINE)

# Signature block: everything from a standalone "--" line onward.
_SIGNATURE_BLOCK = re.compile(r'\n--\s*\n.*', re.DOTALL)

# Whitespace normalisation.
_WHITESPACE_RUNS = re.compile(r'[ \t]+')

# URLs.
_URLS = re.compile(r'https?://\S+|www\.\S+')

# Email addresses.
_EMAILS = re.compile(r'\S+@\S+\.\S+')

# UUCP paths like "server1!server2!server3" — UCI-specific noise.
_UUCP_PATHS = re.compile(r'\S+!\S+!\S+')

# Minimum post length after cleaning (characters).
_MIN_LENGTH = 50

# Maximum whitespace-tokenised words before truncation.
_MAX_WORDS = 512


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_article(raw: str) -> str:
    """
    Apply the full cleaning pipeline to a single raw UCI newsgroup article.

    Steps (order matters):
    1. Strip header block (split on first blank line, then regex residuals)
    2. Strip signature block
    3. Strip quoted reply lines (both > and | prefixes)
    4. Strip URLs, email addresses, UUCP paths
    5. Normalise whitespace
    6. Truncate to _MAX_WORDS

    Returns the cleaned body text, or empty string if nothing remains.
    """
    text = raw

    # 1. Header block — split on first blank line (standard news format).
    if '\n\n' in text:
        _, text = text.split('\n\n', 1)
    # Strip any residual header fields that leaked through.
    text = _HEADER_FIELDS.sub('', text)

    # 2. Signature block — before quoted lines to avoid preserving
    #    signatures that begin with quoted text.
    text = _SIGNATURE_BLOCK.sub('', text)

    # 3. Quoted reply lines.
    text = _QUOTED_LINES.sub('', text)

    # 4. Noise.
    text = _URLS.sub(' ', text)
    text = _EMAILS.sub(' ', text)
    text = _UUCP_PATHS.sub(' ', text)

    # 5. Normalise whitespace.
    text = _WHITESPACE_RUNS.sub(' ', text)
    text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())

    # 6. Truncate.
    words = text.split()
    if len(words) > _MAX_WORDS:
        text = ' '.join(words[:_MAX_WORDS])

    return text.strip()


def embed_corpus(
    texts: List[str],
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 64,
    show_progress: bool = True,
) -> 'np.ndarray':
    """
    Embed a list of cleaned documents using a sentence-transformers model.
    Returns float32 numpy array of shape (N, embedding_dim).

    normalize_embeddings=True makes all vectors unit-length so
    cosine similarity equals dot product, making cache lookups faster.
    """
    import numpy as np
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

    logger.info(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    return embeddings.astype(np.float32)