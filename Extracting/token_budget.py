"""
token_budget.py -- how much text one chunk may hold. ONE rule, used by
both extractors:

    Extracting/extract.py          books     -> book_content
    Extracting/tutorial_chunks.py  tutorials -> tut_content

    from token_budget import MAX_CHUNK_TOKENS, estimate_tokens, split_into_chunks

The embedding model (all-MiniLM-L6-v2) reads only the first 256 tokens of a
chunk and ignores the rest, so a chunk must fit in that window or its second
half can never be found by search. 2 of the 256 are the model's own markers.

Nothing here needs PyMuPDF or the embedding model, so the tests of either
extractor can load it as it is.
"""
import math
import re

EMBED_MAX_TOKENS = 256
MAX_CHUNK_TOKENS = EMBED_MAX_TOKENS - 2
MIN_CHUNK_TOKENS = 80          # a shorter last chunk is evened out with the one before

# Tokens are not words: every punctuation mark is one token and a long or rare
# word is several, so code is far "longer" than its word count
# ("for(i=0;i<n;i++)" is 1 word but 14 tokens). Counting pieces -- a word, a
# number, or one punctuation mark -- follows the real tokenizer closely on all
# four books (real tokens per piece: median 1.05, 95% of the time under 1.2)
# and needs no model to be loaded here.
_PIECE_RE = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")
TOKENS_PER_PIECE = 1.2


def estimate_tokens(text):
    """About how many tokens the embedding model will make of this text
    (deliberately a little high, so a chunk that passes really fits)."""
    return math.ceil(len(_PIECE_RE.findall(text)) * TOKENS_PER_PIECE)


def split_into_units(text, max_tokens):
    """The pieces chunks are built from: the text's lines. A line that is too
    long on its own (a table row, a listing with no line breaks) is cut
    between words."""
    units = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if estimate_tokens(line) <= max_tokens:
            units.append(line)
            continue
        run = []
        for word in line.split():
            if run and estimate_tokens(" ".join(run + [word])) > max_tokens:
                units.append(" ".join(run))
                run = []
            run.append(word)
        units.append(" ".join(run))
    return units


def split_into_chunks(text, max_tokens=MAX_CHUNK_TOKENS):
    """Split text into chunks the embedding model can read in full. Lines are
    packed one after another until the next would not fit, so a cut only ever
    falls between two lines.

    (Book chunks used to be cut at blank lines and aim for 350 words. In three
    of the four books a blank line only appears between PAGES, so a chunk was a
    whole page -- two to three times what the model reads -- and a chapter's
    title page, being short, became a chunk of its own that matched every
    search for that chapter's topic.)"""

    units = [(unit, estimate_tokens(unit)) for unit in split_into_units(text, max_tokens)]
    size = lambda group: sum(tokens for _, tokens in group)

    groups, current = [], []
    for unit in units:
        if current and size(current) + unit[1] > max_tokens:
            groups.append(current)
            current = []
        current.append(unit)
    if current:
        groups.append(current)

    # A short last chunk (it only exists because it did not fit into the one
    # before): share the lines of the last two chunks out evenly instead.
    if len(groups) > 1 and size(groups[-1]) < MIN_CHUNK_TOKENS:
        both = groups[-2] + groups[-1]
        cut = 1
        while size(both[:cut + 1]) <= size(both) / 2:
            cut += 1
        if size(both[cut:]) <= max_tokens:
            groups[-2:] = [both[:cut], both[cut:]]

    return ["\n".join(unit for unit, _ in group) for group in groups]
