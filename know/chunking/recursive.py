import re
from typing import Callable, List, Pattern

from .base import AbstractChunker, Chunk

#  Configuration
PARAGRAPH_RE: Pattern = re.compile(r"\n\s*\n")          # blank line
SENTENCE_RE:  Pattern = re.compile(r"(?<=[.!?])\s+")    # ., ! or ?  + space
PHRASE_RE:    Pattern = re.compile(r"\s*[,;:]\s*")      # , ; :
WORD_RE:      Pattern = re.compile(r"\S+")              # one or more non-whitespace characters

def _segments_with_pos(regex: Pattern, text: str, offset: int):
    """Split *text* by *regex* while preserving absolute positions."""
    last = 0
    for m in regex.finditer(text):
        if m.start() > last:
            yield text[last : m.start()], offset + last, offset + m.start()
        last = m.end()
    if last < len(text):
        yield text[last:], offset + last, offset + len(text)


class RecursiveChunker(AbstractChunker):
    def __init__(
        self,
        *,
        max_tokens: int,
        token_counter: Callable[[str], int] = lambda s: len(s.split()),
        paragraph_re: Pattern = PARAGRAPH_RE,
        sentence_re: Pattern = SENTENCE_RE,
        phrase_re: Pattern = PHRASE_RE,
        word_re: Pattern = WORD_RE,
    ):
        self.max_tokens = max_tokens
        self.token_counter = token_counter
        self.paragraph_re = paragraph_re
        self.sentence_re = sentence_re
        self.phrase_re = phrase_re
        self.word_re = word_re

    def _split_words(self, text: str, start: int) -> List[Chunk]:
        """Final fallback: slice an over-long span on word boundaries."""
        pieces, matches = [], list(self.word_re.finditer(text))
        for i in range(0, len(matches), self.max_tokens):
            s_idx = matches[i].start()
            e_idx = matches[min(i + self.max_tokens - 1, len(matches) - 1)].end()
            span = text[s_idx:e_idx]
            pieces.append(Chunk(start + s_idx, start + e_idx, span))
        return pieces

    def _split_recursively(
        self, text_to_split: str, offset: int, regex: Pattern, next_level_fn: Callable
    ) -> List[Chunk]:
        """
        Generic helper to split text by a regex and recursively call the next-level splitter.
        """
        segments = list(_segments_with_pos(regex, text_to_split, offset))

        # If the regex doesn't split the text, pass the whole text to the next level.
        if len(segments) == 1:
            # segments[0] is a tuple (text, start, end)
            return next_level_fn(segments[0][0], segments[0][1])

        leaves = []
        for span, s, e in segments:
            if self.token_counter(span) > self.max_tokens:
                leaves.extend(next_level_fn(span, s))
            else:
                leaves.append(Chunk(s, e, span))
        return leaves

    def _pack(self, segments: List[Chunk], text: str) -> List[Chunk]:
        """Greedily combine consecutive leaf segments until token cap."""
        if not segments:
            return []
        packed, buf, buf_tokens = [], [], 0
        for seg in segments:
            t = self.token_counter(seg.text)
            if buf_tokens + t <= self.max_tokens:
                buf.append(seg)
                buf_tokens += t
            else:
                first, last = buf[0], buf[-1]
                packed.append(Chunk(first.start, last.end, text[first.start:last.end], buf))
                buf, buf_tokens = [seg], t
        if buf:
            first, last = buf[0], buf[-1]
            packed.append(Chunk(first.start, last.end, text[first.start:last.end], buf))
        return packed

    def _split_phrases(self, sentence: str, sent_start: int) -> List[Chunk]:
        if self.token_counter(sentence) <= self.max_tokens:
            return [Chunk(sent_start, sent_start + len(sentence), sentence)]

        return self._split_recursively(
            sentence, sent_start, self.phrase_re, self._split_words
        )

    def _split_sentences(self, paragraph: str, para_start: int, text: str) -> List[Chunk]:
        if self.token_counter(paragraph) <= self.max_tokens:
            return [Chunk(para_start, para_start + len(paragraph), paragraph)]

        leaves = self._split_recursively(
            paragraph, para_start, self.sentence_re, self._split_phrases
        )
        return self._pack(leaves, text)

    def chunk(self, text: str) -> List[Chunk]:
        """
        Return a *list of top-level Chunk objects* (usually the paragraphs).
        Each Chunk's .children holds the next-deeper level, down to leaves whose
        .text is guaranteed not to exceed *max_tokens*.
        """
        top_nodes: List[Chunk] = []
        for para, p_start, p_end in _segments_with_pos(self.paragraph_re, text, 0):
            if not para.strip():  # skip blank paras
                continue
            children = self._split_sentences(para, p_start, text)
            if len(children) == 1:
                # The whole paragraph fits into a single chunk (either because it was
                # small to begin with, or because its pieces were packed into one).
                top_nodes.append(children[0])
            else:
                # The paragraph was split into multiple chunks. Create a parent for them.
                top_nodes.append(Chunk(p_start, p_end, para, children))

        return top_nodes
