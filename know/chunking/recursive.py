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

    def _split_recursively(
        self, text_to_split: str, offset: int, full_text: str, regex: Pattern, next_level_fn: Callable
    ) -> List[Chunk]:
        """
        Generic helper to split text by a regex, recursively call the next-level splitter,
        and then pack the results.
        """
        # Optimization: if the whole text is small enough, no need to split.
        if self.token_counter(text_to_split) <= self.max_tokens:
            return [Chunk(offset, offset + len(text_to_split), text_to_split)]

        segments = list(_segments_with_pos(regex, text_to_split, offset))

        # If the regex doesn't split the text, pass the whole text to the next level.
        if len(segments) == 1:
            # segments[0] is a tuple (text, start, end)
            return next_level_fn(segments[0][0], segments[0][1], full_text)

        leaves = []
        for span, s, e in segments:
            if not span.strip():  # filter empty/whitespace-only segments
                continue
            if self.token_counter(span) > self.max_tokens:
                leaves.extend(next_level_fn(span, s, full_text))
            else:
                leaves.append(Chunk(s, e, span))
        return self._pack(leaves, full_text)

    def _final_fallback(self, text: str, start: int, full_text: str) -> List[Chunk]:
        """Base case for recursion: text that can't be split further."""
        # This might be an over-long word. Just create a chunk for it.
        return [Chunk(start, start + len(text), text)]

    def _split_words(self, text: str, start: int, full_text: str) -> List[Chunk]:
        """Splits by words, then packs them."""
        return self._split_recursively(
            text, start, full_text, self.word_re, self._final_fallback
        )

    def _split_phrases(self, sentence: str, sent_start: int, full_text: str) -> List[Chunk]:
        """Splits by phrases, then packs them."""
        return self._split_recursively(
            sentence, sent_start, full_text, self.phrase_re, self._split_words
        )

    def _split_sentences(self, paragraph: str, para_start: int, full_text: str) -> List[Chunk]:
        """Splits by sentences, then packs them."""
        return self._split_recursively(
            paragraph, para_start, full_text, self.sentence_re, self._split_phrases
        )

    def chunk(self, text: str) -> List[Chunk]:
        """
        Return a *list of top-level Chunk objects* (usually the paragraphs).
        Each Chunk's .children holds the next-deeper level, down to leaves whose
        .text is guaranteed not to exceed *max_tokens*.
        """
        if not text.strip():
            return []
        return self._split_recursively(
            text, 0, text, self.paragraph_re, self._split_sentences
        )
