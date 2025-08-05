import re
from typing import Optional, Set

from know.models import ProgrammingLanguage
from know.parsers import CodeParserRegistry
from know.settings import ProjectSettings, TokenizerType


_SPLIT_CAMEL_PASCAL = re.compile(r"""
    (?<=[A-Z])(?=[A-Z][a-z]) |  # ABCDef -> ABC | Def
    (?<=[a-z0-9])(?=[A-Z])   |  # fooBar, 9Lives -> foo | Bar, 9 | Lives
    (?<=[A-Za-z])(?=\d)      |  # foo123 -> foo | 123
    (?<=\d)(?=[A-Za-z])         # 123foo -> 123 | foo
""", re.X)

_EXTRACT_WORDS = re.compile(r"[A-Za-z0-9]+")


def noop_tokenizer(src: str, stop_words: Optional[Set[str]] = None) -> str:
    return src


def code_tokenizer(src: str, stop_words: Optional[Set[str]] = None) -> str:
    separated = _SPLIT_CAMEL_PASCAL.sub(" ", src)
    tokens = _EXTRACT_WORDS.findall(separated)
    if stop_words:
        tokens = [token for token in tokens if token.lower() not in stop_words]
    return ' '.join(tokens)


def word_tokenizer(src: str, stop_words: Optional[Set[str]] = None) -> str:
    tokens = _EXTRACT_WORDS.findall(src)
    if stop_words:
        tokens = [token for token in tokens if token.lower() not in stop_words]
    return ' '.join(tokens)


def auto_tokenizer(s: ProjectSettings, src: str, stop_words: Optional[Set[str]] = None) -> str:
    if s.tokenizer.default == TokenizerType.NOOP:
        return noop_tokenizer(src, stop_words=stop_words)
    if s.tokenizer.default == TokenizerType.WORD:
        return word_tokenizer(src, stop_words=stop_words)
    return code_tokenizer(src, stop_words=stop_words)
def search_tokenizer(s: ProjectSettings, lang: ProgrammingLanguage, src: str) -> str:
    stop_words: Optional[Set[str]] = None
    helper = CodeParserRegistry.get_helper(lang)
    if helper:
        stop_words = helper.get_common_syntax_words()

    return auto_tokenizer(s, src, stop_words=stop_words)
