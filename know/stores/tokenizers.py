import re
from know.settings import ProjectSettings, TokenizerType


_SPLIT_CAMEL_PASCAL = re.compile(r"""
    (?<=[A-Z])(?=[A-Z][a-z]) |  # ABCDef -> ABC | Def
    (?<=[a-z0-9])(?=[A-Z])   |  # fooBar, 9Lives -> foo | Bar, 9 | Lives
    (?<=[A-Za-z])(?=\d)      |  # foo123 -> foo | 123
    (?<=\d)(?=[A-Za-z])         # 123foo -> 123 | foo
""", re.X)

_EXTRACT_WORDS = re.compile(r"[A-Za-z0-9]+")


def noop_tokenizer(src: str) -> str:
    return src


def code_tokenizer(src: str) -> str:
    separated = _SPLIT_CAMEL_PASCAL.sub(" ", src)
    return ' '.join(_EXTRACT_WORDS.findall(separated))


def word_tokenizer(src: str) -> str:
    return ' '.join(_EXTRACT_WORDS.findall(src))


def auto_tokenizer(s: ProjectSettings, src: str) -> str:
    if s.tokenizer.default == TokenizerType.NOOP:
        return noop_tokenizer(src)
    if s.tokenizer.default == TokenizerType.WORD:
        return word_tokenizer(src)
    return code_tokenizer(src)
