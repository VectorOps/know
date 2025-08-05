import re
from know.settings import ProjectSettings


_SPLIT_CAMEL_PASCAL = re.compile(r"""
    (?<=[A-Z])(?=[A-Z][a-z]) |  # ABCDef -> ABC | Def
    (?<=[a-z0-9])(?=[A-Z])   |  # fooBar, 9Lives -> foo | Bar, 9 | Lives
    (?<=[A-Za-z])(?=\d)      |  # foo123 -> foo | 123
    (?<=\d)(?=[A-Za-z])         # 123foo -> 123 | foo
""", re.X)

_EXTRACT_WORDS = re.compile(r"[A-Za-z0-9]+")


def noop_tokenizer(s: ProjectSettings, src: str) -> str:
    return src


def coding_tokenizer(s: ProjectSettings, src: str) -> str:
    separated = _SPLIT_CAMEL_PASCAL.sub(" ", src)
    return ' '.join(_EXTRACT_WORDS.findall(separated))


def word_tokenizer(s: ProjectSettings, src: str) -> str:
    return ' '.join(_EXTRACT_WORDS.findall(src))


def auto_tokenizer(s: ProjectSettings, src: str) -> str:
    return coding_tokenizer(s, src)
