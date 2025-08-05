import pytest

from know import tokenizers
from know.settings import ProjectSettings


TOKENIZER_TEST_CASES = [
    # (input_string, expected_noop, expected_word, expected_coding)
    ("hello world", "hello world", "hello world", "hello world"),
    ("HelloWorld", "HelloWorld", "HelloWorld", "HelloWorld Hello World"),
    ("helloWorld", "helloWorld", "helloWorld", "helloWorld hello World"),
    ("hello_world", "hello_world", "hello world", "hello world"),
    ("foo-bar", "foo-bar", "foo bar", "foo bar"),
    ("foo123bar", "foo123bar", "foo123bar", "foo123bar foo 123 bar"),
    ("123foo", "123foo", "123foo", "123foo 123 foo"),
    ("foo123", "foo123", "foo123", "foo123 foo 123"),
    ("HTTPResponse", "HTTPResponse", "HTTPResponse", "HTTPResponse HTTP Response"),
    ("a-b_c.d", "a-b_c.d", "a b c d", "a b c d"),
    ("getHTML", "getHTML", "getHTML", "getHTML get HTML"),
    ("parseURL", "parseURL", "parseURL", "parseURL parse URL"),
    ("APIFactory", "APIFactory", "APIFactory", "APIFactory API Factory"),
    ("GL11_drawLine", "GL11_drawLine", "GL11 drawLine", "GL11 GL 11 drawLine draw Line"),
    (
        "def search(self, query: NodeSearchQuery) -> list[Node]:",
        "def search(self, query: NodeSearchQuery) -> list[Node]:",
        "def search self query NodeSearchQuery list Node",
        "def search self query NodeSearchQuery Node Search Query list Node",
    ),
]


def test_tokenizers():
    # a dummy settings object is needed for the tokenizer functions
    settings = ProjectSettings(project_name="test", repo_name="test")

    for src, expected_noop, expected_word, expected_coding in TOKENIZER_TEST_CASES:
        # test noop_tokenizer
        assert tokenizers.noop_tokenizer(src) == expected_noop

        # test word_tokenizer
        assert tokenizers.word_tokenizer(src) == expected_word

        # test coding_tokenizer
        assert tokenizers.code_tokenizer(src) == expected_coding

        # test auto_tokenizer (should be same as coding)
        assert tokenizers.auto_tokenizer(settings, src) == expected_coding
