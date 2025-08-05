import pytest

from know.stores import tokenizers
from know.settings import ProjectSettings


TOKENIZER_TEST_CASES = [
    # (input_string, expected_noop, expected_word, expected_coding)
    ("hello world", "hello world", "hello world", "hello world"),
    ("HelloWorld", "HelloWorld", "HelloWorld", "Hello World"),
    ("helloWorld", "helloWorld", "helloWorld", "hello World"),
    ("hello_world", "hello_world", "hello world", "hello world"),
    ("foo-bar", "foo-bar", "foo bar", "foo bar"),
    ("foo123bar", "foo123bar", "foo123bar", "foo 123 bar"),
    ("123foo", "123foo", "123foo", "123 foo"),
    ("foo123", "foo123", "foo123", "foo 123"),
    ("HTTPResponse", "HTTPResponse", "HTTPResponse", "HTTP Response"),
    ("a-b_c.d", "a-b_c.d", "a b c d", "a b c d"),
    ("getHTML", "getHTML", "getHTML", "get HTML"),
    ("parseURL", "parseURL", "parseURL", "parse URL"),
    ("APIFactory", "APIFactory", "APIFactory", "API Factory"),
    ("GL11_drawLine", "GL11_drawLine", "GL11 drawLine", "GL 11 draw Line"),
    (
        "def search(self, query: NodeSearchQuery) -> list[Node]:",
        "def search(self, query: NodeSearchQuery) -> list[Node]:",
        "def search self query NodeSearchQuery list Node",
        "def search self query Node Search Query list Node",
    ),
]


@pytest.mark.parametrize("src, expected_noop, expected_word, expected_coding", TOKENIZER_TEST_CASES)
def test_tokenizers(src, expected_noop, expected_word, expected_coding):
    # a dummy settings object is needed for the tokenizer functions
    settings = ProjectSettings(project_name="test", repo_name="test")

    # test noop_tokenizer
    assert tokenizers.noop_tokenizer(settings, src) == expected_noop

    # test word_tokenizer
    assert tokenizers.word_tokenizer(settings, src) == expected_word

    # test coding_tokenizer
    assert tokenizers.coding_tokenizer(settings, src) == expected_coding

    # test auto_tokenizer (should be same as coding)
    assert tokenizers.auto_tokenizer(settings, src) == expected_coding
