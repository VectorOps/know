from know.chunking.recursive import RecursiveChunker

LONG_SENTENCE = (
    "This paragraph contains a single, extremely long sentence: it has commas, clauses, and colons "
    "that will require multiple fallback levels, because otherwise the chunk would be far beyond the "
    "token limit; therefore, we must observe how the algorithm behaves."
)
TEST_TEXT = f"""Short paragraph.

{LONG_SENTENCE}

Last one."""


def test_recursive_chunker_fallback_levels():
    """
    Test that the recursive chunker correctly falls back to smaller chunking
    strategies (paragraph -> sentence -> phrase -> word) when a chunk exceeds
    max_tokens.
    """
    # Use a small max_tokens to trigger all fallback levels. The default
    # token_counter splits by whitespace.
    chunker = RecursiveChunker(max_tokens=10)

    # Execute
    chunks = chunker.chunk(TEST_TEXT)

    # Assert: The output should be three top-level chunks for the three paragraphs.
    assert len(chunks) == 3

    # The first and last paragraphs are short and should be leaf chunks.
    assert chunks[0].text == "Short paragraph."
    assert not chunks[0].children

    assert chunks[2].text == "Last one."
    assert not chunks[2].children

    # The middle paragraph is long and should be broken down.
    # It's one sentence, so the top-level chunk for it will be a sentence-chunk...
    long_para_chunk = chunks[1]
    assert long_para_chunk.text == LONG_SENTENCE
    # ...which is broken down into phrases.
    assert len(long_para_chunk.children) == 7

    phrase_chunks = long_para_chunk.children

    assert phrase_chunks[0].text == "This paragraph contains a single, extremely long sentence"
    assert not phrase_chunks[0].children

    assert phrase_chunks[1].text == "it has commas"
    assert not phrase_chunks[1].children

    assert phrase_chunks[2].text == "clauses"
    assert not phrase_chunks[2].children

    assert phrase_chunks[3].text == "and colons that will require multiple fallback levels"
    assert not phrase_chunks[3].children

    # This phrase is too long and must be chunked by words.
    long_phrase_chunk = phrase_chunks[4]
    assert long_phrase_chunk.text == "because otherwise the chunk would be far beyond the token limit"
    assert len(long_phrase_chunk.children) == 2

    word_chunks = long_phrase_chunk.children
    assert word_chunks[0].text == "because otherwise the chunk would be far beyond the token"
    assert not word_chunks[0].children
    assert word_chunks[1].text == "limit"
    assert not word_chunks[1].children

    assert phrase_chunks[5].text == "therefore"
    assert not phrase_chunks[5].children

    assert phrase_chunks[6].text == "we must observe how the algorithm behaves."
    assert not phrase_chunks[6].children
