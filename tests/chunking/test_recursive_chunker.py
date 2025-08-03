from know.chunking.recursive import RecursiveChunker

LONG_SENTENCE = (
    "This paragraph contains a single, extremely long sentence: it has commas, clauses, and colons "
    "that will require multiple fallback levels, because otherwise the chunk would be far beyond the "
    "token limit; therefore, we must observe how the algorithm behaves."
)
SECOND_LONG_PARA = (
    "It was broken into leaves (phrases, and words for one long phrase), which were then packed back\n"
    "into larger chunks under the token limit. This should result in 5 packed chunks for this paragraph."
)
TEST_TEXT = f"""Short paragraph.

{LONG_SENTENCE}

{SECOND_LONG_PARA}


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

    # Assert: The chunker should produce a flat list of 11 chunks.
    assert len(chunks) == 11

    # Check chunks against expected output
    assert chunks[0].text == (
        "Short paragraph.\n\nThis paragraph contains a single, extremely long sentence"
    )
    assert not chunks[0].children

    assert chunks[1].text == "it has commas, clauses"
    assert not chunks[1].children

    assert chunks[2].text == "and colons that will require multiple fallback levels"
    assert not chunks[2].children

    assert chunks[3].text == "because otherwise the chunk would be far beyond the token"
    assert not chunks[3].children

    assert chunks[4].text == "limit; therefore, we must observe how the algorithm behaves."
    assert not chunks[4].children

    assert chunks[5].text == "It was broken into leaves (phrases"
    assert not chunks[5].children

    assert chunks[6].text == "and words for one long phrase"
    assert not chunks[6].children

    assert chunks[7].text == (
        "which were then packed back\ninto larger chunks under the"
    )
    assert not chunks[7].children

    assert chunks[8].text == "token limit."
    assert not chunks[8].children

    assert (
        chunks[9].text == "This should result in 5 packed chunks for this paragraph."
    )
    assert not chunks[9].children

    assert chunks[10].text == "Last one."
    assert not chunks[10].children
