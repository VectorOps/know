from know.chunking.recursive import RecursiveChunker

TEST_TEXT = f"""Short paragraph.

This paragraph contains a single, extremely long sentence: it has commas, clauses, and colons
that will require multiple fallback levels, because otherwise the chunk would be far beyond the
token limit; therefore, we must observe how the algorithm behaves.

It was broken into leaves (phrases, and words for one long phrase), which were then packed back
into larger chunks under the token limit. This should result in 5 packed chunks for this paragraph.

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

    # Assert: The chunker should produce a flat list of 7 chunks.
    expected_chunks = [
        "Short paragraph.\n\nThis paragraph contains a single, extremely long sentence:",
        " it has commas, clauses, and colons\nthat will require multiple fallback levels,",
        " because otherwise the chunk would be far beyond the\ntoken",
        " limit; therefore, we must observe how the algorithm behaves.\n\n",
        "It was broken into leaves (phrases, and words for one long phrase),",
        " which were then packed back\ninto larger chunks under the token limit.",
        " This should result in 5 packed chunks for this paragraph.\n\nLast one.",
    ]
    actual_chunks = [c.text for c in chunks]
    assert actual_chunks == expected_chunks


def test_recursive_chunker_with_min_tokens():
    """
    Test that the recursive chunker with min_tokens set does not pack
    segments that are over the minimum token count.
    """
    # Use a small max_tokens to trigger all fallback levels. The default
    # token_counter splits by whitespace. min_tokens is set to 9.
    chunker = RecursiveChunker(max_tokens=10, min_tokens=9)

    # Execute
    chunks = chunker.chunk(TEST_TEXT)

    # Assert: The chunker should produce a flat list of 11 chunks.
    # Chunks with >= 9 tokens should not be packed with their neighbors.
    expected_chunks = [
        "Short paragraph.\n\n",
        "This paragraph contains a single, extremely long sentence:",
        " it has commas, clauses,",
        " and colons\nthat will require multiple fallback levels,",
        " because otherwise the chunk would be far beyond the\ntoken",
        " limit; therefore, we must observe how the algorithm behaves.\n\n",
        "It was broken into leaves (phrases, and words for one long phrase),",
        " which were then packed back\ninto larger chunks under the",
        " token limit.",
        " This should result in 5 packed chunks for this paragraph.\n\n",
        "Last one.",
    ]
    actual_chunks = [c.text for c in chunks]
    assert actual_chunks == expected_chunks
