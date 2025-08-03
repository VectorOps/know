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
    expected_chunks = [
        "Short paragraph.\n\nThis paragraph contains a single, extremely long sentence",
        "it has commas, clauses",
        "and colons that will require multiple fallback levels",
        "because otherwise the chunk would be far beyond the token",
        "limit; therefore, we must observe how the algorithm behaves.",
        "It was broken into leaves (phrases",
        "and words for one long phrase",
        "which were then packed back\ninto larger chunks under the",
        "token limit.",
        "This should result in 5 packed chunks for this paragraph.",
        "Last one.",
    ]
    actual_chunks = [c.text for c in chunks]
    assert actual_chunks == expected_chunks
