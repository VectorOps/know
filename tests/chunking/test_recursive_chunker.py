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

    # Assert: The output should be four top-level chunks for the four paragraphs.
    assert len(chunks) == 4

    # The first paragraph is short and should be a single leaf chunk.
    assert chunks[0].text == "Short paragraph."
    assert not chunks[0].children

    # The second paragraph is long and should be broken down.
    long_para_chunk = chunks[1]
    assert long_para_chunk.text == LONG_SENTENCE

    # It was broken into leaves (phrases, and words for one long phrase),
    # which were then packed back into larger chunks under the token limit.
    # This should result in 5 packed chunks for this paragraph.
    assert len(long_para_chunk.children) == 5

    packed_chunks = long_para_chunk.children

    # Packed chunk 1: from a single phrase leaf.
    assert packed_chunks[0].text == "This paragraph contains a single, extremely long sentence"
    assert not packed_chunks[0].children

    # Packed chunk 2: from two phrase leaves that were packed together.
    assert packed_chunks[1].text == "it has commas, clauses"
    assert not packed_chunks[1].children

    # Packed chunk 3: from a single phrase leaf.
    assert packed_chunks[2].text == "and colons that will require multiple fallback levels"
    assert not packed_chunks[2].children

    # Packed chunk 4: from one over-sized phrase that was split into words and repacked.
    # The original phrase "because otherwise..." (11 tokens) was split into two word-chunks.
    # The first word-chunk (10 tokens) becomes a packed chunk by itself.
    assert packed_chunks[3].text == "because otherwise the chunk would be far beyond the token"
    assert not packed_chunks[3].children

    # Packed chunk 5: from the second word-chunk and two more phrase-chunks.
    assert packed_chunks[4].text == "limit; therefore, we must observe how the algorithm behaves."
    assert not packed_chunks[4].children

    # The third paragraph is also long and should be broken down.
    second_long_para_chunk = chunks[2]
    assert second_long_para_chunk.text == SECOND_LONG_PARA
    assert len(second_long_para_chunk.children) == 5
    packed_chunks_2 = second_long_para_chunk.children

    assert packed_chunks_2[0].text == "It was broken into leaves (phrases"
    assert not packed_chunks_2[0].children
    assert packed_chunks_2[1].text == "and words for one long phrase)"
    assert not packed_chunks_2[1].children
    assert packed_chunks_2[2].text == (
        "which were then packed back\n" "into larger chunks under the"
    )
    assert not packed_chunks_2[2].children
    assert packed_chunks_2[3].text == "token limit."
    assert not packed_chunks_2[3].children
    assert (
        packed_chunks_2[4].text
        == "This should result in 5 packed chunks for this paragraph."
    )
    assert not packed_chunks_2[4].children

    # The last paragraph is short and should be a single leaf chunk.
    assert chunks[3].text == "Last one."
    assert not chunks[3].children
