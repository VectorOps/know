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
    long_para_chunk = chunks[1]
    assert long_para_chunk.text == LONG_SENTENCE

    # It was broken into leaves (phrases, and words for one long phrase),
    # which were then packed back into larger chunks under the token limit.
    # This should result in 5 packed chunks for this paragraph.
    assert len(long_para_chunk.children) == 5

    packed_chunks = long_para_chunk.children

    # Packed chunk 1: from a single phrase leaf
    assert packed_chunks[0].text == "This paragraph contains a single, extremely long sentence"
    assert len(packed_chunks[0].children) == 1
    assert packed_chunks[0].children[0].text == "This paragraph contains a single, extremely long sentence"
    assert not packed_chunks[0].children[0].children

    # Packed chunk 2: from two phrase leaves
    assert packed_chunks[1].text == "it has commas, clauses"
    assert len(packed_chunks[1].children) == 2
    assert packed_chunks[1].children[0].text == "it has commas"
    assert not packed_chunks[1].children[0].children
    assert packed_chunks[1].children[1].text == "clauses"
    assert not packed_chunks[1].children[1].children

    # Packed chunk 3: from a single phrase leaf
    assert packed_chunks[2].text == "and colons that will require multiple fallback levels"
    assert len(packed_chunks[2].children) == 1
    assert packed_chunks[2].children[0].text == "and colons that will require multiple fallback levels"
    assert not packed_chunks[2].children[0].children

    # Packed chunk 4: from a single word-split leaf
    # The original phrase "because otherwise..." (11 tokens) was split into two word-chunks.
    # The first word-chunk (10 tokens) is too big to be packed with anything else.
    assert packed_chunks[3].text == "because otherwise the chunk would be far beyond the token"
    assert len(packed_chunks[3].children) == 1
    assert packed_chunks[3].children[0].text == "because otherwise the chunk would be far beyond the token"
    assert not packed_chunks[3].children[0].children

    # Packed chunk 5: from the second word-split leaf and two phrase leaves
    assert packed_chunks[4].text == "limit; therefore, we must observe how the algorithm behaves."
    assert len(packed_chunks[4].children) == 3
    assert packed_chunks[4].children[0].text == "limit"
    assert not packed_chunks[4].children[0].children
    assert packed_chunks[4].children[1].text == "therefore"
    assert not packed_chunks[4].children[1].children
    assert packed_chunks[4].children[2].text == "we must observe how the algorithm behaves."
    assert not packed_chunks[4].children[2].children
