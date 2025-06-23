from typing import List

def quantize_embeddings(
    vector: List[float],
    *,
    num_bits: int = 8,
) -> List[int]:
    """
    Uniform, symmetric min-/max-based quantization of one embedding vector.

    Parameters
    ----------
    vector:     embedding values as floats.
    num_bits:   bit-width (1-16).  Default: 8.

    Returns
    -------
    List[int]   quantized values in range  [−2**(b−1), 2**(b−1)−1].
    """
    if not 1 <= num_bits <= 16:
        raise ValueError("num_bits must be between 1 and 16")

    qmax = (1 << (num_bits - 1)) - 1
    qmin = -1 << (num_bits - 1)

    max_abs = max(abs(v) for v in vector) or 1.0
    scale   = qmax / max_abs

    return [max(min(int(round(v * scale)), qmax), qmin) for v in vector]
