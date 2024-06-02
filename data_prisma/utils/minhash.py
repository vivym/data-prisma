import hashlib
import struct
from dataclasses import dataclass
from itertools import tee
from typing import Any, Callable

import numpy as np
from scipy.integrate import quad as integrate

SEED = 42
RNG = np.random.RandomState(SEED)

# 64 bit config is backwards compatibility mode.
# it uses 64 bit types but almost entirely 32bit data, except for one mersenne prime 2^61
# why legacy implementations used mersenne primes for modulo:
# https://en.wikipedia.org/wiki/Universal_hashing#Hashing_strings
HASH_CONFIG: dict[int, tuple[type[np.uint64], Any, Any]] = {
    64: (np.uint64, np.uint32((1 << 32) - 1), np.uint64((1 << 61) - 1)),
    # 32, 16 bit config does not use a mersenne prime.
    # The original reason for using mersenne prime was speed.
    # Testing reveals, there is no benefit to using a 2^61 mersenne prime for division
    32: (np.uint32, np.uint32((1 << 32) - 1), np.uint32((1 << 32) - 5)),
    16: (np.uint16, np.uint16((1 << 16) - 1), np.uint16((1 << 16) - 15)),
}


def sha1_hash(data: bytes, d: int = 32) -> int:
    if d == 32:
        return struct.unpack(
            "<I", hashlib.sha1(data, usedforsecurity=False).digest()[:4]
        )[0]
    if d == 64:
        return struct.unpack(
            "<Q", hashlib.sha1(data, usedforsecurity=False).digest()[:8]
        )[0]
    # struct is faster but does not support arbitrary bit lengths
    return int.from_bytes(
        hashlib.sha1(data, usedforsecurity=False).digest()[: d // 8],
        byteorder="little",
    )


@dataclass
class MinHashConfig:
    num_perm: int

    ngram_size: int

    min_length: int

    hash_ranges: list[tuple[int, int]]

    permutations: np.ndarray

    hash_func: Callable[[bytes], int]

    dtype: type[np.uint64]

    max_hash: np.uint

    modulo_prime: np.uint

    bands: int

    rows: int


def optimal_param(
    threshold: float,
    num_perm: int,
    false_positive_weight: float = 0.5,
    false_negative_weight: float = 0.5,
):
    """
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative, taken from datasketch.

    You can also refer to the interactive demo at https://huggingface.co/spaces/bigcode/near-deduplication.

    Parameters
    ----------
    threshold : float
        The threshold for similarity.
    num_perm : int
        The number of permutations.
    false_positive_weight : float
        The weight of false positive.
    false_negative_weight : float
        The weight of false negative.

    Returns
    -------
    Tuple[int, int]
        The optimal `b` (bands) and `r` (rows) parameters.

    Examples
    --------
    >>> optimal_param(0.75, 256)
    (21, 12)
    >>> optimal_param(0.75, 256, 0.1, 0.9)
    (28, 9)
    """

    def false_positive_area(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def proba(s):
            return 1 - (1 - s ** float(r)) ** float(b)

        a, _ = integrate(proba, 0.0, threshold)
        return a

    def false_negative_area(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def proba(s):
            return 1 - (1 - (1 - s ** float(r)) ** float(b))

        a, _ = integrate(proba, threshold, 1.0)
        return a

    min_error = float("inf")
    opt = (0, 0)

    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = false_positive_area(threshold, b, r)
            fn = false_negative_area(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)

    return opt


def get_minhash_config(
    num_perm: int = 256,
    ngram_size: int = 3,
    min_length: int = 5,
    hash_func: str = "sha1",
    hash_bits: int = 64,
    threshold: float = 0.7,
) -> MinHashConfig:
    # defaults to backwards compatible HASH_BITS = 64, which is np.uint64 dtypes with 32bit hashes
    dtype, max_hash, modulo_prime = HASH_CONFIG.get(hash_bits, HASH_CONFIG[64])

    match hash_func:
        case "sha1":
            def hash_func(data: bytes) -> int:
                return sha1_hash(data, d=min(hash_bits, 32))

    # Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    # of probabilities of false positive and false negative, taken from datasketch.
    # You can also refer to the interactive demo at https://huggingface.co/spaces/bigcode/near-deduplication.
    # The following assumes a "perfect hash". using 16 bit hashes might challenge this assumption
    # lower precision dtype will cause more collisions, so higher false_positives and less false negatives.
    # Both effects move the result towards more documents being considered duplicates.
    bands, rows = optimal_param(
        threshold,
        num_perm,
        false_positive_weight=0.5,
        false_negative_weight=0.5,
    )

    hash_ranges = [(i * rows, (i + 1) * rows) for i in range(bands)]

    # for minhash, we need to make a lot of hashes(=num_perms).
    # In many previous implementations, this is achieved through a method described in
    # `Universal classes of hash functions` https://doi.org/10.1016/0022-0000(79)90044-8
    # There we start with a know good hash x (=hash_func) and permutate it as the following:
    # `new_hash = (a * x + b) mod prime mod max_hash` we need one a (!=0), b pair per new hash
    # the following produces these a, b pairs
    permutations: tuple[np.ndarray, np.ndarray] = (
        RNG.randint(
            1, modulo_prime, size=(num_perm,), dtype=dtype
        ),  # a is a multiplier so should not be 0
        RNG.randint(0, modulo_prime, size=(num_perm,), dtype=dtype),  # b
    )

    return MinHashConfig(
        num_perm=num_perm,
        ngram_size=ngram_size,
        min_length=min_length,
        hash_ranges=hash_ranges,
        permutations=permutations,
        hash_func=hash_func,
        dtype=dtype,
        max_hash=max_hash,
        modulo_prime=modulo_prime,
        bands=bands,
        rows=rows,
    )


def ngrams(sequence: list[int], n: int, min_length: int = 5) -> list[tuple[int, ...]]:
    if len(sequence) < min_length:
        return []

    if len(sequence) < n:
        return [tuple(sequence)]


    iterables = tee(iter(sequence), n)

    for i, it in enumerate(iterables):
        for _ in range(i):
            next(it, None)

    return list(zip(*iterables))


def compute_minhash(tokens: list[str | int], *, config: MinHashConfig) -> list[bytes]:
    # a, b are each np.ndarray arrays containing {num_perm} pairs of random numbers used for building new hashes
    # the formula is a * x(base hash of each shingle) + b
    a, b = config.permutations

    tokens: set[bytes] = {
        struct.pack("I" * len(t), *t)
        for t in ngrams(tokens, config.ngram_size, min_length=config.min_length)
    }

    hash_values: np.ndarray = np.array(
        [config.hash_func(token) for token in tokens], dtype=config.dtype
    ).reshape(len(tokens), 1)

    # Permute the hash values to produce new universal hashes
    # Element-wise multiplication with 'hashvalues' and a (non 0 random value) and then adding b
    # Then, take modulo 'MODULO_PRIME' and bitwise_and with 'MAX_HASH' to keep only the necessary bits.
    hash_values = (hash_values * a + b) % config.modulo_prime & config.max_hash

    # this part is where the name "min" of minhash comes from
    # this stacks all the hashes and then takes the minimum from each column
    masks: np.ndarray = np.full(
        shape=config.num_perm, dtype=config.dtype, fill_value=config.max_hash
    )
    hash_values = np.vstack([hash_values, masks]).min(axis=0)

    # Originally, byteswap was done for speed. Testing show it has a negligible impact
    # keeping for backward compatibility, even though theoretically and empirically
    # it doesnt matter if it is there or not. github.com/ekzhu/datasketch/issues/114
    hs: list[bytes] = [
        bytes(hash_values[start:end].byteswap().data) for start, end in config.hash_ranges
    ]

    return hs
