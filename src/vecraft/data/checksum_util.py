import hashlib
import json
import zlib
from typing import Any, Dict, Union, List, Callable

# Type for checksum function: takes bytes -> hex string
ChecksumFunc = Callable[[bytes], str]


# Predefined checksum functions
_PREDEFINED_CHECKSUMS: Dict[str, ChecksumFunc] = {
    'crc32': lambda b: format(zlib.crc32(b) & 0xFFFFFFFF, '08x'),
    'md5': lambda b: hashlib.md5(b).hexdigest(),
    'sha1': lambda b: hashlib.sha1(b).hexdigest(),
    'sha256': lambda b: hashlib.sha256(b).hexdigest(),
}


def get_checksum_func(
    algorithm: Union[str, ChecksumFunc]
) -> ChecksumFunc:
    """
    Resolve an algorithm name or accept a custom function.
    """
    if callable(algorithm):
        return algorithm
    alg = algorithm.lower()
    if alg in _PREDEFINED_CHECKSUMS:
        return _PREDEFINED_CHECKSUMS[alg]
    # fallback to hashlib
    try:
        def _fn(b: bytes, name=alg):
            h = hashlib.new(name)
            h.update(b)
            return h.hexdigest()
        _ = _fn(b"test")
        return _fn
    except Exception:
        raise ValueError(f"Unsupported checksum algorithm: {algorithm}")


def _prepare_field_bytes(value: Any) -> bytes:
    """
    Prepare any Python value into bytes for checksum.
    Uses JSON (sorted keys) when possible, else repr().
    """
    try:
        return json.dumps(value, sort_keys=True).encode('utf-8')
    except Exception:
        return repr(value).encode('utf-8')


def _concat_bytes(components: List[bytes]) -> bytes:
    """
    Concatenate multiple byte components deterministically.
    """
    return b"".join(components)