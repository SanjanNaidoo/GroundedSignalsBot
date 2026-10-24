"""Content digests — the comparability guarantee.

Every run manifest records a digest of the item set and of the course config, so
two runs can be *proved* to have seen the same questions under the same settings
rather than merely asserted to have done. That claim is about content, so the
hash is taken over line-ending-normalised bytes.

Without that normalisation the claim is wrong in a way that is hard to see: a
clone made on Windows with `core.autocrlf=true` has CRLF in its working tree, so
`bench/eee4114f.jsonl` is 47 bytes longer than the identical file on macOS and
digests differently. One frozen benchmark then looks like two incomparable ones,
and `ym07 grade` refuses to put the runs side by side — which is exactly the
comparison the open-weights arm exists to make.
"""

from __future__ import annotations

import hashlib


def content_digest(raw_bytes: bytes) -> str:
    """Short SHA-256 of `raw_bytes`, insensitive to line-ending style.

    CRLF and lone CR are folded to LF first. Files that are already LF-only —
    which is every file as committed — hash exactly as they did before this
    function existed, so the digests recorded in manifests written earlier stay
    valid and comparable.
    """
    normalised = raw_bytes.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalised).hexdigest()[:16]
