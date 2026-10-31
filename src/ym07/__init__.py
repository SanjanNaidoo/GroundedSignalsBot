"""YM-07 — course-specific tutor for EEE4114F.

This package is the *measuring instrument*, not the tutor. It runs a frozen set
of benchmark items against each experimental condition, records raw outputs, and
supports blind human grading. See protocols/evaluation.md.
"""

import os as _os

# Set before anything in this package can import torch, because the allocator reads it
# once at import and ignores it afterwards.
#
# It used to be set inside stress.LocalRunner, which runs after `build_retriever` has
# already pulled torch in for the dense half of the retriever, so it never took effect.
# Moving it here makes the setting real - but on Windows it is still inert: torch prints
# "expandable_segments not supported on this platform" and carries on with the default
# allocator. It is kept because the harness is meant to run on Linux too, where it does
# work, and because a setting that silently does nothing in the place it was written is
# worse than one that does nothing somewhere visible.
#
# The out-of-memory failures it was reached for were not fixed by this. They were eased
# by raising --cap-gb, which is a different lever, and the two were changed together in
# one retry - so the recovery of 7 answers belongs to the cap, not to this line.
_os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

__version__ = "0.1.0"
