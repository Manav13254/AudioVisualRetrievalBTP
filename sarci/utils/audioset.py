"""
Helper to import Cnn14 from the vendored audioset_tagging_cnn repo.

Factored out of the old flat scripts (each one duplicated this sys.path
munging block). audioset_tagging_cnn/ lives at the project root and is not
a proper installable package, so we add its subfolders to sys.path once.
"""

import os
import sys
from typing import Optional


def load_cnn14_class(repo_root: Optional[str] = None):
    """Returns the Cnn14 class, importable only after audioset_tagging_cnn's
    pytorch/ and utils/ folders are on sys.path.

    repo_root: path to the project root that contains audioset_tagging_cnn/.
    Defaults to two levels up from this file (sarci/utils/ -> project root).
    """
    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    audioset_repo = os.path.join(repo_root, "audioset_tagging_cnn")
    for sub in ("", "utils", "pytorch"):
        path = os.path.join(audioset_repo, sub) if sub else audioset_repo
        if path not in sys.path:
            sys.path.append(path)

    from pytorch.models import Cnn14  # noqa: E402

    return Cnn14
