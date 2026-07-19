from __future__ import annotations

import importlib
import importlib.util
from typing import Any


def resilient_find_spec(module_name: str, *, attempts: int = 3) -> tuple[Any | None, str | None]:
    """Find a module while tolerating transient Windows directory-cache errors."""
    last_error: OSError | None = None
    for _attempt in range(max(1, attempts)):
        try:
            return importlib.util.find_spec(module_name), None
        except ModuleNotFoundError:
            return None, None
        except OSError as exc:
            last_error = exc
            importlib.invalidate_caches()
    if last_error is None:
        return None, None
    return None, f"{type(last_error).__name__}: {last_error}"
