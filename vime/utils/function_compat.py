import inspect
from collections.abc import Callable
from typing import Any


def call_with_optional_keyword(
    func: Callable[..., Any], *args: Any, keyword: str, value: Any
) -> Any:
    """Call ``func`` with a keyword only when its installed version supports it."""
    if keyword in inspect.signature(func).parameters:
        return func(*args, **{keyword: value})
    return func(*args)
