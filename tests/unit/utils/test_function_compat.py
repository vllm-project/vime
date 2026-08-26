import pytest

from vime.utils.function_compat import call_with_optional_keyword


@pytest.mark.parametrize("supports_keyword", [False, True])
def test_call_with_optional_keyword(supports_keyword: bool) -> None:
    calls = []

    if supports_keyword:

        def target(self, is_checkpoint_format: bool = True) -> None:
            calls.append((self, is_checkpoint_format))

    else:

        def target(self) -> None:
            calls.append((self, "without_keyword"))

    receiver = object()
    call_with_optional_keyword(
        target,
        receiver,
        keyword="is_checkpoint_format",
        value=False,
    )

    assert calls == [(receiver, False if supports_keyword else "without_keyword")]
