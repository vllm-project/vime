"""CPU unit tests for ``vime.utils.data._build_messages``.

``--multimodal-keys`` maps a media type name to the dataset column holding that
media, and the prompt marks insertion points with the type's placeholder
(``<image>``, ``<video>``, ``<audio>``). ``_build_messages`` rewrites a plain
string prompt into the list-of-content-dicts form the VLM chat templates and
``process_vision_info`` expect.

The cases below pin down what happens at the edges of that rewrite: a row of a
multimodal dataset that carries no media at all, a row that is already in
list-of-dicts form, and a ``--multimodal-keys`` mapping that names a type vime
does not support.
"""

from __future__ import annotations

import pytest

from vime.utils.data import _build_messages


NUM_GPUS = 0

IMAGE_KEYS = {"image": "images"}


def _row(prompt, **columns):
    return {"text": prompt, **columns}


def _content(messages):
    assert len(messages) == 1, f"expected a single message, got {len(messages)}"
    return messages[0]["content"]


@pytest.mark.unit
def test_media_placeholders_are_replaced():
    messages = _build_messages(_row("What is in <image>?", images=["a.png"]), "text", True, IMAGE_KEYS)

    assert _content(messages) == [
        {"type": "text", "text": "What is in "},
        {"type": "image", "image": "a.png"},
        {"type": "text", "text": "?"},
    ]


@pytest.mark.unit
def test_media_entries_may_be_rich_dicts():
    item = {"type": "image", "image": "a.png", "max_pixels": 50176}
    messages = _build_messages(_row("<image> here", images=[item]), "text", True, IMAGE_KEYS)

    assert _content(messages) == [item, {"type": "text", "text": " here"}]


@pytest.mark.unit
def test_single_media_entry_as_string_or_dict_is_wrapped():
    # Test single string
    messages = _build_messages(_row("What is in <image>?", images="a.png"), "text", True, IMAGE_KEYS)
    assert _content(messages) == [
        {"type": "text", "text": "What is in "},
        {"type": "image", "image": "a.png"},
        {"type": "text", "text": "?"},
    ]

    # Test single dict
    item = {"type": "image", "image": "a.png", "max_pixels": 50176}
    messages = _build_messages(_row("<image> here", images=item), "text", True, IMAGE_KEYS)
    assert _content(messages) == [item, {"type": "text", "text": " here"}]


@pytest.mark.unit
def test_row_without_media_keeps_its_prompt_intact():
    """A mixed dataset has rows with no media; those prompts must stay intact.

    Building the split pattern from an empty placeholder set yields ``"()"``,
    which matches the empty string at every position, so the prompt used to come
    back as one ``{"type": "text"}`` dict per character.
    """
    prompt = "Describe this image."
    messages = _build_messages(_row(prompt, images=None), "text", True, IMAGE_KEYS)

    # Same representation a text-only dataset gets, i.e. no `--multimodal-keys`.
    assert _content(messages) == prompt


@pytest.mark.unit
def test_row_with_an_empty_media_column_keeps_one_text_segment():
    prompt = "Describe this image."
    messages = _build_messages(_row(prompt, images=[]), "text", True, IMAGE_KEYS)

    assert _content(messages) == [{"type": "text", "text": prompt}]


@pytest.mark.unit
def test_unknown_media_type_is_rejected():
    """A typo like ``images`` used to be skipped silently: the media never made
    it into the prompt and training ran text-only against a VLM dataset."""
    with pytest.raises(ValueError, match="Unknown multimodal type 'images'"):
        _build_messages(_row("What is in <image>?", images=["a.png"]), "text", True, {"images": "images"})


@pytest.mark.unit
def test_list_content_row_is_passed_through():
    """Content already in list-of-dicts form embeds its media inline, so there
    is no placeholder for this function to spend the row's media on."""
    content = [{"type": "image", "image": "a.png"}, {"type": "text", "text": "What is in it?"}]
    prompt = [{"role": "user", "content": list(content)}]

    messages = _build_messages(_row(prompt, images=["a.png"]), "text", True, IMAGE_KEYS)

    assert _content(messages) == content


@pytest.mark.unit
def test_more_media_than_placeholders_still_raises():
    with pytest.raises(AssertionError, match="Multimodal data count mismatch"):
        _build_messages(_row("What is in <image>?", images=["a.png", "b.png"]), "text", True, IMAGE_KEYS)


@pytest.mark.unit
def test_more_placeholders_than_media_still_raises():
    with pytest.raises(AssertionError, match="Not enough image data"):
        _build_messages(_row("<image> vs <image>", images=["a.png"]), "text", True, IMAGE_KEYS)


@pytest.mark.unit
def test_without_multimodal_keys_the_prompt_is_untouched():
    prompt = "Describe this image."

    assert _build_messages(_row(prompt), "text", False, None) == prompt
    assert _content(_build_messages(_row(prompt), "text", True, None)) == prompt
