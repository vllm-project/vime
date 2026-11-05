"""Per-test scratch ownership and failure-disposal policies."""

import errno

import pytest

from vime.utils.external_utils import command_utils as U

NUM_GPUS = 0


@pytest.mark.parametrize("error_number", [errno.ENOTEMPTY, errno.EACCES])
@pytest.mark.parametrize("transient", [False, True])
def test_scratch_cleanup_retries_only_transient_nonempty_directories(monkeypatch, tmp_path, error_number, transient):
    monkeypatch.setattr(U, "get_test_tmp_dir", lambda: str(tmp_path))
    monkeypatch.setattr(U.time, "sleep", lambda _: None)
    original = U.tempfile.TemporaryDirectory.cleanup
    calls = []
    directories = []

    def cleanup(directory):
        calls.append(directory.name)
        directories.append(directory)
        if not transient or len(calls) == 1:
            raise OSError(error_number, "fixture", directory.name)
        original(directory)

    monkeypatch.setattr(U.tempfile.TemporaryDirectory, "cleanup", cleanup)
    sibling = tmp_path / "another-test"
    sibling.mkdir()
    (sibling / "sentinel").write_text("preserve")
    try:
        if error_number == errno.ENOTEMPTY and transient:
            with U.test_work_directory("cleanup_", keep=False) as path:
                (path / "checkpoint").write_text("owned")
            assert not path.exists()
            assert len(calls) == 2
        else:
            with pytest.raises(OSError) as failure, U.test_work_directory("cleanup_", keep=False):
                pass
            assert failure.value.errno == error_number
            assert len(calls) == (4 if error_number == errno.ENOTEMPTY else 1)
        assert (sibling / "sentinel").read_text() == "preserve"
    finally:
        for directory in directories:
            original(directory)


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("keep", [False, True])
def test_temporary_directory_policy_cleans_failures_and_supports_keep(monkeypatch, tmp_path, keep, failure):
    monkeypatch.setattr(U, "get_test_tmp_dir", lambda: str(tmp_path))
    monkeypatch.setenv("VIME_TEST_KEEP_ARTIFACTS", "1" if keep else "0")
    path = None
    try:
        with U.test_work_directory("fixture_") as path:
            (path / "output").write_text("fixture")
            if failure:
                raise RuntimeError("fixture")
    except RuntimeError:
        assert failure
    assert path.exists() is keep


def test_explicit_directory_is_preserved_and_cli_keep_overrides_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(U, "get_test_tmp_dir", lambda: str(tmp_path))
    monkeypatch.setenv("VIME_TEST_KEEP_ARTIFACTS", "0")
    explicit = tmp_path / "explicit"
    with pytest.raises(RuntimeError), U.test_work_directory("fixture_", directory=explicit) as path:
        (path / "sentinel").write_text("preserve")
        raise RuntimeError("fixture")
    assert (explicit / "sentinel").read_text() == "preserve"
    with U.test_work_directory("fixture_", keep=True) as kept:
        pass
    assert kept.exists()
    monkeypatch.setenv("VIME_TEST_KEEP_ARTIFACTS", "1")
    with U.test_work_directory("fixture_", keep=False) as deleted:
        pass
    assert not deleted.exists()
