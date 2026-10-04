import numpy as np
import pytest
import subprocess
import sys
import types

import thymus_quant as tq
from thymus_quant.exceptions import SegmentorConfigurationError
from thymus_quant.inputs import make_image_context


def test_known_alias_and_unknown_alias_rejection():
    seg = tq.load_segmentor("trqseg_v1", members=[0])
    assert seg.info.name == "trqseg_v1"
    assert seg.info.selected_members == ("fold-0",)
    with pytest.raises(SegmentorConfigurationError, match="unknown segmentor"):
        tq.load_segmentor("someone/random-repo")


@pytest.mark.parametrize("members", [[], [0, 0], [-1], [99], ["fold-99"]])
def test_invalid_members_rejected(members):
    with pytest.raises(SegmentorConfigurationError):
        tq.load_segmentor("trqseg_v1", members=members)


def test_cache_reuse_for_same_config():
    a = tq.load_segmentor("trqseg_v1", members=[0], revision="rev-a", local_files_only=True)
    b = tq.load_segmentor("trqseg_v1", members=[0], revision="rev-a", local_files_only=True)
    assert a is b


def test_unspecified_hf_revision_is_resolved_before_download(tmp_path, monkeypatch):
    calls = {}
    resolved_sha = "a" * 40
    downloaded_weight = tmp_path / "model.safetensors"
    downloaded_weight.write_bytes(b"fake")

    class FakeHfApi:
        def model_info(self, *, repo_id, revision):
            calls["model_info"] = {"repo_id": repo_id, "revision": revision}
            return types.SimpleNamespace(sha=resolved_sha)

    def fake_hf_hub_download(**kwargs):
        calls["download"] = kwargs
        return str(downloaded_weight)

    fake_hf = types.SimpleNamespace(HfApi=FakeHfApi, hf_hub_download=fake_hf_hub_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    seg = tq.load_segmentor("trqseg_v1", members=[0], cache_dir=tmp_path / "cache")
    path = seg._resolve_weight_path(seg.members[0])

    assert path == str(downloaded_weight)
    assert calls["model_info"] == {"repo_id": "yuki-okamura-hf/TRQseg-v1", "revision": None}
    assert calls["download"]["revision"] == resolved_sha
    assert seg.info.requested_revision is None
    assert seg.info.resolved_revision == resolved_sha
    assert seg.members[0].resolved_revision == resolved_sha


def test_local_weight_provenance_from_env(tmp_path, monkeypatch):
    weight = tmp_path / "weights" / "fold-0" / "model.safetensors"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"fake")
    monkeypatch.setenv("THYQ_TRQSEG_V1_LOCAL_REPO", str(tmp_path))
    seg = tq.load_segmentor("trqseg_v1", members=[0], local_files_only=True)
    assert seg.members[0].local_path == str(weight)
    assert seg.info.local_source == str(tmp_path)
    assert seg.info.weight_source == "local"


def test_pinned_revision_does_not_use_mirror_without_git_head(tmp_path, monkeypatch):
    weight = tmp_path / "weights" / "fold-0" / "model.safetensors"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"fake")
    monkeypatch.setenv("THYQ_TRQSEG_V1_LOCAL_REPO", str(tmp_path))
    seg = tq.load_segmentor("trqseg_v1", members=[0], revision="rev-local", local_files_only=True)
    assert seg.members[0].local_path is None
    assert seg.info.local_source is None
    assert seg.info.resolved_revision == "rev-local"


def _git_mirror(tmp_path):
    weight = tmp_path / "weights" / "fold-0" / "model.safetensors"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"fake")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "add", "weights/fold-0/model.safetensors"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-m", "add weight"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()
    return weight, head


def test_pinned_revision_uses_local_mirror_only_at_that_commit(tmp_path, monkeypatch, caplog):
    weight, head = _git_mirror(tmp_path)
    monkeypatch.setenv("THYQ_TRQSEG_V1_LOCAL_REPO", str(tmp_path))

    for same in (head, head.upper()):
        seg = tq.load_segmentor("trqseg_v1", members=[0], revision=same, cache_dir=tmp_path / "c1", local_files_only=True)
        assert seg.members[0].local_path == str(weight)
        assert seg.info.local_source == str(tmp_path)
        assert seg.info.local_revision == head
        assert seg.info.resolved_revision == same

    calls = {}

    def fake_hf_hub_download(**kwargs):
        calls["download"] = kwargs
        return str(weight)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=None, hf_hub_download=fake_hf_hub_download))
    for other in ("b" * 40, "main", head[:7]):
        caplog.clear()
        with caplog.at_level("WARNING", logger="thymus_quant.api"):
            seg = tq.load_segmentor("trqseg_v1", members=[0], revision=other, cache_dir=tmp_path / "c2", local_files_only=True)
        assert seg.members[0].local_path is None
        assert seg.info.local_source is None
        assert seg.info.weight_source == "unknown"
        assert seg.info.resolved_revision == other
        assert "not at the requested revision" in caplog.text

        calls.clear()
        seg._resolve_weight_path(seg.members[0])
        assert calls["download"]["revision"] == other
        assert seg.members[0].weight_source == "downloaded"


def test_local_git_mirror_revision_is_recorded(tmp_path, monkeypatch):
    weight = tmp_path / "weights" / "fold-0" / "model.safetensors"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"fake")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "add", "weights/fold-0/model.safetensors"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-m", "add weight"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    monkeypatch.setenv("THYQ_TRQSEG_V1_LOCAL_REPO", str(tmp_path))
    seg = tq.load_segmentor("trqseg_v1", members=[0], cache_dir=tmp_path / "cache", local_files_only=True)

    # The mirror's HEAD is recorded as the local revision, not as a Hugging Face revision.
    assert seg.info.local_revision == expected
    assert seg.info.resolved_revision is None
    assert seg.members[0].resolved_revision is None


def test_heuristic_is_explicit_and_warns(tmp_path):
    import nibabel as nib

    ct = np.zeros((8, 8, 8), dtype=np.float32)
    img = nib.Nifti1Image(ct, np.eye(4))
    with pytest.warns(UserWarning, match="heuristic_trq"):
        out = tq.segment_trq(img, segmentor="heuristic_trq", study_id="s")
    assert out.segmentor.name == "heuristic_trq"


def test_model_load_failure_does_not_fallback(monkeypatch):
    seg = tq.load_segmentor("trqseg_v1", members=[0], local_files_only=True)
    monkeypatch.setattr(type(seg), "_ensure_models_loaded", lambda self: None)
    img = make_image_context(ct_hu=np.zeros((4, 4, 4)), spacing_mm=(1, 1, 1))
    with pytest.raises(RuntimeError, match="heuristic fallback is disabled"):
        seg.segment_trq(img, study_id="s")


def test_weight_checksums_and_source_are_recorded_when_loaded(tmp_path, monkeypatch):
    import hashlib

    weight = tmp_path / "weights" / "fold-0" / "model.safetensors"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"fake weights")
    monkeypatch.setenv("THYQ_TRQSEG_V1_LOCAL_REPO", str(tmp_path))
    seg = tq.load_segmentor("trqseg_v1", members=[0], cache_dir=tmp_path / "c", local_files_only=True)
    seg._resolve_weight_path(seg.members[0])
    seg._record_weight_provenance(seg.members)
    assert seg.info.weight_sha256 == (hashlib.sha256(b"fake weights").hexdigest(),)
    assert seg.info.weight_source == "local"

    downloaded = tmp_path / "dl.safetensors"
    downloaded.write_bytes(b"other")
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=None, hf_hub_download=lambda **kw: str(downloaded)))
    seg2 = tq.load_segmentor("trqseg_v1", members=[1], revision="c" * 40, cache_dir=tmp_path / "c2", local_files_only=True)
    seg2._resolve_weight_path(seg2.members[0])
    seg2._record_weight_provenance(seg2.members)
    assert seg2.info.weight_source == "downloaded"
    assert seg2.info.weight_sha256 == (hashlib.sha256(b"other").hexdigest(),)
