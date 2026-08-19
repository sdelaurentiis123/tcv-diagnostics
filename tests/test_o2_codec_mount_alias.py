from __future__ import annotations

from pathlib import Path

import pytest

from tcv_diagnostics.phase3_5.equivariance import canonical_existing_development_path


def test_phase3_5_codec_path_resolves_mount_alias(tmp_path: Path) -> None:
    checkpoint = tmp_path / "codec.pt"
    checkpoint.write_bytes(b"immutable codec bytes")
    alias = tmp_path / "ceph-alias.pt"
    alias.symlink_to(checkpoint)
    assert canonical_existing_development_path(alias) == checkpoint.resolve(strict=True)
    with pytest.raises(FileNotFoundError):
        canonical_existing_development_path(tmp_path / "missing.pt")


def test_codec_provenance_rejects_held_out_path_before_resolution(tmp_path: Path) -> None:
    checkpoint = tmp_path / "codec.pt"
    checkpoint.write_bytes(b"immutable codec bytes")
    held_out_spelling = tmp_path / "85606-codec.pt"
    held_out_spelling.symlink_to(checkpoint)

    with pytest.raises(ValueError, match="85606"):
        canonical_existing_development_path(held_out_spelling)
