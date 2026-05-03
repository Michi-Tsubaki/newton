# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Download third-party example assets without touching Newton-managed assets."""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ThirdPartyArchive:
    name: str
    url: str
    sha256: str
    root_dir: str


NEXTAGE_DESCRIPTION = ThirdPartyArchive(
    name="nextage_description",
    url=(
        "https://raw.githubusercontent.com/iory/scikit-robot-models/"
        "35f450dde137629b641206be7cee5f262976b07d/nextage_description.tar.gz"
    ),
    sha256="df4eb9debfa0eb60d1ed14b6bd3c0a14bddb31d4f754ea766c2337671dd7e003",
    root_dir="nextage_description",
)

_ARCHIVES = {
    NEXTAGE_DESCRIPTION.name: NEXTAGE_DESCRIPTION,
}


def _default_cache_dir() -> Path:
    return Path(__file__).resolve().parents[3] / ".cache" / "thirdparty_assets"


def _safe_extract_tar(tar: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in tar.getmembers():
        if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
            raise RuntimeError(f"Refusing to extract unsupported archive member: {member.name}")
        member_path = (destination / member.name).resolve()
        if os.path.commonpath([destination, member_path]) != str(destination):
            raise RuntimeError(f"Refusing to extract unsafe archive member: {member.name}")
    tar.extractall(destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive_asset(
    name: str,
    cache_dir: str | os.PathLike[str] | None = None,
    force_refresh: bool = False,
) -> Path:
    """Download and extract a registered third-party archive asset.

    Args:
        name: Registered third-party asset name.
        cache_dir: Directory to cache downloads. Defaults to ``.cache/thirdparty_assets``
            in the repository root.
        force_refresh: If True, remove the cached extraction and download again.

    Returns:
        Path to the extracted asset root directory.
    """
    if name not in _ARCHIVES:
        available = ", ".join(sorted(_ARCHIVES))
        raise ValueError(f"Unknown third-party asset '{name}'. Available assets: {available}")

    archive = _ARCHIVES[name]
    cache_path = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    final_dir = cache_path / f"{archive.name}_{archive.sha256[:8]}" / archive.root_dir
    marker = final_dir / ".newton_thirdparty_asset"

    if final_dir.exists() and marker.exists() and not force_refresh:
        return final_dir

    package_dir = final_dir.parent
    cache_path.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{archive.name}_", dir=cache_path))
    archive_path = temp_dir / f"{archive.name}.tar.gz"

    try:
        print(f"Downloading third-party asset {archive.name}...")
        urllib.request.urlretrieve(archive.url, archive_path)

        actual_sha256 = _sha256(archive_path)
        if actual_sha256 != archive.sha256:
            raise RuntimeError(
                f"Checksum mismatch for {archive.name}: expected {archive.sha256}, got {actual_sha256}"
            )

        extract_dir = temp_dir / "extract"
        extract_dir.mkdir()
        with tarfile.open(archive_path, "r:gz") as tar:
            _safe_extract_tar(tar, extract_dir)

        extracted_root = extract_dir / archive.root_dir
        if not extracted_root.exists():
            raise RuntimeError(f"Archive {archive.name} does not contain {archive.root_dir}/")

        if package_dir.exists():
            shutil.rmtree(package_dir)
        package_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted_root), str(final_dir))
        marker.write_text(f"{archive.url}\n{archive.sha256}\n", encoding="utf-8")
        return final_dir
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def download_nextage_description(
    cache_dir: str | os.PathLike[str] | None = None,
    force_refresh: bool = False,
) -> Path:
    """Download and extract the Nextage description package."""
    return download_archive_asset(NEXTAGE_DESCRIPTION.name, cache_dir=cache_dir, force_refresh=force_refresh)


__all__ = [
    "download_archive_asset",
    "download_nextage_description",
]
