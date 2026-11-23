#!/usr/bin/env python3
"""Export an explicit, size-limited package snapshot; never traverse result trees."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

SOURCE = Path(__file__).resolve().parents[1]
SKIP = {".DS_Store", "__pycache__", ".git", ".idea", ".vscode"}
LOCAL_PATH = re.compile(r"/(?:Users|home)/[^/\s]+/|/(?:Applications)/")
CREDENTIAL = re.compile(
    r"gh[pousr]_[A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{30,}|"
    r"AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
)


def relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Selection must be a relative package path: {value}")
    return path


def selected_files() -> dict[Path, Path]:
    config = json.loads((SOURCE / "packaging/upload.json").read_text())
    selection: dict[Path, Path] = {}
    for name in config["directories"]:
        directory = SOURCE / relative_path(name)
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"Missing directory or symlink: {name}")
        for item in sorted(directory.rglob("*")):
            relative = item.relative_to(SOURCE)
            if any(part in SKIP for part in relative.parts):
                continue
            if item.is_symlink():
                raise ValueError(f"Symlinks are not exported: {relative}")
            if item.is_file():
                selection[relative] = item
    for name in config["files"]:
        relative = relative_path(name)
        selection[relative] = SOURCE / relative
    for destination, origin in config.get("overrides", {}).items():
        relative = relative_path(destination)
        if relative not in selection:
            raise ValueError(f"Override not in selection: {destination}")
        selection[relative] = SOURCE / relative_path(origin)

    total = 0
    for relative, source in sorted(selection.items()):
        # Reject a symlink anywhere along an explicitly selected path.
        if any(parent.is_symlink() for parent in [source, *source.parents]
               if parent == SOURCE or SOURCE in parent.parents):
            raise ValueError(f"Symlink in selected path: {relative}")
        if not source.is_file():
            raise ValueError(f"Missing selected file: {relative}")
        if source.suffix.lower() in {".bin", ".mat", ".gif", ".mp4", ".zip", ".pem", ".key"}:
            raise ValueError(f"Unexpected binary/credential file: {relative}")
        if source.name.startswith(".env"):
            raise ValueError(f"Environment file is not exported: {relative}")
        size = source.stat().st_size
        if size > config["max_file_bytes"]:
            raise ValueError(f"Selected file exceeds size limit: {relative}")
        total += size
        text = source.read_text(encoding="utf-8")
        if LOCAL_PATH.search(text):
            raise ValueError(f"Machine-specific path in selected file: {relative}")
        if CREDENTIAL.search(text):
            raise ValueError(f"Possible credential in selected file: {relative}")
    if total > config["max_total_bytes"]:
        raise ValueError(f"Package exceeds total size limit: {total} bytes")
    print(f"Validated {len(selection)} selected files ({total:,} bytes).")
    return dict(sorted(selection.items()))


def export_package(output: Path, archive: bool) -> None:
    output = output.expanduser().resolve()
    if output == SOURCE or SOURCE in output.parents or output in SOURCE.parents:
        raise ValueError("Choose a new destination outside the source tree")
    if output.exists():
        raise ValueError(f"Destination already exists; choose a new path: {output}")
    zip_path = output.with_name(output.name + ".zip")
    if archive and zip_path.exists():
        raise ValueError(f"Archive already exists; choose a new path: {zip_path}")
    selection = selected_files()  # Complete validation before creating anything.
    output.mkdir(parents=True)
    manifest = []
    for relative, source in selection.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        data = destination.read_bytes()
        manifest.append({"path": relative.as_posix(), "bytes": len(data),
                         "sha256": hashlib.sha256(data).hexdigest()})
    report = {"package": "TensorNetworksJulia", "license": "MIT", "files": manifest,
              "note": "This manifest excludes itself; generated data and Git metadata are not exported."}
    manifest_path = output / "UPLOAD_CONTENTS.json"
    manifest_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if archive:
        with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=9) as stream:
            for relative in [*selection, Path("UPLOAD_CONTENTS.json")]:
                info = zipfile.ZipInfo("TensorNetworksJulia/" + relative.as_posix(),
                                       date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                stream.writestr(info, (output / relative).read_bytes())
        print(f"Archive: {zip_path} ({zip_path.stat().st_size:,} bytes)")
    print(f"Clean package: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate the explicit selection only")
    parser.add_argument("--output", type=Path, help="New clean package directory")
    parser.add_argument("--archive", action="store_true", help="Also make a ZIP without Git metadata")
    args = parser.parse_args()
    if args.check and (args.output is not None or args.archive):
        parser.error("Use --check alone, or --output with optional --archive")
    if args.check:
        selected_files()
    elif args.output is not None:
        export_package(args.output, args.archive)
    else:
        parser.error("Use --check or --output PATH")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, UnicodeError) as error:
        print(f"Export refused: {error}", file=sys.stderr)
        sys.exit(1)
