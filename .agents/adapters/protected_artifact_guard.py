#!/usr/bin/env python3
"""Metadata-only baseline guard for ignored protected artifacts."""

import argparse
import json
import os
import stat
import sys
import tempfile
from pathlib import Path


MANIFEST_VERSION = 1
PROTECTED_DIRECTORIES = (
    Path("packages_win") / "py312",
    Path("mlruns"),
    Path("out"),
)
PROTECTED_SUFFIXES = {".bat", ".parquet"}
PRUNED_TOP_LEVEL = {".git", ".worktrees"}


class GuardError(Exception):
    def __init__(self, error, message):
        super().__init__(message)
        self.error = error
        self.message = message


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise GuardError("invalid_arguments", message)


def emit(payload):
    sys.stdout.write(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    )


def canonical_root(root_argument):
    try:
        root = Path(root_argument).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise GuardError("invalid_root", str(exc)) from exc
    if not root.is_dir():
        raise GuardError("invalid_root", f"not a directory: {root}")
    return root


def canonical_state(state_argument, root):
    try:
        state = Path(state_argument).expanduser().resolve(strict=False)
        inside_root = os.path.commonpath((str(root), str(state))) == str(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise GuardError("invalid_state_path", str(exc)) from exc
    if inside_root:
        raise GuardError(
            "state_inside_root",
            "state path must be outside the repository root",
        )
    return state


def metadata(path):
    try:
        file_stat = path.lstat()
        record = {
            "mode": stat.S_IMODE(file_stat.st_mode),
            "size": file_stat.st_size,
            "mtime_ns": file_stat.st_mtime_ns,
        }
        if stat.S_ISLNK(file_stat.st_mode):
            record["symlink_target"] = os.readlink(path)
        return record
    except OSError as exc:
        raise GuardError("scan_error", f"{path}: {exc}") from exc


def relative_name(path, root):
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise GuardError("scan_error", f"path escaped root: {path}") from exc


def scan_tree(path, root, entries, include_all):
    if include_all or path.suffix.casefold() in PROTECTED_SUFFIXES:
        entries[relative_name(path, root)] = metadata(path)
    try:
        path_stat = path.lstat()
    except OSError as exc:
        raise GuardError("scan_error", f"{path}: {exc}") from exc
    if not stat.S_ISDIR(path_stat.st_mode) or stat.S_ISLNK(path_stat.st_mode):
        return

    try:
        children = sorted(os.scandir(path), key=lambda item: item.name)
    except OSError as exc:
        raise GuardError("scan_error", f"{path}: {exc}") from exc
    for child in children:
        child_path = Path(child.path)
        try:
            is_directory = child.is_dir(follow_symlinks=False)
            is_symlink = child.is_symlink()
        except OSError as exc:
            raise GuardError("scan_error", f"{child.path}: {exc}") from exc
        if include_all or child_path.suffix.casefold() in PROTECTED_SUFFIXES:
            entries[relative_name(child_path, root)] = metadata(child_path)
        if is_directory and not is_symlink:
            scan_tree(child_path, root, entries, include_all)


def scan_repository(root):
    entries = {}
    for relative in PROTECTED_DIRECTORIES:
        protected_root = root / relative
        if os.path.lexists(protected_root):
            scan_tree(protected_root, root, entries, include_all=True)

    try:
        top_level = sorted(os.scandir(root), key=lambda item: item.name)
    except OSError as exc:
        raise GuardError("scan_error", f"{root}: {exc}") from exc
    for child in top_level:
        if child.name in PRUNED_TOP_LEVEL or child.name.startswith(".venv"):
            continue
        child_path = Path(child.path)
        try:
            is_directory = child.is_dir(follow_symlinks=False)
            is_symlink = child.is_symlink()
        except OSError as exc:
            raise GuardError("scan_error", f"{child.path}: {exc}") from exc
        if child_path.suffix.casefold() in PROTECTED_SUFFIXES:
            entries[relative_name(child_path, root)] = metadata(child_path)
        if is_directory and not is_symlink:
            scan_tree(child_path, root, entries, include_all=False)
    return dict(sorted(entries.items()))


def atomic_write_manifest(state, manifest):
    parent = state.parent
    if not parent.is_dir():
        raise GuardError("state_write_error", f"missing directory: {parent}")
    temporary_name = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{state.name}.", dir=parent
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, state)
        temporary_name = None
        os.chmod(state, 0o600)
    except OSError as exc:
        raise GuardError("state_write_error", str(exc)) from exc
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass


def validate_manifest(manifest, root):
    if not isinstance(manifest, dict):
        raise GuardError("malformed_state", "manifest must be an object")
    if manifest.get("version") != MANIFEST_VERSION:
        raise GuardError("malformed_state", "unsupported manifest version")
    if manifest.get("root") != str(root):
        raise GuardError("malformed_state", "manifest root does not match")
    entries = manifest.get("entries")
    if not isinstance(entries, dict):
        raise GuardError("malformed_state", "manifest entries must be an object")
    for relative, record in entries.items():
        if not isinstance(relative, str) or not isinstance(record, dict):
            raise GuardError("malformed_state", "invalid manifest entry")
        required = {"mode", "size", "mtime_ns"}
        if not required.issubset(record):
            raise GuardError("malformed_state", "incomplete manifest entry")
        if not all(isinstance(record[key], int) for key in required):
            raise GuardError("malformed_state", "invalid manifest metadata")
        allowed = required | {"symlink_target"}
        if set(record) - allowed:
            raise GuardError("malformed_state", "unknown manifest metadata")
        if "symlink_target" in record and not isinstance(
            record["symlink_target"], str
        ):
            raise GuardError("malformed_state", "invalid symlink target")
    return entries


def load_manifest(state, root):
    if not state.exists():
        raise GuardError("missing_state", f"state does not exist: {state}")
    try:
        with state.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise GuardError("malformed_state", str(exc)) from exc
    entries = validate_manifest(manifest, root)
    return manifest, entries


def snapshot(root, state):
    entries = scan_repository(root)
    manifest = {
        "version": MANIFEST_VERSION,
        "guard": "metadata-only",
        "root": str(root),
        "entries": entries,
    }
    atomic_write_manifest(state, manifest)
    return 0, {
        "status": "ok",
        "action": "snapshot",
        "guard": "metadata-only",
        "version": MANIFEST_VERSION,
        "entries": len(entries),
        "state": str(state),
    }


def verify(root, state):
    manifest, previous = load_manifest(state, root)
    current = scan_repository(root)
    previous_paths = set(previous)
    current_paths = set(current)
    added = sorted(current_paths - previous_paths)
    removed = sorted(previous_paths - current_paths)
    modified = sorted(
        path
        for path in previous_paths & current_paths
        if previous[path] != current[path]
    )
    if added or removed or modified:
        return 1, {
            "status": "changed",
            "action": "verify",
            "guard": manifest.get("guard", "metadata-only"),
            "added": added,
            "removed": removed,
            "modified": modified,
        }
    return 0, {
        "status": "ok",
        "action": "verify",
        "guard": manifest.get("guard", "metadata-only"),
        "version": manifest["version"],
        "entries": len(current),
    }


def parse_args(arguments):
    parser = JsonArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("snapshot", "verify"):
        command = subparsers.add_parser(action)
        command.add_argument("--root", required=True)
        command.add_argument("--state", required=True)
    return parser.parse_args(arguments)


def main(arguments=None):
    try:
        args = parse_args(arguments)
        root = canonical_root(args.root)
        state = canonical_state(args.state, root)
        if args.action == "snapshot":
            return_code, payload = snapshot(root, state)
        else:
            return_code, payload = verify(root, state)
    except GuardError as exc:
        return_code = 2
        payload = {
            "status": "error",
            "error": exc.error,
            "message": exc.message,
        }
    except Exception as exc:
        return_code = 3
        payload = {
            "status": "error",
            "error": "internal_error",
            "message": str(exc),
        }
    emit(payload)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
