from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import socket
import stat
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .commands import run_checked


MAX_REQUEST_BYTES = 16_384
UUID_PATTERN = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _secure_regular_file(path: Path, label: str) -> None:
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} is not a regular file: {path}")
    if metadata.st_uid != 0:
        raise ValueError(f"{label} must be owned by root: {path}")
    if metadata.st_mode & 0o022:
        raise ValueError(f"{label} cannot be group/world writable: {path}")


def _secure_executable(path: Path) -> None:
    if not path.is_absolute():
        raise ValueError("radio driver executable paths must be absolute")
    _secure_regular_file(path, "radio driver executable")
    if not os.access(path, os.X_OK):
        raise ValueError(f"radio driver executable is not executable: {path}")


@dataclass(frozen=True)
class ApprovedRadioProfile:
    profile_id: str
    sdr_profile_id: str
    connection_uuid: str
    evidence_path: Path
    activate_argv: tuple[str, ...]
    verify_argv: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ApprovedRadioProfile":
        profile_id = str(raw.get("profile_id", "")).strip()
        sdr_profile_id = str(raw.get("sdr_profile_id", "")).strip()
        connection_uuid = str(raw.get("connection_uuid", "")).strip()
        if not profile_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", profile_id):
            raise ValueError("profile_id is invalid")
        if not sdr_profile_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", sdr_profile_id):
            raise ValueError("sdr_profile_id is invalid")
        if not UUID_PATTERN.fullmatch(connection_uuid):
            raise ValueError(f"connection_uuid is invalid for {profile_id}")
        evidence_path = Path(str(raw.get("evidence_path", "")))
        if not evidence_path.is_absolute():
            raise ValueError(f"evidence_path must be absolute for {profile_id}")
        activate_argv = tuple(str(value) for value in raw.get("activate_argv", []))
        verify_argv = tuple(str(value) for value in raw.get("verify_argv", []))
        if not activate_argv or not verify_argv:
            raise ValueError(f"activation and verification commands are required for {profile_id}")
        _secure_executable(Path(activate_argv[0]))
        _secure_executable(Path(verify_argv[0]))
        return cls(profile_id, sdr_profile_id, connection_uuid, evidence_path, activate_argv, verify_argv)


@dataclass(frozen=True)
class RadioAgentConfig:
    socket_path: Path
    evidence_public_key_path: Path
    profiles: tuple[ApprovedRadioProfile, ...]

    @classmethod
    def load(cls, path: str | Path) -> "RadioAgentConfig":
        config_path = Path(path)
        _secure_regular_file(config_path, "radio agent configuration")
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        socket_path = Path(str(raw.get("socket_path", "")))
        key_path = Path(str(raw.get("evidence_public_key_path", "")))
        if not socket_path.is_absolute() or not key_path.is_absolute():
            raise ValueError("socket and public-key paths must be absolute")
        _secure_regular_file(key_path, "evidence public key")
        profiles = tuple(ApprovedRadioProfile.from_dict(item) for item in raw.get("profiles", []))
        identifiers = [profile.profile_id for profile in profiles]
        if not profiles or len(identifiers) != len(set(identifiers)):
            raise ValueError("at least one uniquely named approved profile is required")
        return cls(socket_path, key_path, profiles)


class SignedEvidenceStore:
    def __init__(self, public_key_path: Path) -> None:
        loaded = serialization.load_pem_public_key(public_key_path.read_bytes())
        if not isinstance(loaded, Ed25519PublicKey):
            raise ValueError("radio evidence public key must be Ed25519")
        self._public_key = loaded

    def load(self, path: Path, *, profile_id: str, connection_uuid: str) -> dict[str, Any]:
        _secure_regular_file(path, "signed radio evidence")
        envelope = json.loads(path.read_text(encoding="utf-8"))
        payload = envelope.get("payload")
        signature_text = envelope.get("signature")
        if not isinstance(payload, dict) or not isinstance(signature_text, str):
            raise ValueError("signed radio evidence envelope is malformed")
        try:
            signature = base64.b64decode(signature_text, validate=True)
            self._public_key.verify(signature, _canonical_json(payload))
        except (ValueError, InvalidSignature) as exc:
            raise ValueError("radio evidence signature verification failed") from exc
        if payload.get("profile_id") != profile_id or payload.get("connection_uuid") != connection_uuid:
            raise ValueError("signed radio evidence does not match the approved profile")
        report_path = Path(str(payload.get("conformance_report_path", "")))
        expected_hash = str(payload.get("conformance_report_sha256", "")).lower()
        if not report_path.is_absolute() or not SHA256_PATTERN.fullmatch(expected_hash):
            raise ValueError("radio evidence has invalid conformance report metadata")
        _secure_regular_file(report_path, "conformance report")
        actual_hash = hashlib.sha256(report_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError("conformance report hash does not match signed evidence")
        capability = payload.get("capability")
        if not isinstance(capability, dict) or capability.get("connection_uuid") != connection_uuid:
            raise ValueError("signed radio evidence omitted matching capability data")
        capability = dict(capability)
        capability["conformance_report_sha256"] = actual_hash
        return capability


class RadioAgent:
    def __init__(self, config: RadioAgentConfig) -> None:
        self.config = config
        self._profiles = {profile.profile_id: profile for profile in config.profiles}
        self._evidence = SignedEvidenceStore(config.evidence_public_key_path)

    def _profile(self, request: dict[str, Any]) -> ApprovedRadioProfile:
        profile_id = str(request.get("profile_id", ""))
        profile = self._profiles.get(profile_id)
        if profile is None:
            raise ValueError("profile is not approved by the radio agent")
        if request.get("connection_uuid") != profile.connection_uuid:
            raise ValueError("connection UUID does not match the approved radio profile")
        return profile

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        profile = self._profile(request)
        operation = request.get("operation")
        if operation == "capabilities":
            capability = self._evidence.load(
                profile.evidence_path,
                profile_id=profile.profile_id,
                connection_uuid=profile.connection_uuid,
            )
            return {"schema": "kval-radio-agent/v1", "success": True, "capability": capability}
        if operation != "activate":
            raise ValueError("unsupported operation")
        catalog_hash = str(request.get("catalog_hash", ""))
        if not SHA256_PATTERN.fullmatch(catalog_hash):
            raise ValueError("a lowercase SHA-256 signed-catalog hash is required")
        if request.get("sdr_profile_id") != profile.sdr_profile_id:
            raise ValueError("SDR profile does not match the approved radio profile")
        capability = self._evidence.load(
            profile.evidence_path,
            profile_id=profile.profile_id,
            connection_uuid=profile.connection_uuid,
        )
        run_checked(profile.activate_argv, timeout_seconds=30.0)
        verification = json.loads(run_checked(profile.verify_argv, timeout_seconds=10.0))
        if verification.get("active_profile_id") != profile.sdr_profile_id:
            raise RuntimeError("radio driver did not verify the exact approved profile")
        return {
            "schema": "kval-radio-agent/v1",
            "success": True,
            "active_profile_id": profile.sdr_profile_id,
            "connection_uuid": profile.connection_uuid,
            "catalog_hash": catalog_hash,
            "implementation_id": capability.get("implementation_id"),
        }

    @staticmethod
    def _peer_uid(connection: socket.socket) -> int:
        credentials = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
        _, uid, _ = struct.unpack("3i", credentials)
        return uid

    def serve_forever(self) -> None:
        self.config.socket_path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
        if self.config.socket_path.exists() or self.config.socket_path.is_symlink():
            metadata = self.config.socket_path.lstat()
            if not stat.S_ISSOCK(metadata.st_mode) or metadata.st_uid != 0:
                raise RuntimeError("refusing to replace an unsafe radio-agent socket path")
            self.config.socket_path.unlink()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(self.config.socket_path))
            os.chmod(self.config.socket_path, 0o600)
            server.listen(16)
            while True:
                connection, _ = server.accept()
                with connection:
                    response: dict[str, Any]
                    try:
                        if self._peer_uid(connection) != 0:
                            raise PermissionError("radio agent accepts only root peers")
                        raw = bytearray()
                        while len(raw) <= MAX_REQUEST_BYTES and b"\n" not in raw:
                            chunk = connection.recv(4096)
                            if not chunk:
                                break
                            raw.extend(chunk)
                        if len(raw) > MAX_REQUEST_BYTES:
                            raise ValueError("request exceeded size limit")
                        request = json.loads(bytes(raw).split(b"\n", 1)[0])
                        if not isinstance(request, dict):
                            raise ValueError("request must be a JSON object")
                        response = self.handle(request)
                    except Exception as exc:
                        response = {"schema": "kval-radio-agent/v1", "success": False, "error": str(exc)}
                    connection.sendall(_canonical_json(response) + b"\n")


def write_signed_evidence(path: Path, payload: dict[str, Any], signature: bytes) -> None:
    """Atomically install an offline-signed evidence envelope."""
    envelope = {"payload": payload, "signature": base64.b64encode(signature).decode("ascii")}
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(envelope, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
