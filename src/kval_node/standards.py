from __future__ import annotations

import json
import re
import socket
import stat
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .commands import run_command


class StandardIdentifier(str, Enum):
    THREE_GPP_TS_23_003 = "3gpp-ts-23.003"
    THREE_GPP_RELEASE_17 = "3gpp-release-17"
    THREE_GPP_RELEASE_18 = "3gpp-release-18"
    ETSI_EN_303_648 = "etsi-en-303-648"
    ETSI_EN_303_681_1 = "etsi-en-303-681-1"
    ETSI_EN_303_681_2 = "etsi-en-303-681-2"
    ETSI_EN_303_681_3 = "etsi-en-303-681-3"
    ETSI_EN_303_681_4 = "etsi-en-303-681-4"
    IEEE_1900_5_1_2020 = "ieee-1900.5.1-2020"
    IEEE_1900_5_2_2024 = "ieee-1900.5.2-2024"
    ANSI_VITA_49_2_2017_R2024 = "ansi-vita-49.2-2017-r2024"


class RadioAccessTechnology(str, Enum):
    GERAN = "geran"
    UTRAN = "utran"
    EUTRAN = "eutran"
    NR = "nr"
    NR_NTN = "nr-ntn"


MODEM_TECHNOLOGY_MAP: dict[RadioAccessTechnology, frozenset[str]] = {
    RadioAccessTechnology.GERAN: frozenset({"gsm", "gsm-compact", "gprs", "edge"}),
    RadioAccessTechnology.UTRAN: frozenset({"umts", "hsdpa", "hsupa", "hspa", "hspa-plus"}),
    RadioAccessTechnology.EUTRAN: frozenset({"lte", "lte-cat-m", "lte-nb-iot"}),
    RadioAccessTechnology.NR: frozenset({"5gnr", "nr", "nr5g", "5g", "5g-nsa", "5g-sa"}),
    RadioAccessTechnology.NR_NTN: frozenset({"nr-ntn", "5gnr-ntn"}),
}


@dataclass(frozen=True)
class SliceSelection:
    sst: int
    sd: str | None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SliceSelection":
        sst = int(raw["sst"])
        if not 0 <= sst <= 255:
            raise ValueError("S-NSSAI SST must fit one octet")
        sd_raw = raw.get("sd")
        sd = str(sd_raw).upper() if sd_raw is not None else None
        if sd is not None and not re.fullmatch(r"[0-9A-F]{6}", sd):
            raise ValueError("S-NSSAI SD must be six hexadecimal characters")
        return cls(sst, sd)

    def to_dict(self) -> dict[str, Any]:
        return {"sst": self.sst, "sd": self.sd}


@dataclass(frozen=True)
class StandardsRequirement:
    required_identifiers: frozenset[StandardIdentifier]
    required_rat: RadioAccessTechnology | None
    minimum_3gpp_release: int | None
    dnn: str | None
    s_nssai: SliceSelection | None
    capability_socket: Path | None
    sdr_profile_id: str | None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StandardsRequirement":
        identifiers = frozenset(StandardIdentifier(value) for value in raw.get("required_identifiers", []))
        required_rat = RadioAccessTechnology(raw["required_rat"]) if raw.get("required_rat") else None
        release = int(raw["minimum_3gpp_release"]) if raw.get("minimum_3gpp_release") is not None else None
        if release is not None and not 15 <= release <= 19:
            raise ValueError("minimum_3gpp_release must be between 15 and 19")
        dnn = str(raw["dnn"]).strip() if raw.get("dnn") else None
        if dnn is not None:
            if "REPLACE_" in dnn or len(dnn) > 100 or not re.fullmatch(r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*", dnn):
                raise ValueError("DNN must be a provisioned dot-separated 3GPP-style name")
        s_nssai = SliceSelection.from_dict(raw["s_nssai"]) if raw.get("s_nssai") else None
        capability_socket = Path(raw["capability_socket"]) if raw.get("capability_socket") else None
        if capability_socket is not None and not capability_socket.is_absolute():
            raise ValueError("radio capability socket path must be absolute")
        sdr_profile_id = str(raw["sdr_profile_id"]).strip() if raw.get("sdr_profile_id") else None
        if sdr_profile_id and "REPLACE_" in sdr_profile_id:
            raise ValueError("sdr_profile_id must be provisioned")
        requires_attestation = bool(release is not None or s_nssai is not None or identifiers - {StandardIdentifier.THREE_GPP_TS_23_003} or sdr_profile_id)
        if requires_attestation and capability_socket is None:
            raise ValueError("release, slice, SDR, VITA, ETSI, and IEEE requirements require a live capability socket")
        return cls(identifiers, required_rat, release, dnn, s_nssai, capability_socket, sdr_profile_id)

    def is_empty(self) -> bool:
        return not any(
            (
                self.required_identifiers,
                self.required_rat,
                self.minimum_3gpp_release,
                self.dnn,
                self.s_nssai,
                self.capability_socket,
                self.sdr_profile_id,
            )
        )


@dataclass(frozen=True)
class StandardsEvidence:
    verified: bool
    evidence: tuple[str, ...]
    errors: tuple[str, ...]
    capability: dict[str, Any] | None


class RadioStandardsClient:
    """Fail-closed client for a real local modem/SDR standards agent."""

    MAX_RESPONSE_BYTES = 65_536

    @staticmethod
    def _validate_socket(path: Path) -> None:
        try:
            metadata = path.stat()
        except FileNotFoundError as exc:
            raise RuntimeError(f"radio capability socket is missing: {path}") from exc
        if not stat.S_ISSOCK(metadata.st_mode):
            raise RuntimeError(f"radio capability endpoint is not a Unix socket: {path}")
        if metadata.st_uid != 0:
            raise RuntimeError("radio capability socket must be owned by root")
        if metadata.st_mode & 0o022:
            raise RuntimeError("radio capability socket cannot be group/world writable")

    def request(self, path: Path, request: dict[str, Any]) -> dict[str, Any]:
        self._validate_socket(path)
        encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(3.0)
            client.connect(str(path))
            client.sendall(encoded)
            response = bytearray()
            while len(response) <= self.MAX_RESPONSE_BYTES:
                chunk = client.recv(4096)
                if not chunk:
                    break
                response.extend(chunk)
                if b"\n" in chunk:
                    break
        if len(response) > self.MAX_RESPONSE_BYTES:
            raise RuntimeError("radio capability response exceeded size limit")
        try:
            parsed = json.loads(bytes(response).split(b"\n", 1)[0])
        except json.JSONDecodeError as exc:
            raise RuntimeError("radio capability agent returned malformed JSON") from exc
        if not isinstance(parsed, dict) or parsed.get("schema") != "kval-radio-agent/v1":
            raise RuntimeError("radio capability agent returned an unsupported schema")
        return parsed

    def capabilities(self, path: Path, profile_id: str, connection_uuid: str) -> dict[str, Any]:
        response = self.request(
            path,
            {
                "operation": "capabilities",
                "profile_id": profile_id,
                "connection_uuid": connection_uuid,
            },
        )
        if response.get("success") is not True:
            raise RuntimeError(f"radio capability agent rejected request: {response.get('error', 'unknown error')}")
        capability = response.get("capability")
        if not isinstance(capability, dict):
            raise RuntimeError("radio capability agent omitted capability evidence")
        if capability.get("connection_uuid") != connection_uuid:
            raise RuntimeError("radio capability evidence is for a different connection UUID")
        return capability

    def activate(
        self,
        path: Path,
        profile_id: str,
        sdr_profile_id: str,
        connection_uuid: str,
        catalog_hash: str,
    ) -> dict[str, Any]:
        response = self.request(
            path,
            {
                "operation": "activate",
                "profile_id": profile_id,
                "sdr_profile_id": sdr_profile_id,
                "connection_uuid": connection_uuid,
                "catalog_hash": catalog_hash,
            },
        )
        if response.get("success") is not True or response.get("active_profile_id") != sdr_profile_id:
            raise RuntimeError(f"radio standards agent did not confirm profile activation: {response}")
        return response


class StandardsVerifier:
    def __init__(self, client: RadioStandardsClient | None = None) -> None:
        self.client = client or RadioStandardsClient()

    @staticmethod
    def _rat_matches(required: RadioAccessTechnology, observed: str | None) -> bool:
        if observed is None:
            return False
        observed_values = {value.strip().lower() for value in observed.split(",")}
        return bool(observed_values & MODEM_TECHNOLOGY_MAP[required])

    @staticmethod
    def _connection_dnn(connection_uuid: str) -> str | None:
        result = run_command(
            ["nmcli", "--get-values", "gsm.apn", "connection", "show", "uuid", connection_uuid],
            timeout_seconds=10.0,
        )
        if result.returncode != 0:
            return None
        value = result.stdout.strip()
        return value or None

    def verify(
        self,
        requirement: StandardsRequirement,
        *,
        profile_id: str,
        connection_uuid: str,
        observed_access_technology: str | None,
    ) -> StandardsEvidence:
        if requirement.is_empty():
            return StandardsEvidence(True, ("NO_ADDITIONAL_STANDARDS_CONSTRAINT",), (), None)
        evidence: list[str] = []
        errors: list[str] = []
        capability: dict[str, Any] | None = None
        if requirement.capability_socket is not None:
            try:
                capability = self.client.capabilities(requirement.capability_socket, profile_id, connection_uuid)
                implementation_id = str(capability.get("implementation_id", "")).strip()
                conformance_hash = str(capability.get("conformance_report_sha256", ""))
                if not implementation_id:
                    errors.append("radio agent omitted its implementation identity")
                if not re.fullmatch(r"[0-9a-fA-F]{64}", conformance_hash):
                    errors.append("radio agent omitted a valid conformance report hash")
                if implementation_id and re.fullmatch(r"[0-9a-fA-F]{64}", conformance_hash):
                    evidence.append("ROOT_OWNED_RADIO_AGENT_CAPABILITY_WITH_PROVENANCE")
            except Exception as exc:
                errors.append(str(exc))

        capability_rat = str(capability.get("rat")) if capability and capability.get("rat") else None
        observed_rat = observed_access_technology or capability_rat
        if requirement.required_rat is not None:
            if self._rat_matches(requirement.required_rat, observed_rat):
                evidence.append(f"3GPP_RAT_{requirement.required_rat.value.upper()}")
            else:
                errors.append(f"required RAT {requirement.required_rat.value} was not observed")

        if requirement.dnn is not None:
            observed_dnn = self._connection_dnn(connection_uuid)
            capability_dnn = str(capability.get("dnn")) if capability and capability.get("dnn") else None
            if requirement.dnn in {observed_dnn, capability_dnn}:
                evidence.append("3GPP_DNN_MATCH")
            else:
                errors.append("configured DNN does not match live connection/capability evidence")

        if requirement.minimum_3gpp_release is not None:
            release = int(capability.get("3gpp_release", 0)) if capability else 0
            if release >= requirement.minimum_3gpp_release:
                evidence.append(f"3GPP_RELEASE_{release}_ATTESTED")
            else:
                errors.append("radio agent did not attest the required 3GPP release")

        if requirement.s_nssai is not None:
            observed_slice = capability.get("s_nssai") if capability else None
            if observed_slice == requirement.s_nssai.to_dict():
                evidence.append("3GPP_S_NSSAI_MATCH")
            else:
                errors.append("radio agent did not attest the required S-NSSAI")

        if requirement.required_identifiers:
            observed_identifiers = set(capability.get("standard_identifiers", [])) if capability else set()
            for identifier in requirement.required_identifiers:
                if identifier is StandardIdentifier.THREE_GPP_TS_23_003 and requirement.dnn is not None:
                    evidence.append(identifier.value)
                elif identifier.value in observed_identifiers:
                    evidence.append(identifier.value)
                else:
                    errors.append(f"radio agent did not attest {identifier.value}")

        if requirement.sdr_profile_id is not None:
            profiles = set(capability.get("approved_sdr_profile_ids", [])) if capability else set()
            if requirement.sdr_profile_id in profiles:
                evidence.append("APPROVED_SDR_PROFILE_PRESENT")
            else:
                errors.append("required SDR profile was not reported by the radio agent")

        return StandardsEvidence(not errors, tuple(sorted(set(evidence))), tuple(sorted(set(errors))), capability)
