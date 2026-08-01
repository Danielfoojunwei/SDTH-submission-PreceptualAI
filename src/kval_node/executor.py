from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from .commands import CommandError, run_checked
from .models import Profile
from .standards import RadioStandardsClient


@dataclass(frozen=True)
class ActivationResult:
    profile_id: str
    connection_uuid: str
    interface: str
    success: bool
    networkmanager_output: str
    radio_agent_output: dict[str, object] | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class NetworkManagerExecutor:
    """Activates only a pre-validated signed-catalogue NetworkManager UUID."""

    @staticmethod
    def _active_uuid_for_interface(interface: str) -> str | None:
        output = run_checked(
            [
                "nmcli",
                "--terse",
                "--escape",
                "no",
                "--fields",
                "UUID,DEVICE,STATE",
                "connection",
                "show",
                "--active",
            ],
            timeout_seconds=10.0,
        )
        for line in output.splitlines():
            fields = line.split(":", 2)
            if len(fields) == 3 and fields[1] == interface and fields[2] == "activated":
                return fields[0]
        return None

    def __init__(self) -> None:
        self._standards_client = RadioStandardsClient()

    def activate(self, profile: Profile, *, catalog_hash: str) -> ActivationResult:
        if os.environ.get("KVAL_ENABLE_APPLY") != "YES":
            raise PermissionError("network activation requires KVAL_ENABLE_APPLY=YES")
        radio_agent_output: dict[str, object] | None = None
        if profile.standards.capability_socket is not None and profile.standards.sdr_profile_id is not None:
            radio_agent_output = self._standards_client.activate(
                profile.standards.capability_socket,
                profile.profile_id,
                profile.standards.sdr_profile_id,
                profile.connection_uuid,
                catalog_hash,
            )
        output = run_checked(
            [
                "nmcli",
                "--wait",
                "30",
                "connection",
                "up",
                "uuid",
                profile.connection_uuid,
                "ifname",
                profile.interface,
            ],
            timeout_seconds=40.0,
        ).strip()
        active_uuid = self._active_uuid_for_interface(profile.interface)
        if active_uuid != profile.connection_uuid:
            raise CommandError(
                f"NetworkManager returned success but {profile.connection_uuid} is not active on {profile.interface}"
            )
        return ActivationResult(
            profile_id=profile.profile_id,
            connection_uuid=profile.connection_uuid,
            interface=profile.interface,
            success=True,
            networkmanager_output=output,
            radio_agent_output=radio_agent_output,
        )
