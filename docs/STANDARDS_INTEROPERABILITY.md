# Standards interoperability

Kval separates a standards-aligned integration contract from a certification
claim. A profile becomes eligible only when its required evidence is observed
from NetworkManager/ModemManager or a root-owned local radio standards agent.
Self-declared catalogue text alone is not sufficient.

## Supported standards vocabulary

| Area | Identifier | Kval use |
|---|---|---|
| Cellular identity | 3GPP TS 23.003 | Validates configured DNN and S-NSSAI structure |
| NTN baseline | 3GPP Release 17/18 | Capability gate for standards-agent-attested NR-NTN profiles |
| SDR architecture | ETSI EN 303 648 | Radio-equipment reconfiguration architecture identifier |
| SDR interfaces | ETSI EN 303 681-1 through -4 | gMURI, gRRFI, gURAI, and gRPI capability identifiers |
| Spectrum policy | IEEE 1900.5.1-2020 | Vendor-independent policy-language provenance |
| Spectrum model | IEEE 1900.5.2-2024 | Spectrum-consumption-model provenance |
| Radio transport | ANSI/VITA 49.2-2017 (R2024) | VRT control/context capability provenance |

Official references:

- 3GPP Release 18: https://www.3gpp.org/specifications-technologies/releases/release-18
- ETSI TC RRS: https://www.etsi.org/technical-groups/rrs/
- ETSI RRS activity and EN 303 681 interface family: https://portal.etsi.org/TBSiteMap/rrs/activityreport
- IEEE 1900.5.1-2020: https://standards.ieee.org/standard/1900_5_1-2020.html
- IEEE 1900.5.2-2024: https://standards.ieee.org/ieee/1900.5.2/10713/
- VITA 49 overview: https://www.vita.com/page-1855484

## 3GPP operation

Commercial modem profiles use real NetworkManager UUIDs and ModemManager
access-technology evidence. The following mappings are enforced:

- 3G: UTRAN-family observations.
- 4G: E-UTRAN/LTE observations.
- 5G: NR observations.
- 5G NTN: NR-NTN attested by a standards agent.

DNN is compared against the live NetworkManager `gsm.apn` value. S-NSSAI and a
minimum 3GPP release are not exposed consistently by generic Linux modem APIs,
so Kval requires them from the local radio standards agent. If that agent or
evidence is absent, the profile is rejected.

This means Kval can select a pre-provisioned 3GPP network slice; it does not
create a network slice or allocate spectrum itself. Operators and networks own
the S-NSSAI, DNN, subscription, QoS, and regulatory configuration.

## SDR operation

Kval does not load arbitrary waveforms. A signed catalogue references an
operator-approved SDR profile ID. A separate radio implementation—such as a
vendor ETSI RRS gateway—must expose a root-owned Unix socket and the contract
below. The Kval client verifies capabilities and requests activation of only
that approved ID.

Capability request:

```json
{"operation":"capabilities","profile_id":"ntn-primary","connection_uuid":"..."}
```

Minimum response shape:

```json
{
  "schema": "kval-radio-agent/v1",
  "success": true,
  "capability": {
    "connection_uuid": "...",
    "rat": "nr-ntn",
    "3gpp_release": 18,
    "standard_identifiers": [
      "3gpp-release-18",
      "etsi-en-303-648",
      "etsi-en-303-681-1",
      "ansi-vita-49.2-2017-r2024"
    ],
    "approved_sdr_profile_ids": ["operator-ntn-profile-01"],
    "implementation_id": "vendor-and-version",
    "conformance_report_sha256": "64-hex-character-report-hash"
  }
}
```

Activation uses the same socket and includes the signed catalogue hash. The
agent must return `success: true` and the exact active profile ID. Kval then
activates and verifies the corresponding NetworkManager UUID.

## What still requires laboratory evidence

- 3GPP modem/GCF or PTCRB certification and operator acceptance.
- ETSI RRS interface conformance against the licensed standards.
- IEEE 1900.5 policy/model conformance.
- VITA 49.2 packet interoperability with the chosen radio.
- Country-specific spectrum and radio-equipment approval.

Kval's software gates make these integrations possible and fail closed when
their evidence is absent. They do not replace external conformance testing.

