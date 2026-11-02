"""L0-L2: packet-error curves for the slot simulator and the layered-bits study.

Link curves come from the ns-3 NIST error model already archived by the v2
study (VHT80 / HT20, two streams, 1,200-byte payloads). The layered study
(moonshot M1) runs a real Sionna chain: 5G LDPC, a hierarchical 16-QAM
constellation, flat Rayleigh block fading and AWGN, against time sharing at
equal symbols and equal average power. It is a simulation/SDR result, not a
Wi-Fi-compatible waveform.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CURVES = ROOT / "artifacts" / "warehouse_v2" / "studies" / "wifi_link_curves.json"


class LinkCurves:
    """Packet success vs SNR for each MCS of a band; vectorised lookups."""

    def __init__(self, path: Path = CURVES):
        data = json.loads(Path(path).read_text())
        self.source = data["source"]
        self.tables = {}
        for band in ("2g", "5g"):
            rows = [p for p in data["points"] if p["band"] == band]
            mcs = sorted({p["mcs"] for p in rows})
            snr = np.array(sorted({p["snr_db"] for p in rows}), dtype=float)
            succ = np.zeros((len(mcs), len(snr)))
            rate = np.zeros(len(mcs))
            for p in rows:
                m = mcs.index(p["mcs"])
                succ[m, np.searchsorted(snr, p["snr_db"])] = p["packet_success"]
                rate[m] = p["rate_mbps"]
            self.tables[band] = (snr, succ, rate)
        # 6 GHz: 160 MHz, same per-MCS error behaviour vs SNR, double rate.
        snr, succ, rate = self.tables["5g"]
        self.tables["6g"] = (snr, succ, 2.0 * rate)

    def success(self, band: str, mcs: np.ndarray, snr_db: np.ndarray) -> np.ndarray:
        """Packet success for integer MCS indices at the given SNRs."""
        snr, succ, _ = self.tables[band]
        idx = np.clip(np.round((np.asarray(snr_db) - snr[0]) / (snr[1] - snr[0])).astype(int),
                      0, len(snr) - 1)
        return succ[np.asarray(mcs), idx]

    def rates(self, band: str) -> np.ndarray:
        return self.tables[band][2]

    def n_mcs(self, band: str) -> int:
        return len(self.tables[band][2])

    def threshold_db(self, band: str, target: float = 0.9) -> np.ndarray:
        """SNR at which each MCS first reaches `target` packet success."""
        snr, succ, _ = self.tables[band]
        out = np.empty(succ.shape[0])
        for m in range(succ.shape[0]):
            hit = np.nonzero(succ[m] >= target)[0]
            out[m] = snr[hit[0]] if len(hit) else np.inf
        return out


def hierarchical_points(alpha: float) -> np.ndarray:
    """16 points, label bits (I_base, Q_base, I_enh, Q_enh), unit energy.

    Per dimension the base bit picks the sign and the enhancement bit picks
    inner/outer: amplitude alpha + 1 or alpha - 1. alpha = 2 is uniform
    Gray 16-QAM; larger alpha separates quadrants (protects base bits)."""
    pts = np.zeros(16, dtype=complex)
    for idx in range(16):
        bi, bq, ei, eq = (idx >> 3) & 1, (idx >> 2) & 1, (idx >> 1) & 1, idx & 1
        ai = (1 - 2 * bi) * (alpha + (1 if ei == 0 else -1))
        aq = (1 - 2 * bq) * (alpha + (1 if eq == 0 else -1))
        pts[idx] = ai + 1j * aq
    return pts / math.sqrt(np.mean(np.abs(pts) ** 2))


def layered_study(*, seed=0, n_sym=648, batch=64, batches=8,
                  snr_db=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26),
                  alphas=(2.0, 3.0, 4.0), fading="rayleigh") -> dict:
    """Critical bits (base, LDPC r=1/2) + bulk bits (enhancement, r=3/4) in
    n_sym symbols, versus time sharing: critical 16-QAM r=1/2 in n_sym/2
    symbols, bulk 16-QAM r=3/4 in the other n_sym/2. Same bits, symbols and
    average power. Reports block error per stream."""
    import torch
    from sionna.phy import config
    from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder
    from sionna.phy.mapping import BinarySource, Constellation, Demapper, Mapper

    config.seed = seed
    torch.manual_seed(seed)
    src = BinarySource()
    nc = 2 * n_sym                       # coded bits per stream
    kc, kb = nc // 2, (3 * nc) // 4
    enc_c, enc_b = LDPC5GEncoder(kc, nc), LDPC5GEncoder(kb, nc)
    dec_c, dec_b = LDPC5GDecoder(enc_c, num_iter=20), LDPC5GDecoder(enc_b, num_iter=20)

    def channel(x, no):
        if fading == "rayleigh":   # one complex gain per codeword block
            h = (torch.randn(x.shape[0], 1) + 1j * torch.randn(x.shape[0], 1)) / math.sqrt(2)
        else:
            h = torch.ones(x.shape[0], 1, dtype=torch.complex64)
        h = h.to(torch.complex64)
        n = (torch.randn(x.shape) + 1j * torch.randn(x.shape)) * math.sqrt(no / 2)
        y = h * x + n.to(torch.complex64)
        # Perfect CSI zero-forcing: per-block effective noise no/|h|^2.
        return y / h, (no / (h.abs() ** 2)).real.expand(x.shape).to(torch.float32)

    def run_scheme(points, critical_bit_positions, layered):
        const = Constellation("custom", 4, points=torch.tensor(points, dtype=torch.complex64))
        mapper, demapper = Mapper(constellation=const), Demapper("app", constellation=const)
        rows = []
        for snr in snr_db:
            no = 10.0 ** (-snr / 10.0)
            err = {"critical": 0, "bulk": 0}
            total = 0
            with torch.inference_mode():
                for _ in range(batches):
                    mc, mb = src([batch, kc]), src([batch, kb])
                    cc, cb = enc_c(mc), enc_b(mb)
                    if layered:
                        bits = torch.stack([cc[:, 0::2], cc[:, 1::2], cb[:, 0::2], cb[:, 1::2]], -1)
                        x = mapper(bits.reshape(batch, -1))
                        xe, noe = channel(x, no)
                        llr = demapper(xe, noe).reshape(batch, n_sym, 4)
                        lc = torch.stack([llr[..., 0], llr[..., 1]], -1).reshape(batch, nc)
                        lb = torch.stack([llr[..., 2], llr[..., 3]], -1).reshape(batch, nc)
                    else:   # time sharing: each stream fills its own n_sym/2 symbols
                        xc, xb = mapper(cc), mapper(cb)
                        x = torch.cat([xc, xb], 1)        # one fading block per frame
                        xe, noe = channel(x, no)
                        half = xc.shape[1]
                        lc = demapper(xe[:, :half], noe[:, :half])
                        lb = demapper(xe[:, half:], noe[:, half:])
                    err["critical"] += int((dec_c(lc) != mc).any(-1).sum())
                    err["bulk"] += int((dec_b(lb) != mb).any(-1).sum())
                    total += batch
            rows.append({"snr_db": float(snr), "critical_bler": err["critical"] / total,
                         "bulk_bler": err["bulk"] / total, "blocks": total})
        return rows

    out = {"source": "sionna_ldpc_hierarchical_16qam", "fading": fading, "n_sym": n_sym,
           "critical_info_bits": kc, "bulk_info_bits": kb, "seed": seed,
           "time_sharing": run_scheme(hierarchical_points(2.0), None, False)}
    for a in alphas:
        out[f"layered_alpha{a:g}"] = run_scheme(hierarchical_points(a), None, True)

    def snr_at(rows, key, target=0.01):
        for r in rows:
            if r[key] <= target:
                return r["snr_db"]
        return None

    out["snr_for_1pct_bler"] = {
        name: {"critical": snr_at(rows, "critical_bler"), "bulk": snr_at(rows, "bulk_bler")}
        for name, rows in out.items() if isinstance(rows, list)}
    return out
