# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Sionna tutorials/phy/Hello_World.ipynb; see third_party/README.md.
"""Execute a real coded Sionna PHY chain, without pretending it is Wi-Fi."""
from __future__ import annotations

from importlib.metadata import version
import time

import torch


def run_phy(*, seed=0, batch_size=128, batches=8, snr_db=(-2, 0, 2, 4, 6, 8, 10, 14, 18)):
    from sionna.phy import config
    from sionna.phy.mapping import BinarySource, Mapper, Demapper
    from sionna.phy.channel import AWGN
    from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
    config.seed = seed
    config.device = 'cpu'
    source, channel = BinarySource(), AWGN()
    rows = []
    started = time.perf_counter()
    for bits, rate in [(2, .5), (4, .5), (6, .75)]:
        n = 1296
        k = int(n*rate)
        encoder = LDPC5GEncoder(k, n, num_bits_per_symbol=bits)
        decoder = LDPC5GDecoder(encoder, num_iter=12)
        mapper, demapper = Mapper('qam', bits), Demapper('app', 'qam', bits)
        for snr in snr_db:
            bit_errors = block_errors = total_bits = total_blocks = 0
            no = torch.tensor(10.**(-snr/10.))  # Es/N0, unit-energy constellation.
            with torch.inference_mode():
                for _ in range(batches):
                    message = source([batch_size, k])
                    transmitted = mapper(encoder(message))
                    received = channel(transmitted, no)
                    estimate = decoder(demapper(received, no))
                    errors = message != estimate
                    bit_errors += int(errors.sum())
                    block_errors += int(errors.any(-1).sum())
                    total_bits += errors.numel()
                    total_blocks += batch_size
            row = {'bits_per_symbol': bits, 'code_rate': rate, 'information_bits': k,
                   'codeword_bits': n, 'es_n0_db': float(snr), 'bit_errors': bit_errors,
                   'bits': total_bits, 'block_errors': block_errors, 'blocks': total_blocks,
                   'ber': bit_errors/total_bits, 'bler': block_errors/total_blocks}
            print(f'Sionna QAM={2**bits} rate={rate} Es/N0={snr}: BLER={row["bler"]:.4f}', flush=True)
            rows.append(row)
    return {'source': 'sionna_coded_phy_simulation', 'sionna_version': version('sionna'),
            'seed': seed, 'device': 'cpu', 'channel': 'AWGN', 'coding': '3GPP 5G LDPC',
            'wifi_link_curve': False, 'elapsed_s': time.perf_counter()-started, 'points': rows,
            'limits': ['Not an IEEE 802.11 code or packet/MAC model',
                       'Zero observed errors are a finite-sample result, not zero error probability']}
