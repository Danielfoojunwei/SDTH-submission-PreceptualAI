/*
 * Adapted from ns-3 examples/wireless/wifi-ofdm-eht-validation.cc.
 * Copyright (c) 2021 DERONNE SOFTWARE ENGINEERING
 * SPDX-License-Identifier: GPL-2.0-only
 * Modifications: VHT 80 MHz / HT 20 MHz, two streams, packet success JSON.
 */
#include "ns3/core-module.h"
#include "ns3/wifi-module.h"
#include <cmath>
#include <fstream>
#include <iomanip>
using namespace ns3;
int main(int argc, char** argv)
{
    std::string output;
    uint32_t bytes = 1200;
    CommandLine cmd(__FILE__);
    cmd.AddValue("output", "JSON curve output", output);
    cmd.AddValue("bytes", "PHY payload bytes", bytes);
    cmd.Parse(argc, argv);
    NS_ABORT_MSG_IF(output.empty() || bytes == 0, "Explicit output and positive payload required");
    auto model = CreateObject<NistErrorRateModel>();
    std::ofstream out(output);
    NS_ABORT_MSG_IF(!out, "Cannot write output");
    out << std::setprecision(12) << "{\"source\":\"ns3_NistErrorRateModel\",\"ns3\":\"3.49\","
        << "\"payload_bytes\":" << bytes << ",\"streams\":2,\"guard_interval_ns\":800,\"points\":[";
    bool comma = false;
    for (bool band5 : {false, true})
    {
        for (uint8_t mcs = 0; mcs < (band5 ? 10 : 8); ++mcs)
        {
            auto mode = band5 ? VhtPhy::GetVhtMcs(mcs) : HtPhy::GetHtMcs(mcs+8);
            WifiTxVector tx;
            tx.SetMode(mode);
            tx.SetChannelWidth(band5 ? MHz_u{80} : MHz_u{20});
            tx.SetGuardInterval(NanoSeconds(800));
            tx.SetNss(2);
            tx.SetNTx(2);
            for (double snr = -5.; snr <= 45.; snr += .25)
            {
                double success = model->GetChunkSuccessRate(mode, tx, std::pow(10., snr/10.), bytes*8);
                if (comma) out << ',';
                comma = true;
                out << "{\"band\":\"" << (band5 ? "5g" : "2g") << "\",\"mcs\":" << unsigned(mcs)
                    << ",\"snr_db\":" << snr << ",\"rate_mbps\":" << mode.GetDataRate(tx)/1e6
                    << ",\"packet_success\":" << success << '}';
            }
        }
    }
    out << "]}\n";
}
