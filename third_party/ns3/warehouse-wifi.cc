/*
 * Adapted from ns-3 examples/wireless/wifi-he-network.cc and wifi-simple-infra.cc.
 * Copyright (c) 2016 SEBASTIEN DERONNE; Copyright (c) 2009 The Boeing Company.
 * Modifications: warehouse topology input, per-robot uplink traffic and metrics,
 * pinned client association, per-BSS channels, MinstrelHt rate control, JSON.
 * SPDX-License-Identifier: GPL-2.0-only
 */
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/wifi-module.h"
#include <fstream>
#include <iomanip>
#include <vector>

using namespace ns3;

static double Quantile(const Histogram& h, double q)
{
    uint64_t count = 0;
    for (uint32_t i = 0; i < h.GetNBins(); ++i) count += h.GetBinCount(i);
    if (!count) return -1;
    uint64_t cumulative = 0;
    for (uint32_t i = 0; i < h.GetNBins(); ++i)
    {
        cumulative += h.GetBinCount(i);
        if (cumulative >= q * count) return h.GetBinEnd(i);
    }
    return -1;
}

int main(int argc, char** argv)
{
    std::string input, output;
    uint32_t seed = 1, run = 1;
    double duration = 8.;
    CommandLine cmd(__FILE__);
    cmd.AddValue("input", "Topology: N M; M rows x y channel is5; N rows x y Mbps AP", input);
    cmd.AddValue("output", "JSON metrics file", output);
    cmd.AddValue("seed", "ns-3 seed", seed);
    cmd.AddValue("run", "Independent ns-3 run", run);
    cmd.AddValue("duration", "Application traffic duration in seconds", duration);
    cmd.Parse(argc, argv);
    NS_ABORT_MSG_IF(input.empty() || output.empty() || duration <= 0, "Explicit paths and positive duration required");
    RngSeedManager::SetSeed(seed);
    RngSeedManager::SetRun(run);
    std::ifstream source(input);
    uint32_t n = 0, m = 0;
    source >> n >> m;
    NS_ABORT_MSG_IF(!source || n == 0 || m == 0 || m > 200, "Invalid topology header");
    std::vector<double> ax(m), ay(m), x(n), y(n), demand(n);
    std::vector<uint32_t> channels(m), band5(m), association(n);
    for (uint32_t j = 0; j < m; ++j) source >> ax[j] >> ay[j] >> channels[j] >> band5[j];
    for (uint32_t i = 0; i < n; ++i)
    {
        source >> x[i] >> y[i] >> demand[i] >> association[i];
        NS_ABORT_MSG_IF(!source || association[i] >= m || demand[i] <= 0, "Invalid station record");
    }
    NodeContainer aps, stations;
    aps.Create(m);
    stations.Create(n);
    InternetStackHelper internet;
    internet.Install(aps);
    internet.Install(stations);
    MobilityHelper mobility;
    auto positions = CreateObject<ListPositionAllocator>();
    for (uint32_t j = 0; j < m; ++j) positions->Add(Vector(ax[j], ay[j], 5.));
    for (uint32_t i = 0; i < n; ++i) positions->Add(Vector(x[i], y[i], .3));
    mobility.SetPositionAllocator(positions);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    NodeContainer all;
    all.Add(aps); all.Add(stations);
    mobility.Install(all);

    YansWifiChannelHelper channel5, channel2;
    channel5.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    channel2.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    channel5.AddPropagationLoss("ns3::LogDistancePropagationLossModel", "Exponent", DoubleValue(3.3),
                                "ReferenceLoss", DoubleValue(46.7));
    channel2.AddPropagationLoss("ns3::LogDistancePropagationLossModel", "Exponent", DoubleValue(3.3),
                                "ReferenceLoss", DoubleValue(40.0));
    auto radioChannel5 = channel5.Create();
    auto radioChannel2 = channel2.Create();
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211ac);
    wifi.SetRemoteStationManager("ns3::MinstrelHtWifiManager");
    std::vector<Ipv4Address> stationAddresses(n);
    for (uint32_t j = 0; j < m; ++j)
    {
        // 802.11n on 2.4 GHz, 802.11ac on 5 GHz; shared propagation channel.
        wifi.SetStandard(band5[j] ? WIFI_STANDARD_80211ac : WIFI_STANDARD_80211n);
        YansWifiPhyHelper phy;
        phy.SetChannel(band5[j] ? radioChannel5 : radioChannel2);
        phy.Set("TxPowerStart", DoubleValue(16.));
        phy.Set("TxPowerEnd", DoubleValue(16.));
        phy.Set("ChannelSettings", StringValue("{" + std::to_string(channels[j]) +
                ", 20, " + (band5[j] ? std::string("BAND_5GHZ") : std::string("BAND_2_4GHZ")) + ", 0}"));
        WifiMacHelper mac;
        Ssid ssid("warehouse-" + std::to_string(j));
        mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
        NetDeviceContainer devices = wifi.Install(phy, mac, aps.Get(j));
        std::vector<uint32_t> members;
        for (uint32_t i = 0; i < n; ++i)
        {
            if (association[i] != j) continue;
            mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid), "ActiveProbing", BooleanValue(false));
            devices.Add(wifi.Install(phy, mac, stations.Get(i)));
            members.push_back(i);
        }
        Ipv4AddressHelper addresses;
        std::string subnet = "10." + std::to_string(j+1) + ".0.0";
        addresses.SetBase(subnet.c_str(), "255.255.0.0");
        auto interfaces = addresses.Assign(devices);
        for (uint32_t k = 0; k < members.size(); ++k)
        {
            uint32_t i = members[k];
            stationAddresses[i] = interfaces.GetAddress(k+1);
            uint16_t port = 9000+i;
            PacketSinkHelper sink("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
            auto receiver = sink.Install(aps.Get(j));
            receiver.Start(Seconds(0));
            OnOffHelper sender("ns3::UdpSocketFactory", InetSocketAddress(interfaces.GetAddress(0), port));
            sender.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
            sender.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
            sender.SetAttribute("DataRate", DataRateValue(DataRate(uint64_t(demand[i]*1e6))));
            sender.SetAttribute("PacketSize", UintegerValue(1200));
            auto application = sender.Install(stations.Get(i));
            application.Start(Seconds(2.));
            application.Stop(Seconds(2.+duration));
        }
    }
    FlowMonitorHelper helper;
    auto monitor = helper.InstallAll();
    Simulator::Stop(Seconds(duration+4.));
    Simulator::Run();
    monitor->CheckForLostPackets();
    auto classifier = DynamicCast<Ipv4FlowClassifier>(helper.GetClassifier());
    auto stats = monitor->GetFlowStats();
    std::ofstream out(output);
    NS_ABORT_MSG_IF(!out, "Cannot write report");
    out << std::setprecision(12) << "{\"source\":\"ns3_packet_simulation\",\"ns3\":\"3.49\","
        << "\"seed\":" << seed << ",\"run\":" << run << ",\"robots\":" << n
        << ",\"aps\":" << m << ",\"duration_s\":" << duration
        << ",\"rate_control\":\"MinstrelHt\",\"channel_width_mhz\":20,\"per_robot\":[";
    for (uint32_t i = 0; i < n; ++i)
    {
        if (i) out << ',';
        uint64_t tx = 0, rx = 0, bytes = 0;
        double delay = 0, p95 = -1, p99 = -1;
        for (const auto& [id, stat] : stats)
        {
            auto tuple = classifier->FindFlow(id);
            if (tuple.sourceAddress != stationAddresses[i]) continue;
            tx += stat.txPackets; rx += stat.rxPackets; bytes += stat.rxBytes;
            delay += stat.delaySum.GetSeconds();
            p95 = Quantile(stat.delayHistogram, .95);
            p99 = Quantile(stat.delayHistogram, .99);
        }
        out << "{\"robot\":" << i << ",\"ap\":" << association[i] << ",\"demand_mbps\":" << demand[i]
            << ",\"throughput_mbps\":" << bytes*8./duration/1e6 << ",\"tx_packets\":" << tx
            << ",\"rx_packets\":" << rx << ",\"unreceived_packets\":" << (tx-rx)
            << ",\"mean_delay_s\":" << (rx ? std::to_string(delay/rx) : "null")
            << ",\"p95_delay_s_upper_bin\":" << (p95 >= 0 ? std::to_string(p95) : "null")
            << ",\"p99_delay_s_upper_bin\":" << (p99 >= 0 ? std::to_string(p99) : "null") << '}';
    }
    out << "]}\n";
    Simulator::Destroy();
}
