/*
 * Adapted from ns-3 wifi-simple-infra.cc and this project's warehouse-wifi.cc.
 * Copyright (c) 2009 The Boeing Company; Copyright (c) 2016 SEBASTIEN DERONNE.
 * Modifications: committed routes, burst traffic, exact endpoint deadlines,
 * PHY retry/busy and MAC queue telemetry, paired RTS experiments.
 * SPDX-License-Identifier: GPL-2.0-only
 */
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/wifi-module.h"
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <sstream>
using namespace ns3;

struct Counter
{
    uint64_t tx[2]{}, rx[2]{}, bytes[2]{}, late[2]{};
    uint64_t attempts{}, retries{};
    double busy{}, delaySum{};
    std::vector<double> delays;
    std::vector<double> criticalArrivals;
};
static std::vector<Counter> totals, bins;
static std::vector<Ptr<WifiNetDevice>> stationDevices;
static std::vector<std::string> traceRows;
static double trafficStart = 2.;
static double trafficEnd = 14.;

static void Sent(uint32_t robot, uint32_t flow, Ptr<const Packet> packet)
{
    totals[robot].tx[flow]++; bins[robot].tx[flow]++;
}
static void Received(uint32_t robot, uint32_t flow, Ptr<const Packet> packet,
                     const Address& from, const Address& to, const SeqTsSizeHeader& header)
{
    double delay = (Simulator::Now()-header.GetTs()).GetSeconds();
    for (auto* item : {&totals[robot], &bins[robot]})
    {
        item->rx[flow]++; item->bytes[flow] += header.GetSize();
        item->late[flow] += delay > (flow == 0 ? .05 : .25);
        item->delaySum += delay; item->delays.push_back(delay);
        if (flow==0 && Simulator::Now().GetSeconds()<=trafficEnd) item->criticalArrivals.push_back(Simulator::Now().GetSeconds());
    }
}
static void Transmit(uint32_t robot, Ptr<const Packet> packet, double power)
{
    WifiMacHeader header; packet->PeekHeader(header);
    if (!header.IsData()) return;
    totals[robot].attempts++; bins[robot].attempts++;
    totals[robot].retries += header.IsRetry(); bins[robot].retries += header.IsRetry();
}
static void State(uint32_t robot, Time start, Time duration, WifiPhyState state)
{
    if (state == WifiPhyState::IDLE || state == WifiPhyState::SLEEP || state == WifiPhyState::OFF) return;
    double begin = std::max(start.GetSeconds(), trafficStart);
    double end = std::min(trafficEnd,start.GetSeconds()+duration.GetSeconds());
    double busy = std::max(0., end-begin);
    totals[robot].busy += busy; bins[robot].busy += busy;
}
static double Quantile(std::vector<double> values, double q)
{
    if (values.empty()) return -1.;
    std::sort(values.begin(), values.end());
    return values[std::min(values.size()-1, size_t(q*(values.size()-1)))];
}
static void WriteCounter(std::ostream& out, const Counter& c)
{
    out << "\"critical_tx\":" << c.tx[0] << ",\"critical_rx\":" << c.rx[0]
        << ",\"critical_late\":" << c.late[0] << ",\"bulk_tx\":" << c.tx[1]
        << ",\"bulk_rx\":" << c.rx[1] << ",\"bulk_late\":" << c.late[1]
        << ",\"delivered_bytes\":" << c.bytes[0]+c.bytes[1]
        << ",\"phy_data_attempts\":" << c.attempts << ",\"phy_retry_attempts\":" << c.retries
        << ",\"busy_seconds\":" << c.busy;
    double p95 = Quantile(c.delays, .95), p99 = Quantile(c.delays, .99);
    out << ",\"p95_delay_s\":" << (p95 < 0 ? "null" : std::to_string(p95))
        << ",\"p99_delay_s\":" << (p99 < 0 ? "null" : std::to_string(p99));
}
static void Sample(uint32_t index, uint32_t count)
{
    std::ostringstream row; row << std::setprecision(12) << "{\"second\":" << index << ",\"robots\":[";
    for (uint32_t i=0; i<bins.size(); ++i)
    {
        if (i) row << ',';
        row << "{\"robot\":" << i << ','; WriteCounter(row, bins[i]);
        uint32_t queue = 0;
        for (auto ac : {AC_BE,AC_VO,AC_VI,AC_BK}) queue += stationDevices[i]->GetMac()->GetTxopQueue(ac)->GetNPackets();
        row << ",\"queue_packets\":" << queue << '}'; bins[i] = Counter();
    }
    row << "]}"; traceRows.push_back(row.str());
    if (index < count) Simulator::Schedule(Seconds(1), &Sample, index+1, count);
}

int main(int argc, char** argv)
{
    std::string input, output;
    uint32_t seed=1, run=1; double duration=12.; bool rts=false, burst=true;
    bool matchedPhy=true;
    CommandLine cmd(__FILE__);
    cmd.AddValue("input", "N M; AP x y channel band5 busy; STA demand critical outage then seven xy route points", input);
    cmd.AddValue("output", "Actual event report", output);
    cmd.AddValue("seed", "Random seed", seed); cmd.AddValue("run", "Random run", run);
    cmd.AddValue("duration", "Traffic duration", duration); cmd.AddValue("rts", "Enable RTS/CTS", rts);
    cmd.AddValue("matchedPhy", "VHT80/HT20 with two streams instead of the preliminary single-stream 20MHz profile", matchedPhy);
    cmd.AddValue("burst", "Bursty bulk traffic", burst); cmd.Parse(argc, argv);
    trafficEnd=trafficStart+duration;
    NS_ABORT_MSG_IF(input.empty() || output.empty() || duration<5, "Explicit valid inputs required");
    RngSeedManager::SetSeed(seed); RngSeedManager::SetRun(run);
    std::ifstream source(input); uint32_t n=0,m=0; double clientPower=15.; std::string schema;
    source >> schema >> n >> m >> clientPower;
    NS_ABORT_MSG_IF(!source || schema!="WIRELESSJEV_PACKET_V2" || n<1 || n>200 || m<1 || m>128, "Invalid header");
    std::vector<double> ax(m),ay(m),busy(m),apPower(m),demand(n),critical(n),outage(n);
    std::vector<uint32_t> channel(m),band(m),association(n);
    for (uint32_t j=0;j<m;++j) source >> ax[j] >> ay[j] >> channel[j] >> band[j] >> busy[j] >> apPower[j];
    std::vector<std::vector<Vector>> routes(n);
    for (uint32_t i=0;i<n;++i)
    {
        source >> demand[i] >> critical[i] >> association[i] >> outage[i];
        for (uint32_t s=0;s<7;++s) { double x,y; source >> x >> y; routes[i].push_back(Vector(x,y,.3)); }
        NS_ABORT_MSG_IF(!source || association[i]>=m || critical[i]<=0 || demand[i]<critical[i], "Invalid robot");
    }
    NodeContainer aps,stations,external; aps.Create(m); stations.Create(n); external.Create(m);
    InternetStackHelper internet; internet.Install(aps); internet.Install(stations); internet.Install(external);
    MobilityHelper fixed; fixed.SetMobilityModel("ns3::ConstantPositionMobilityModel"); fixed.Install(aps);
    for (uint32_t j=0;j<m;++j) aps.Get(j)->GetObject<MobilityModel>()->SetPosition(Vector(ax[j],ay[j],5.));
    fixed.Install(external);
    for (uint32_t j=0;j<m;++j) external.Get(j)->GetObject<MobilityModel>()->SetPosition(Vector(std::min(119.,ax[j]+8.),std::min(79.,ay[j]+8.),.3));
    MobilityHelper moving; moving.SetMobilityModel("ns3::WaypointMobilityModel"); moving.Install(stations);
    for (uint32_t i=0;i<n;++i)
    {
        auto motion=stations.Get(i)->GetObject<WaypointMobilityModel>();
        motion->AddWaypoint(Waypoint(Seconds(0),routes[i][0]));
        for (uint32_t s=0;s<7;++s) motion->AddWaypoint(Waypoint(Seconds(2.+5*s),routes[i][s]));
    }
    YansWifiChannelHelper c5,c2;
    for (auto* c : {&c5,&c2}) c->SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    c5.AddPropagationLoss("ns3::LogDistancePropagationLossModel","Exponent",DoubleValue(3.3),"ReferenceLoss",DoubleValue(46.7));
    c2.AddPropagationLoss("ns3::LogDistancePropagationLossModel","Exponent",DoubleValue(3.3),"ReferenceLoss",DoubleValue(40.));
    auto radio5=c5.Create(),radio2=c2.Create();
    totals.resize(n); bins.resize(n); stationDevices.resize(n);
    WifiHelper wifi;
    for (uint32_t j=0;j<m;++j)
    {
        wifi.SetStandard(band[j] ? WIFI_STANDARD_80211ac : WIFI_STANDARD_80211n);
        wifi.SetRemoteStationManager("ns3::MinstrelHtWifiManager","RtsCtsThreshold",UintegerValue(rts ? 0 : 65535));
        YansWifiPhyHelper phy; phy.SetChannel(band[j] ? radio5 : radio2);
        phy.SetErrorRateModel("ns3::NistErrorRateModel"); phy.Set("RxNoiseFigure",DoubleValue(7.));
        phy.Set("TxPowerStart",DoubleValue(apPower[j])); phy.Set("TxPowerEnd",DoubleValue(apPower[j]));
        uint32_t centre=channel[j],width=20;
        if (matchedPhy)
        {
            phy.Set("Antennas",UintegerValue(2));
            phy.Set("MaxSupportedTxSpatialStreams",UintegerValue(2));
            phy.Set("MaxSupportedRxSpatialStreams",UintegerValue(2));
            if (band[j])
            {
                width=80;
                if (centre==36) centre=42; else if (centre==52) centre=58;
                else if (centre==100) centre=106; else if (centre==116) centre=122;
                else if (centre==132) centre=138; else if (centre==149) centre=155;
                else NS_ABORT_MSG("Unsupported 80MHz primary channel");
            }
        }
        phy.Set("ChannelSettings",StringValue("{"+std::to_string(centre)+", "+std::to_string(width)+", "+(band[j] ? std::string("BAND_5GHZ") : std::string("BAND_2_4GHZ"))+", 0}"));
        WifiMacHelper mac; Ssid ssid("contract-"+std::to_string(j));
        mac.SetType("ns3::ApWifiMac","Ssid",SsidValue(ssid));
        auto devices=wifi.Install(phy,mac,aps.Get(j)); wifi.AssignStreams(devices,100000+j*20);
        phy.Set("TxPowerStart",DoubleValue(clientPower)); phy.Set("TxPowerEnd",DoubleValue(clientPower));
        mac.SetType("ns3::StaWifiMac","Ssid",SsidValue(ssid),"ActiveProbing",BooleanValue(false));
        auto background=wifi.Install(phy,mac,external.Get(j)); wifi.AssignStreams(background,200000+j*20);
        devices.Add(background); std::vector<uint32_t> members;
        for (uint32_t i=0;i<n;++i) if (association[i]==j)
        {
            mac.SetType("ns3::StaWifiMac","Ssid",SsidValue(ssid),"ActiveProbing",BooleanValue(false));
            auto station=wifi.Install(phy,mac,stations.Get(i)); devices.Add(station); members.push_back(i);
            wifi.AssignStreams(station,300000+i*20);
            stationDevices[i]=DynamicCast<WifiNetDevice>(station.Get(0));
            stationDevices[i]->GetPhy()->TraceConnectWithoutContext("PhyTxBegin",MakeBoundCallback(&Transmit,i));
            stationDevices[i]->GetPhy()->GetState()->TraceConnectWithoutContext("State",MakeBoundCallback(&State,i));
        }
        Ipv4AddressHelper addresses; addresses.SetBase(("10."+std::to_string(j+1)+".0.0").c_str(),"255.255.0.0");
        auto interfaces=addresses.Assign(devices);
        PacketSinkHelper noiseSink("ns3::UdpSocketFactory",InetSocketAddress(Ipv4Address::GetAny(),8000));
        auto noiseReceiver=noiseSink.Install(external.Get(j)); noiseReceiver.Start(Seconds(0));
        OnOffHelper noise("ns3::UdpSocketFactory",InetSocketAddress(interfaces.GetAddress(1),8000));
        double duty=std::clamp(busy[j],.02,.95);
        noise.SetAttribute("OnTime",StringValue("ns3::ExponentialRandomVariable[Mean="+std::to_string(duty)+"]"));
        noise.SetAttribute("OffTime",StringValue("ns3::ExponentialRandomVariable[Mean="+std::to_string(1-duty)+"]"));
        noise.SetAttribute("DataRate",DataRateValue(DataRate("8Mbps"))); noise.SetAttribute("PacketSize",UintegerValue(1200));
        auto interferer=noise.Install(aps.Get(j)); DynamicCast<OnOffApplication>(interferer.Get(0))->AssignStreams(400000+j*4);
        interferer.Start(Seconds(2)); interferer.Stop(Seconds(2+duration));
        for (uint32_t k=0;k<members.size();++k)
        {
            uint32_t i=members[k];
            for (uint32_t f=0;f<2;++f)
            {
                double rate=f==0 ? critical[i] : demand[i]-critical[i]; if (rate<=0) continue;
                uint16_t port=9000+2*i+f;
                PacketSinkHelper sink("ns3::UdpSocketFactory",InetSocketAddress(Ipv4Address::GetAny(),port));
                sink.SetAttribute("EnableSeqTsSizeHeader",BooleanValue(true));
                auto receiver=sink.Install(aps.Get(j)); receiver.Start(Seconds(0));
                receiver.Get(0)->TraceConnectWithoutContext("RxWithSeqTsSize",MakeBoundCallback(&Received,i,f));
                OnOffHelper sender("ns3::UdpSocketFactory",InetSocketAddress(interfaces.GetAddress(0),port));
                bool onoff=burst && f==1;
                sender.SetAttribute("OnTime",StringValue(onoff ? "ns3::ExponentialRandomVariable[Mean=0.3]" : "ns3::ConstantRandomVariable[Constant=1]"));
                sender.SetAttribute("OffTime",StringValue(onoff ? "ns3::ExponentialRandomVariable[Mean=0.2]" : "ns3::ConstantRandomVariable[Constant=0]"));
                sender.SetAttribute("DataRate",DataRateValue(DataRate(uint64_t(rate*1e6/(onoff ? .6 : 1.)))));
                sender.SetAttribute("PacketSize",UintegerValue(f==0 ? 240 : 1200));
                sender.SetAttribute("EnableSeqTsSizeHeader",BooleanValue(true));
                auto application=sender.Install(stations.Get(i));
                DynamicCast<OnOffApplication>(application.Get(0))->AssignStreams(500000+i*8+f*2);
                application.Get(0)->TraceConnectWithoutContext("Tx",MakeBoundCallback(&Sent,i,f));
                application.Start(Seconds(2.+outage[i])); application.Stop(Seconds(2.+duration));
                if (f==0)
                {
                    auto app=DynamicCast<OnOffApplication>(application.Get(0));
                    Simulator::Schedule(Seconds(2.+outage[i]+.000001),[app]() {
                        if (app->GetSocket()) app->GetSocket()->SetPriority(6);
                    });
                }
            }
        }
    }
    // Random access includes naturally hidden transmitters. This paired arm
    // changes RTS/CTS; it does not assert every topology has a hidden terminal.
    Simulator::Schedule(Seconds(3.),&Sample,1,uint32_t(duration));
    Simulator::Stop(Seconds(duration+4.)); Simulator::Run();
    std::ofstream out(output); NS_ABORT_MSG_IF(!out,"Cannot write report");
    out << std::setprecision(12) << "{\"source\":\"ns3_packet_events\",\"version\":\"3.49\",\"robots\":" << n
        << ",\"seed\":" << seed << ",\"run\":" << run << ",\"duration_s\":" << duration
        << ",\"rts_cts\":" << (rts ? "true" : "false") << ",\"bursty_bulk\":" << (burst ? "true" : "false")
        << ",\"phy_profile\":\"" << (matchedPhy ? "vht80_ht20_2ss" : "preliminary_20mhz_1ss")
        << "\",\"client_tx_dbm\":" << clientPower << ",\"per_robot\":[";
    for (uint32_t i=0;i<n;++i)
    {
        if (i) out << ',';
        out << "{\"robot\":" << i << ",\"ap\":" << association[i] << ",\"offered_mbps\":" << demand[i]
            << ",\"assumed_application_mute_s\":" << outage[i] << ','; WriteCounter(out,totals[i]);
        double previous=trafficStart,longest=0.;
        for (double at : totals[i].criticalArrivals) { longest=std::max(longest,at-previous); previous=at; }
        longest=std::max(longest,trafficEnd-previous);
        out << ",\"longest_critical_silence_s\":" << longest << ",\"first_critical_packet_s\":"
            << (totals[i].criticalArrivals.empty() ? "null" : std::to_string(totals[i].criticalArrivals[0]-trafficStart)) << '}';
    }
    out << "],\"telemetry\":[";
    for (uint32_t i=0;i<traceRows.size();++i) { if (i) out << ','; out << traceRows[i]; }
    out << "],\"limits\":[\"Log-distance propagation lacks rack/material calibration\","
        << "\"Switch cost is assumed application muting, not association or authentication emulation\","
        << "\"Absent transmissions during mute are outside generated-packet deadline denominators\","
        << "\"External duty drives an independent 8Mbps burst source per BSS; this is not measured airtime calibration\","
        << "\"Busy time includes RX/TX/CCA and is collected from completed PHY state intervals\"]}\n";
    Simulator::Destroy();
}
