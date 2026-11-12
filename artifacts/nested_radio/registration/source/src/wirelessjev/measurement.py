"""Validate recorded measurements without turning missing values into data."""
from __future__ import annotations
import math
from statistics import median

SWITCH_STAGES=('request_submitted','association','authentication','ip_ready','dhcp','first_packet')


def switch_timing(events):
    """Analyse actual same-clock events, retaining absent and optional stages.

    The caller must supply device-generated events. Client request completion
    is not reinterpreted as authentication, IP readiness or first delivery.
    """
    seen={}
    for event in events:
        if event.get('stage') not in SWITCH_STAGES:raise ValueError('Unknown switch stage')
        if event.get('source') not in ['hardware','ns3_packet_events']:raise ValueError('Explicit event source required')
        if not event.get('clock_id') or type(event.get('monotonic_ns')) is not int or event['monotonic_ns']<0:
            raise ValueError('Explicit monotonic event clock required')
        if event['stage'] in seen:raise ValueError('Duplicate switch stage')
        if event['stage']=='first_packet' and (not isinstance(event.get('endpoint'),str) or not event['endpoint']
                or type(event.get('received_bytes')) is not int or event['received_bytes']<=0):
            raise ValueError('First-packet evidence needs an explicit endpoint and positive received-byte count')
        if seen and (event['clock_id']!=next(iter(seen.values()))['clock_id'] or event['source']!=next(iter(seen.values()))['source']):
            raise ValueError('Clock domains or measurement origins cannot be mixed')
        seen[event['stage']]=event
    if 'request_submitted' not in seen:raise ValueError('No submission event recorded')
    start=seen['request_submitted']['monotonic_ns']
    elapsed={stage:None if stage not in seen else (seen[stage]['monotonic_ns']-start)/1e9 for stage in SWITCH_STAGES}
    if any(value is not None and value<0 for value in elapsed.values()):raise ValueError('A stage predates submission')
    # Some networks preserve IP, skip DHCP, or report association/auth in a
    # different order. Only IP-to-endpoint ordering is checked when observed.
    if elapsed['ip_ready'] is not None and elapsed['first_packet'] is not None and elapsed['first_packet']<elapsed['ip_ready']:
        raise ValueError('First-packet event predates reported IP readiness')
    return {'source':seen['request_submitted']['source'],'clock_id':seen['request_submitted']['clock_id'],
        'seconds_since_submission':elapsed,'dhcp_required':False,'missing_stages':[k for k,v in elapsed.items() if v is None],
        'first_packet_observed':elapsed['first_packet'] is not None,'stages_imputed':False}


def packet_features(row):
    """Observed per-second ns-3 counters; never infer retries from loss."""
    keys=['critical_tx','critical_rx','critical_late','bulk_tx','bulk_rx','bulk_late',
          'delivered_bytes','phy_data_attempts','phy_retry_attempts','queue_packets','busy_seconds']
    if any(key not in row or not isinstance(row[key],(int,float)) or not math.isfinite(row[key]) or row[key]<0 for key in keys):
        raise ValueError('Missing or invalid actual packet counters')
    if row['phy_retry_attempts']>row['phy_data_attempts'] or row['critical_late']>row['critical_rx'] or row['bulk_late']>row['bulk_rx']:
        raise ValueError('Impossible retry or endpoint-delay counters')
    # RX counts in one wall-clock bin may exceed that bin's TX counts because
    # queued packets from previous bins can arrive; do not reject or relabel it.
    return [math.log1p(row['critical_tx']),math.log1p(row['bulk_tx']),math.log1p(row['delivered_bytes']),
        row['phy_retry_attempts']/max(1,row['phy_data_attempts']),math.log1p(row['queue_packets']),
        min(1.,row['busy_seconds']),row['critical_late']/max(1,row['critical_rx']),
        row['bulk_late']/max(1,row['bulk_rx']),float(row['phy_data_attempts']>0),
        float(row['critical_rx']>0),float(row['bulk_rx']>0)]
