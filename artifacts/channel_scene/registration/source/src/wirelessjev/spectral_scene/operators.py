"""Link-specific physical baselines and public axis-aligned CAD features.

No operator identifies material dispersion or absolute receiver bias from RSSI
alone. Frequency transport is conditional on a fixed physical link/response.
"""
from __future__ import annotations

import math
import numpy as np
import torch

C_M_S = 299_792_458.
REFERENCE_CARRIER_HZ = 2.4e9
GEOMETRY_FEATURE_NAMES = (
    'rx_x_over_width','rx_y_over_depth','rx_z_over_height',
    'ap_x_over_width','ap_y_over_depth','ap_z_over_height',
    'log10_distance_over_2','log_carrier_over_2p4GHz','direction_sign',
    'cad_box_crossings_over_8','cad_box_path_fraction',
    'calibration_pair_count_clip16_over16','calibration_unknown',
    'context_estimated_quantization_over4dB','receiver_noise_floor_over100dBm',
    'known_transmit_power_over30dBm',
)
CONTEXT_FEATURE_NAMES = ('observed_residual_over30dB','censor_residual_over30dB',
                         'heard','censored') + GEOMETRY_FEATURE_NAMES
PHYSICS_BASIS_NAMES = ('intercept','log10_distance_over2','box_crossings_over8',
    'inside_length_over10m','log_frequency_ratio','logf_times_logdistance',
    'logf_times_box_crossings','logf_times_inside_length')


def _real_inputs(*values):
    like = next((x for x in values if isinstance(x,torch.Tensor)),None)
    dtype=like.dtype if like is not None and like.is_floating_point() else torch.float64
    device=like.device if like is not None else None
    result=[]
    for value in values:
        if isinstance(value,torch.Tensor):
            if value.is_complex():raise ValueError('Physical coordinates/powers must be real')
            tensor=value.to(dtype=dtype,device=device)
        else:
            if np.iscomplexobj(value):raise ValueError('Physical coordinates/powers must be real')
            tensor=torch.as_tensor(np.array(value,copy=True),dtype=dtype,device=device)
        if not torch.isfinite(tensor).all():raise ValueError('Finite physical inputs are required')
        result.append(tensor)
    return result,like is not None


def fspl_loss_db(rx_xyz, ap_xyz, carrier_hz):
    """Far-field isotropic free-space loss for each physical AP/RX link."""
    (rx,ap,frequency),tensor=_real_inputs(rx_xyz,ap_xyz,carrier_hz)
    if rx.shape[-1:]!=(3,) or ap.shape[-1:]!=(3,):raise ValueError('Use physical XYZ endpoint coordinates')
    distance=torch.linalg.vector_norm(rx-ap,dim=-1)
    if (distance<=0).any() or (frequency<=0).any():raise ValueError('Distance/carrier must be positive')
    value=20.*(math.log10(4*math.pi/C_M_S)+torch.log10(distance)+torch.log10(frequency))
    return value if tensor else value.detach().cpu().numpy()


def fspl_received_power_dbm(rx_xyz,ap_xyz,carrier_hz,tx_power_dbm=20.):
    (rx,ap,frequency,power),tensor=_real_inputs(rx_xyz,ap_xyz,carrier_hz,tx_power_dbm)
    value=power-fspl_loss_db(rx,ap,frequency)
    return value if tensor else value.detach().cpu().numpy()


def thermal_noise_dbm(bandwidth_hz, *, temperature_k=290.,noise_figure_db=0.):
    (bandwidth,temperature,nf),tensor=_real_inputs(bandwidth_hz,temperature_k,noise_figure_db)
    if (bandwidth<=0).any() or (temperature<=0).any():raise ValueError('Bandwidth/temperature must be positive')
    noise=10.*(math.log10(1.380649e-23*1000.)+torch.log10(temperature)+torch.log10(bandwidth))+nf
    return noise if tensor else noise.detach().cpu().numpy()


def receiver_snr_db(received_power_dbm,noise_floor_dbm, *, reference_gauge_assumed=False):
    """SNR in an explicitly calibrated/assumed receiver reference gauge."""
    if reference_gauge_assumed is not True:
        raise ValueError('Relative device calibration alone does not identify absolute SNR')
    (power,noise),tensor=_real_inputs(received_power_dbm,noise_floor_dbm)
    snr=power-noise
    return snr if tensor else snr.detach().cpu().numpy()


def frequency_transport_db(reference_received_dbm,reference_hz,target_hz, *, same_physical_link=False,
                           fixed_material_antenna_response=False):
    """Conditional FSPL transport, never a blind mean across AP links."""
    if same_physical_link is not True or fixed_material_antenna_response is not True:
        raise ValueError('FSPL-only transport requires the same link and fixed response assumption')
    (power,source,target),tensor=_real_inputs(reference_received_dbm,reference_hz,target_hz)
    if (source<=0).any() or (target<=0).any():raise ValueError('Carriers must be positive')
    value=power+20*(torch.log10(source)-torch.log10(target))
    return value if tensor else value.detach().cpu().numpy()


def segment_box_features(rx_xyz,ap_xyz,cad_boxes_m):
    """Intersection count and union path fraction inside public AABB solids.

    These RT scenes contain axis-aligned box meshes. This is not an arbitrary
    triangle-mesh or unknown-material attenuation operator.
    """
    rx,ap=np.asarray(rx_xyz,dtype=float),np.asarray(ap_xyz,dtype=float)
    boxes=np.asarray(cad_boxes_m,dtype=float)
    if rx.shape!=ap.shape or rx.ndim!=2 or rx.shape[1]!=3 or not np.isfinite(rx).all() or not np.isfinite(ap).all():
        raise ValueError('Use matching finite Q x 3 endpoint arrays')
    if boxes.size==0:boxes=np.empty((0,2,3))
    if boxes.ndim!=3 or boxes.shape[1:]!=(2,3) or not np.isfinite(boxes).all() or np.any(boxes[:,1]<boxes[:,0]):
        raise ValueError('CAD boxes require ordered finite lower/upper XYZ corners')
    counts=np.zeros(len(rx),dtype=float);fractions=np.zeros(len(rx),dtype=float)
    for row,(start,finish) in enumerate(zip(ap,rx)):
        delta=finish-start;intervals=[]
        for low,high in boxes:
            lo,hi=0.,1.;valid=True
            for axis in range(3):
                if abs(delta[axis])<1e-14:
                    if start[axis]<low[axis] or start[axis]>high[axis]:valid=False;break
                else:
                    a,b=(low[axis]-start[axis])/delta[axis],(high[axis]-start[axis])/delta[axis]
                    lo=max(lo,min(a,b));hi=min(hi,max(a,b))
            # Touching a face at a single point has zero material path length.
            if valid and hi>lo+1e-12:intervals.append((lo,hi))
        counts[row]=len(intervals)
        if intervals:
            intervals.sort();lo,hi=intervals[0];length=0.
            for a,b in intervals[1:]:
                if a<=hi:hi=max(hi,b)
                else:length+=hi-lo;lo,hi=a,b
            fractions[row]=length+hi-lo
    return counts,fractions


def link_geometry_features(rx_xyz,ap_xyz,carrier_hz,direction,bbox_dims_m,cad_boxes_m, *,
                           calibration_pair_count=0,calibration_unknown=True,
                           quantization_width_db=0.,noise_floor_dbm=-100.,tx_power_dbm=20.):
    """Exactly sixteen public features, excluding IDs, seed and material."""
    rx,ap=np.asarray(rx_xyz,dtype=float),np.asarray(ap_xyz,dtype=float)
    dims=np.asarray(bbox_dims_m,dtype=float)
    if rx.shape!=ap.shape or rx.ndim!=2 or rx.shape[1]!=3 or dims.shape!=(3,) or not np.isfinite(dims).all() or np.any(dims<=0):
        raise ValueError('Geometry requires matching Q x XYZ and positive room dimensions')
    n=len(rx)
    def vector(value,name):
        try: result=np.broadcast_to(np.asarray(value,dtype=float),(n,))
        except ValueError as exc:raise ValueError(name+' must broadcast per query') from exc
        if not np.isfinite(result).all():raise ValueError(name+' must be finite')
        return result
    frequency=vector(carrier_hz,'Carrier');which=vector(direction,'Direction')
    count=vector(calibration_pair_count,'Calibration count');unknown=vector(calibration_unknown,'Calibration status')
    quant=vector(quantization_width_db,'Context-derived quantization');noise=vector(noise_floor_dbm,'Receiver noise')
    tx=vector(tx_power_dbm,'Known transmit power')
    if np.any(frequency<=0) or not np.isin(which,(0.,1.)).all() or np.any(count<0) or np.any(count!=np.floor(count)) or not np.isin(unknown,(0.,1.)).all() or np.any(quant<0):
        raise ValueError('Invalid physical view/calibration metadata')
    distance=np.linalg.norm(rx-ap,axis=-1)
    if np.any(distance<=0):raise ValueError('AP and RX positions must differ')
    crossings,fraction=segment_box_features(rx,ap,cad_boxes_m)
    features=np.column_stack((rx/dims,ap/dims,np.log10(distance)/2,
        np.log(frequency/REFERENCE_CARRIER_HZ),2*which-1,crossings/8,fraction,
        np.minimum(count,16)/16,unknown,quant/4,noise/100,tx/30))
    if features.shape!=(n,len(GEOMETRY_FEATURE_NAMES)) or not np.isfinite(features).all():
        raise ValueError('Malformed public geometry features')
    return features.astype('f4')


def query_physics_basis(geometry_features):
    """Eight known physical regressors; their coefficients need not be independent.

    These are not asserted to be ground-truth material/scatterer coordinates.
    A positive residual relative to direct-path FSPL is physically possible.
    """
    tensor=isinstance(geometry_features,torch.Tensor)
    x=geometry_features if tensor else np.asarray(geometry_features,dtype=float)
    if x.ndim<2 or x.shape[-1]!=len(GEOMETRY_FEATURE_NAMES):raise ValueError('Use the sixteen-column public geometry schema')
    if tensor:
        if not torch.isfinite(x).all():raise ValueError('Geometry features must be finite')
        one=torch.ones_like(x[...,6]);distance=torch.pow(10.,2*x[...,6])
        length=x[...,10]*distance/10.;frequency=x[...,7]
        return torch.stack((one,x[...,6],x[...,9],length,frequency,
                            frequency*x[...,6],frequency*x[...,9],frequency*length),-1)
    if not np.isfinite(x).all():raise ValueError('Geometry features must be finite')
    length=x[...,10]*np.power(10.,2*x[...,6])/10.;frequency=x[...,7]
    return np.stack((np.ones_like(frequency),x[...,6],x[...,9],length,frequency,
                     frequency*x[...,6],frequency*x[...,9],frequency*length),-1).astype('f4')
