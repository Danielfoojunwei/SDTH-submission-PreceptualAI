"""Opt-in JEPA controller profiles selected on development data only.

The default RadioMap controller is unchanged. This interface is for simulation
research with the tested bulk-upload mission and five frozen JEPA members.
"""
from dataclasses import replace
import hashlib,json
from .contracted.data import ROOT,READINGS
from .native.intent_policy import intent_policy
from .gate_repair.gate import GainCalibrator,GainGate


def controller(bank=None,*,method='development_best',mission='bulk_uploads'):
    if mission!='bulk_uploads':raise ValueError('This profile has only been validated for bulk_uploads')
    if method not in {'development_best','scale','gain'}:raise ValueError('Unknown JEPA gate profile')
    root=ROOT/'artifacts/gate_repair';choice=json.loads((root/'selection.json').read_text())
    if not choice.get('sealed'):raise RuntimeError('Development selection is not sealed')
    for relative,expected in choice['checkpoint_sha256'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=expected:raise RuntimeError('The profile does not cover these JEPA checkpoints')
    for relative in ['artifacts/native/intent_flip.json','artifacts/contracted/calibration.json','src/wirelessjev/contracted/model.py','src/wirelessjev/native/intent_policy.py']:
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=choice['source_sha256'][relative]:raise RuntimeError('The profile does not cover changed model calibration or mission semantics')
    if method=='development_best':method=min(['scale','gain'],key=lambda k:(choice[k]['score'],0 if k=='scale' else 1))
    parameter=choice[method]['parameter']
    pol=replace(intent_policy(READINGS[mission]['weights'],READINGS[mission]['focus'],dual_hardware='none'),name='jepa_'+method+'_gate',online_map=False,gate=True)
    hook=None
    if method=='scale':pol=replace(pol,belief_scale=parameter)
    else:
        if bank is None:raise ValueError('The gain gate requires a label-free ForecastBank')
        path=root/'gain_model.json'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=choice['gain_model_sha256']:raise RuntimeError('Gain model differs from the sealed choice')
        hook=GainGate(bank,GainCalibrator.from_dict(json.loads(path.read_text())),parameter)
    return pol,hook,{'method':method,'parameter':parameter,'selected_on':'closed-loop development scenes','research_only':True}
