"""Execute the registered scene-model pilot, preserving the RadioMap default."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import tempfile
import traceback

from wirelessjev.radiojepa import common as c


def progress(phase,state,**detail):
    value={'phase':phase,'state':state,'utc':datetime.now(timezone.utc).isoformat(),**detail}
    c.WORK.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.status-',dir=c.WORK)
    with os.fdopen(fd,'w') as stream:
        json.dump(value,stream,sort_keys=True,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,c.WORK/'status.json')
    print(json.dumps(value,sort_keys=True),flush=True)


def phase(name,call):
    progress(name,'running')
    result=call()
    if isinstance(result,dict) and 'passes' in result and result['passes'] is not True:
        raise RuntimeError('Phase returned a failed proof: '+name)
    progress(name,'completed')
    return result


def markdown(report):
    aggregate=report['aggregate']; factory='dichasus-cf06'
    lines=['# RadioJEPA measured-view pilot','',
        'These are measured factory-channel array-power results. They are not SmartCar command-service results or a global SOTA claim.','',
        'The RadioMap application default and all production qualification flags remain unchanged.','',
        '| Model | Factory RF RMSE, dB (mean ± run SD) | Frozen XY probe error, m |',
        '|---|---:|---:|']
    for cell,row in aggregate['cells'].items():
        value=row['domains'][factory]
        rmse=value['prediction_metrics']['rmse_db']
        probe=value['frozen_position_probe']['mean_error_m']
        lines.append(f"| {cell} | {rmse['mean']:.3f} ± {rmse['training_seed_sd']:.3f} | {probe['mean']:.3f} |")
    lines+=['','Three independently trained members per cell; member metrics are averaged, predictions are not ensembled. The split has 519 common training rows, 130 validation rows and 155 spatially held-out factory rows.','',
        'The report also retains both held-out recordings, constant/linear controls, projector and backbone ranks, cross-view alignment, and pooled/token/power interventions. All metrics use referenced channel power, not dBm or SINR.','',
        'PPR remains ineligible for this power-only experiment: phases/path factors are not identified. The 24 service-confirmation geometry specifications remain unopened.','',
        'The first planner implementation reproduces RadioMap utility; it does not qualify a new learned outage head. Multi-step dynamics, actual device onboarding and matched native packet-service qualification are further requirements.','',
        '[Raw report](../artifacts/radiojepa/report.json) · [Execution scope](RADIOJEPA_EXECUTION.md) · [User plan](plans/RADIOJEPA_USER_PLAN.md)','']
    return '\n'.join(lines)


def _run():
    from wirelessjev.radiojepa import training,baselines,evaluation,verification,rt_data
    c.check_sources()
    if (c.OUT/'verification/manifest.json').exists():
        return phase('existing_byte_closure',verification.verify_closure)
    if (c.OUT/'verification/raw_audit.json').exists():
        report=c.read_json(c.OUT/'report.json')
    else:
        phase('prepare_measured_data',training.prepare_data)
        for stage in ('training','validation','development'):
            phase('sionna_rt_'+stage,lambda s=stage:rt_data.collect_stage(s,c.OUT/'rt_data',authorize=c.authorize_stage))
        phase('eighteen_independent_neural_fits',training.train_all)
        controls=phase('measured_power_and_position_controls',baselines.run_baselines)
        p=c.protocol()
        fits=[c.read_json(c.OUT/'fits'/f'{cell}-seed{seed}.json') for cell in p['cells'] for seed in p['training_seeds']]
        report={'schema':1,'aggregate':evaluation.aggregate_fit_reports(fits,controls),
            'source_receipt_sha256':c.digest(c.OUT/'registration/source.json'),
            'protocol_sha256':c.digest(c.PROTOCOL),'new_neural_backbones':18,
            'reserved_confirmation_opened':False,'radio_map_default':True,
            'learned_heads_qualified':False,'robot_service_qualified':False,'hardware_qualified':False,
            'production_qualified':False,'global_sota_claimed':False,
            'vision_track':'separate exploratory evidence; exact full-recording/video clock proof required',
            'phase_scope':'measured-view pilot plus physical data/operators/interface; the entire foundation-model program is not complete'}
        c.write_once(c.OUT/'report.json',report)
        phase('passive_raw_audit',verification.run_audit)
    document=c.ROOT/'docs/RADIOJEPA_RESULTS.md'
    c.write_bytes_once(document,markdown(report).encode())
    phase('seal',lambda:verification.seal(documents=(document,)))
    result=phase('byte_closure',verification.verify_closure)
    progress('scene_pilot','completed',qualified=False,report_sha256=c.digest(c.OUT/'report.json'))
    return result


def run():
    c.WORK.mkdir(parents=True,exist_ok=True)
    with (c.WORK/'supervisor.lock').open('a') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('Scene pilot supervisor is already active') from exc
        try:return _run()
        except Exception as exc:
            progress('scene_pilot','failed',error_type=type(exc).__name__,error=str(exc))
            raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('register','run','status','verify'))
    args=parser.parse_args(argv)
    if args.command=='register':return c.register()
    if args.command=='status':
        path=c.WORK/'status.json';print(path.read_text() if path.exists() else '{"state":"not_started"}')
        return
    if args.command=='verify':
        from wirelessjev.radiojepa.verification import verify_closure
        print(json.dumps(verify_closure()));return
    run()


if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();raise SystemExit(1)
