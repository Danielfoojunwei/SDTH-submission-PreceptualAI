"""Publish a finite protocol before any measured summary, fit or prediction."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import scipy

from . import common as c
from .baselines import configuration_grid
from .data import inspect_metadata, METADATA_MEMBER_SHA256


def register():
    if c.OUT.exists():
        raise FileExistsError('Preserve the existing study; registration cannot overwrite it')
    if c.digest(c.SOURCE_METADATA) != c.METADATA_SHA256:
        raise RuntimeError('Official source/license metadata differs')
    catalogue = inspect_metadata(c.SOURCE)
    source_files = [c.ROOT/'src/wirelessjev/__init__.py', c.ROOT/'pyproject.toml', c.ROOT/'uv.lock',
                    c.ROOT/'scripts/run_measured_radio.py']
    for directory in ('src/wirelessjev/measured_radio', 'src/wirelessjev/smartcar_numpy', 'tests/measured_radio'):
        source_files += sorted((c.ROOT/directory).glob('*.py'))
    source_files += sorted((c.ROOT/'artifacts/smartcar_numpy/models').glob('*.npz'))
    source_files += [c.ROOT/'artifacts/smartcar_numpy/models/manifest.json']
    files = {str(p.relative_to(c.ROOT)): c.digest(p) for p in sorted(set(source_files))}
    receipt = {'schema': 1, 'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
        cwd=c.ROOT, text=True).strip(), 'files_sha256': files,
        'scope': 'Implementation and three released array members, frozen before official label preparation',
        'official_summary_fit_prediction_calls_before_registration': 0,
        'software': {'python': sys.version.split()[0], 'numpy': np.__version__, 'scipy': scipy.__version__}}
    c.write(c.OUT/'registration/source.json', receipt)
    for name in files:
        destination = c.OUT/'registration/source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(c.ROOT/name, destination)
    c.write(c.OUT/'registration/catalogue.json', catalogue)
    shutil.copyfile(c.SOURCE_METADATA, c.OUT/'registration/zenodo-20051265.json')
    protocol = {
        'schema': 1, 'namespace': c.NAMESPACE, 'registered_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_receipt_sha256': c.digest(c.OUT/'registration/source.json'),
        'catalogue_sha256': c.digest(c.OUT/'registration/catalogue.json'),
        'source': {'doi': '10.5281/zenodo.20051265', 'version': 'v1', 'license': 'CC-BY-4.0',
            'prior_exposure': 'Previously audited public corpus; this finite estimator/support/split assay is newly registered, not a globally blind dataset',
            'citation': 'Matey-Sanz et al., UVIndoorLoc-RTT&RSSI, Zenodo, 2026',
            'download_url': 'https://zenodo.org/records/20051265/files/UVIndoorLoc-RTT%26RSSI.zip?download=1',
            'archive_bytes': 32899133, 'official_metadata_sha256': c.METADATA_SHA256},
        'source_sha256': c.SOURCE_SHA256, 'metadata_member_sha256': METADATA_MEMBER_SHA256,
        'validation_fraction': .2, 'support_budgets': list(c.BUDGETS), 'rows_per_group': 8,
        'expected_audit': {'source_files': 944, 'source_rows': 1644814, 'failed_status_rows': 55686,
            'zero_success_rows': 1, 'successful_rows': 1589127, 'eligible_rows': 758456,
            'eligible_files': 629, 'eligible_groups': 4255, 'summary_records': 4255, 'selected_rows': 33901},
        'admitted_phones': list(c.PHONES), 'physical_AP_units': 8, 'logical_AP_placements': 20,
        'eligibility': 'status0; successful>0; finite reported RSSI !=-200; explicit frequency5180MHz; verified device/MAC/catalogue',
        'missingness': 'Failed/unheard/unmeasured/unknown-frequency results have no RSSI target; never impute a detector bound',
        'target': 'One arithmetic dB mean of at most8 eligible reported RSSI results per raw file/AP, selected by smallest opaque SHA256(archive NUL member NUL CSV-line); not raw packets, watts or SEM',
        'partitions': {'unit': 'whole receiver location, all phones/APs/sessions grouped',
            'test': 'provider TEST locations, untouched for selection',
            'validation': 'per scenario ceil(.2 * provider TRAIN), smallest SHA256(namespace:validation:scenario:location)',
            'counts': {'train': 112, 'validation': 29, 'test': 87}},
        'support': {'session': '01', 'unit': 'receiver locations, not RF samples',
            'order': 'TRAIN catalogue: smallest SHA256(namespace:support:scenario:location), then maximin Euclidean providerXYZ, same hash tie break',
            'same_planned_locations_all_phones': True, 'missing_files_replaced': False,
            'maximum_summaries_at_16_locations': 128, 'maximum_selected_reported_results_at_16_locations': 1024,
            'known_missing_file': 'raw/S03_S24U_020_01.csv (budget16 yields15 actual support locations)'},
        'geometry': {'known': 'provider AP/receiver XYZ and all public reference catalogue bounds',
            'transductive_query_catalogue_access': True, 'frame': 'XY midpoint of combined AP/reference bounds; Zmin; dimensions max(range,[1,1,3])',
            'frame_is_physical_floor': False, 'unknown_CAD': 'two ACTIVE CAD features zero-imputed; no CAD unknown bit',
            'inactive_unknown_slots': 'TX/noise/quantization zero padding; semantic unknown mask stored separately',
            'full_predictor_translation_invariant': False,
            'effective_global_physics_rank_at_most': 2},
        'nuisance_reference': {'group': 'scenario/logicalAP/phone/Android version',
            'formula': 'c_a=mean_admitted_TRAIN01(y+FSPL_loss); context residual=y+loss-c_a; query baseline=-loss+c_a',
            'identified_TX_or_receiver_calibration': False, 'absorbs_mean_environment_attenuation': True,
            'uncertainty_of_plugin_reference_in_model_spread': False, 'TEST02_reuses_TRAIN01_reference': True},
        'neural': {'members': [0, 1, 2], 'weights': 'exact three released nested_joint NumPy states',
            'training_on_this_dataset': False, 'weight_member_or_hyperparameter_selection': False,
            'aggregation': 'compute RMSE per member, then mean three members; never ensemble-mean prediction RMSE'},
        'numerical_grid': configuration_grid(),
        'selection': {'unit': 'ONE global numerical configuration per budget by VAL01 equal-scenario metric',
            'tie': 'first config in registered ordered grid', 'family_winners_also_retained': 6,
            'all_VAL_config_predictions_retained': True, 'TEST_runs_even_if_neural_VAL_bad': True,
            'neural_selection': False, 'no_TEST_target_for_selection': True},
        'metric': 'Within phone/location/session RMSE across supported observed AP summaries, then equal phones, equal locations within scenario, equal three scenarios',
        'coverage': 'Report all planned87 TEST locations x2phones and per-unit observed/supported AP counts; metric conditions on observed supported cohort and reports any missing units',
        'primary': {'budget_locations': 16, 'session': '01', 'contrast': 'mean-member RadioJEPA RMSE minus VAL-selected best numerical RMSE',
            'resampling': '20000 paired whole-location resamples within fixed scenarios, seed40517, percentile95%',
            'sensitivity': '20000 paired5mXY block resamples within fixed scenarios, seed40518, percentile95%; origin min ALL public receiverXY; preserve block multiplicities and sizes',
            'block_minimum': '>=4 occupied scored blocks per scenario, otherwise inconclusive',
            'conditional_advantage_rule': 'BOTH interval upper bounds <0; no population/site/SOTA conclusion',
            'weights_support_members_fixed_during_bootstrap': True},
        'secondary': ['budgets4/8', 'TEST02 later-campaign frozen map, not causal trajectory forecast',
            'per-scenario and per-member RMSE', 'observed/support coverage and report cost', 'conditional working-spread diagnostic'],
        'limitations': ['Eight hardware AP units reused across three logical placements/scenarios; not three independent sites',
            'Phone day labels refer to different actual dates; Android versions differ by scenario',
            'Known successful observable5180MHz links only; no calibrated weak-link/outage/interference endpoint',
            'Partial-input frozen simulation-to-measurement transfer; no CAD, material, TX or detector calibration',
            'No SmartCar, service, mobility-control, packet-risk, protocol-generalization or SOTA qualification'],
        'immutable_outcome_policy': 'Retain every registered arm/result, including negative transfer; do not tune after TEST; append separately registered future studies',
    }
    c.write(c.OUT/'protocol.json', protocol)
    return {'protocol_sha256': c.digest(c.OUT/'protocol.json'), 'source_files': len(files),
            'official_measured_summaries_fits_predictions': 0}


def verify_published(source_commit):
    receipt = c.read(c.OUT/'registration/source.json')
    protocol = c.read(c.OUT/'protocol.json')
    files = {}
    for name, digest in receipt['files_sha256'].items():
        files[name] = digest
        files[str((c.OUT/'registration/source'/name).relative_to(c.ROOT))] = digest
    for name in ('protocol.json', 'registration/source.json'):
        files[str((c.OUT/name).relative_to(c.ROOT))] = c.digest(c.OUT/name)
    files[str((c.OUT/'registration/catalogue.json').relative_to(c.ROOT))] = protocol['catalogue_sha256']
    files[str((c.OUT/'registration/zenodo-20051265.json').relative_to(c.ROOT))] = c.METADATA_SHA256
    for name, expected in files.items():
        raw = subprocess.check_output(['git', 'show', source_commit+':'+name], cwd=c.ROOT)
        if __import__('hashlib').sha256(raw).hexdigest() != expected:
            raise RuntimeError('Published implementation/registration differs: '+name)
    return files


def publication(source_commit):
    """Require a source-published branch before enabling measured execution."""
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=c.ROOT, text=True).strip()
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/'+branch], cwd=c.ROOT, text=True).split()
    current = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=c.ROOT, text=True).strip()
    if not remote or remote[0] != source_commit or current != source_commit:
        raise RuntimeError('Current exact source commit must be published on the origin branch first')
    files = verify_published(source_commit)
    result = {'schema': 1, 'source_commit': source_commit, 'remote_branch': branch,
        'published_and_verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': c.digest(c.OUT/'protocol.json'),
        'source_receipt_sha256': c.digest(c.OUT/'registration/source.json'),
        'published_files_sha256': files,
        'official_measured_summary_fit_prediction_calls_at_publication': 0}
    c.write(c.OUT/'registration/git.json', result)
    return result
