import importlib.util
from pathlib import Path


def test_packet_jepa_evidence_is_complete_and_rejection_preserves_holdout():
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('packet_value_archive_audit',root/'scripts/report_packet_value.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.audit()
    assert result['passed'] and result['status']=='development_rejected'
    assert result['independent_fits']==6 and result['development_runs']==64
    assert result['test_runs']==0
