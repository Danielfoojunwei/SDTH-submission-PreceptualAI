"""Query-time clocks cannot erase or invent the last delivered RF sample."""
from types import SimpleNamespace
import numpy as np
import pytest
from wirelessjev.service_world.study import select_focal


def fixture():
    links=np.zeros((4,3,3,12),dtype='f4');mask=np.zeros_like(links,dtype=bool)
    links[:,0,:,3]=np.array([6.,2.,-999.,4.])[:,None]/30.
    mask[:,0,:,3]=True
    history=SimpleNamespace(rows=[(550,links,None,mask,.1)],last_slot=550)
    public={'slot':558,'current':np.array([0,0,1,0]),
            'outage_remaining':np.zeros(4),'scan_remaining':np.zeros(4),
            'pilot_valid':np.zeros((4,3),dtype=bool),'pilot_snr':np.full((4,3),np.nan)}
    return public,history


def test_last_observed_selection_survives_irregular_query_and_masks_prior_ap():
    public,history=fixture()
    ids=select_focal(public,history,np.random.default_rng(7))
    assert ids[:2]==[1,3] and len(set(ids))==3
    public['pilot_snr'][:]=-99999.;public['misses']=np.arange(4)*100000
    assert select_focal(public,history,np.random.default_rng(7))==ids


def test_current_availability_remains_binding_after_historical_ranking():
    public,history=fixture();public['outage_remaining'][1]=100;public['scan_remaining'][3]=9
    assert select_focal(public,history,np.random.default_rng(7))==[0,2]


def test_future_and_absent_measurement_histories_are_rejected():
    public,history=fixture();public['slot']=549
    with pytest.raises(ValueError,match='future'):select_focal(public,history,np.random.default_rng(7))
    history.rows=[]
    with pytest.raises(ValueError,match='past'):select_focal(public,history,np.random.default_rng(7))
