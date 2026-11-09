"""Analytic software fixtures only; these are not benchmark result scores."""
import numpy as np
import pytest

from wirelessjev.channel_scene.baselines import (
    TrainLMMSE, circular_fourier_basis, fourier_ridge, linear_interpolate,
)


def grid(n=64):
    return (np.arange(n)-n/2)*50.e6/n


def test_complex_interpolation_is_exact_for_affine_frequency_and_has_no_target_argument():
    f=grid();p=np.array([2,9,24,42,58]);scale=f/1.e6
    complete=np.stack([2+1j+(.1-.3j)*scale,-.2+2j+(-.07+.1j)*scale])
    before=complete[:,p].copy();pred=linear_interpolate(before,p,f)
    np.testing.assert_allclose(pred,complete,atol=3.e-15)
    np.testing.assert_array_equal(before,complete[:,p])
    np.testing.assert_allclose(linear_interpolate(before[:,::-1],p[::-1],f),complete,atol=3.e-15)


def test_circular_basis_preserves_actual_hz_delay_scale_and_fourier_convention():
    f=grid();basis=circular_fourier_basis(f,n_taps=8)
    tau=np.arange(8)/(len(f)*(f[1]-f[0]))
    np.testing.assert_allclose(basis,np.exp(-2j*np.pi*(f-f[0])[:,None]*tau),atol=1.e-15)
    # The dimensionless basis is unchanged if bandwidth and canonical delays
    # scale together; the actual delay coordinates are not physical ray labels.
    np.testing.assert_allclose(circular_fourier_basis(f*4,n_taps=8),basis,atol=1.e-15)


def test_unregularized_fourier_reconstructs_representable_channels_from_pilots():
    f=grid();p=np.arange(0,64,4);basis=circular_fourier_basis(f,n_taps=7)
    rng=np.random.default_rng(8);a=rng.normal(size=(4,7))+1j*rng.normal(size=(4,7))
    full=a@basis.T;pred=fourier_ridge(full[:,p],p,f,n_taps=7,ridge=0.)
    np.testing.assert_allclose(pred,full,rtol=1.e-12,atol=1.e-12)
    regularized=fourier_ridge(full[:,p],p,f,n_taps=7,ridge=.03)
    assert np.isfinite(regularized).all()
    assert np.linalg.norm(regularized)<np.linalg.norm(pred)


def test_fourier_ridge_matches_explicit_augmented_least_squares():
    f=grid();p=np.array([0,5,10,14,21,28,36,45,54,63]);rng=np.random.default_rng(5)
    x=rng.normal(size=(3,len(p)))+1j*rng.normal(size=(3,len(p)));ridge=.07
    a=circular_fourier_basis(f,n_taps=5);matrix=np.concatenate([a[p],np.sqrt(ridge*len(p))*np.eye(5)])
    targets=np.concatenate([x.T,np.zeros((5,len(x)),complex)])
    expected=(np.linalg.lstsq(matrix,targets,rcond=None)[0].T)@a.T
    np.testing.assert_allclose(fourier_ridge(x,p,f,n_taps=5,ridge=ridge),expected,atol=1.e-14)


def test_train_lmmse_recovers_complex_train_subspace_from_pilots():
    f=grid();rng=np.random.default_rng(21);basis=circular_fourier_basis(f,n_taps=4)
    train=(rng.normal(size=(50,4))+1j*rng.normal(size=(50,4)))@basis.T
    test=(rng.normal(size=(6,4))+1j*rng.normal(size=(6,4)))@basis.T;p=np.arange(0,64,8)
    estimator=TrainLMMSE.fit(train,f,shrinkage=0.,noise_fraction=0.)
    pred=estimator.predict(test[:,p],p)
    np.testing.assert_allclose(pred,test,atol=3.e-14)
    assert estimator.training_count==len(train)
    assert not estimator.covariance.flags.writeable and not estimator.frequency_offsets_hz.flags.writeable
    np.testing.assert_allclose(estimator.covariance,train.T@train.conj()/len(train),atol=1.e-14)


def test_train_lmmse_shrinkage_noise_and_test_gain_are_explicit():
    f=grid();rng=np.random.default_rng(3)
    train=rng.normal(size=(20,64))+1j*rng.normal(size=(20,64));p=np.arange(0,64,4)
    fit=TrainLMMSE.fit(train,f,shrinkage=.2,noise_fraction=.01)
    pilots=train[:3,p].copy();a=fit.predict(pilots,p)
    np.testing.assert_allclose(fit.predict(pilots*(2+.3j),p),a*(2+.3j),atol=2.e-14)
    np.testing.assert_array_equal(pilots,train[:3,p])
    order=np.arange(len(p))[::-1]
    np.testing.assert_allclose(fit.predict(pilots[:,order],p[order]),a,atol=1.e-14)
    zero=TrainLMMSE.fit(np.zeros((2,64),complex),f)
    np.testing.assert_array_equal(zero.predict(pilots,p),np.zeros_like(a))


@pytest.mark.parametrize('bad',[np.array([0,0,3]),np.array([-1,2,3]),np.array([0.,2.,3.])])
def test_predictors_reject_ambiguous_or_invalid_pilot_indices(bad):
    x=np.ones((2,3),complex)
    with pytest.raises(ValueError,match='distinct actual bin'):
        linear_interpolate(x,bad,grid())
    with pytest.raises(ValueError,match='distinct actual bin'):
        fourier_ridge(x,bad,grid())


def test_estimators_reject_unobserved_nan_or_fake_frequency_grid():
    f=grid();x=np.ones((2,16),complex);x[0,2]=complex(float('nan'),0)
    with pytest.raises(ValueError,match='finite batch'):
        linear_interpolate(x,np.arange(16)*4,f)
    with pytest.raises(ValueError,match='complete TRAIN'):
        TrainLMMSE.fit(np.ones((2,64)),f)
    broken=f.copy();broken[4]+=1.
    with pytest.raises(ValueError,match='uniform'):
        fourier_ridge(np.ones((2,16),complex),np.arange(16)*4,broken)
