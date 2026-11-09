"""Explicit-prior linear target observability and bounded sensing design."""

from .operators import (GaussianPrior,IdentifiabilityCertificate,SubspaceCertificate,
                        certify_targets,expected_posterior_covariance,target_variance)
from .pilots import PilotChoice,PilotDesign,greedy_targeted_pilots

__all__=['GaussianPrior','IdentifiabilityCertificate','SubspaceCertificate',
         'certify_targets','expected_posterior_covariance','target_variance',
         'PilotChoice','PilotDesign','greedy_targeted_pilots']
