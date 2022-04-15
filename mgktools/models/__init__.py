#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
from .regression import GPR, ConsensusRegressor, LRAGPR, NLEGPR, SVR
from .classification import GPC, SVC
from ..kernels.PreComputed import PreComputedKernel


def set_model(model_type: Literal['gpr', 'gpr-sod', 'gpr-nystrom', 'gpr-nle', 'svr', 'gpc', 'svc'],
              kernel,
              # gpr
              alpha: float = None,
              # svm
              C: float = None,
              # sod
              n_estimators: int = None,
              n_samples: int = None,
              consensus_rule: Literal['smallest_uncertainty', 'weight_uncertainty', 'mean'] = 'weight_uncertainty',
              # nystrom
              n_jobs: int = 1):
    if model_type in ['gpr', 'gpr-sod', 'gpr-nystrom', 'gpr-nle']:
        assert alpha is not None
        gpr = GPR(kernel=kernel,
                  optimizer=None,
                  alpha=alpha,
                  normalize_y=True)
        if model_type == 'gpr':
            return gpr
        elif model_type == 'gpr-sod':
            assert n_samples is not None
            return ConsensusRegressor(
                gpr,
                n_estimators=n_estimators,
                n_sample_per_model=n_samples,
                n_jobs=n_jobs,
                consensus_rule=consensus_rule
            )
        elif model_type == 'gpr-nystrom':
            return LRAGPR(kernel=kernel,
                          optimizer=None,
                          alpha=alpha,
                          normalize_y=True)
        elif model_type == 'gpr-nle':
            if kernel.__class__ != PreComputedKernel:
                assert n_jobs == 1
            return NLEGPR(kernel=kernel,
                          alpha=alpha,
                          n_local=n_samples,
                          n_jobs=n_jobs)  # must be 1 for mgk. could be larger for pre-computed kernel.)
    elif model_type == 'svr':
        assert C is not None
        return SVR(kernel=kernel,
                   C=C)
    elif model_type == 'gpc':
        return GPC(kernel=kernel,
                   optimizer=None,
                   n_jobs=n_jobs)
    elif model_type == 'svc':
        assert C is not None
        return SVC(kernel=kernel,
                   C=C,
                   probability=True)
    else:
        raise RuntimeError(f'unsupported model_type: {model_type}')
