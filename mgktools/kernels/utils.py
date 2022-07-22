# !/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import pickle
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
from ..data import Dataset


def get_features_hyperparameters(
        N_RBF: int,
        features_hyperparameters: List[float] = None,
        features_hyperparameters_bounds: Union[List[Tuple[float, float]], Literal['fixed']] = None,
        features_hyperparameters_file: str = None,
) -> Tuple[Optional[List[float]], Optional[List[Tuple[float, float]]]]:
    """

    Parameters
    ----------
    N_RBF: dimension of RBF kernel
    features_hyperparameters
    features_hyperparameters_bounds
    features_hyperparameters_file

    Returns
    -------

    """
    if N_RBF == 0:
        sigma_RBF, sigma_RBF_bounds = None, None
    elif features_hyperparameters_file is not None:
        rbf = json.load(open(features_hyperparameters_file))
        sigma_RBF = rbf['sigma_RBF']
        sigma_RBF_bounds = rbf['sigma_RBF_bounds']
    else:
        sigma_RBF = features_hyperparameters
        sigma_RBF_bounds = features_hyperparameters_bounds
        if len(sigma_RBF) != 1 and len(sigma_RBF) != N_RBF:
            raise ValueError(f'The number of features({N_RBF}) not equal to the number of hyperparameters'
                             f'({len(sigma_RBF)})')
    return sigma_RBF, sigma_RBF_bounds


def get_kernel_config(dataset: Dataset,
                      graph_kernel_type: Literal['graph', 'pre-computed', None],
                      # arguments for vectorized features.
                      features_hyperparameters: List[float] = None,
                      features_hyperparameters_bounds: List[Tuple[float]] = None,
                      features_hyperparameters_file: str = None,
                      # arguments for marginalized graph kernel
                      mgk_hyperparameters_files: List[str] = None,
                      # arguments for pre-computed kernel
                      kernel_dict: Dict = None,
                      kernel_pkl: str = 'kernel.pkl'):
    if graph_kernel_type is None:
        N_RBF = dataset.N_features_mol + dataset.N_features_add
        assert N_RBF != 0
        sigma_RBF, sigma_RBF_bounds = get_features_hyperparameters(
            N_RBF,
            features_hyperparameters,
            features_hyperparameters_bounds,
            features_hyperparameters_file,
        )
        params = {
            'N_RBF': N_RBF,
            'sigma_RBF': sigma_RBF,  # np.concatenate(sigma_RBF),
            'sigma_RBF_bounds': sigma_RBF_bounds,  # * N_RBF,
        }
        from mgktools.kernels.BaseKernelConfig import BaseKernelConfig
        return BaseKernelConfig(**params)
    elif graph_kernel_type == 'graph':
        mgk_hyperparameters_files = [
            json.load(open(j)) for j in mgk_hyperparameters_files
        ]
        assert dataset.N_MGK + dataset.N_conv_MGK == len(mgk_hyperparameters_files)

        N_RBF = dataset.N_features_mol + dataset.N_features_add
        sigma_RBF, sigma_RBF_bounds = get_features_hyperparameters(
            N_RBF,
            features_hyperparameters,
            features_hyperparameters_bounds,
            features_hyperparameters_file,
        )

        params = {
            'N_MGK': dataset.N_MGK,
            'N_conv_MGK': dataset.N_conv_MGK,
            'graph_hyperparameters': mgk_hyperparameters_files,
            'unique': False,
            'N_RBF': N_RBF,
            'sigma_RBF': sigma_RBF,  # np.concatenate(sigma_RBF),
            'sigma_RBF_bounds': sigma_RBF_bounds,  # * N_RBF,
        }
        from mgktools.kernels.GraphKernel import GraphKernelConfig
        return GraphKernelConfig(**params)
    else:
        N_RBF = 0 if dataset.data[0].features_add is None \
            else dataset.data[0].features_add.shape[1]
        sigma_RBF, sigma_RBF_bounds = get_features_hyperparameters(
            N_RBF,
            features_hyperparameters,
            features_hyperparameters_bounds,
            features_hyperparameters_file,
        )

        if kernel_dict is None:
            kernel_dict = pickle.load(open(kernel_pkl, 'rb'))
        params = {
            'kernel_dict': kernel_dict,
            'N_RBF': N_RBF,
            'sigma_RBF': sigma_RBF,
            'sigma_RBF_bounds': sigma_RBF_bounds,  # * N_RBF,
        }
        from mgktools.kernels.PreComputed import PreComputedKernelConfig
        return PreComputedKernelConfig(**params)
