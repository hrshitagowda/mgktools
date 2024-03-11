#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil

import pandas as pd
import pytest
from mgktools.data import Dataset
from mgktools.kernels.utils import get_kernel_config
from mgktools.hyperparameters import *
from mgktools.models import set_model
from mgktools.evaluators.cross_validation import Evaluator


pure = ['CCCC', 'CCCCCO', 'c1ccccc1', 'CCNCCO', 'CCCCN', 'NCCCCCO', 'c1ccccc1N', 'NCCNCCO',
        'CNC(CC)CC', 'c1ccccc1', 'c1ccccc1CCCCc1ccccc1', 'CC(=O)OCCO']
targets_regression = [3.1, 14.5, 25.6, 56.7, 9.1, 17.5, 22.6, 36.7, 23.1, 32.1, 1.4, 7.6]
df_regression = pd.DataFrame({'pure': pure, 'targets': targets_regression})
targets_classification = [1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]
df_classification = pd.DataFrame({'pure': pure, 'targets': targets_classification})


@pytest.mark.parametrize('mgk_file', [additive_norm, additive_pnorm, additive_msnorm,
                                      product_norm, product_pnorm, product_msnorm])
@pytest.mark.parametrize('model', ['gpc', 'svc'])
@pytest.mark.parametrize('split_type', ['random', 'scaffold_order', 'scaffold_random'])
def test_only_graph_classification(mgk_file, model, split_type):
    dataset = Dataset.from_df(df=df_classification,
                              pure_columns=['pure'],
                              target_columns=['targets'])
    dataset.graph_kernel_type = 'graph'
    kernel_config = get_kernel_config(dataset=dataset,
                                      graph_kernel_type='graph',
                                      mgk_hyperparameters_files=[mgk_file])
    C = 1.0 if model == 'svc' else None
    model = set_model(model, kernel=kernel_config.kernel, C=C)
    Evaluator(save_dir='tmp',
              dataset=dataset,
              model=model,
              task_type='binary',
              metrics=['roc-auc', 'accuracy', 'precision', 'recall', 'f1_score', 'mcc'],
              split_type=split_type,
              split_sizes=[0.75, 0.25],
              num_folds=2,
              return_std=False,
              verbose=True).evaluate()
    shutil.rmtree('tmp')


@pytest.mark.parametrize('nfold', [4, 5])
def test_nfold_cv_classification(nfold):
    model = 'gpc'
    mgk_file = additive_norm
    dataset = Dataset.from_df(df=df_classification,
                              pure_columns=['pure'],
                              target_columns=['targets'])
    dataset.graph_kernel_type = 'graph'
    kernel_config = get_kernel_config(dataset=dataset,
                                      graph_kernel_type='graph',
                                      mgk_hyperparameters_files=[mgk_file])
    C = 1.0 if model == 'svc' else None
    model = set_model(model, kernel=kernel_config.kernel, C=C)
    Evaluator(save_dir='tmp',
              dataset=dataset,
              model=model,
              task_type='binary',
              metrics=['roc-auc', 'accuracy', 'precision', 'recall', 'f1_score', 'mcc'],
              cross_validation='nfold',
              nfold=nfold,
              num_folds=2,
              return_std=False,
              verbose=True).evaluate()
    shutil.rmtree('tmp')


@pytest.mark.parametrize('mgk_file', [additive_norm, additive_pnorm, additive_msnorm,
                                      product_norm, product_pnorm, product_msnorm])
@pytest.mark.parametrize('modelsets', [('gpr', None, None, None),
                                       ('gpr-sod', 2, 3, 'smallest_uncertainty'),
                                       # ('gpr-sod', 2, 3, 'weight_uncertainty'),
                                       ('gpr-sod', 2, 3, 'mean'),
                                       ('gpr-nystrom', None, 3, None),
                                       ('gpr-nle', None, 3, None)])
@pytest.mark.parametrize('split_type', ['random', 'scaffold_order', 'scaffold_random'])
def test_only_graph_scalable_gps(mgk_file, modelsets, split_type):
    model_type, n_estimators, n_samples, consensus_rule = modelsets
    dataset = Dataset.from_df(df=df_regression,
                              pure_columns=['pure'],
                              target_columns=['targets'])
    dataset.graph_kernel_type = 'graph'
    kernel_config = get_kernel_config(dataset=dataset,
                                      graph_kernel_type='graph',
                                      mgk_hyperparameters_files=[mgk_file])
    model = set_model(model_type,
                      kernel=kernel_config.kernel,
                      alpha=0.01,
                      n_estimators=n_estimators,
                      n_samples=n_samples,
                      consensus_rule=consensus_rule
                      )
    Evaluator(save_dir='tmp',
              dataset=dataset,
              model=model,
              task_type='regression',
              metrics=['rmse', 'mae', 'mse', 'r2', 'max', 'spearman', 'kendall', 'pearson'],
              split_type=split_type,
              split_sizes=[0.75, 0.25],
              num_folds=2,
              return_std=True,
              verbose=True,
              n_core=n_samples if model_type == 'gpr-nystrom' else None).evaluate()
    shutil.rmtree('tmp')
