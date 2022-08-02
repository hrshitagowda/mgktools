#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import pandas as pd
import numpy as np
from mgktools.data import Dataset
from mgktools.kernels.utils import get_kernel_config
from mgktools.hyperparameters import *

pure = ['CCCC', 'CCCCCO', 'c1ccccc1', 'CCNCCO']
targets = [3.1, 14.5, 25.6, 56.7]
df = pd.DataFrame({'pure': pure, 'targets': targets})
dataset = Dataset.from_df(df=df,
                          pure_columns=['pure'],
                          target_columns=['targets'])
dataset.graph_kernel_type = 'graph'


@pytest.mark.parametrize('testset', [
    (additive),
    (additive_norm),
    (additive_pnorm),
    (additive_msnorm),
    (product),
    (product_norm),
    (product_pnorm),
    (product_msnorm)
])
def test_only_graph(testset):
    N = len(dataset)
    mgk_file = testset
    print(mgk_file)
    kernel_config = get_kernel_config(dataset=dataset,
                                      graph_kernel_type='graph',
                                      mgk_hyperparameters_files=[mgk_file])
    K = kernel_config.kernel(dataset.X)
    assert K.shape == (N, N)
    print(K)
    # invertable
    assert np.linalg.det(K) > 10 ** -3
    for i in range(N):
        for j in range(i + 1, N):
            # symmetric
            assert abs(K[i, j] - K[j, i]) < 10 ** -10
            # diagonal largest
            assert np.sqrt(K[i, i] * K[j, j]) > K[i, j]
