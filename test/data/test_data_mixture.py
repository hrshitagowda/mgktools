#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import pytest
import json
from mgktools.data import Dataset


mixture = [['CCCC', 0.5, 'CCOCCN', 0.5],
           ['CCCCCO', 0.3, 'c1ccccc1O', 0.7],
           ['c1ccccc1', 0.2, 'CCCCC', 0.8],
           ['CCNCCO', 0.4, 'c1ccccc1CC', 0.6]]
mixture = [json.dumps(m) for m in mixture]
targets_1 = [3.4, 4.5, 5.6, 6.7]
targets_2 = [3.1, 14.5, 25.6, 56.7]
df = pd.DataFrame({'mixture': mixture, 'targets_1': targets_1, 'targets_2': targets_2})


@pytest.mark.parametrize('testset', [
    (['targets_1'], (4, 1)),
    (['targets_1', 'targets_2'], (4, 2)),
])
def test_only_graph(testset):
    targets_column, yshape = testset
    dataset = Dataset.from_df(df=df,
                              mixture_columns=['mixture'],
                              target_columns=targets_column)
    dataset.graph_kernel_type = 'graph'
    assert dataset.X.shape == (4, 1)
    assert dataset.y.shape == yshape


@pytest.mark.parametrize('testset', [
    ('morgan', 2048 * 2, 'concat'),
    ('rdkit_2d', 200 * 2, 'concat'),
    ('rdkit_2d_normalized', 200 * 2, 'concat'),
    ('morgan', 2048, 'mean'),
    ('rdkit_2d', 200, 'mean'),
    ('rdkit_2d_normalized', 200, 'mean'),
])
def test_only_fingerprints(testset):
    features_generator, n_features, features_combination = testset
    dataset = Dataset.from_df(df=df,
                              mixture_columns=['mixture'],
                              target_columns=['targets_1'],
                              features_generator=[features_generator],
                              features_combination=features_combination)
    assert dataset.X.shape == (4, n_features)
    assert dataset.y.shape == (4, 1)


@pytest.mark.parametrize('testset', [
    ('morgan', 2048 * 2, 'concat'),
    ('rdkit_2d', 200 * 2, 'concat'),
    ('rdkit_2d_normalized', 200 * 2, 'concat'),
    ('morgan', 2048, 'mean'),
    ('rdkit_2d', 200, 'mean'),
    ('rdkit_2d_normalized', 200, 'mean'),
])
def test_graph_fingerprints(testset):
    features_generator, n_features, features_combination = testset
    dataset = Dataset.from_df(df=df,
                              mixture_columns=['mixture'],
                              target_columns=['targets_1'],
                              features_generator=[features_generator],
                              features_combination=features_combination)
    dataset.graph_kernel_type = 'graph'
    assert dataset.X.shape == (4, 1 + n_features)
    assert dataset.y.shape == (4, 1)
