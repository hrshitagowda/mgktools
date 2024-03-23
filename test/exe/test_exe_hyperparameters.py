# !/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os

CWD = os.path.dirname(os.path.abspath(__file__))
import sys

sys.path.append("%s/.." % CWD)
from mgktools.hyperparameters import (
    additive,
    additive_pnorm,
    additive_msnorm,
    additive_norm,
    product,
    product_pnorm,
    product_msnorm,
    product_norm,
)
from mgktools.exe.run import (
    mgk_gradientopt,
    mgk_hyperopt,
    mgk_hyperopt_multi_datasets,
    mgk_optuna,
    mgk_optuna_multi_datasets,
)


@pytest.mark.parametrize(
    "dataset",
    [
        ("freesolv", ["smiles"], ["freesolv"]),
    ],
)
@pytest.mark.parametrize(
    "split_set",
    [
        ("leave-one-out", None, None, "1"),
        ("Monte-Carlo", None, "random", "10"),
        ("n-fold", "5", None, "1"),
    ],
)
@pytest.mark.parametrize("num_splits", ["1", "2"])
@pytest.mark.parametrize("metric", ["r2", "mae", "rmse"])
@pytest.mark.parametrize(
    "graph_hyperparameters",
    [
        additive_msnorm,
    ],
)
@pytest.mark.parametrize("optimize_alpha", [True, False])
def test_hyperopt_PureGraph_regression(
    dataset, split_set, num_splits, metric, graph_hyperparameters, optimize_alpha
):
    task = "regression"
    model = "gpr"
    dataset, pure_columns, target_columns = dataset
    save_dir = "%s/data/_%s_%s_%s" % (
        CWD,
        dataset,
        ",".join(pure_columns),
        ",".join(target_columns),
    )
    cross_validation, nfold, split, num_folds = split_set
    assert not os.path.exists("%s/graph_hyperparameters.json" % save_dir)
    assert not os.path.exists("%s/alpha" % save_dir)
    arguments = [
        "--save_dir",
        save_dir,
        "--graph_kernel_type",
        "graph",
        "--task_type",
        task,
        "--model_type",
        model,
        "--cross_validation",
        cross_validation,
        "--metric",
        metric,
        "--num_folds",
        num_folds,
        "--graph_hyperparameters",
        "%s" % graph_hyperparameters,
        "--num_iters",
        "10",
        "--alpha",
        "0.01",
        "--num_splits",
        num_splits,
    ]
    if nfold is not None:
        arguments += ["--nfold", nfold]
    if split is not None:
        arguments += ["--split_type", split, "--split_sizes", "0.8", "0.2"]
    if optimize_alpha:
        arguments += ["--alpha_bounds", "0.008", "0.02"]
    mgk_hyperopt(arguments)
    if optimize_alpha:
        assert 0.008 < float(open("%s/alpha" % save_dir).readline()) < 0.02
        os.remove("%s/alpha" % save_dir)
    else:
        assert not os.path.exists("%s/alpha" % save_dir)
    os.remove("%s/graph_hyperparameters.json" % save_dir)


@pytest.mark.parametrize(
    "dataset",
    [
        ("bace", ["smiles"], ["bace"]),
        ("np", ["smiles1", "smiles2"], ["np"]),
    ],
)
@pytest.mark.parametrize(
    "modelset",
    [
        ("gpr", True),
        ("gpr", False),
        ("gpc", False),
        ("svc", True),
        ("svc", False),
    ],
)
@pytest.mark.parametrize(
    "testset",
    [
        ("random", "10"),
    ],
)
@pytest.mark.parametrize(
    "metric", ["roc-auc", "accuracy", "precision", "recall", "f1_score", "mcc"]
)
@pytest.mark.parametrize("graph_hyperparameters", [additive_msnorm])
def test_hyperopt_PureGraph_binary(
    dataset, modelset, testset, metric, graph_hyperparameters
):
    task = "binary"
    dataset, pure_columns, target_columns = dataset
    save_dir = "%s/data/_%s_%s_%s" % (
        CWD,
        dataset,
        ",".join(pure_columns),
        ",".join(target_columns),
    )
    model, optimize_C = modelset
    split, num_folds = testset
    if len(pure_columns) == 1:
        assert not os.path.exists("%s/graph_hyperparameters.json" % save_dir)
    else:
        for i in range(len(pure_columns)):
            assert not os.path.exists("%s/kernel_%d.json" % (save_dir, i))
    assert not os.path.exists("%s/alpha" % save_dir)
    assert not os.path.exists("%s/C" % save_dir)
    arguments = (
        [
            "--save_dir",
            save_dir,
            "--graph_kernel_type",
            "graph",
            "--task_type",
            task,
            "--model_type",
            model,
            "--split_type",
            split,
            "--split_sizes",
            "0.8",
            "0.2",
            "--metric",
            metric,
            "--num_folds",
            num_folds,
            "--graph_hyperparameters",
        ]
        + ["%s" % graph_hyperparameters] * len(pure_columns)
        + [
            "--num_iters",
            "10",
            "--C",
            "1",
        ]
    )
    if model == "gpr":
        arguments += ["--alpha", "0.01"]
        if optimize_C:
            arguments += ["--alpha_bounds", "0.001", "0.02"]
    elif model == "svc":
        arguments += [
            "--C",
            "1",
        ]
        if optimize_C:
            arguments += ["--C_bounds", "0.01", "10.0"]
    mgk_hyperopt(arguments)
    if optimize_C and model == "svc":
        assert 0.01 < float(open("%s/C" % save_dir).readline()) < 10.0
        os.remove("%s/C" % save_dir)
    else:
        assert not os.path.exists("%s/C" % save_dir)

    if optimize_C and model == "gpr":
        assert 0.001 < float(open("%s/alpha" % save_dir).readline()) < 0.02
        os.remove("%s/alpha" % save_dir)
    else:
        assert not os.path.exists("%s/alpha" % save_dir)

    if len(pure_columns) == 1:
        os.remove("%s/graph_hyperparameters.json" % save_dir)
    else:
        for i in range(len(pure_columns)):
            os.remove("%s/kernel_%d.json" % (save_dir, i))


@pytest.mark.parametrize(
    "dataset",
    [
        ("clintox", ["smiles"], ["FDA_APPROVED", "CT_TOX"]),
    ],
)
@pytest.mark.parametrize(
    "modelset",
    [
        ("gpc", False),
        ("svc", True),
        ("svc", False),
    ],
)
@pytest.mark.parametrize(
    "testset",
    [
        ("random", "10"),
    ],
)
@pytest.mark.parametrize(
    "metric", ["roc-auc", "accuracy", "precision", "recall", "f1_score", "mcc"]
)
@pytest.mark.parametrize("graph_hyperparameters", [additive_msnorm])
def test_hyperopt_PureGraph_binary_multitask(
    dataset, modelset, testset, metric, graph_hyperparameters
):
    pass
    # TODO


@pytest.mark.parametrize(
    "dataset",
    [
        ("st", ["smiles"], ["st"], ["T"]),
    ],
)
@pytest.mark.parametrize("group_reading", [True, False])
@pytest.mark.parametrize("features_scaling", [True, False])
@pytest.mark.parametrize("metric", ["rmse"])
@pytest.mark.parametrize("graph_hyperparameters", [additive_pnorm])
@pytest.mark.parametrize("features_kernel_type", ["rbf", "dot_product"])
@pytest.mark.parametrize("features_hyperparameter_fix", [True, False])
@pytest.mark.parametrize("exe", [mgk_hyperopt, mgk_optuna])
def test_hyperopt_optuna_PureGraph_FeaturesAdd_regression(
    dataset,
    group_reading,
    features_scaling,
    metric,
    graph_hyperparameters,
    features_kernel_type,
    features_hyperparameter_fix,
    exe,
):
    optimize_alpha = features_hyperparameter_fix
    task = "regression"
    model = "gpr"
    dataset, pure_columns, target_columns, features_columns = dataset
    save_dir = "%s/data/_%s_%s_%s_%s_%s" % (
        CWD,
        dataset,
        ",".join(pure_columns),
        ",".join(target_columns),
        group_reading,
        features_scaling,
    )
    if optimize_alpha:
        assert not os.path.exists("%s/alpha" % save_dir)
    assert not os.path.exists("%s/kernel_0.json" % save_dir)
    assert not os.path.exists("%s/kernel_1.json" % save_dir)
    assert not os.path.exists("%s/optuna.db" % save_dir)
    arguments = [
        "--save_dir",
        save_dir,
        "--graph_kernel_type",
        "graph",
        "--task_type",
        task,
        "--model_type",
        model,
        "--cross_validation",
        "leave-one-out",
        "--metric",
        metric,
        "--num_folds",
        "1",
        "--graph_hyperparameters",
        graph_hyperparameters,
        "--num_iters",
        "10",
        "--alpha",
        "0.01",
        "--features_kernel_type",
        features_kernel_type,
    ]
    if features_hyperparameter_fix:
        arguments += [
            "--features_hyperparameters",
            "1.0",
        ]
    else:
        arguments += [
            "--features_hyperparameters",
            "1.0",
            "--features_hyperparameters_min",
            "0.1",
            "--features_hyperparameters_max",
            "20.0",
        ]
    if optimize_alpha:
        arguments += ["--alpha_bounds", "0.008", "0.02"]
    exe(arguments)
    if optimize_alpha:
        assert 0.008 < float(open("%s/alpha" % save_dir).readline()) < 0.02
        os.remove("%s/alpha" % save_dir)
    else:
        assert not os.path.exists("%s/alpha" % save_dir)
    os.remove("%s/kernel_0.json" % save_dir)
    os.remove("%s/kernel_1.json" % save_dir)
    if exe == mgk_optuna:
        os.remove("%s/optuna.db" % save_dir)


def test_hyperopt_PureGraph_FeauturesAdd_binary():
    # TODO
    return


def test_hyperopt_PureGraph_FeauturesAdd_multiclass():
    # TODO
    return


def test_hyperopt_PureGraph_FeaturesMol_regression():
    # TODO
    return


def test_hyperopt_PureGraph_FeaturesMol_binary():
    # TODO
    return


def test_hyperopt_PureGraph_FeaturesMol_multiclass():
    # TODO
    return


def test_hyperopt_PureGraph_FeaturesAddMol_regression():
    # TODO
    return


def test_hyperopt_PureGraph_FeaturesAddMol_binary():
    # TODO
    return


def test_hyperopt_PureGraph_FeaturesAddMol_multiclass():
    # TODO
    return


@pytest.mark.parametrize(
    "graph_hyperparameters",
    [
        additive,
        additive_norm,
        additive_pnorm,
        additive_msnorm,
        product,
        product_norm,
        product_pnorm,
        product_msnorm,
    ],
)
@pytest.mark.parametrize(
    "exe", [mgk_hyperopt_multi_datasets, mgk_optuna_multi_datasets]
)
@pytest.mark.parametrize("features_generator", [None, "rdkit_2d_normalized"])
def test_hyperopt_optuna_Multi_Datasets(graph_hyperparameters, exe, features_generator):
    if features_generator is None:
        save_dir = "%s/data/_hyperopt_multi_datasets" % CWD
        assert not os.path.exists("%s/graph_hyperparameters.json" % save_dir)
    else:
        save_dir = "%s/data/_hyperopt_multi_datasets_%s" % (CWD, features_generator)
        assert not os.path.exists("%s/kernel_0.json" % save_dir)
        assert not os.path.exists("%s/kernel_1.json" % save_dir)
    assert not os.path.exists("%s/optuna.db" % save_dir)
    arguments = [
        "--save_dir",
        save_dir,
        "--data_paths",
        "%s/data/freesolv.csv" % CWD,
        "%s/data/bace.csv" % CWD,
        "--pure_columns",
        "smiles;smiles",
        "--target_columns",
        "freesolv;bace",
        "--tasks_type",
        "regression",
        "binary",
        "--alpha",
        "0.01",
        "--metrics",
        "r2",
        "roc-auc",
        "--graph_kernel_type",
        "graph",
        "--graph_hyperparameters",
        graph_hyperparameters,
        "--num_iters",
        "5",
    ]
    if features_generator is not None:
        arguments += [
            "--features_generator",
            features_generator,
            "--features_kernel_type",
            "rbf",
            "--features_hyperparameters",
            "10.0",
            "--features_hyperparameters_min",
            "0.1",
            "--features_hyperparameters_max",
            "30.0",
        ]
    exe(arguments)
    if features_generator is None:
        os.remove("%s/graph_hyperparameters.json" % save_dir)
    else:
        os.remove("%s/kernel_0.json" % save_dir)
        os.remove("%s/kernel_1.json" % save_dir)
    if exe == mgk_optuna_multi_datasets:
        os.remove("%s/optuna.db" % save_dir)
    for i in range(2):
        os.remove("%s/dataset_%d.pkl" % (save_dir, i))


@pytest.mark.parametrize(
    "dataset",
    [
        ("freesolv", ["smiles"], ["freesolv"]),
    ],
)
@pytest.mark.parametrize("graph_hyperparameters", [additive_pnorm])
@pytest.mark.parametrize("optimizer", ["L-BFGS-B", "SLSQP"])
@pytest.mark.parametrize("loss", ["loocv", "likelihood"])
def test_GradientOpt_PureGraph_regression(
    dataset, graph_hyperparameters, optimizer, loss
):
    model = "gpr"
    dataset, pure_columns, target_columns = dataset
    save_dir = "%s/data/_%s_%s_%s" % (
        CWD,
        dataset,
        ",".join(pure_columns),
        ",".join(target_columns),
    )
    assert not os.path.exists("%s/graph_hyperparameters.json" % save_dir)
    arguments = [
        "--save_dir",
        "%s" % save_dir,
        "--graph_kernel_type",
        "graph",
        "--model_type",
        model,
        "--graph_hyperparameters",
        graph_hyperparameters,
        "--alpha",
        "0.01",
        "--optimizer",
        optimizer,
        "--loss",
        loss,
    ]
    mgk_gradientopt(arguments)
    os.remove("%s/graph_hyperparameters.json" % save_dir)


@pytest.mark.parametrize(
    "dataset",
    [
        ("st", ["smiles"], ["st"], ["T"]),
    ],
)
@pytest.mark.parametrize("group_reading", [True, False])
@pytest.mark.parametrize("features_scaling", [True, False])
@pytest.mark.parametrize("graph_hyperparameters", [additive_pnorm])
@pytest.mark.parametrize("optimizer", ["L-BFGS-B", "SLSQP"])
@pytest.mark.parametrize("loss", ["loocv", "likelihood"])
@pytest.mark.parametrize("features_kernel_type", ["rbf", "dot_product"])
@pytest.mark.parametrize("features_hyperparameter_fix", [True, False])
def test_GradientOpt_PureGraph_FeaturesAdd_regression(
    dataset,
    group_reading,
    features_scaling,
    graph_hyperparameters,
    optimizer,
    loss,
    features_kernel_type,
    features_hyperparameter_fix,
):
    model = "gpr"
    dataset, pure_columns, target_columns, features_columns = dataset
    save_dir = "%s/data/_%s_%s_%s_%s_%s" % (
        CWD,
        dataset,
        ",".join(pure_columns),
        ",".join(target_columns),
        group_reading,
        features_scaling,
    )
    assert not os.path.exists("%s/kernel_0.json" % save_dir)
    assert not os.path.exists("%s/kernel_1.json" % save_dir)
    arguments = [
        "--save_dir",
        "%s" % save_dir,
        "--graph_kernel_type",
        "graph",
        "--model_type",
        model,
        "--graph_hyperparameters",
        graph_hyperparameters,
        "--alpha",
        "0.01",
        "--optimizer",
        optimizer,
        "--loss",
        loss,
        "--features_kernel_type",
        features_kernel_type,
    ]
    if features_hyperparameter_fix:
        arguments += [
            "--features_hyperparameters",
            "1.0",
        ]
    else:
        arguments += [
            "--features_hyperparameters",
            "1.0",
            "--features_hyperparameters_min",
            "0.1",
            "--features_hyperparameters_max",
            "20.0",
        ]
    mgk_gradientopt(arguments)
    os.remove("%s/kernel_0.json" % save_dir)
    os.remove("%s/kernel_1.json" % save_dir)


def test_GradientOpt_PureGraph_FeaturesMol_regression():
    # TODO
    return


def test_GradientOpt_PureGraph_FeaturesAddMol_regression():
    # TODO
    return
