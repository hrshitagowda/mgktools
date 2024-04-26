#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from tap import Tap
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import numpy as np
from mgktools.evaluators.metric import Metric


class CommonArgs(Tap):
    save_dir: str
    """The output directory."""
    n_jobs: int = 1
    """The cpu numbers used for parallel computing."""
    data_path: str = None
    """The Path of input data CSV file."""
    pure_columns: List[str] = None
    """
    For pure compounds.
    Name of the columns containing single SMILES or InChI string.
    """
    mixture_columns: List[str] = None
    """
    For mixtures.
    Name of the columns containing multiple SMILES or InChI string and 
    corresponding concentration.
    example: ["C", 0.5, "CC", 0.3]
    """
    mixture_type: Literal["single_graph", "multi_graph"] = "single_graph"
    """How the mixture is represented."""
    reaction_columns: List[str] = None
    """
    For chemical reactions.
    Name of the columns containing single reaction smarts string.
    """
    reaction_type: Literal["reaction", "agent", "reaction+agent"] = "reaction"
    """How the chemical reaction is represented."""
    feature_columns: List[str] = None
    """
    Name of the columns containing additional features_mol such as temperature, 
    pressuer.
    """
    features_generator: List[str] = None
    """Method(s) of generating additional features_mol."""
    features_combination: Literal["concat", "mean"] = None
    """How to combine features vector for mixtures."""
    target_columns: List[str] = None
    """
    Name of the columns containing target values.
    """
    features_mol_normalize: bool = False
    """Nomralize the molecular features_mol."""
    features_add_normalize: bool = False
    """Nomralize the additonal features_mol."""
    group_reading: bool = False
    """Find unique input strings first, then read the data."""
    def __init__(self, *args, **kwargs):
        super(CommonArgs, self).__init__(*args, **kwargs)

    @property
    def graph_columns(self) -> List[str]:
        graph_columns = []
        if self.pure_columns is not None:
            graph_columns += self.pure_columns
        if self.mixture_columns is not None:
            graph_columns += self.mixture_columns
        if self.reaction_columns is not None:
            graph_columns += self.reaction_columns
        return graph_columns

    def update_columns(self, keys: List[str]):
        """Add all undefined columns to target_columns"""
        if self.target_columns is not None:
            return
        else:
            used_columns = self.graph_columns
            if self.feature_columns is not None:
                used_columns += self.feature_columns
            for key in used_columns:
                keys.remove(key)
            self.target_columns = keys

    def process_args(self) -> None:
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        if self.group_reading:
            if self.feature_columns is None:
                raise ValueError("feature_columns must be assigned when using group_reading.")


class KArgs(Tap):
    graph_kernel_type: Literal["graph", "pre-computed"] = None
    """The type of kernel to use."""
    graph_hyperparameters: List[str] = None
    """hyperparameters file for graph kernel."""
    features_kernel_type: Literal["dot_product", "rbf"] = None
    """choose dot product kernel or rbf kernel for features."""
    features_hyperparameters: List[float] = None
    """hyperparameters for molecular features."""
    features_hyperparameters_min: float = None
    """hyperparameters for molecular features."""
    features_hyperparameters_max: float = None
    """hyperparameters for molecular features."""
    features_hyperparameters_file: str = None
    """JSON file contains features hyperparameters"""
    single_features_hyperparameter: bool = True
    """Use the same hyperparameter for all features."""

    @property
    def features_hyperparameters_bounds(self):
        if self.features_hyperparameters_min is None or self.features_hyperparameters_max is None:
            if self.features_hyperparameters is None:
                return None
            else:
                return "fixed"
        else:
            return (self.features_hyperparameters_min, self.features_hyperparameters_max)

    @property
    def ignore_features_add(self) -> bool:
        if self.feature_columns is None and \
                self.features_hyperparameters is None and \
                self.features_hyperparameters_file is None:
            return True
        else:
            return False


class KernelArgs(CommonArgs, KArgs):
    augment_data: bool = False
    """If True, the dataset will be augmented by shuffling the order of the equivalent columns of the data."""

    def process_args(self) -> None:
        super().process_args()


class TrainArgs(KernelArgs):
    task_type: Literal["regression", "binary", "multi-class"] = None
    """
    Type of task.
    """
    model_type: Literal["gpr", "svc", "svr", "gpc", "gpr_nystrom", "gpr_nle"]
    """Type of model to use"""
    loss: Literal["loocv", "likelihood"] = "loocv"
    """The target loss function to minimize or maximize."""
    cross_validation: Literal["n-fold", "leave-one-out", "Monte-Carlo"] = "Monte-Carlo"
    """The way to split data for cross-validation."""
    nfold: int = None
    """The number of fold for n-fold CV."""
    split_type: Literal["random", "scaffold_order", "scaffold_random", "stratified"] = None
    """Method of splitting the data into train/test sets."""
    split_sizes: List[float] = None
    """Split proportions for train/test sets."""
    num_folds: int = 1
    """Number of folds when performing cross validation."""
    alpha: str = None
    """data noise used in gpr."""
    C: str = None
    """C parameter used in Support Vector Machine."""
    seed: int = 0
    """Random seed."""
    ensemble: bool = False
    """use ensemble model."""
    n_estimator: int = 1
    """Ensemble model with n estimators."""
    n_sample_per_model: int = None
    """The number of samples use in each estimator."""
    ensemble_rule: Literal["smallest_uncertainty", "weight_uncertainty",
                           "mean"] = "weight_uncertainty"
    """The rule to combining prediction from estimators."""
    n_local: int = 500
    """The number of samples used in Naive Local Experts."""
    n_core: int = None
    """The number of samples used in Nystrom core set."""
    metric: Metric = None
    """metric"""
    extra_metrics: List[Metric] = []
    """Metrics"""
    no_proba: bool = False
    """Use predict_proba for classification task."""
    evaluate_train: bool = False
    """If set True, evaluate the model on training set."""
    detail: bool = False
    """If set True, 5 most similar molecules in the training set will be save in the test_*.log."""
    save_model: bool = False
    """Save the trained model file."""
    separate_test_path: str = None
    """Path to separate test set, optional."""
    atomic_attribution: bool = False
    """Output interpretability."""

    @property
    def metrics(self) -> List[Metric]:
        return [self.metric] + self.extra_metrics

    @property
    def alpha_(self) -> float:
        if self.alpha is None:
            return None
        elif isinstance(self.alpha, float):
            return self.alpha
        elif os.path.exists(self.alpha):
            return float(open(self.alpha, "r").read())
        else:
            return float(self.alpha)

    @property
    def C_(self) -> float:
        if self.C is None:
            return None
        elif isinstance(self.C, float):
            return self.C
        elif os.path.exists(self.C):
            return float(open(self.C, "r").read())
        else:
            return float(self.C)

    def kernel_args(self):
        return super()

    def process_args(self) -> None:
        super().process_args()
        if self.task_type == "regression":
            assert self.model_type in ["gpr", "gpr_nystrom", "gpr_nle", "svr"]
            for metric in self.metrics:
                assert metric in ["rmse", "mae", "mse", "r2", "max"]
        elif self.task_type == "binary":
            assert self.model_type in ["gpc", "svc", "gpr"]
            for metric in self.metrics:
                assert metric in ["roc-auc", "accuracy", "precision", "recall", "f1_score", "mcc"]
        elif self.task_type == "multi-class":
            assert self.model_type in ["gpc", "svc"]
            for metric in self.metrics:
                assert metric in ["accuracy", "precision", "recall", "f1_score"]

        if self.cross_validation == "leave-one-out":
            assert self.num_folds == 1
            assert self.model_type == "gpr"

        if self.model_type in ["gpr", "gpr_nystrom"]:
            assert self.alpha is not None

        if self.model_type == "svc":
            assert self.C is not None

        if not hasattr(self, "optimizer"):
            self.optimizer = None
        if not hasattr(self, "batch_size"):
            self.batch_size = None

        if self.save_model:
            assert self.num_folds == 1
            assert self.split_sizes[0] > 0.99999
            assert self.model_type == "gpr"

        if self.ensemble:
            assert self.n_sample_per_model is not None

        if self.atomic_attribution:
            assert self.graph_kernel_type == "graph", "Set graph_kernel_type to graph for interpretability"
            assert self.model_type == "gpr", "Set model_type to gpr for interpretability"
            assert self.ensemble is False


class GradientOptArgs(TrainArgs):
    optimizer: Literal["SLSQP", "L-BFGS-B", "BFGS", "fmin_l_bfgs_b", "sgd", "rmsprop", "adam"] = None
    """Optimizer"""

    def process_args(self) -> None:
        super().process_args()
        assert self.model_type == "gpr"


class HyperoptArgs(TrainArgs):
    num_iters: int = 100
    """Number of hyperparameter choices to try."""
    alpha_bounds: Tuple[float, float] = None
    """Bounds of alpha used in GPR."""
    d_alpha: float = None
    """The step size of alpha to be optimized."""
    C_bounds: Tuple[float, float] = None #  (1e-3, 1e3)
    """Bounds of C used in SVC."""
    d_C: float = None
    """The step size of C to be optimized."""
    batch_size: int = None
    """batch_size"""
    num_splits: int = 1
    """split the dataset randomly into no. subsets."""
    save_all: bool = False
    """save all hyperparameters during bayesian optimization."""

    @property
    def minimize_score(self) -> bool:
        """Whether the model should try to minimize the score metric or maximize it."""
        return self.metric in {"rmse", "mae", "mse", "r2"}

    @property
    def opt_alpha(self) -> bool:
        if self.alpha_bounds is not None and \
                self.model_type in ["gpr", "gpr_nystrom"]:
            return True
        else:
            return False

    @property
    def opt_C(self) -> bool:
        if self.C_bounds is not None and \
                self.model_type == "svc":
            return True
        else:
            return False

    def process_args(self) -> None:
        super().process_args()
        if self.optimizer in ["L-BFGS-B"]:
            assert self.model_type == "gpr"


class EmbeddingArgs(KernelArgs):
    embedding_algorithm: Literal["tSNE", "kPCA"] = "tSNE"
    """Algorithm for data embedding."""
    n_components: int = 2
    """Dimension of the embedded space."""
    perplexity: float = 30.0
    """
    The perplexity is related to the number of nearest neighbors that
    is used in other manifold learning algorithms. Larger datasets
    usually require a larger perplexity. Consider selecting a value
    different results.
    """
    n_iter: int = 1000
    """Maximum number of iterations for the optimization. Should be at least 250."""
    save_png: bool = False
    """If True, save the png file of the data embedding."""

    def process_args(self) -> None:
        super().process_args()
        if self.save_png:
            assert self.n_components == 2


class HyperoptMultiDatasetArgs(KArgs):
    save_dir: str
    """The output directory."""
    n_jobs: int = 1
    """The cpu numbers used for parallel computing."""
    data_paths: List[str]
    """The Path of input data CSV file."""
    pure_columns: str = None
    """
    For pure compounds.
    Name of the columns containing single SMILES or InChI string.
    """
    mixture_columns: str = None
    """
    For mixtures.
    Name of the columns containing multiple SMILES or InChI string and 
    corresponding concentration.
    example: ["C", 0.5, "CC", 0.3]
    """
    mixture_type: Literal["single_graph", "multi_graph"] = "single_graph"
    """How the mixture is represented."""
    reaction_columns: str = None
    """
    For chemical reactions.
    Name of the columns containing single reaction smarts string.
    """
    reaction_type: Literal["reaction", "agent", "reaction+agent"] = "reaction"
    """How the chemical reaction is represented."""
    feature_columns: str = None
    """
    Name of the columns containing additional features_mol such as temperature, 
    pressuer.
    """
    features_generator: List[str] = None
    """Method(s) of generating additional features_mol."""
    features_combination: Literal["concat", "mean"] = None
    """How to combine features vector for mixtures."""
    target_columns: str = None
    """
    Name of the columns containing target values.
    """
    features_mol_normalize: bool = False
    """Nomralize the molecular features_mol."""
    features_add_normalize: bool = False
    """Nomralize the additonal features_mol."""
    group_reading: bool = False
    """Find unique input strings first, then read the data."""
    tasks_type: List[Literal["regression", "binary", "multi-class"]]
    """
    Type of task.
    """
    metrics: List[Metric]
    """taget metrics to be optimized."""
    num_iters: int = 100
    """Number of hyperparameter choices to try."""
    alpha: str = None
    """data noise used in gpr."""
    alpha_bounds: Tuple[float, float] = None
    """Bounds of alpha used in GPR."""
    d_alpha: float = None
    """The step size of alpha to be optimized."""
    seed: int = 0
    """Random seed."""

    @property
    def alpha_(self) -> float:
        if self.alpha is None:
            return None
        elif isinstance(self.alpha, float):
            return self.alpha
        elif os.path.exists(self.alpha):
            return float(open(self.alpha, "r").read())
        else:
            return float(self.alpha)
        
    @property
    def graph_columns(self) -> List[str]:
        graph_columns = []
        if self.pure_columns is not None:
            graph_columns += self.pure_columns
        if self.mixture_columns is not None:
            graph_columns += self.mixture_columns
        if self.reaction_columns is not None:
            graph_columns += self.reaction_columns
        return graph_columns

    def update_columns(self, keys: List[str]):
        """Add all undefined columns to target_columns"""
        if self.target_columns is not None:
            return
        else:
            used_columns = self.graph_columns
            if self.feature_columns is not None:
                used_columns += self.feature_columns
            for key in used_columns:
                keys.remove(key)
            self.target_columns = keys

    def process_args(self) -> None:
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        none_list = [None] * len(self.data_paths)
        self.pure_columns_ = [i.split(",") for i in self.pure_columns.split(";")] if self.pure_columns is not None else none_list
        self.mixture_columns_ = [i.split(",") for i in self.mixture_columns.split(";")] if self.mixture_columns is not None else none_list
        self.reaction_columns_ = [i.split(",") for i in self.reaction_columns.split(";")] if self.reaction_columns is not None else none_list
        self.feature_columns_ = [i.split(",") for i in self.feature_columns.split(";")] if self.feature_columns is not None else none_list
        self.target_columns_ = [i.split(",") for i in self.target_columns.split(";")]

        if self.group_reading:
            if self.feature_columns is None:
                raise ValueError("feature_columns must be assigned when using group_reading.")
