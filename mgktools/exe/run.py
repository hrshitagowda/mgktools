#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
from mgktools.data import Dataset
from mgktools.kernels.utils import get_kernel_config
from mgktools.kernels.PreComputed import calc_precomputed_kernel_config
from mgktools.evaluators.cross_validation import Evaluator
from mgktools.data.split import dataset_split
from mgktools.hyperparameters.hyperopt import (
    bayesian_optimization,
    bayesian_optimization_gpr_multi_datasets,
)
from mgktools.hyperparameters.optuna import (
    bayesian_optimization as optuna_bayesian_optimization,
    bayesian_optimization_gpr_multi_datasets as optuna_bayesian_optimization_gpr_multi_datasets,
)
from mgktools.exe.args import (
    CommonArgs,
    KernelArgs,
    TrainArgs,
    GradientOptArgs,
    HyperoptArgs,
    HyperoptMultiDatasetArgs,
    EmbeddingArgs,
)
from mgktools.exe.model import set_model
from mgktools.evaluators.cross_validation import data_augmentation


def mgk_read_data(arguments=None):
    args = CommonArgs().parse_args(arguments)
    print("Reading the datasets and saving them in pickle files.")
    if args.data_path is not None:
        dataset = Dataset.from_df(
            df=pd.read_csv(args.data_path),
            pure_columns=args.pure_columns,
            mixture_columns=args.mixture_columns,
            reaction_columns=args.reaction_columns,
            feature_columns=args.feature_columns,
            target_columns=args.target_columns,
            features_generator=args.features_generator,
            features_combination=args.features_combination,
            mixture_type=args.mixture_type,
            reaction_type=args.reaction_type,
            group_reading=args.group_reading,
            n_jobs=args.n_jobs,
        )
    else:
        raise ValueError("Please provide data_path.")
    if args.features_mol_normalize:
        dataset.normalize_features_mol()
    if args.features_add_normalize:
        dataset.normalize_features_add()
    dataset.save(args.save_dir, overwrite=True)
    print("Reading Datasets Finished.")


def mgk_kernel_calc(arguments=None):
    args = KernelArgs().parse_args(arguments)
    assert args.graph_kernel_type == "graph"
    assert args.feature_columns is None
    assert args.n_jobs == 1
    # load data set.
    dataset = Dataset.load(path=args.save_dir)
    dataset.graph_kernel_type = "graph"
    # set kernel_config
    kernel_config = get_kernel_config(
        dataset=dataset,
        graph_kernel_type=args.graph_kernel_type,
        features_kernel_type=args.features_kernel_type,
        features_hyperparameters=args.features_hyperparameters,
        features_hyperparameters_bounds=args.features_hyperparameters_bounds,
        features_hyperparameters_file=args.features_hyperparameters_file,
        mgk_hyperparameters_files=args.graph_hyperparameters,
    )
    if args.augment_data:
        datasets = data_augmentation(dataset)
        for d in data_augmentation(dataset)[1:]:
            dataset.data.extend(d.data)
    print("**\tCalculating kernel matrix\t**")
    kernel_config = calc_precomputed_kernel_config(kernel_config=kernel_config, dataset=dataset)
    print("**\tEnd Calculating kernel matrix\t**")
    kernel_pkl = os.path.join(args.save_dir, "kernel.pkl")
    pickle.dump(kernel_config, open(kernel_pkl, "wb"), protocol=4)


def mgk_model_evaluate(arguments=None):
    args = TrainArgs().parse_args(arguments)
    dataset = Dataset.load(path=args.save_dir)
    dataset.graph_kernel_type = args.graph_kernel_type
    kernel_config = get_kernel_config(
        dataset=dataset,
        graph_kernel_type=args.graph_kernel_type,
        features_kernel_type=args.features_kernel_type,
        features_hyperparameters=args.features_hyperparameters,
        features_hyperparameters_bounds=args.features_hyperparameters_bounds,
        features_hyperparameters_file=args.features_hyperparameters_file,
        mgk_hyperparameters_files=args.graph_hyperparameters,
        kernel_pkl=os.path.join(args.save_dir, "kernel.pkl"),
    )
    model = set_model(args, kernel=kernel_config.kernel)
    if args.separate_test_path is not None:
        df = pd.read_csv(args.separate_test_path)
        if args.target_columns is None:
            df["null_target"] = 0.0
        dataset_test = Dataset.from_df(
            df=df,
            pure_columns=args.pure_columns,
            mixture_columns=args.mixture_columns,
            reaction_columns=args.reaction_columns,
            feature_columns=args.feature_columns,
            target_columns=args.target_columns or ["null_target"],
            features_generator=args.features_generator,
            features_combination=args.features_combination,
            mixture_type=args.mixture_type,
            reaction_type=args.reaction_type,
            group_reading=args.group_reading,
            n_jobs=args.n_jobs,
        )
        dataset_test.graph_kernel_type = args.graph_kernel_type
        dataset.unify_datatype(dataset_test.X_graph)
    else:
        dataset_test = None
    evaluator = Evaluator(
        save_dir=args.save_dir,
        dataset=dataset,
        model=model,
        task_type=args.task_type,
        metrics=args.metrics,
        cross_validation=args.cross_validation,
        nfold=args.nfold,
        split_type=args.split_type,
        split_sizes=args.split_sizes,
        num_folds=args.num_folds,
        return_std=True if args.model_type == "gpr" else False,
        return_proba=True if args.task_type == "binary" else False,
        evaluate_train=False,
        augment_data=args.augment_data,
        n_similar=None,
        kernel=None,
        n_core=args.n_core,
        atomic_attribution=args.atomic_attribution,
        seed=args.seed,
        verbose=True,
    )

    if args.separate_test_path is not None and args.target_columns is None:
        evaluator.fit(X=dataset.X, y=dataset.y)
        evaluator.predict(
            X=dataset_test.X, y=None, repr=dataset_test.repr.ravel()
        ).to_csv(
            "%s/pred_ext.csv" % args.save_dir,
            sep="\t",
            index=False,
            float_format="%15.10f",
        )
        if args.atomic_attribution:
            evaluator.interpret(dataset_test=dataset_test, output_tag="ext")
    else:
        evaluator.evaluate(external_test_dataset=dataset_test)


def mgk_embedding(arguments=None):
    def plotmap(ax, X, Y, c, cmap="viridis", size=1, min=None, max=None):
        if min is None:
            min = c.min()
        if max is None:
            max = c.max()
        style = dict(s=size, cmap=cmap)
        sc = ax.scatter(X, Y, c=c, vmin=min, vmax=max, **style)
        return sc

    args = EmbeddingArgs().parse_args(arguments)
    dataset = Dataset.load(args.save_dir)
    dataset.graph_kernel_type = args.graph_kernel_type
    kernel_config = get_kernel_config(
        dataset=dataset,
        graph_kernel_type=args.graph_kernel_type,
        features_kernel_type=args.features_kernel_type,
        features_hyperparameters=args.features_hyperparameters,
        features_hyperparameters_bounds=args.features_hyperparameters_bounds,
        features_hyperparameters_file=args.features_hyperparameters_file,
        mgk_hyperparameters_files=args.graph_hyperparameters,
        kernel_pkl=os.path.join(args.save_dir, "kernel.pkl"),
    )
    if args.embedding_algorithm == "tSNE":
        # compute data embedding.
        R = kernel_config.kernel(dataset.X)
        d = R.diagonal() ** -0.5
        K = d[:, None] * R * d[None, :]
        D = np.sqrt(np.maximum(0, 2 - 2 * K**2))
        embed = TSNE(
            n_components=args.n_components,
            perplexity=args.perplexity,
            n_iter=args.n_iter,
            n_jobs=args.n_jobs,
        ).fit_transform(D)
    else:
        R = kernel_config.kernel(dataset.X)
        embed = KernelPCA(
            n_components=args.n_components, kernel="precomputed", n_jobs=args.n_jobs
        ).fit_transform(R)
    # embedding dataframe.
    df = pd.DataFrame({"repr": dataset.repr.ravel()})
    for i in range(args.n_components):
        df["embedding_%d" % i] = embed[:, i]
    num_tasks = dataset.N_tasks
    if num_tasks == 1:
        df["y_0"] = dataset.y
    else:
        for i in range(num_tasks):
            df["y_%d" % i] = dataset.y[:, i]
    df.to_csv(
        "%s/%s.csv" % (args.save_dir, args.embedding_algorithm), sep="\t", index=False
    )

    if args.save_png:
        for i in range(num_tasks):
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            p = "y_%d" % i
            ymin = df[p].min()
            ymax = df[p].max()
            sc = plotmap(
                ax,
                df["embedding_0"],
                df["embedding_1"],
                df[p],
                size=5,
                min=ymin,
                max=ymax,
            )
            ax.set_xlabel("Embedding 1")
            ax.set_ylabel("Embedding 2")
            ax.set_title(args.embedding_algorithm)
            fig.savefig(
                "%s/%s_%s.png" % (args.save_dir, args.embedding_algorithm, p), dpi=300
            )


def mgk_gradientopt(arguments=None):
    args = GradientOptArgs().parse_args(arguments)
    # read data
    dataset = Dataset.load(args.save_dir)
    dataset.graph_kernel_type = args.graph_kernel_type
    # set kernel_config
    kernel_config = get_kernel_config(
        dataset=dataset,
        graph_kernel_type=args.graph_kernel_type,
        features_kernel_type=args.features_kernel_type,
        features_hyperparameters=args.features_hyperparameters,
        features_hyperparameters_bounds=args.features_hyperparameters_bounds,
        features_hyperparameters_file=args.features_hyperparameters_file,
        mgk_hyperparameters_files=args.graph_hyperparameters,
    )
    model = set_model(args, kernel=kernel_config.kernel)
    model.fit(dataset.X, dataset.y, loss=args.loss, verbose=True)
    kernel_config.update_from_theta()
    kernel_config.update_kernel()
    kernel_config.save(args.save_dir)


def mgk_hyperopt(arguments=None):
    args = HyperoptArgs().parse_args(arguments)
    # read data
    dataset = Dataset.load(args.save_dir)
    dataset.graph_kernel_type = args.graph_kernel_type
    if args.num_splits == 1:
        datasets = [dataset]
    else:
        datasets = dataset_split(
            dataset=dataset,
            split_type="random",
            sizes=[1 / args.num_splits] * args.num_splits,
        )
    # set kernel_config
    kernel_config = get_kernel_config(
        dataset=dataset,
        graph_kernel_type=args.graph_kernel_type,
        features_kernel_type=args.features_kernel_type,
        features_hyperparameters=args.features_hyperparameters,
        features_hyperparameters_bounds=args.features_hyperparameters_bounds,
        features_hyperparameters_file=args.features_hyperparameters_file,
        mgk_hyperparameters_files=args.graph_hyperparameters,
    )
    best_hyperdict, results, hyperdicts = bayesian_optimization(
        save_dir=args.save_dir,
        datasets=datasets,
        kernel_config=kernel_config,
        task_type=args.task_type,
        model_type=args.model_type,
        metric=args.metric,
        cross_validation=args.cross_validation,
        nfold=args.nfold,
        split_type=args.split_type,
        split_sizes=args.split_sizes,
        num_folds=args.num_folds,
        num_iters=args.num_iters,
        alpha=args.alpha_,
        alpha_bounds=args.alpha_bounds,
        d_alpha=args.d_alpha,
        C=args.C_,
        C_bounds=args.C_bounds,
        d_C=args.d_C,
        seed=args.seed,
    )
    if args.save_all:
        for i, hyperdict in enumerate(hyperdicts):
            if not os.path.exists("%s/%d" % (args.save_dir, i)):
                os.mkdir("%s/%d" % (args.save_dir, i))
            kernel_config.update_from_space(hyperdict)
            kernel_config.save_hyperparameters("%s/%d" % (args.save_dir, i))
            open("%s/%d/loss" % (args.save_dir, i), "w").write(str(results[i]))


def mgk_hyperopt_multi_datasets(arguments=None):
    args = HyperoptMultiDatasetArgs().parse_args(arguments)
    print("Preprocessing Dataset.")
    datasets = []
    for i, data_path in enumerate(args.data_paths):
        dataset_pkl = f"{args.save_dir}/dataset_{i}.pkl"
        if os.path.exists(dataset_pkl):
            dataset = Dataset.load(path=args.save_dir, filename=f"dataset_{i}.pkl")
        else:
            dataset = Dataset.from_df(
                df=pd.read_csv(data_path),
                pure_columns=args.pure_columns_[i],
                mixture_columns=args.mixture_columns_[i],
                reaction_columns=args.reaction_columns_[i],
                feature_columns=args.feature_columns_[i],
                target_columns=args.target_columns_[i],
                features_generator=args.features_generator,
                features_combination=args.features_combination,
                reaction_type=args.reaction_type,
                group_reading=args.group_reading,
                n_jobs=args.n_jobs,
            )
            dataset.graph_kernel_type = args.graph_kernel_type
            dataset.save(
                path=args.save_dir, filename=f"dataset_{i}.pkl", overwrite=False
            )
        datasets.append(dataset)
    print("Preprocessing Dataset Finished.")
    # set kernel_config
    kernel_config = get_kernel_config(
        dataset=dataset,
        graph_kernel_type=args.graph_kernel_type,
        features_kernel_type=args.features_kernel_type,
        features_hyperparameters=args.features_hyperparameters,
        features_hyperparameters_bounds=args.features_hyperparameters_bounds,
        features_hyperparameters_file=args.features_hyperparameters_file,
        mgk_hyperparameters_files=args.graph_hyperparameters,
    )
    best_hyperdict, results, hyperdicts = bayesian_optimization_gpr_multi_datasets(
        save_dir=args.save_dir,
        datasets=datasets,
        kernel_config=kernel_config,
        tasks_type=args.tasks_type,
        metrics=args.metrics,
        num_iters=args.num_iters,
        alpha=args.alpha_,
        alpha_bounds=args.alpha_bounds,
        d_alpha=args.d_alpha,
        seed=args.seed,
    )
    pd.DataFrame(results, columns=args.data_paths).to_csv(
        "%s/hyperopt_traj.csv" % args.save_dir, index=False
    )


def mgk_optuna(arguments=None):
    args = HyperoptArgs().parse_args(arguments)
    # read data
    dataset = Dataset.load(args.save_dir)
    dataset.graph_kernel_type = args.graph_kernel_type
    if args.num_splits == 1:
        datasets = [dataset]
    else:
        datasets = dataset_split(
            dataset=dataset,
            split_type="random",
            sizes=[1 / args.num_splits] * args.num_splits,
        )
    # set kernel_config
    kernel_config = get_kernel_config(
        dataset=dataset,
        graph_kernel_type=args.graph_kernel_type,
        features_kernel_type=args.features_kernel_type,
        features_hyperparameters=args.features_hyperparameters,
        features_hyperparameters_bounds=args.features_hyperparameters_bounds,
        features_hyperparameters_file=args.features_hyperparameters_file,
        mgk_hyperparameters_files=args.graph_hyperparameters,
    )
    optuna_bayesian_optimization(
        save_dir=args.save_dir,
        datasets=datasets,
        kernel_config=kernel_config,
        task_type=args.task_type,
        model_type=args.model_type,
        metric=args.metric,
        cross_validation=args.cross_validation,
        nfold=args.nfold,
        split_type=args.split_type,
        split_sizes=args.split_sizes,
        num_folds=args.num_folds,
        num_iters=args.num_iters,
        alpha=args.alpha_,
        alpha_bounds=args.alpha_bounds,
        d_alpha=args.d_alpha,
        C=args.C_,
        C_bounds=args.C_bounds,
        d_C=args.d_C,
        seed=args.seed,
    )


def mgk_optuna_multi_datasets(arguments=None):
    args = HyperoptMultiDatasetArgs().parse_args(arguments)
    print("Preprocessing Dataset.")
    datasets = []
    for i, data_path in enumerate(args.data_paths):
        dataset_pkl = f"{args.save_dir}/dataset_{i}.pkl"
        if os.path.exists(dataset_pkl):
            dataset = Dataset.load(path=args.save_dir, filename=f"dataset_{i}.pkl")
        else:
            dataset = Dataset.from_df(
                df=pd.read_csv(data_path),
                pure_columns=args.pure_columns_[i],
                mixture_columns=args.mixture_columns_[i],
                reaction_columns=args.reaction_columns_[i],
                feature_columns=args.feature_columns_[i],
                target_columns=args.target_columns_[i],
                features_generator=args.features_generator,
                features_combination=args.features_combination,
                reaction_type=args.reaction_type,
                group_reading=args.group_reading,
                n_jobs=args.n_jobs,
            )
            dataset.graph_kernel_type = args.graph_kernel_type
            dataset.save(
                path=args.save_dir, filename=f"dataset_{i}.pkl", overwrite=False
            )
        datasets.append(dataset)
    print("Preprocessing Dataset Finished.")
    # set kernel_config
    kernel_config = get_kernel_config(
        dataset=dataset,
        graph_kernel_type=args.graph_kernel_type,
        features_kernel_type=args.features_kernel_type,
        features_hyperparameters=args.features_hyperparameters,
        features_hyperparameters_bounds=args.features_hyperparameters_bounds,
        features_hyperparameters_file=args.features_hyperparameters_file,
        mgk_hyperparameters_files=args.graph_hyperparameters,
    )
    optuna_bayesian_optimization_gpr_multi_datasets(
        save_dir=args.save_dir,
        datasets=datasets,
        kernel_config=kernel_config,
        tasks_type=args.tasks_type,
        metrics=args.metrics,
        num_iters=args.num_iters,
        alpha=args.alpha_,
        alpha_bounds=args.alpha_bounds,
        d_alpha=args.d_alpha,
        seed=args.seed,
    )

