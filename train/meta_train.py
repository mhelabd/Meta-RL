#!/usr/bin/env python3
# pylint: disable=no-value-for-parameter
import click
import torch
import akro
import os
import pickle
import json

from env import MetaRLEnv, EnvConfig
from train.base_train import BaseTrainer
from utils import TrainConfig
from typing import Tuple, List

from garage import wrap_experiment, EnvSpec, Environment
from garage.envs import GymEnv
from garage.experiment import MetaEvaluator
from garage.experiment.experiment import ExperimentContext
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import ConstructEnvsSampler
from garage.sampler import LocalSampler, RaySampler, MultiprocessingSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import MAMLPPO
from garage.torch.algos import PEARL
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

import numpy as np
# from tqdm.rich import tqdm
# import tensorboard


envConfig = EnvConfig()
trainConfig = TrainConfig()


class MetaRLAlgo(BaseTrainer):
    def __init__(
            self,
            # Environment
            ctxt=None,
            env_xml_path: str = "env.xml",
            do_render: bool = True,
            # TODO: Fix env to have objects added variably
            num_objs: Tuple[int] = (1, 1),
            num_samples: int = 12,
            meta_algo: str = "maml",
            # Learning Algo
            policy_hidden_states: Tuple[int] = (
                64, 64),
            value_fn_hidden_states: Tuple[int] = (
                32, 32),
            #  Learning Hyperparameters
            meta_batch_size: int = 1,
            discount: float = 0.99,
            gae_lambda: float = 1.0,
            inner_lr: float = 0.1,
            outer_lr: float = 1e-3,
            num_grad_updates: int = 1,
            lr_clip_range: float = 5e-1,
            entropy_method='no_entropy',
            #  Meta Learning
            n_test_tasks: int = 1,
            n_test_episodes: int = 1,
            #  Misc
            snapshot_dir: str = "saves/MAMLPPO/",
            seed: int = 2,
            use_gpu=False,
    ):
        set_seed(seed)
        # Environment Setup
        self.num_objs = num_objs
        self.meta_batch_size = meta_batch_size
        self.env_xml_path = env_xml_path
        self.snapshot_dir = snapshot_dir
        self.do_render = do_render
        self.env = MetaRLEnv(self.env_xml_path, do_render=self.do_render)
        self.env = GymEnv(self.env, max_episode_length=envConfig.MAX_NUM_STEPS)
        assert isinstance(self.env, Environment)

        # Learning Models

        self.policy = GaussianMLPPolicy(
            env_spec=self.env.spec,
            hidden_sizes=policy_hidden_states,
            hidden_nonlinearity=torch.tanh,
            output_nonlinearity=None,
        )
        self.value_function = GaussianMLPValueFunction(
            env_spec=self.env.spec,
            hidden_sizes=value_fn_hidden_states,
            hidden_nonlinearity=torch.tanh,
            output_nonlinearity=None,
        )
        # Meta Learning Setup
        self.envs = self.get_envs_list(
            num_objs=num_objs, num_samples=num_samples)
        self.task_sampler = ConstructEnvsSampler(self.envs)
        self.meta_evaluator = MetaEvaluator(
            test_task_sampler=self.task_sampler,
            n_test_tasks=n_test_tasks,
            n_test_episodes=n_test_episodes,
        )
        # Trainer Setup
        if ctxt == None:
            ctxt = ExperimentContext(
                snapshot_dir=f"saves/MAMLPPO/policy_hidden_states={policy_hidden_states}"
                f"value_fn_hidden_states={value_fn_hidden_states} , n_test_tasks={n_test_tasks}"
                f"n_test_episodes={n_test_episodes}, seed={seed}",
                snapshot_mode="last",
                snapshot_gap=1,
            )

        self.sampler = LocalSampler(
            agents=self.policy,
            envs=[env() for env in self.envs],
            n_workers=num_samples,
            max_episode_length=self.env.spec.max_episode_length)

        self.trainer = Trainer(ctxt)
        self.algo = MAMLPPO(
            env=self.env,
            policy=self.policy,
            sampler=self.sampler,
            task_sampler=self.task_sampler,
            value_function=self.value_function,
            meta_batch_size=self.meta_batch_size,
            discount=discount,
            gae_lambda=gae_lambda,
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            num_grad_updates=num_grad_updates,
            meta_evaluator=self.meta_evaluator,
            lr_clip_range=lr_clip_range,
            entropy_method=entropy_method
        )

        super().__init__(
            trainer=self.trainer,
            algo=self.algo,
            env=self.env,
            env_spec=self.env.spec,
            env_xml_path=self.env_xml_path,
            do_render=self.do_render,
            meta_batch_size=self.meta_batch_size,
            num_objs=self.num_objs
        )


@wrap_experiment(
    snapshot_mode='all',
    log_dir="saves/MAMLPPO/policy_hidden_states={policy_hidden_states}"
    "value_fn_hidden_states={value_fn_hidden_states} , n_test_tasks={n_test_tasks}"
    "n_test_episodes={n_test_episodes}, seed={seed}"
)
def my_experiment(ctxt):
    algo = MetaRLAlgo(ctxt=ctxt,
                      num_samples=trainConfig.num_samples,
                      meta_batch_size=trainConfig.meta_batch_size,
                      n_test_tasks=trainConfig.n_test_tasks,
                      n_test_episodes=trainConfig.n_test_episodes,
                      do_render=trainConfig.do_render,
                      inner_lr=trainConfig.inner_lr,
                      outer_lr=trainConfig.outer_lr,
                      lr_clip_range=trainConfig.lr_clip_range,
                      entropy_method=trainConfig.entropy_method
                      )
    try:
        print("about to train")
        algo.train(trainConfig.epochs)
    except KeyboardInterrupt:
        print("Interrupted, evaluatings now")
    # Evaluation
    results = algo.evaluate(
        num_evals=trainConfig.num_evals,
        num_samples_per_eval=trainConfig.num_samples_per_eval,
        do_render=trainConfig.do_render,
    )
    print(results)
    metrics_path = os.path.join(ctxt.snapshot_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    my_experiment()
