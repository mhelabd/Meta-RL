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
from garage.experiment.task_sampler import ConstructEnvsSampler, EnvPoolSampler
from garage.sampler import LocalSampler, RaySampler, MultiprocessingSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import MAMLPPO
from garage.torch.algos import PEARL
from garage.torch.algos.pearl import PEARLWorker
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
from garage.torch.policies import (
    ContextConditionedPolicy, TanhGaussianMLPPolicy)
from garage.torch.embeddings import MLPEncoder
from garage.torch.q_functions import ContinuousMLPQFunction

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
            # Learning Algo
            latent_size=5,
            net_size=300,
            encoder_hidden_size=200,
            num_steps_per_epoch=2000,
            num_initial_steps=2000,
            #  Learning Hyperparameters
            meta_batch_size: int = 1,
            discount: int = 0.99,
            gae_lambda: int = 1.0,
            inner_lr: int = 0.1,
            num_grad_updates: int = 1,
            #  Meta Learning
            n_test_tasks: int = 1,
            n_test_episodes: int = 1,
            #  Misc
            snapshot_dir: str = "saves/PEARL/",
            seed: int = 2,
            use_gpu = False
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

        # Meta Learning Setup
        self.envs = self.get_envs_list(
            num_objs=num_objs, num_samples=num_samples)

        self.env_sampler = ConstructEnvsSampler(self.envs)
        env = self.env_sampler.sample(num_samples)
        self.test_env_sampler = ConstructEnvsSampler(self.envs)

        # Trainer Setup
        if ctxt == None:
            ctxt = ExperimentContext(
                snapshot_dir=f"saves/PEARL/run",
                snapshot_mode="last",
                snapshot_gap=1,
            )
        self.trainer = Trainer(ctxt)

        # instantiate networks
        encoder_hidden_sizes = (encoder_hidden_size, encoder_hidden_size,
                                encoder_hidden_size)
        augmented_env = PEARL.augment_env_spec(self.env, latent_size)
        qf = ContinuousMLPQFunction(env_spec=augmented_env,
                                    hidden_sizes=[net_size, net_size, net_size])

        vf_env = PEARL.get_env_spec(self.env, latent_size, 'vf')
        vf = ContinuousMLPQFunction(env_spec=vf_env,
                                    hidden_sizes=[net_size, net_size, net_size])

        inner_policy = TanhGaussianMLPPolicy(
            env_spec=augmented_env, hidden_sizes=[net_size, net_size, net_size])

        self.sampler = LocalSampler(
            agents=None,
            envs=[env() for env in self.envs],
            n_workers=num_samples,
            max_episode_length=self.env.spec.max_episode_length,
            worker_class=PEARLWorker)

        self.algo = PEARL(
            env=env,
            policy_class=ContextConditionedPolicy,
            encoder_class=MLPEncoder,
            inner_policy=inner_policy,
            qf=qf,
            vf=vf,
            sampler=self.sampler,
            num_train_tasks=num_samples,
            num_test_tasks=n_test_tasks,
            latent_dim=latent_size,
            encoder_hidden_sizes=encoder_hidden_sizes,
            test_env_sampler=self.test_env_sampler,
            meta_batch_size=meta_batch_size,
            num_tasks_sample=num_samples,
            num_steps_per_epoch=envConfig.MAX_NUM_STEPS,
            batch_size=1,
            embedding_batch_size=1,
            embedding_mini_batch_size=1
            # num_steps_per_epoch=num_steps_per_epoch,
            # num_initial_steps=num_initial_steps,
            # num_tasks_sample=num_tasks_sample,
            # num_steps_prior=num_steps_prior,
            # num_extra_rl_steps_posterior=num_extra_rl_steps_posterior,
            # batch_size=batch_size,
            # embedding_batch_size=embedding_batch_size,
            # embedding_mini_batch_size=embedding_mini_batch_size,
            # reward_scale=reward_scale,
        )

        set_gpu_mode(use_gpu)
        self.algo.to()

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
    log_dir="saves/PEARL/run"
)
def my_experiment(ctxt):
    
    algo = MetaRLAlgo(ctxt=ctxt,
                      num_samples=trainConfig.num_samples,
                      meta_batch_size=trainConfig.meta_batch_size,
                      n_test_tasks=trainConfig.n_test_tasks,
                      n_test_episodes=trainConfig.n_test_episodes,
                      do_render=trainConfig.do_render,
                      use_gpu=True,)
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
