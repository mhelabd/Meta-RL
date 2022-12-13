#!/usr/bin/env python3
# pylint: disable=no-value-for-parameter
import click
import torch
import akro
import os
import json

from env import MetaRLEnv, EnvConfig
from train.base_train import BaseTrainer
from utils import TrainConfig

from garage import wrap_experiment, EnvSpec, Environment
from garage.envs import GymEnv, normalize
from garage.experiment import MetaEvaluator
from garage.experiment.experiment import ExperimentContext
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import ConstructEnvsSampler
from garage.sampler import LocalSampler, RaySampler, MultiprocessingSampler
from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

import numpy as np
from tqdm.rich import tqdm
from typing import Tuple, List

# import tensorboard


envConfig = EnvConfig()
trainConfig = TrainConfig()


class MultiTaskRLAlgo(BaseTrainer):
    def __init__(
        self,
        # Environment
        ctxt=None,
        env_xml_path: str = "env.xml",
        do_render: bool = True,
        # TODO: Fix env to have objects added variably
        num_objs: Tuple[int] = (1, 1),
        num_samples: int = 1,
        # Learning Algo
        policy_hidden_states: Tuple[int] = (
            64, 64),
        value_fn_hidden_states: Tuple[int] = (
            32, 32),
        #  Learning Hyperparameters
        meta_batch_size: int = 1,
        discount: int = 0.99,
        gae_lambda: int = 1.0,
        inner_lr: int = 0.1,
        num_grad_updates: int = 1,
        entropy_method: str = "no_entropy",
        #  Misc
        snapshot_dir: str = "saves/PPO/",
        seed: int = 1,
    ):
        set_seed(seed)
        # Environment Setup
        self.env_xml_path = env_xml_path
        self.snapshot_dir = snapshot_dir
        self.do_render = do_render
        self.env = MetaRLEnv(self.env_xml_path, do_render=self.do_render)
        self.env = GymEnv(self.env, max_episode_length=envConfig.MAX_NUM_STEPS)
        assert isinstance(self.env, Environment)
        # self.env = normalize(
        #     self.env, expected_action_scale=10.0
        # )  # TODO: expected_action_scale
        # Learning Models
        self.env_spec = EnvSpec(
            self.env.observation_space, self.env.action_space, max_episode_length=envConfig.MAX_NUM_STEPS)
        # self.env.spec = self.env_spec
        self.policy = GaussianMLPPolicy(
            env_spec=self.env_spec,
            hidden_sizes=policy_hidden_states,
            hidden_nonlinearity=torch.tanh,
            output_nonlinearity=None,
        )
        self.value_function = GaussianMLPValueFunction(
            env_spec=self.env_spec,
            hidden_sizes=value_fn_hidden_states,
            hidden_nonlinearity=torch.tanh,
            output_nonlinearity=None,
        )
        # Meta Learning Setup
        env_samples = self.get_envs_list(
            num_objs=num_objs, num_samples=num_samples)
        self.task_sampler = ConstructEnvsSampler(env_samples)
        # Trainer Setup
        if ctxt == None:
            ctxt = ExperimentContext(
                snapshot_dir=f"saves/MAMLPPO/policy_hidden_states={policy_hidden_states}"
                f"value_fn_hidden_states={value_fn_hidden_states}, seed={seed}",
                snapshot_mode="last",
                snapshot_gap=1,
            )

        self.sampler = LocalSampler(
            agents=self.policy,
            envs=self.env,
            max_episode_length=self.env_spec.max_episode_length)

        self.trainer = Trainer(ctxt)
        self.algo = PPO(
            env_spec=self.env_spec,
            policy=self.policy,
            value_function=self.value_function,
            sampler=self.sampler,
            # task_sampler=self.task_sampler,
            # meta_batch_size=self.meta_batch_size,
            discount=discount,
            gae_lambda=gae_lambda,
            # inner_lr=inner_lr,
            # num_grad_updates=num_grad_updates
            # meta_evaluator=self.meta_evaluator,
            entropy_method=entropy_method
        )

        super().__init__(
            trainer=self.trainer,
            algo=self.algo,
            env=self.env,
            env_spec=self.env_spec,
            env_xml_path=self.env_xml_path,
            do_render=self.do_render,
            meta_batch_size=meta_batch_size,
            num_objs=num_objs
        )

@wrap_experiment(
    snapshot_mode='all',
    log_dir="saves/MultitaskPPO/policy_hidden_states={policy_hidden_states}"
    "value_fn_hidden_states={value_fn_hidden_states} , n_test_tasks={n_test_tasks}"
    "n_test_episodes={n_test_episodes}, seed={seed}"
)
def my_experiment(ctxt):
    algo = MultiTaskRLAlgo(ctxt=ctxt,
                            num_samples=trainConfig.num_samples,
                            meta_batch_size=trainConfig.meta_batch_size,
                            entropy_method=trainConfig.entropy_method,
                            do_render=trainConfig.do_render)
    try:
        algo.train(trainConfig.epochs)
    except KeyboardInterrupt:
        print("Interrupted, evaluatings now")   

    # Evaluation
    results = algo.evaluate(
        num_evals=trainConfig.num_evals,
        num_samples_per_eval=trainConfig.num_samples_per_eval
    )
    print(results)
    metrics_path = os.path.join(ctxt.snapshot_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    my_experiment()
    # metaRLAlgo = MetaRLAlgo(do_render=False)
    # metaRLAlgo.train(100)
