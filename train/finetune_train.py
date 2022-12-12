from typing import Tuple, List
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

import numpy as np
import os
import json
from tqdm.rich import tqdm

from env import MetaRLEnv, EnvConfig
from utils import make_sequential_log_dir, TrainConfig

envConfig = EnvConfig()
trainConfig = TrainConfig()


class FineTuneRLAlgo:
    def __init__(
            self,
            # Environment
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
            #  Meta Learning
            n_test_tasks: int = 1,
            n_test_episodes: int = 1,
            #  Misc
            snapshot_dir: str = "saves/FINETUNEPPO/",
            seed: int = 1,
    ):
        # Environment Setup
        self.meta_batch_size = meta_batch_size
        self.env_xml_path = env_xml_path
        self.snapshot_dir = snapshot_dir
        self.do_render = do_render
        self.num_objs = num_objs
        self.num_samples = num_samples
        self.env = MetaRLEnv(
            self.env_xml_path, do_render=self.do_render, use_akro=False)
        check_env(self.env, warn=True)
        self.envs = self.get_envs_list(
            num_objs=num_objs, num_samples=num_samples)

    def get_envs_list(
            self,
            # TODO: Fix env to have objects added variably
            num_objs: Tuple[int] = (1, 1),
            num_samples: int = 5,
    ) -> List:
        envs = []
        num_samples = int(num_samples / (1 + num_objs[1] - num_objs[0]))
        for _ in range(num_samples):
            for number_of_objects in range(num_objs[0], num_objs[1] + 1):
                envs.append(
                    lambda: MetaRLEnv(
                        self.env_xml_path,
                        do_render=self.do_render,
                        actor_loc=envConfig.WORLD_MAX_LOC * np.random.rand(3),
                        objects_loc=envConfig.WORLD_MAX_LOC
                        * np.random.rand(number_of_objects, 3),
                        number_of_objects=number_of_objects,
                    )
                )
        return envs

    def train(self, epochs):
        self.model = PPO("MlpPolicy", self.env, verbose=1, device="cuda:0",
                         n_steps=envConfig.MAX_NUM_STEPS, tensorboard_log=self.snapshot_dir, ent_coef=0.01)
        self.model.learn(total_timesteps=epochs, progress_bar=True)
        self.pretrained_model_path = os.path.join(
            self.snapshot_dir, "pretrained.zip")
        self.model.save(self.pretrained_model_path)

    def evaluate(self, num_evals, num_samples_per_eval, finetune_epochs=1000, lr=0.0003/10., pretrained_model_path=None):
        rewards = []
        num_success = 0
        num_attempts = 0
        self.envs = self.get_envs_list(
            num_objs=self.num_objs, num_samples=num_evals)

        if pretrained_model_path is None:
            pretrained_model_path = self.pretrained_model_path

        for env in self.envs:  # repeats num_evals times
            env = env()
            model = PPO.load(pretrained_model_path)
            model.set_env(env)
            print("Finetuning...")
            model.learn(total_timesteps=finetune_epochs, progress_bar=True)
            print("Evaluating...")
            for _ in tqdm(range(num_samples_per_eval)):
                reward, done = self.run_single_eval(env, model)
                rewards.append(reward)
                if done:
                    num_success += 1.
                num_attempts += 1.
        results = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'num_success': num_success,
            'percent_success': num_success/num_attempts
        }
        return results

    def run_single_eval(self, env, model, num_steps=envConfig.MAX_NUM_STEPS):
        episode_rewards = 0.0
        obs = env.reset()
        for i in range(num_steps):
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_rewards += reward
            if done:
                obs = env.reset()
        return episode_rewards, done


if __name__ == "__main__":
    snapshot_dir = "saves/FinetunePPO/run"
    snapshot_dir = make_sequential_log_dir(snapshot_dir)

    algo = FineTuneRLAlgo(do_render=False, snapshot_dir=snapshot_dir)
    algo.train(trainConfig.pretrain_epochs)
    results = algo.evaluate(
        finetune_epochs=trainConfig.finetune_epochs,
        num_evals=trainConfig.num_evals,
        num_samples_per_eval=trainConfig.num_samples_per_eval)

    print(results)
    metrics_path = os.path.join(snapshot_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        print(metrics_path)
        json.dump(results, f)
    print("success")
