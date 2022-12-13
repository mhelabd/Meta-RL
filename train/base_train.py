import time
from typing import Tuple, List
import numpy as np
from tqdm.rich import tqdm

from garage import Environment, EnvSpec
from garage.envs import GymEnv
from garage.trainer import Trainer
from garage._dtypes import StepType

from env import MetaRLEnv, EnvConfig
from utils import TrainConfig

envConfig = EnvConfig()
trainConfig = TrainConfig()


class BaseTrainer:
    def __init__(
        self,
        # Required
        trainer: Trainer,
        algo,   # from garage.torch.algos
        env: Environment,
        env_spec: EnvSpec,
        # Environment
        env_xml_path: str = "env.xml",
        do_render: bool = True,
        num_objs: Tuple[int, int] = (1, 1),
        #  Learning Hyperparameters
        meta_batch_size: int = 1,
    ):
        self.trainer = trainer
        self.algo = algo
        self.env = env
        self.env_spec = env_spec
        self.env_xml_path = env_xml_path
        self.do_render = do_render
        self.meta_batch_size = meta_batch_size
        self.num_objs = num_objs

    def get_envs_list(
            self,
            # TODO: Fix env to have objects added variably
            num_objs: Tuple[int] = (1, 1),  # range of num objects to include
            num_samples: int = 5,
    ) -> List:
        envs = []
        num_samples = int(num_samples / (1 + num_objs[1] - num_objs[0]))
        actor_loc = [0.55, 0.55, 0.15]
        objects_loc = [[0.75, 0.75, 0.125]]
        for _ in range(num_samples):
            for number_of_objects in range(num_objs[0], num_objs[1] + 1):
                if self.do_render:
                    envs.append(
                        lambda: MetaRLEnv(
                            self.env_xml_path,
                            do_render=False, #TEMPORARY
                            # envConfig.WORLD_MAX_LOC * np.random.rand(3),
                            actor_loc=actor_loc,
                            # envConfig.WORLD_MAX_LOC* np.random.rand(3),
                            objects_loc=objects_loc,
                            # 0-5 0-5
                            # ( [1,4], [0,5] 0.5)
                            # ( [1,4], [0,4] 0.05)
                            target_loc= [
                                1 + (envConfig.WORLD_MAX_LOC-2) * np.random.rand(), 
                                1 + (envConfig.WORLD_MAX_LOC-2) * np.random.rand(),
                                0.05,
                            ],
                            number_of_objects=number_of_objects,
                        )
                    )
                else:
                    envs.append(
                        lambda: GymEnv(MetaRLEnv(
                            self.env_xml_path,
                            do_render=self.do_render,
                            # envConfig.WORLD_MAX_LOC * np.random.rand(3),
                            actor_loc=actor_loc,
                            # envConfig.WORLD_MAX_LOC* np.random.rand(3),
                            objects_loc=objects_loc,
                            target_loc= [
                                1 + (envConfig.WORLD_MAX_LOC-2) * np.random.rand(), 
                                1 + (envConfig.WORLD_MAX_LOC-2) * np.random.rand(),
                                0.05,
                            ],
                            number_of_objects=number_of_objects,
                        ), max_episode_length=envConfig.MAX_NUM_STEPS,
                        ))
        return envs

    def train(self, epochs):
        self.trainer.setup(self.algo, self.env)
        self.trainer.train(
            n_epochs=epochs,
            batch_size=self.meta_batch_size * self.env_spec.max_episode_length
        )

    def evaluate(self, num_evals, num_samples_per_eval, do_render: bool = False):
        rewards = []
        num_success = 0
        num_attempts = 0
        self.do_render = True
        self.envs = self.get_envs_list(
            num_objs=self.num_objs, num_samples=num_evals)
        for env in self.envs:
            for _ in tqdm(range(num_samples_per_eval)):
                reward, done = self.run_single_eval(env(), self.algo.policy)
                rewards.append(reward)
                if done:
                    num_success += 1.
                num_attempts += 1
        results = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'num_success': num_success,
            'percent_success': num_success/num_attempts
        }
        return results

    def run_single_eval(self, env, policy, num_steps=envConfig.MAX_NUM_STEPS):
        episode_rewards = 0.0
        obs = env.reset()
        for i in range(num_steps):
            action, _ = policy.get_action(obs)
            obs, rewards, done, _ = env.step(action)
            env.render()
            if done:
                print("done")
                obs = env.reset()
                if rewards > 0:
                    return episode_rewards, True
                return episode_rewards, False                
        return episode_rewards, False

        #     env_step = env.step(action)

        #     obs, reward, step_type = env_step.observation, env_step.reward, env_step.step_type
        #     episode_rewards += reward
        #     if step_type == StepType.TERMINAL:
        #         obs, _ = env.reset()
        #         return episode_rewards, step_type
        # return episode_rewards, step_type
