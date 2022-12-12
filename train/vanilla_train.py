import gym
import numpy as np
from env import MetaRLEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from gym.utils import play
# Parallel environments

xml_path = "env.xml"
env = MetaRLEnv(xml_path, do_render=False)
check_env(env, warn=True)
env = make_vec_env(lambda : env, n_envs=8)
# env = make_vec_env(lambda : env, n_envs=16)

# gym.utils.play.play(env, zoom=3)

# SAVES_TO = "saves/meta-rl-agent-to-cube"
SAVES_TO = "saves/meta-rl-agent-push-4cube"
TRAIN = True
model = PPO("MlpPolicy", env, verbose=1, device="cuda:1", n_steps=2048, tensorboard_log="./ppo_single_cube/", ent_coef=0.01)
if TRAIN:
    print("Starting learning...")
    try:
        model.learn(total_timesteps=1_000_000_000, progress_bar=True)
        print("Saving model...")
        model.save(SAVES_TO)
    except KeyboardInterrupt:
        print(f"Saving model... to {SAVES_TO}")
        model.save(SAVES_TO)

print("Loading model...")
model = PPO.load(SAVES_TO)
env = MetaRLEnv(xml_path)
obs = env.reset()
print("Rendering...")
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

