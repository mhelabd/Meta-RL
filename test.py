from stable_baselines3 import PPO
from env import MetaRLEnv
from tqdm.rich import tqdm


xml_path = "env.xml"
SAVES_TO = "saves/meta-rl-agent-push-4cube"
N_TEST_TRAJECTORIES = 100

print("Loading model...")
model = PPO.load(SAVES_TO)
env = MetaRLEnv(xml_path, do_render=True)

num_trajectories = 0
num_success = 0
obs = env.reset()
pbar = tqdm(total=N_TEST_TRAJECTORIES)
while num_trajectories < N_TEST_TRAJECTORIES:
	action, _states = model.predict(obs)
	obs, rewards, done, info = env.step(action)
	env.render()
	if done:
		if rewards > 0: num_success += 1
		print(rewards)
		obs = env.reset()
		num_trajectories += 1
		pbar.update(1)
pbar.close()
print(f"Accuracy: {num_success/num_trajectories}")
