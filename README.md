# Assessing Meta-Reinforcement Learning Algorithms in Complex Environments

Final Project for CS 330: Deep Multi-Task and Meta Learning. Using randomly sampled Mujoco world environments to test leading Meta-RL algorithms on task distributions that require both navigation and object manipulation. Link to final paper: [link]

## Environment
To set up the environment, install necessary requirements from `requirements.txt`.
```
pip install -r requirements.txt
```
Our testing was done in a conda environment using Python 3.8.

## Training

All Meta-RL algorithms inherit the base class `BaseTrainer` found in `train/base_train.py`.
All training hyperparameters can be found `TrainConfig` in `utils.py`.
All environment parameters can be found in `EnvConfig` in `env.py`

To run a trainer, simply run the train file associated with the model
```
python train/meta_train.py         # MAML + PPO
python train/pearl_train.py        # PEARL
python train/finetune_train.py     # Finetune on pretrained PPO
python train/multitask_train.py    # Multitask PPO
python train/vanilla_train.py      # Train PPO on one environment, for debugging environments
```

Training is automatically stopped after the specified number of epochs, or alternatively, can be stopped manually with `ctrl + c`, and then testing metrics will automatically be run on the last saved model. 

# Logging

Models checkpoints and metrics are automatically logged to the `log_dir` variable specified in each trainer, which by default is set to `saves/{model_name}/`