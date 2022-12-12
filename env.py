import gym
import mujoco
import mujoco_viewer
from gym import spaces
import numpy as np
from dataclasses import dataclass

from garage import EnvSpec
from dowel import tabular, logger

from utils import dist
from typing import List
import akro


def setattrs(_self, **kwargs):
    for k, v in kwargs.items():
        setattr(_self, k, v)


@dataclass(frozen=True)
class EnvConfig:
    RENDER_MODES = ["human"]
    POS_DIM = 3
    XQUAT_DIM = 4
    WORLD_MIN_LOC, WORLD_MAX_LOC = 0, 5
    X_MOTOR_NAME = "particle_hinge0:motortx"
    Y_MOTOR_NAME = "particle_hinge0:motorty"
    Z_MOTOR_NAME = "particle_hinge0:motorrz"
    ADHESION_MOTOR_NAME = "particle_hinge0:adhesion"
    AGENT_NAME = "particle_hinge0:particle"
    TARGET_NAME = "target"
    MAX_NUM_STEPS = 1024
    MIN_DIST_TO_TARGET = 1


envConfig = EnvConfig()


class MetaRLEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self,
                 env_xml_path: str,
                 obj_types: List[str] = ['cube'],
                 target_name: str = 'target',
                 actor_loc: List[float] = [0., 1., 1.],
                 objects_loc: List[List[float]] = [[0., 1., 1.]],
                 target_loc: List[float] = [0., 1., 1.],
                 number_of_objects: int = 1,
                 do_render: bool = True,
                 use_akro: bool = True,
                 ):
        super(MetaRLEnv, self).__init__()
        self.metadata = {'render.modes': 'human'}

        self.obj_types = obj_types
        self.number_of_objects = number_of_objects
        self.target_name = target_name
        self.do_render = do_render
        self.num_steps = 0
        self.use_akro = use_akro
        self.target_loc = np.array(target_loc)
        self.actor_loc = actor_loc
        self.objects_loc = objects_loc

        self.model = mujoco.MjModel.from_xml_path(env_xml_path)
        self.data = mujoco.MjData(self.model)

        self.action_space = self._set_action_space()
        # self.entity_names = self._get_object_names()
        #TEMPORARY OVERRIDE
        self.entity_names = np.array(['particle_hinge0:particle', 'target', 'cube0'])

        self.object_names = self._get_object_names(
            include_agent=False, include_target=False)
        self.objects_completed = np.array([False]*len(self.object_names))
        self._initialize_locations(actor_loc, objects_loc, target_loc)
        self.observation_space = self._set_observation_space()
        if use_akro:
            self.spec = EnvSpec(self.observation_space, self.action_space,
                                max_episode_length=envConfig.MAX_NUM_STEPS)
        self.max_episode_length = envConfig.MAX_NUM_STEPS

        if self.do_render:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        
        print("Init.", end="")

    def _initialize_locations(self, actor_loc: List[float], objects_loc: List[List[float]], target_loc: List[float]):
        # self.data.body(envConfig.AGENT_NAME).xpos = actor_loc
        self.data.site(envConfig.TARGET_NAME).xpos = target_loc
        # for i, object_loc in enumerate(objects_loc):
        #     self.data.body(self.object_names[i]).xpos = object_loc

    def _set_action_space(self):
        """Sets action space to be the space of movement of each motor."""
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        if self.use_akro:
            return akro.Box(low=low, high=high, dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _set_observation_space(self, include_agent: bool = True, include_target: bool = True):
        """Sets observation space to be location and orientation of objects, person, and target."""
        num_objects = len(self.entity_names)
        if self.use_akro:
            obs = akro.Box(
                low=envConfig.WORLD_MIN_LOC - 0.01,
                high=envConfig.WORLD_MAX_LOC + 0.01,
                shape=(num_objects * (envConfig.POS_DIM + envConfig.XQUAT_DIM), ),
                dtype=np.float64
            )
        else:
            obs = spaces.Box(
                low=envConfig.WORLD_MIN_LOC,
                high=envConfig.WORLD_MAX_LOC,
                shape=(num_objects * (envConfig.POS_DIM + envConfig.XQUAT_DIM), ),
                dtype=np.float64
            )
        return obs

    def is_object(self, name: str, obj: str = 'cube') -> bool:
        """Checks if the object with `name` is of type `obj`.

        Checks if:
        - Name of object starts with `obj`
        - Length of the name is less than the maximum length of a `obj` name
        """
        return name.startswith(obj) and len(name) <= len(obj) + len(str(self.number_of_objects))

    def _get_object_names(self, include_agent: bool = True, include_target: bool = True) -> np.ndarray:
        """Gets all the objects in the environment"""
        objects = set()
        for obj_type in self.obj_types:
            objects.update(set(filter(lambda x: self.is_object(x, obj_type),
                                      str(self.model.names).split('\\x00'))))
        if include_agent:
            objects.add(envConfig.AGENT_NAME)
        if include_target:
            objects.add(envConfig.TARGET_NAME)
        return np.array(list(objects))

    def get_object_state(self, name: str) -> np.ndarray:
        """Get the location and orientation of an object from data."""
        if name == envConfig.TARGET_NAME:
            return np.concatenate((self.target_loc, [0]*envConfig.XQUAT_DIM))
            # return np.concatenate((self.data.site(envConfig.TARGET_NAME).xpos, [0]*envConfig.XQUAT_DIM))
        return np.concatenate((self.data.body(name).xpos, self.data.body(name).xquat))

    def set_action(self, action: list) -> None:
        """Sets the action of the agent from a list of actions."""
        self.data.actuator(envConfig.X_MOTOR_NAME).ctrl[0] = action[0]
        self.data.actuator(envConfig.Y_MOTOR_NAME).ctrl[0] = action[1]
        self.data.actuator(envConfig.Z_MOTOR_NAME).ctrl[0] = action[2]

    def get_observation(self, names, flatten=True) -> np.ndarray:
        """Gets the observation to be the location of all objects and agent."""
        obs = np.array(list(map(self.get_object_state, names)))
        return obs.reshape(-1) if flatten else obs

    def get_reward(self, observation: np.ndarray) -> float:
        """Gets the reward function of trying to get objects as close to target as possible."""

        target_xpos = self.target_loc
        # get all object observations
        objects_xpos = self.get_observation(self.object_names, flatten=False)
        # discard rotation data, only keep xpos data
        objects_xpos = objects_xpos[:, :-envConfig.XQUAT_DIM]

        dists_cubes_to_targets = dist(objects_xpos, target_xpos)
        # Truth array for which cubes are newly successes
        new_successes = (dists_cubes_to_targets <
                         envConfig.MIN_DIST_TO_TARGET) & ~self.objects_completed
        if np.sum(new_successes) > 0:
            self.objects_completed = self.objects_completed | new_successes
            return 100 * np.sum(new_successes)
        return -0.001 * np.sum(dists_cubes_to_targets[~self.objects_completed])

        # agent_xpos = self.data.body(envConfig.AGENT_NAME).xpos
        # dist_agent_to_target = dist(agent_xpos, target_xpos)
        # if dist_agent_to_target < envConfig.MIN_DIST_TO_TARGET:
        # 	print("Success!", end="")
        # 	self.objects_completed = [True]  #terminal
        # 	return 100
        # return -0.001 * dist_agent_to_target

    def get_done(self, observation: np.ndarray) -> bool:
        """Gets whether the terminal condition has been reached.

        Checks if:
        - Maximum number of steps has been reached.
        - Or all objectives have been met.
        """
        if self.num_steps > envConfig.MAX_NUM_STEPS:
            return True
        elif all(self.objects_completed):
            print("Success!", end="")
            return True
        return False

    def step(self, action: list) -> list:
        """Steps the environment by taking action, running physics, seeing new obs and reward."""
        self.num_steps += 1
        self.set_action(action)
        mujoco.mj_step(self.model, self.data)
        observation = self.get_observation(self.entity_names)
        reward = self.get_reward(observation)
        # tabular.record('reward', reward)
        # tabular.record('num_steps', self.num_steps)
        # logger.log(tabular)
        done = self.get_done(observation)
        return [observation, reward, done, {}]

    def reset(self) -> np.ndarray:
        """Resets the environment and gives the new observation."""
        self.num_steps = 0
        self.objects_completed = np.array([False]*len(self.object_names))
        mujoco.mj_resetData(self.model, self.data)
        self._initialize_locations(self.actor_loc, self.objects_loc, self.target_loc)
        observation = self.get_observation(self.entity_names)
        return observation

    def render(self, mode: str = "human") -> None:
        """Renders the environment."""
        if self.do_render and self.viewer.is_alive:
            self.viewer.render()

    def close(self) -> None:
        """Ends the environment and the viewer."""
        if self.do_render and self.viewer:
            self.viewer.close()
