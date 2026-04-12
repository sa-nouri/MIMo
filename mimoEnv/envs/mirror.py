"""MIMo Mirror Environment: MIMo stands in front of a vertical mirror.

This environment places MIMo facing a reflective vertical surface. The agent
can observe its own reflection through its binocular eye cameras. The scene is
otherwise empty (no task object), making it suitable for self-exploration and
future mirror self-recognition (MSR) studies.

The environment is a minimal extension of :class:`~mimoEnv.envs.mimo_env.MIMoEnv`.
No task reward is defined (reward is always 0); the environment is intended for
intrinsic-motivation or curiosity-driven training, or for controlled
verification of the mirror rendering pipeline.

The class with the environment is :class:`MIMoMirrorEnv` and the path to the
scene XML is :data:`MIRROR_XML`.
"""

import os
import numpy as np

from mimoEnv.envs.mimo_env import (
    MIMoEnv,
    DEFAULT_PROPRIOCEPTION_PARAMS,
    DEFAULT_VISION_PARAMS,
    SCENE_DIRECTORY,
)
from mimoActuation.actuation import SpringDamperModel


MIRROR_XML = os.path.join(SCENE_DIRECTORY, "mirror_scene.xml")
"""Path to the mirror scene XML.

:meta hide-value:
"""


class MIMoMirrorEnv(MIMoEnv):
    """MIMo in front of a vertical mirror.

    MIMo is placed facing a vertical reflective surface at approximately 1.2 m
    distance. Vision is enabled by default (binocular, 256x256) so that the
    agent receives its own reflection as part of the observation. All body
    parts are free to move (no welds except those in the base MIMo model).

    This environment defines no external task. The reward is always 0 and
    episodes never terminate due to success or failure (only via the maximum
    episode step limit). This makes it suitable for:

    - Verifying that the mirror renders correctly in the agent's cameras
    - Intrinsic-motivation or curiosity-driven exploration
    - Future mirror self-recognition (MSR) experiments

    Attributes:
        mirror_geom_id (int): MuJoCo geom ID of the mirror surface, for
            programmatic access (e.g., checking visibility in camera frames).
    """

    def __init__(
        self,
        model_path=MIRROR_XML,
        initial_qpos=None,
        frame_skip=2,
        proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
        touch_params=None,
        vision_params=DEFAULT_VISION_PARAMS,
        vestibular_params=None,
        actuation_model=SpringDamperModel,
        goals_in_observation=False,
        done_active=False,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            initial_qpos=initial_qpos,
            frame_skip=frame_skip,
            proprio_params=proprio_params,
            touch_params=touch_params,
            vision_params=vision_params,
            vestibular_params=vestibular_params,
            actuation_model=actuation_model,
            goals_in_observation=goals_in_observation,
            done_active=done_active,
            **kwargs,
        )
        import mujoco
        self.mirror_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "mirror_surface"
        )

    def sample_goal(self):
        """No goal in this environment.

        Returns:
            numpy.ndarray: An empty array.
        """
        return np.array([])

    def get_achieved_goal(self):
        """No goal in this environment.

        Returns:
            numpy.ndarray: An empty array.
        """
        return np.array([])

    def is_success(self, achieved_goal, desired_goal):
        """Always returns False (no task).

        Returns:
            bool: False.
        """
        return False

    def is_failure(self, achieved_goal, desired_goal):
        """Always returns False (no failure condition).

        Returns:
            bool: False.
        """
        return False

    def is_truncated(self):
        """Always returns False.

        Returns:
            bool: False.
        """
        return False

    def compute_reward(self, achieved_goal, desired_goal, info):
        """No extrinsic reward. Always returns 0.

        Returns:
            float: 0.0.
        """
        return 0.0

    def reset_model(self):
        """Reset to initial standing position.

        Returns:
            dict: Observations after reset.
        """
        qpos = self.init_qpos.copy()
        qvel = np.zeros(self.data.qvel.shape)
        self.set_state(qpos, qvel)
        return self._get_obs()
