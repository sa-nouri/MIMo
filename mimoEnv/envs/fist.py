"""MIMo single-hand fist environment for COMPOSER evaluation.

Scene consists of MIMo with the left arm in a fixed posture; only the
hand joints are actuated. Used by the hierarchical COMPOSER experiments
to study finger-level latent representations.

Originally received from the Trieschlab COMPOSER team (May 2026); ported
into MIMo proper to register the env under ``MIMoFist-v0``.
"""
import os

import numpy as np

from mimoEnv.envs.mimo_env import MIMoEnv, SCENE_DIRECTORY
from mimoActuation.muscle import MuscleModel


FIST_XML = os.path.join(SCENE_DIRECTORY, "fist_scene.xml")
""" Path to the fist scene.

:meta hide-value:
"""


TOUCH_PARAMS = {
    "scales": {
        "left_ffdistal": 0.02,
        "left_mfdistal": 0.02,
        "left_rfdistal": 0.02,
        "left_lfdistal": 0.02,
        "left_thdistal": 0.02,
    },
    "touch_function": "force_vector",
    "response_function": "spread_linear",
}
""" Touch parameters for the fist environment. One sensor per distal fingertip.

:meta hide-value:
"""


JOINT_NAMES = [
    "robot:left_th_pivot",
    "robot:left_th_middle",
    "robot:left_th_distal",
    "robot:left_th_swivel",

    "robot:left_mf_side",
    "robot:left_mf_knuckle",
    "robot:left_mf_middle",
    "robot:left_mf_distal",

    "robot:left_rf_side",
    "robot:left_rf_knuckle",
    "robot:left_rf_middle",
    "robot:left_rf_distal",

    "robot:left_lf_side",
    "robot:left_lf_meta",
    "robot:left_lf_knuckle",
    "robot:left_lf_middle",
    "robot:left_lf_distal",

    "robot:left_ff_side",
    "robot:left_ff_knuckle",
    "robot:left_ff_middle",
    "robot:left_ff_distal",
]
""" Joint names governing the left-hand fingers. Used by the COMPOSER
state encoder to read a low-dimensional proprio slice.

:meta hide-value:
"""


class MIMoFistEnv(MIMoEnv):
    """Single-hand env for COMPOSER experiments.

    Reward is constant zero; success / failure / termination are
    disabled. The env is exploration-only and consumers (composer_hier)
    drive learning via their own goal-conditioned wrapper.
    """

    def __init__(self,
                 model_path=FIST_XML,
                 frame_skip=5,
                 proprio_params=None,
                 touch_params=TOUCH_PARAMS,
                 vision_params=None,
                 vestibular_params=None,
                 actuation_model=MuscleModel,
                 done_active=True,
                 joint_names=JOINT_NAMES,
                 **kwargs):

        super().__init__(model_path=model_path,
                         frame_skip=frame_skip,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         actuation_model=actuation_model,
                         done_active=done_active,
                         **kwargs)

        self.joint_names = joint_names
        self.qpos_ids = [
            self.model.jnt_qposadr[self.model.joint(name).id]
            for name in self.joint_names
        ]
        self.state_dim = len(self.qpos_ids)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return 0

    def is_success(self, achieved_goal, desired_goal):
        return False

    def is_failure(self, achieved_goal, desired_goal):
        return False

    def is_truncated(self):
        return False

    def reset_model(self):
        return self._get_obs()

    def sample_goal(self):
        return np.zeros((1,), dtype=np.float32)

    def get_achieved_goal(self):
        return np.zeros((1,), dtype=np.float32)
