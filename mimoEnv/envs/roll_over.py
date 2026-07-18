"""
This module contains a simple experiment in which MIMo tries to roll over.

MIMo starts either in prone or supine position. This can be adjusted below.
The task is to roll over to the opposite position.

The scene consists only of MIMo. His head is fixed.
Sensory input consists of proprioceptive and vestibular inputs,
using the default configurations for both.

MIMo initial position is determined by slightly randomizing all joint
positions from a standing position and then letting the simulation settle.
This leads to MIMo being in a slightly random prone or supine position each
episode. All episodes have a fixed length, there are no goal or failure states.

Reward shaping is employed, such that MIMo is penalized for using muscle
inputs and large inputs in particular. Additionally, he is rewarded each step
for the current rotation of his hip.

The class with the env is :class:`~mimoEnv.envs.standup.MIMoRollOverEnv` while
the path to the scene XML is defined in :data:`ROLL_OVER_XML`.
"""

from mimoEnv.envs.mimo_env import MIMoEnv, SCENE_DIRECTORY, \
    DEFAULT_PROPRIOCEPTION_PARAMS, DEFAULT_VESTIBULAR_PARAMS
from mimoActuation.actuation import SpringDamperModel
import mujoco
import numpy as np
import os

STARTING_POSITION = "prone"
""" Default initial position of MIMo for the legacy module-level constant
behaviour. Can be 'prone' or 'supine'. The class also accepts
``starting_position="random"`` as a constructor kwarg, which causes the
pose to be re-sampled (50/50) at every reset; this is the right setting
for representation-learning experiments that want both poses in the
training data.

:meta hide-value:
"""

ROLL_OVER_XML = os.path.join(SCENE_DIRECTORY, "roll_over_scene.xml")
""" Path to the roll over scene.

:meta hide-value:
"""


class MIMoRollOverEnv(MIMoEnv):
    """
    MIMo learns to roll over from prone or supine position.

    Attributes and parameters are the same as in the base class, but the
    default arguments are adapted for the scenario. Specifically we have
    :attr:`.done_active` and :attr:`.goals_in_observation` as ``False`` and
    touch and vision sensors disabled.

    Even though we define a success condition in :meth:
    `~mimoEnv.envs.standup.MIMoStandupEnv._is_success`, it is disabled since
    :attr:`.done_active` is set to ``False``. The purpose of this is to enable
    extra information for the logging features of stable baselines.

    Attributes:
        init_position (numpy.ndarray): The initial position.
    """

    def __init__(self,
                 model_path=ROLL_OVER_XML,
                 initial_qpos=None,
                 frame_skip=2,
                 age=None,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=None,
                 vision_params=None,
                 vestibular_params=DEFAULT_VESTIBULAR_PARAMS,
                 actuation_model=SpringDamperModel,
                 starting_position=None,
                 done_active=False,
                 **kwargs):

        # Resolve starting_position: explicit kwarg overrides the legacy
        # module-level constant so existing callers stay working.
        sp = STARTING_POSITION if starting_position is None else starting_position
        if sp not in ["prone", "supine", "random"]:
            msg = (f"Unknown starting position '{sp}'. "
                   "Needs to be 'prone', 'supine', or 'random'.")
            raise ValueError(msg)
        self._starting_position_mode = sp

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         frame_skip=frame_skip,
                         age=age,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         actuation_model=actuation_model,
                         goals_in_observation=False,
                         done_active=done_active,
                         **kwargs)

        self.model.body("hip").pos = [0, 0, 0.2]

        # Precompute the settled qpos for BOTH prone and supine starting poses
        # so reset_model can switch between them cheaply when in "random" mode.
        #
        # BOTH settles must reset to a clean (init_qpos, init_qvel) state first.
        # The prone settle previously ran with reset_state=False (settling from
        # the contaminated post-super().__init__ state); that diverged and
        # produced a garbage prone pose with the hip free-joint at z ~= 3.46 m,
        # so every prone (and ~50% of "random") episode started with MIMo
        # falling from ~2 m and reaching high achieved_goal purely from the
        # fall. Resetting first makes prone symmetric with the (correct) supine
        # settle and yields a stable lying pose (hip z ~= 0.05 m).
        base_quat = np.array([0, -0.7071068, 0, 0.7071068])
        self._hip_quat_prone = base_quat.copy()
        self._hip_quat_supine = base_quat * np.array([1, -1, 1, 1])
        self._init_position_prone = self._stabilise_to_pose(
            self._hip_quat_prone, reset_state=True)
        self._init_position_supine = self._stabilise_to_pose(
            self._hip_quat_supine, reset_state=True)

        # Pick the active init_position used by the *first* reset.
        if sp == "supine":
            self.init_position = self._init_position_supine
            self._current_starting_position = "supine"
        else:
            # 'prone' or 'random': prone is also the first-episode default for
            # 'random' mode (subsequent episodes get re-sampled in reset_model).
            self.init_position = self._init_position_prone
            self._current_starting_position = "prone"

    def _stabilise_to_pose(self, hip_quat, reset_state=False):
        """Set the hip body quaternion, simulate 100 steps, return settled qpos.

        When ``reset_state=True`` the simulation is first set back to
        ``(init_qpos, init_qvel)`` to clear residual dynamics from a prior
        settle (used for the supine settle, which runs after prone).
        Otherwise the function settles from the current sim state — used
        for the FIRST settle in ``__init__`` so contact resolution matches
        the pre-5dc3553 behaviour. See the explanatory comment in
        ``__init__`` above for why.

        The quat assignment happens BEFORE ``set_state`` so that
        ``set_state``'s internal ``mj_forward`` runs with the new
        kinematic frame already in place. Otherwise the forward
        kinematics would run once with the stale (old) quat, perturbing
        contact resolution and leading to an over-deep settle.
        """
        self.model.body("hip").quat = hip_quat
        if reset_state:
            self.set_state(self.init_qpos, self.init_qvel)
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        return self.data.qpos.copy()

    def is_success(self, achieved_goal, desired_goal):
        """ Did we reach our goal rotation.

        Arguments:
            achieved_goal (float): The achieved hip rotation.
            desired_goal (float): This target hip rotation.

        Returns:
            bool: If the achieved hip rotation exceeds the desired rotation.
        """

        success = (achieved_goal >= desired_goal)

        return success

    def is_failure(self, achieved_goal, desired_goal):
        """ Dummy function. Always returns ``False``.

        Arguments:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.

        Returns:
            bool: ``False``
        """
        return False

    def is_truncated(self):
        """ Dummy function. Always returns ``False``.

        Returns:
            bool: ``False``.
        """
        return False

    def reset_model(self):
        """ Resets the simulation.

        Return the simulation to the XML state, then slightly randomize all
        joint positions. Afterwards we let the simulation settle for a fixed
        number of steps. This leads to MIMo settling into a slightly random
        prone or supine position.

        When ``starting_position == "random"`` the prone / supine choice is
        re-sampled 50/50 at every reset.

        Returns:
            Dict: Observations after reset.
        """

        self.set_state(self.init_qpos, self.init_qvel)
        if self._starting_position_mode == "random":
            if self.np_random.uniform() < 0.5:
                self._current_starting_position = "prone"
                self.init_position = self._init_position_prone
            else:
                self._current_starting_position = "supine"
                self.init_position = self._init_position_supine
        # The stored init_position qpos was settled under a specific hip body
        # frame; the model's hip.quat is left at whichever settle ran last in
        # __init__ (supine), so it must be re-set to match the chosen pose or
        # the loaded prone qpos is interpreted in the wrong frame.
        if self._current_starting_position == "prone":
            self.model.body("hip").quat = self._hip_quat_prone
        else:
            self.model.body("hip").quat = self._hip_quat_supine
        qpos = self.init_position.copy()

        # Set initial positions stochastically.
        random = self.np_random.uniform(
            low=-0.01, high=0.01, size=len(qpos[7:])
        )
        qpos[7:] = qpos[7:] + random

        # Set initial velocities to zero.
        qvel = np.zeros(self.data.qvel.shape)

        self.set_state(qpos, qvel)

        # Perform 100 steps with no actions to stabilize initial position.
        actions = np.zeros(self.action_space.shape)
        self._set_action(actions)
        mujoco.mj_step(self.model, self.data, nstep=100)

        return self._get_obs()

    def sample_goal(self):
        """ Returns the goal rotation.

        We use a fixed goal rotation of 0.8.

        Returns:
            float: 0.8
        """
        return 0.8

    def get_achieved_goal(self):
        """ Get the standardized hip rotation of MIMo.

        Returns:
            float: The standardized hip rotation of MIMo.
        """

        # Get the rotation matrix of the hip.
        xmat = self.data.body("hip").xmat.reshape(3, 3)

        # Calculate the euler angle of the y-axis.
        angle = np.arctan2(
            -xmat[2, 0],
            np.sqrt(xmat[2, 1] ** 2 + xmat[2, 2] ** 2)
        )
        angle *= (180 / np.pi)

        # Normalize the angle to [0, 1].
        angle_norm = (angle - (-90)) / (90 - (-90))

        if self._current_starting_position == "prone":
            angle_norm = 1 - angle_norm

        return angle_norm

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Computes the reward.

        The reward consists of the standardized hip rotation with a
        penalty of the square of the control signal.

        Arguments:
            achieved_goal (float): The achieved hip rotation.
            desired_goal (float): This parameter is ignored.
            info (dict): This parameter is ignored.

        Returns:
            float: The reward as described above.
        """

        # Use the hip rotation as the main reward.
        reward = achieved_goal  # [0, 1]

        # Penalize excessive use of force.
        quad_ctrl_cost = 0.01 * np.square(self.data.ctrl).sum()  # [0, 0.44]
        reward -= quad_ctrl_cost

        return reward
