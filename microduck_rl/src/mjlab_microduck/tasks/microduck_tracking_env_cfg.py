"""Microduck motion-tracking task: imitate a reference choreography.

Ports mjlab's BeyondMimic-style tracking family (shipped only for the Unitree
G1) to the Microduck, so a keyframed reference motion (npz from
scripts/motion_csv_to_npz.py) can be trained as a policy. First use: the
Toosie Slide (see scripts/author_toosie_reference.py).

Differences from the stock G1 config, and why:
  - The microduck robot cfg defines no builtin imu velocity sensors, so the
    base velocity observations use the standard data-backed terms instead of
    builtin_sensor reads. Sim-only for now; a hardware deployment would swap
    the actor's base_lin_vel for an estimator or drop it (the G1 config's
    no-state-estimation variant shows the pattern).
  - Every distance-like constant is rescaled for a 25 cm, 780 g robot
    (roughly 1/6 of G1 scale): tracking reward stds, RSI pose noise,
    termination thresholds, push velocities, CoM DR.
  - The BAM actuator invariants from AGENTS.md are honored: standalone env
    cfgs must register expand_bam_friction_fields, joint-friction DR must
    scale the actuator's friction_scale, and the NaN guard termination is
    wired.

The tracking obs contract is its own (not the 61-D hot-swap family): this
task's export is a dance policy, not a runtime hot-swap sibling.
"""

from dataclasses import replace as _dc_replace
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import observations as base_obs
from mjlab.managers import (
    EventTermCfg,
    ObservationTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp

# Default reference motion, committed with the repo (regenerate via
# scripts/author_toosie_reference.py + scripts/motion_csv_to_npz.py).
DEFAULT_MOTION_FILE = str(
    Path(__file__).resolve().parents[3] / "motions" / "toosie_slide.npz"
)

# Trunk + feet + head: the bodies whose reference poses the reward tracks.
TRACKED_BODY_NAMES = (
    "trunk_base",
    "ankle_left",
    "ankle_right",
    "neck_pitch",
    "yaw_roll_motion",
)

# Duck-scale perturbation envelope (G1 uses +-0.5 m/s; the courier task
# proved +-0.25 m/s pushes are survivable for this robot).
DUCK_VELOCITY_RANGE = {
    "x": (-0.25, 0.25),
    "y": (-0.25, 0.25),
    "z": (-0.1, 0.1),
    "roll": (-0.3, 0.3),
    "pitch": (-0.3, 0.3),
    "yaw": (-0.78, 0.78),
}


def make_microduck_tracking_env_cfg(
    play: bool = False,
    motion_file: str = DEFAULT_MOTION_FILE,
) -> ManagerBasedRlEnvCfg:
    """Create the microduck motion-tracking configuration."""
    cfg = make_tracking_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (self_collision_cfg,)

    cfg.actions["joint_pos"].scale = 1.0

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = motion_file
    motion_cmd.anchor_body_name = "trunk_base"
    motion_cmd.body_names = TRACKED_BODY_NAMES
    # RSI noise at duck scale: centimeters, not decimeters.
    motion_cmd.pose_range = {
        "x": (-0.02, 0.02),
        "y": (-0.02, 0.02),
        "z": (-0.005, 0.005),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
    }
    motion_cmd.velocity_range = dict(DUCK_VELOCITY_RANGE)
    motion_cmd.joint_position_range = (-0.1, 0.1)

    # No builtin imu sensors on this robot cfg: use data-backed velocities.
    cfg.observations["actor"].terms["base_lin_vel"] = ObservationTermCfg(
        func=base_obs.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5)
    )
    cfg.observations["actor"].terms["base_ang_vel"] = ObservationTermCfg(
        func=base_obs.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
    )
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=base_obs.base_lin_vel
    )
    cfg.observations["critic"].terms["base_ang_vel"] = ObservationTermCfg(
        func=base_obs.base_ang_vel
    )

    # Tracking reward stds at duck scale (G1 position stds divided by ~6;
    # orientation and angular-velocity stds are scale-free and stay).
    cfg.rewards["motion_global_root_pos"].params["std"] = 0.05
    cfg.rewards["motion_body_pos"].params["std"] = 0.05
    cfg.rewards["motion_body_lin_vel"].params["std"] = 0.3

    # Terminations at duck scale.
    cfg.terminations["anchor_pos"].params["threshold"] = 0.06
    cfg.terminations["ee_body_pos"].params["threshold"] = 0.06
    cfg.terminations["ee_body_pos"].params["body_names"] = (
        "ankle_left",
        "ankle_right",
    )
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # Events: duck-scale pushes and DR, plus the BAM invariants.
    cfg.events["push_robot"].params["velocity_range"] = dict(DUCK_VELOCITY_RANGE)
    cfg.events["push_robot"].interval_range_s = (2.0, 4.0)
    cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.events["base_com"].params["ranges"] = {
        0: (-0.003, 0.003),
        1: (-0.003, 0.003),
        2: (-0.003, 0.003),
    }
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        "left_foot_collision",
        "right_foot_collision",
    )
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.events["randomize_joint_friction"] = EventTermCfg(
        func=microduck_mdp.randomize_bam_friction,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "scale_range": (0.9, 1.1),
        },
    )

    cfg.viewer.body_name = "trunk_base"
    cfg.viewer.distance = 0.8
    cfg.viewer.elevation = -15.0

    # Two full 2-bar Toosie loops (5.85 s each) per episode.
    cfg.episode_length_s = 12.0

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        # Disable RSI randomization: play always starts the clip clean.
        motion_cmd.pose_range = {}
        motion_cmd.velocity_range = {}
        motion_cmd.joint_position_range = (0.0, 0.0)
        motion_cmd.sampling_mode = "start"
    return cfg


MicroduckTrackingRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_tracking",
    run_name="microduck_tracking",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=30_000,
)
