"""Microduck book-courier task: pick, carry, place a duck-scale paperback.

The duck starts standing with *The Courage to Be Disliked* on the floor in
front of it and a seated reader a few body-lengths away. One 8 s phase
cycle:

  phase ∈ [0, 0.35)   pick   mouth to the book
  phase ∈ [0.35, 0.75) carry  walk the grasped book toward the reader
  phase ∈ [0.75, 1)    place  set it down at their feet

Grasp is kinematic (mouth-proximity latch), matching ground-pick's payload
trick: the real robot's beak is a linkage, not a 15th actuator. The actor
stays on the unified 61-D obs contract so a trained courier ONNX can
hot-swap with walk / ground-pick. The existing command slots carry the book
position plus grasp flag and the reader position, so the actor can observe the
delivery task without changing that contract.

Reward the delivery, not looking cute: pick proximity, a one-shot grasp
edge, potential-based carry progress, and a place Gaussian. Standing /
upright / smoothness are regularizers, not the objective.
"""

import math
from copy import deepcopy

ENABLE_SYMMETRY = False

ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_KP_RANDOMIZATION              = False
ENABLE_KD_RANDOMIZATION              = False
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

COM_RANDOMIZATION_RANGE             = 0.003
HEAD_COM_RANDOMIZATION_RANGE        = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
VELOCITY_PUSH_INTERVAL_S            = (3.0, 6.0)
VELOCITY_PUSH_RANGE                 = (-0.25, 0.25)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

EPISODE_LENGTH_S = 8.0
COURIER_PERIOD = 8.0
STAND_Z = 0.115

BOOK_OFFSET = (0.16, 0.0)
PERSON_OFFSET = (0.55, 0.0)
BOOK_NOISE_XY = 0.02
PERSON_NOISE_XY = 0.03

_LEG_JOINTS  = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_APARTMENT_PROPS_CFG,
    MICRODUCK_BOOK_CFG,
    MICRODUCK_READER_CFG,
    MICRODUCK_STANDUP_ROBOT_CFG,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_courier_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the book-courier environment (flat terrain only)."""
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    cfg = make_velocity_env_cfg()
    cfg.scene.entities = {
        "robot": MICRODUCK_STANDUP_ROBOT_CFG,
        "book": MICRODUCK_BOOK_CFG,
        "reader": MICRODUCK_READER_CFG,
        "apartment": MICRODUCK_APARTMENT_PROPS_CFG,
    }
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"
    cfg.viewer.origin_type = cfg.viewer.OriginType.WORLD
    cfg.viewer.lookat = (0.28, 0.0, 0.08)
    cfg.viewer.distance = 0.70
    cfg.viewer.elevation = -18.0
    cfg.viewer.azimuth = 120.0
    cfg.viewer.max_extra_envs = 0
    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.sim.nconmax = 50

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
        "soft_landing",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    cfg.rewards["pick_proximity"] = RewardTermCfg(
        func=microduck_mdp.courier_pick_proximity,
        weight=8.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=["mouth_tip"]),
            "book_name": "book",
            "std": 0.06,
        },
    )
    cfg.rewards["grasp_edge"] = RewardTermCfg(
        # This non-zero term also updates the kinematic grasp before all
        # carry/place rewards. RewardManager skips weight-0 terms entirely.
        func=microduck_mdp.courier_update_grasp_edge,
        weight=12.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=["mouth_tip"]),
            "book_name": "book",
        },
    )
    cfg.rewards["carry_progress"] = RewardTermCfg(
        func=microduck_mdp.courier_carry_progress,
        weight=20.0,
        params={
            "book_name": "book",
            "asset_cfg": SceneEntityCfg("robot"),
            "max_tilt_deg": 45.0,
        },
    )
    cfg.rewards["place_success"] = RewardTermCfg(
        func=microduck_mdp.courier_place_success,
        weight=250.0,
        params={"book_name": "book", "std": 0.08},
    )
    cfg.rewards["failed_episode"] = RewardTermCfg(
        func=microduck_mdp.courier_failed_episode,
        weight=-500.0,
    )

    cfg.rewards["feet_grounded"] = RewardTermCfg(
        func=microduck_mdp.feet_grounded_reward,
        weight=0.25,
        params={"sensor_name": feet_ground_cfg.name},
    )
    cfg.rewards["pose_stand_legs"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=0.15,
        params={"std": 0.5, "joint_indices": _LEG_JOINTS, "target_overrides": None},
    )
    cfg.rewards["pose_stand_neck"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=0.08,
        params={"std": 0.35, "joint_indices": _NECK_JOINTS, "target_overrides": None},
    )
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 0.25
    cfg.rewards["upright"].params["std"] = math.sqrt(0.05)
    cfg.rewards["height_stand"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=0.1,
        params={
            "std": 0.04,
            "target_height": STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["action_rate_l2"].weight = -0.1
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    del cfg.observations["actor"].terms["base_lin_vel"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # Preserve the unified 61-D actor contract while making the task observable:
    # head-command slot = book xyz in base frame + grasp flag (4D)
    # body-command slot = reader xyz in base frame + zero padding (6D)
    # A deployed courier must fill the same slots from perception.
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.courier_book_command,
            params={"asset_name": "book"},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.courier_person_command,
        )

    cfg.observations["critic"].terms["book_position"] = ObservationTermCfg(
        func=microduck_mdp.courier_book_pos_in_base, params={"asset_name": "book"},
    )
    cfg.observations["critic"].terms["person_position"] = ObservationTermCfg(
        func=microduck_mdp.courier_person_pos_in_base,
    )
    cfg.observations["critic"].terms["grasp_flag"] = ObservationTermCfg(
        func=microduck_mdp.courier_grasp_flag,
    )

    command: object = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.debug_vis = False
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(
        **{
            **vars(command),
            "class_type": microduck_mdp.GroundPickPhaseCommand,
            "period": COURIER_PERIOD,
            "randomize_phase": False,
        }
    )

    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob": 0.0,
            "face_up_prob": 0.0,
            "sitting_prob": 0.0,
            "standing_prob": 1.0,
            "sitting_tilt_max": math.radians(5),
            "standing_z_min": 0.11,
            "standing_z_max": 0.12,
        },
    )
    cfg.events["reset_courier"] = EventTermCfg(
        func=microduck_mdp.reset_courier_props,
        mode="reset",
        params={
            "book_offset": BOOK_OFFSET,
            "person_offset": PERSON_OFFSET,
            "book_noise_xy": BOOK_NOISE_XY,
            "person_noise_xy": PERSON_NOISE_XY,
            "asset_name": "book",
            "reader_name": "reader",
            "pregrasp_fraction": 0.0 if play else 0.5,
            "near_reader_fraction": 0.0 if play else 0.25,
        },
    )

    if not play:
        cfg.terminations["delivered"] = TerminationTermCfg(
            func=microduck_mdp.courier_is_delivered,
            time_out=False,
        )

    if ENABLE_VELOCITY_PUSHES:
        interval = (2.0, 4.0) if play else VELOCITY_PUSH_INTERVAL_S
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=interval,
            params={
                "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )
    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )
    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )
    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.1},
                {"step": 500 * 24, "weight": -0.2},
                {"step": 1000 * 24, "weight": -0.5},
                {"step": 1500 * 24, "weight": -1.0},
            ],
        },
    )
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0, "range": 0.003},
                    {"step": 1000 * 24, "range": 0.01},
                    {"step": 1500 * 24, "range": 0.015},
                ],
            },
        )
    if ENABLE_VELOCITY_PUSHES:
        cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
            func=microduck_mdp.push_curriculum,
            params={
                "event_name": "push_robot",
                "push_stages": [
                    {"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
                    {"step": 750 * 24, "velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}},
                    {"step": 1500 * 24, "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}},
                ],
            },
        )
    if play:
        # Evaluation/recording should exercise the configured perturbation.
        # Leaving the training curricula active would immediately rewrite the
        # play-mode push range to the step-0 value of exactly zero.
        cfg.curriculum.pop("action_rate_weight", None)
        cfg.curriculum.pop("com_range", None)
        cfg.curriculum.pop("push_magnitude", None)
    return cfg


MicroduckCourierRlCfg = RslRlOnPolicyRunnerCfg(
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
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="book_courier",
    run_name="book_courier",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=10_000,
)
