from mjlab_microduck.tasks.microduck_courier_env_cfg import (
    make_microduck_courier_env_cfg,
)
from mjlab_microduck.tasks.mdp import GroundPickPhaseCommand
from mjlab_microduck.tasks import mdp

import torch


def test_courier_cfg_rewards_delivery_not_cute():
    cfg = make_microduck_courier_env_cfg()
    r = cfg.rewards
    assert "pick_proximity" in r and r["pick_proximity"].weight == 8.0
    assert "grasp_edge" in r and r["grasp_edge"].weight == 12.0
    assert "carry_progress" in r and r["carry_progress"].weight == 20.0
    assert "place_success" in r and r["place_success"].weight == 250.0
    assert "failed_episode" in r and r["failed_episode"].weight == -500.0
    assert r["failed_episode"].func is mdp.courier_failed_episode
    assert "courier_grasp_update" not in r
    assert r["grasp_edge"].func is mdp.courier_update_grasp_edge
    assert r["pick_proximity"].weight + r["carry_progress"].weight + r["place_success"].weight > (
        r["pose_stand_legs"].weight + r["pose_stand_neck"].weight
    )


def test_courier_command_is_phase():
    cfg = make_microduck_courier_env_cfg()
    cmd = cfg.commands["twist"]
    assert cmd.class_type is GroundPickPhaseCommand
    assert cmd.period == 8.0
    assert cmd.randomize_phase is False
    assert cmd.debug_vis is False


def test_courier_has_book_entity():
    cfg = make_microduck_courier_env_cfg()
    assert "robot" in cfg.scene.entities
    assert "book" in cfg.scene.entities
    assert "reader" in cfg.scene.entities
    assert "apartment" in cfg.scene.entities


def test_courier_command_slots_padded():
    cfg = make_microduck_courier_env_cfg()
    for group in ("actor", "critic"):
        assert "head_command" in cfg.observations[group].terms
        assert cfg.observations[group].terms["head_command"].func is mdp.courier_book_command
        assert "body_command" in cfg.observations[group].terms
        assert cfg.observations[group].terms["body_command"].func is mdp.courier_person_command
    assert "book_position" in cfg.observations["critic"].terms
    assert "person_position" in cfg.observations["critic"].terms
    assert "grasp_flag" in cfg.observations["critic"].terms
    assert "book_position" not in cfg.observations["actor"].terms


def test_courier_play_variant_builds():
    cfg = make_microduck_courier_env_cfg(play=True)
    assert "place_success" in cfg.rewards
    assert "reset_courier" in cfg.events
    assert cfg.events["reset_courier"].params["pregrasp_fraction"] == 0.0
    assert cfg.events["reset_courier"].params["near_reader_fraction"] == 0.0
    assert "delivered" not in cfg.terminations
    assert "push_magnitude" not in cfg.curriculum
    assert cfg.events["push_robot"].params["velocity_range"] == {
        "x": (-0.25, 0.25),
        "y": (-0.25, 0.25),
    }


def test_courier_training_uses_reverse_curriculum():
    cfg = make_microduck_courier_env_cfg()
    reset = cfg.events["reset_courier"].params
    assert reset["pregrasp_fraction"] == 0.5
    assert reset["near_reader_fraction"] == 0.25
    assert cfg.terminations["delivered"].func is mdp.courier_is_delivered


def test_ground_pick_command_reset_preserves_courier_curriculum_phase():
    command = mdp.GroundPickPhaseCommand.__new__(mdp.GroundPickPhaseCommand)
    command._env = type(
        "Env",
        (),
        {"_courier_phase_start": torch.tensor([0.0, 0.36, 0.76])},
    )()
    command._gp_phase = torch.ones(3)
    command._randomize_phase = False

    command.reset(torch.tensor([0, 1, 2]))

    assert torch.allclose(command._gp_phase, torch.tensor([0.0, 0.36, 0.76]))


class _Command:
    _gp_phase = torch.tensor([0.9])


class _CommandManager:
    def get_term(self, _name):
        return _Command()


class _BookData:
    root_link_pos_w = torch.tensor([[0.55, 0.0, mdp.COURIER_BOOK_HALF_Z]])


class _Book:
    data = _BookData()


class _Scene(dict):
    pass


class _CourierEnv:
    num_envs = 1
    device = "cpu"
    command_manager = _CommandManager()
    scene = _Scene(book=_Book())


def test_courier_failure_cost_excludes_successful_delivery():
    env = _CourierEnv()
    env.termination_manager = type(
        "TerminationManager", (), {"terminated": torch.tensor([True])}
    )()
    mdp._courier_buffers(env)

    assert mdp.courier_failed_episode(env).item() == 1.0
    env._courier_delivered[:] = True
    assert mdp.courier_failed_episode(env).item() == 0.0


def test_courier_edges_pay_once_not_every_step():
    env = _CourierEnv()
    mdp._courier_buffers(env)
    env._courier_grasped[:] = True
    assert mdp.courier_grasp_edge(env).item() == 1.0
    assert mdp.courier_grasp_edge(env).item() == 0.0

    env._courier_grasped[:] = False
    env._courier_delivered[:] = True
    assert mdp.courier_place_success(env).item() > 0.99
    assert mdp.courier_place_success(env).item() == 0.0


def test_courier_actor_command_slots_keep_61d_layout():
    env = _CourierEnv()
    mdp._courier_buffers(env)
    env._courier_person_xy[:] = torch.tensor([[0.55, 0.0]])
    env._courier_grasped[:] = True

    class _RobotData:
        root_link_pos_w = torch.tensor([[0.0, 0.0, 0.125]])
        root_link_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    class _Robot:
        data = _RobotData()

    class _Terrain:
        env_origins = torch.zeros(1, 3)

    env.scene["robot"] = _Robot()
    env.scene.terrain = _Terrain()
    assert mdp.courier_book_command(env).shape == (1, 4)
    assert mdp.courier_book_command(env)[0, 3].item() == 1.0
    assert mdp.courier_person_command(env).shape == (1, 6)
    assert torch.count_nonzero(mdp.courier_person_command(env)[0, 3:]) == 0
