from mjlab_microduck.tasks.microduck_courier_env_cfg import (
    make_microduck_courier_env_cfg,
)
from mjlab_microduck.tasks.mdp import GroundPickPhaseCommand


def test_courier_cfg_rewards_delivery_not_cute():
    cfg = make_microduck_courier_env_cfg()
    r = cfg.rewards
    assert "pick_proximity" in r and r["pick_proximity"].weight == 4.0
    assert "grasp_edge" in r and r["grasp_edge"].weight == 6.0
    assert "carry_progress" in r and r["carry_progress"].weight == 8.0
    assert "place_success" in r and r["place_success"].weight == 5.0
    assert r["courier_grasp_update"].weight == 0.0
    assert r["pick_proximity"].weight + r["carry_progress"].weight + r["place_success"].weight > (
        r["pose_stand_legs"].weight + r["pose_stand_neck"].weight
    )


def test_courier_command_is_phase():
    cfg = make_microduck_courier_env_cfg()
    cmd = cfg.commands["twist"]
    assert cmd.class_type is GroundPickPhaseCommand
    assert cmd.period == 8.0
    assert cmd.randomize_phase is False


def test_courier_has_book_entity():
    cfg = make_microduck_courier_env_cfg()
    assert "robot" in cfg.scene.entities
    assert "book" in cfg.scene.entities


def test_courier_command_slots_padded():
    cfg = make_microduck_courier_env_cfg()
    for group in ("actor", "critic"):
        assert "head_command" in cfg.observations[group].terms
        assert cfg.observations[group].terms["head_command"].params["dim"] == 4
        assert "body_command" in cfg.observations[group].terms
        assert cfg.observations[group].terms["body_command"].params["dim"] == 6
    assert "book_position" in cfg.observations["critic"].terms
    assert "person_position" in cfg.observations["critic"].terms
    assert "grasp_flag" in cfg.observations["critic"].terms
    assert "book_position" not in cfg.observations["actor"].terms


def test_courier_play_variant_builds():
    cfg = make_microduck_courier_env_cfg(play=True)
    assert "place_success" in cfg.rewards
    assert "reset_courier" in cfg.events
