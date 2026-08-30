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


def test_courier_wide_cfg_distinct_from_v1():
    v1 = make_microduck_courier_env_cfg()
    wide = make_microduck_courier_env_cfg(wide=True)
    assert v1.commands["twist"].class_type is GroundPickPhaseCommand
    assert wide.commands["twist"].class_type is mdp.CourierPhaseCommand
    assert wide.commands["twist"].period == 14.0
    assert wide.episode_length_s == 14.0
    assert v1.episode_length_s == 8.0
    # v1 must stay exactly what the shipped checkpoint was trained on.
    assert v1.rewards["grasp_edge"].params.get("settle_steps", 0) == 0
    assert "randomize_motor_gains" not in v1.events
    assert "randomize_book_inertia" not in v1.events
    assert "book_friction" not in v1.events
    assert v1.observations["actor"].terms["head_command"].noise is None
    assert "book_radius_range" not in v1.events["reset_courier"].params
    # The wide task carries the full v2 stack.
    assert wide.rewards["grasp_edge"].params["settle_steps"] == 5
    assert "randomize_motor_gains" in wide.events
    assert "randomize_book_inertia" in wide.events
    assert "book_friction" in wide.events
    # Pick shaping: v1 keeps the plain absolute Gaussian (bounded by its
    # wall-clock pick window); wide bounds the same Gaussian with an explicit
    # time budget so the held pick phase cannot be farmed, and pays the latch
    # as a salient one-shot edge.
    assert "time_budget_s" not in v1.rewards["pick_proximity"].params
    assert "potential" not in v1.rewards["pick_proximity"].params
    assert v1.rewards["pick_proximity"].weight == 8.0
    assert v1.rewards["grasp_edge"].weight == 12.0
    assert wide.rewards["pick_proximity"].params["time_budget_s"] == 5.0
    assert wide.rewards["pick_proximity"].weight == 8.0
    assert wide.rewards["grasp_edge"].weight == 250.0


def test_courier_wide_spawns_polar_with_curriculum():
    train = make_microduck_courier_env_cfg(wide=True)
    play = make_microduck_courier_env_cfg(play=True, wide=True)
    tp = train.events["reset_courier"].params
    pp = play.events["reset_courier"].params
    # Training starts at the narrow first stage; the curriculum widens it.
    assert tp["book_radius_range"] == (0.14, 0.22)
    assert tp["book_bearing_deg"] == 15.0
    # Play/eval always runs the full target distribution.
    assert pp["book_radius_range"] == (0.12, 0.35)
    assert pp["book_bearing_deg"] == 60.0
    assert pp["person_radius_range"] == (0.40, 0.90)
    assert pp["person_bearing_deg"] == 90.0
    assert tp["book_yaw_random"] and pp["book_yaw_random"]
    assert "spawn_width" in train.curriculum
    assert "spawn_width" not in play.curriculum
    assert train.curriculum["spawn_width"].func is mdp.courier_spawn_curriculum


def test_courier_wide_actor_slots_noisy_critic_clean():
    wide = make_microduck_courier_env_cfg(wide=True)
    for name in ("head_command", "body_command"):
        actor_term = wide.observations["actor"].terms[name]
        assert actor_term.noise is not None
        assert actor_term.delay_max_lag == 2
        critic_term = wide.observations["critic"].terms[name]
        assert critic_term.noise is None


def _make_phase_command(prev, grasped, book_xy, person_xy, period=14.0):
    cmd = mdp.CourierPhaseCommand.__new__(mdp.CourierPhaseCommand)
    cmd._period = period
    cmd._pick_end = mdp.COURIER_PICK_END
    cmd._carry_end = mdp.COURIER_CARRY_END
    cmd._hold_eps = 0.02
    cmd._handoff_dist = 0.18
    cmd._book_name = "book"
    cmd._gp_phase = prev.clone()
    cmd.vel_command_b = torch.zeros(len(prev), 3)

    n = len(prev)
    book_pos = torch.zeros(n, 3)
    book_pos[:, :2] = book_xy

    class _BookData:
        root_link_pos_w = book_pos

    class _Book:
        data = _BookData()

    class _Env:
        _courier_grasped = grasped
        _courier_person_xy = person_xy
        scene = {"book": _Book()}

    cmd._env = _Env()
    return cmd


def test_courier_phase_holds_pick_until_grasped():
    prev = torch.tensor([0.30, 0.30])
    grasped = torch.tensor([False, True])
    far_book = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    person = torch.tensor([[0.5, 0.0], [0.5, 0.0]])
    cmd = _make_phase_command(prev, grasped, far_book, person)
    cmd.compute(dt=1.4)  # 1.4/14 = 0.10 of phase, enough to cross pick_end
    pick_cap = mdp.COURIER_PICK_END - 0.02
    assert torch.isclose(cmd._gp_phase[0], torch.tensor(pick_cap))
    assert cmd._gp_phase[1] > mdp.COURIER_PICK_END


def test_courier_phase_holds_carry_until_handoff():
    prev = torch.tensor([0.70, 0.70])
    grasped = torch.tensor([True, True])
    book = torch.tensor([[1.0, 0.0], [0.55, 0.0]])  # far vs at the reader
    person = torch.tensor([[0.5, 0.0], [0.5, 0.0]])
    cmd = _make_phase_command(prev, grasped, book, person)
    cmd.compute(dt=1.4)
    carry_cap = mdp.COURIER_CARRY_END - 0.02
    assert torch.isclose(cmd._gp_phase[0], torch.tensor(carry_cap))
    assert cmd._gp_phase[1] > mdp.COURIER_CARRY_END


def test_courier_phase_never_pulls_back_and_never_wraps():
    # A delivered episode has released the latch (grasped False) at high
    # phase: gating must not yank it back into the pick segment, and the
    # one-shot task must clamp below 1.0 instead of wrapping to a second lap.
    prev = torch.tensor([0.90, 0.998])
    grasped = torch.tensor([False, False])
    book = torch.tensor([[0.5, 0.0], [0.5, 0.0]])
    person = torch.tensor([[0.5, 0.0], [0.5, 0.0]])
    cmd = _make_phase_command(prev, grasped, book, person)
    cmd.compute(dt=0.5)
    assert cmd._gp_phase[0] > 0.90
    assert cmd._gp_phase[1] < 1.0


class _SettleBookData:
    def __init__(self):
        self.root_link_pos_w = torch.tensor([[0.55, 0.0, 0.05]])
        self.root_link_vel_w = torch.zeros(1, 6)


class _SettleBook:
    def __init__(self):
        self.data = _SettleBookData()


class _SettleRobotData:
    site_pos_w = torch.full((1, 1, 3), 10.0)  # mouth far away: no re-grasp


class _SettleRobot:
    data = _SettleRobotData()


class _SettleTerrain:
    env_origins = torch.zeros(1, 3)


class _SettleScene(dict):
    terrain = _SettleTerrain()


class _PlacePhaseCommand:
    _gp_phase = torch.tensor([0.9])


class _PlaceCommandManager:
    def get_term(self, _name):
        return _PlacePhaseCommand()


class _SettleEnv:
    num_envs = 1
    device = "cpu"

    def __init__(self):
        self.command_manager = _PlaceCommandManager()
        self.scene = _SettleScene(book=_SettleBook(), robot=_SettleRobot())


_SITE_CFG = type("Cfg", (), {"site_ids": [0]})()


def test_courier_settle_checked_delivery_requires_rest():
    env = _SettleEnv()
    mdp._courier_buffers(env)
    env._courier_grasped[:] = True
    book = env.scene["book"]

    # Release step: within the place radius but still at beak height.
    mdp.courier_update_grasp(env, asset_cfg=_SITE_CFG, settle_steps=3)
    assert bool(env._courier_released[0])
    assert not bool(env._courier_grasped[0])
    assert not bool(env._courier_delivered[0])

    # Book comes to rest on the floor: three settled steps latch delivery.
    book.data.root_link_pos_w = torch.tensor([[0.55, 0.0, mdp.COURIER_BOOK_HALF_Z]])
    for _ in range(2):
        mdp.courier_update_grasp(env, asset_cfg=_SITE_CFG, settle_steps=3)
        assert not bool(env._courier_delivered[0])
    mdp.courier_update_grasp(env, asset_cfg=_SITE_CFG, settle_steps=3)
    assert bool(env._courier_delivered[0])


def test_courier_settle_bounced_drop_never_counts():
    env = _SettleEnv()
    mdp._courier_buffers(env)
    env._courier_grasped[:] = True
    book = env.scene["book"]

    mdp.courier_update_grasp(env, asset_cfg=_SITE_CFG, settle_steps=3)
    # The drop bounces out of the place radius and rests there.
    book.data.root_link_pos_w = torch.tensor([[0.75, 0.0, mdp.COURIER_BOOK_HALF_Z]])
    for _ in range(10):
        mdp.courier_update_grasp(env, asset_cfg=_SITE_CFG, settle_steps=3)
    assert not bool(env._courier_delivered[0])
    assert int(env._courier_settle_count[0]) == 0
    # And the released book cannot be silently re-grasped either.
    assert not bool(env._courier_grasped[0])


def test_courier_pick_potential_pays_approach_not_hover():
    class _PickPhaseCommand:
        _gp_phase = torch.tensor([0.2])

    class _PickCommandManager:
        def get_term(self, _name):
            return _PickPhaseCommand()

    class _RobotData:
        def __init__(self):
            self.site_pos_w = torch.tensor([[[0.0, 0.0, 0.10]]])

    class _Robot:
        def __init__(self):
            self.data = _RobotData()

    class _BookData:
        root_link_pos_w = torch.tensor([[0.16, 0.0, 0.006]])

    class _Book:
        data = _BookData()

    class _Env:
        num_envs = 1
        device = "cpu"
        step_dt = 0.02

        def __init__(self):
            self.command_manager = _PickCommandManager()
            self.scene = {"book": _Book(), "robot": _Robot()}

    env = _Env()
    mdp._courier_buffers(env)
    robot = env.scene["robot"]

    # First call after reset: the sentinel pays zero and records the distance.
    first = mdp.courier_pick_proximity(env, asset_cfg=_SITE_CFG, potential=True)
    assert first.item() == 0.0
    # Approaching the book pays the (positive) distance decrease per second.
    robot.data.site_pos_w = torch.tensor([[[0.10, 0.0, 0.05]]])
    approach = mdp.courier_pick_proximity(env, asset_cfg=_SITE_CFG, potential=True)
    assert approach.item() > 1.0
    # Hovering in place pays exactly nothing: the farm is closed.
    hover = mdp.courier_pick_proximity(env, asset_cfg=_SITE_CFG, potential=True)
    assert hover.item() == 0.0
    # Retreating pays nothing either (clamped, like carry_progress).
    robot.data.site_pos_w = torch.tensor([[[0.0, 0.0, 0.10]]])
    retreat = mdp.courier_pick_proximity(env, asset_cfg=_SITE_CFG, potential=True)
    assert retreat.item() == 0.0

    # Time-budgeted absolute mode: the Gaussian pays inside the budget and
    # goes silent after it, so a held-open pick phase cannot be farmed.
    robot.data.site_pos_w = torch.tensor([[[0.13, 0.0, 0.03]]])
    env.episode_length_buf = torch.tensor([0])
    inside = mdp.courier_pick_proximity(env, asset_cfg=_SITE_CFG, time_budget_s=5.0)
    assert inside.item() > 0.3
    env.episode_length_buf = torch.tensor([300])  # 6 s at 50 Hz
    outside = mdp.courier_pick_proximity(env, asset_cfg=_SITE_CFG, time_budget_s=5.0)
    assert outside.item() == 0.0


def test_courier_legacy_settle_zero_keeps_v1_semantics():
    env = _SettleEnv()
    mdp._courier_buffers(env)
    env._courier_grasped[:] = True
    mdp.courier_update_grasp(env, asset_cfg=_SITE_CFG, settle_steps=0)
    assert bool(env._courier_delivered[0])


def test_courier_spawn_curriculum_mutates_live_event_cfg():
    stages = [
        {"step": 0, "book_radius_range": (0.14, 0.22), "book_bearing_deg": 15.0},
        {"step": 100, "book_radius_range": (0.12, 0.35), "book_bearing_deg": 60.0},
    ]

    class _EventCfg:
        params = {"book_radius_range": (0.14, 0.22), "book_bearing_deg": 15.0}

    class _EventManager:
        cfg = _EventCfg()

        def get_term_cfg(self, _name):
            return self.cfg

    class _Env:
        device = "cpu"
        common_step_counter = 0
        event_manager = _EventManager()

    env = _Env()
    out = mdp.courier_spawn_curriculum(env, None, "reset_courier", stages)
    assert env.event_manager.cfg.params["book_bearing_deg"] == 15.0
    assert abs(float(out) - 0.22) < 1e-6
    env.common_step_counter = 101
    out = mdp.courier_spawn_curriculum(env, None, "reset_courier", stages)
    assert env.event_manager.cfg.params["book_bearing_deg"] == 60.0
    assert env.event_manager.cfg.params["book_radius_range"] == (0.12, 0.35)
    assert abs(float(out) - 0.35) < 1e-6


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
