from pathlib import Path

from mjlab_microduck.tasks.microduck_tracking_env_cfg import (
    DEFAULT_MOTION_FILE,
    TRACKED_BODY_NAMES,
    make_microduck_tracking_env_cfg,
)
from mjlab_microduck.tasks import mdp


def test_tracking_cfg_duck_scale_and_bam_invariants():
    cfg = make_microduck_tracking_env_cfg()
    # BAM invariants from AGENTS.md for standalone env cfgs.
    assert "expand_bam_friction_fields" in cfg.events
    assert cfg.events["randomize_joint_friction"].func is mdp.randomize_bam_friction
    assert cfg.terminations["nan_state"].func is mdp.robot_state_is_nan
    # Duck-scale tracking stds and thresholds, not G1-scale.
    assert cfg.rewards["motion_global_root_pos"].params["std"] == 0.05
    assert cfg.rewards["motion_body_pos"].params["std"] == 0.05
    assert cfg.terminations["anchor_pos"].params["threshold"] == 0.06
    assert cfg.terminations["ee_body_pos"].params["body_names"] == (
        "ankle_left",
        "ankle_right",
    )
    # The motion command tracks trunk + feet + head and anchors on the trunk.
    motion = cfg.commands["motion"]
    assert motion.anchor_body_name == "trunk_base"
    assert motion.body_names == TRACKED_BODY_NAMES
    assert motion.motion_file == DEFAULT_MOTION_FILE
    # Duck-scale pushes (G1 default is +-0.5 m/s).
    assert cfg.events["push_robot"].params["velocity_range"]["x"] == (-0.25, 0.25)


def test_tracking_play_variant_is_clean_start():
    play = make_microduck_tracking_env_cfg(play=True)
    assert "push_robot" not in play.events
    assert play.commands["motion"].sampling_mode == "start"
    assert play.commands["motion"].pose_range == {}
    assert play.observations["actor"].enable_corruption is False


def test_tracking_reference_motion_is_committed_and_loops():
    import numpy as np

    path = Path(DEFAULT_MOTION_FILE)
    assert path.is_file(), (
        "Reference motion missing; regenerate with "
        "scripts/author_toosie_reference.py + scripts/motion_csv_to_npz.py"
    )
    data = np.load(path)
    assert int(data["fps"][0]) == 50
    joint_pos = data["joint_pos"]
    assert joint_pos.shape[1] == 14
    # Seamless loop: the clip must end where it starts (pose and world xy),
    # because the tracking command restarts the clip when it runs out.
    assert np.abs(joint_pos[0] - joint_pos[-1]).max() < 0.06
    body_pos = data["body_pos_w"]
    assert np.abs(body_pos[0, :, :2] - body_pos[-1, :, :2]).max() < 0.01
    assert np.isfinite(joint_pos).all() and np.isfinite(body_pos).all()


def test_spedup_tracking_task_registered_with_own_motion():
    from mjlab_microduck.tasks.microduck_tracking_env_cfg import SPEDUP_MOTION_FILE
    import numpy as np

    cfg = make_microduck_tracking_env_cfg(motion_file=SPEDUP_MOTION_FILE)
    assert cfg.commands["motion"].motion_file == SPEDUP_MOTION_FILE
    assert Path(SPEDUP_MOTION_FILE).is_file()
    data = np.load(SPEDUP_MOTION_FILE)
    # 4 bars of 108.7 BPM at 50 fps (compiled by ducktok).
    assert data["joint_pos"].shape[0] in (441, 442)
    assert np.isfinite(data["joint_pos"]).all()
