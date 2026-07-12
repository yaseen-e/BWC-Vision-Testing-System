from pathlib import Path

import cv2

from src.vision.display_layouts import ContextNode, apply_navigation_command
from src.vision.vision_engine import _build_display_mask, _find_display_contour


def test_display_detector_recovers_real_samples():
    sample_dir = Path(__file__).resolve().parents[1] / "src" / "vision" / "capture_samples"
    sample_paths = sorted(sample_dir.glob("fail*.jpg"))
    assert sample_paths, "expected sample captures to be present"

    hits = 0
    for sample_path in sample_paths:
        frame = cv2.imread(str(sample_path))
        assert frame is not None, f"failed to load {sample_path}"
        mask = _build_display_mask(frame)
        contour = _find_display_contour(mask)
        if contour is not None:
            hits += 1

    assert hits >= 2, f"expected at least 2 detected displays, got {hits}"


def test_apply_navigation_command_supports_starred_repeat_tokens():
    child = ContextNode(key="child", label="child", route_here=(("DOWN*", "SELECT"),))
    root = ContextNode(key="root", label="root", children=(child,))

    current_menu, transition_buffer, sequence_broken = apply_navigation_command(root, root, (), "SELECT")
    assert current_menu is child
    assert transition_buffer == ()
    assert sequence_broken is False

    current_menu, transition_buffer, sequence_broken = apply_navigation_command(root, root, (), "DOWN")
    assert current_menu is root
    assert transition_buffer == ("DOWN",)
    assert sequence_broken is False

    current_menu, transition_buffer, sequence_broken = apply_navigation_command(root, root, ("DOWN",), "DOWN")
    assert current_menu is root
    assert transition_buffer == ("DOWN", "DOWN")
    assert sequence_broken is False

    current_menu, transition_buffer, sequence_broken = apply_navigation_command(root, root, ("DOWN", "DOWN"), "SELECT")
    assert current_menu is child
    assert transition_buffer == ()
    assert sequence_broken is False


def test_apply_navigation_command_supports_optional_leading_downs_in_route():
    confirmation = ContextNode(key="confirmation", label="confirmation")
    child = ContextNode(
        key="child",
        label="child",
        route_here=(("DOWN*", "SELECT", "DOWN", "SELECT") + ("DOWN",) * 7 + ("SELECT", "SELECT"),),
        children=(confirmation,),
    )
    root = ContextNode(key="root", label="root", children=(child,))

    current_menu, transition_buffer, sequence_broken = apply_navigation_command(root, root, (), "DOWN")
    assert current_menu is root
    assert transition_buffer == ("DOWN",)
    assert sequence_broken is False

    current_menu, transition_buffer, sequence_broken = apply_navigation_command(root, root, ("DOWN",), "SELECT")
    assert current_menu is root
    assert transition_buffer == ("DOWN", "SELECT")
    assert sequence_broken is False

    current_menu, transition_buffer, sequence_broken = apply_navigation_command(root, root, ("DOWN", "SELECT"), "DOWN")
    assert current_menu is root
    assert transition_buffer == ("DOWN", "SELECT", "DOWN")
    assert sequence_broken is False

    current_menu, transition_buffer, sequence_broken = apply_navigation_command(root, root, ("DOWN", "SELECT", "DOWN"), "SELECT")
    assert current_menu is root
    assert transition_buffer == ("DOWN", "SELECT", "DOWN", "SELECT")
    assert sequence_broken is False

    stream = ("DOWN", "SELECT", "DOWN", "SELECT") + ("DOWN",) * 7 + ("SELECT", "SELECT")
    current_menu = root
    transition_buffer = ()
    for token in stream:
        current_menu, transition_buffer, sequence_broken = apply_navigation_command(root, current_menu, transition_buffer, token)
        assert sequence_broken is False

    assert current_menu is child
    assert transition_buffer == ()
