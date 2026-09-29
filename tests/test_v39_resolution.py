"""V3.9 resolution policy, runtime forwarding, and production frontend parity."""
from pathlib import Path
import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from ComfyUI_H3_Continuum_Join.v3 import resolution as r
from ComfyUI_H3_Continuum_Join.v3.driving_nodes import (
    H3ContinuumSamplerV37, H3ContinuumSamplerV38, H3ContinuumSamplerV39,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    r.H3_ASPECT_SQUARE: [(544, 544), (768, 768), (768, 768)],
    "4:3": [(640, 480), (896, 672), (1024, 768)],
    "3:2": [(672, 448), (960, 640), (1152, 768)],
    "16:10": [(704, 448), (992, 608), (1216, 768)],
    r.H3_ASPECT_LANDSCAPE: [(736, 416), (1024, 576), (1344, 768)],
    "21:9": [(832, 352), (1184, 512), (1344, 576)],
    "3:4": [(480, 640), (672, 896), (768, 1024)],
    "2:3": [(448, 672), (640, 960), (768, 1152)],
    "10:16": [(448, 704), (608, 992), (768, 1216)],
    r.H3_ASPECT_PORTRAIT: [(416, 736), (576, 1024), (768, 1344)],
}
PRESETS = (r.H3_PRESET_DRAFT, r.H3_PRESET_BALANCED, r.H3_PRESET_NATIVE)


def image(width, height):
    return SimpleNamespace(shape=(1, height, width, 3))


def resolve(**kwargs):
    return r.resolve_h3_v39_size_source(size_source=r.H3_V39_MP_MODE, **kwargs)


@pytest.mark.parametrize("aspect", EXPECTED)
@pytest.mark.parametrize("index,preset", enumerate(PRESETS))
def test_expected_dimensions_and_runtime_forwarding(monkeypatch, aspect, index, preset):
    expected = EXPECTED[aspect][index]
    plan = resolve(aspect=aspect, preset=preset, first_frame=image(300, 800))
    assert (plan.width, plan.height) == expected
    assert plan.actual_mp == expected[0] * expected[1] / 1_000_000
    captured = {}

    def record(_self, **kwargs):
        captured.update(kwargs)
        return "runtime reached"

    monkeypatch.setattr(H3ContinuumSamplerV37, "run", record)
    assert H3ContinuumSamplerV39().run(
        size_source=r.H3_V39_MP_MODE, aspect=aspect, preset=preset,
        first_frame=image(300, 800),
    ) == "runtime reached"
    assert (captured["width"], captured["height"]) == expected
    assert not {"aspect", "preset", "custom_mp", "size_source"} & captured.keys()


@pytest.mark.parametrize("aspect", EXPECTED)
@pytest.mark.parametrize("mp", [0.10, 0.37, 2.50, 16.0])
def test_custom_uses_existing_math_and_bounds(aspect, mp):
    plan = resolve(aspect=aspect, preset=r.H3_PRESET_CUSTOM, custom_mp=mp)
    ratio = r.H3_V39_ASPECT_RATIOS[aspect]
    # Compare against the unchanged legacy Auto path with the same input ratio.
    legacy = r.resolve_h3_resolution(
        aspect=r.H3_ASPECT_AUTO, preset=r.H3_PRESET_CUSTOM, custom_mp=mp,
        first_frame=image(round(ratio * 9000), 9000),
    )
    assert (plan.width, plan.height, plan.actual_mp) == (legacy.width, legacy.height, legacy.actual_mp)
    assert plan.width % 32 == plan.height % 32 == 0


@pytest.mark.parametrize("mp", [0.09, 16.01, True])
def test_invalid_custom_mp_retains_validation(mp):
    with pytest.raises((ValueError, TypeError)):
        resolve(aspect="3:2", preset=r.H3_PRESET_CUSTOM, custom_mp=mp)


@pytest.mark.parametrize("preset", (*PRESETS, r.H3_PRESET_CUSTOM))
@pytest.mark.parametrize("wide,tall", [
    ("4:3", "3:4"), ("3:2", "2:3"), ("16:10", "10:16"),
    (r.H3_ASPECT_LANDSCAPE, r.H3_ASPECT_PORTRAIT),
])
def test_reciprocals(wide, tall, preset):
    a, b = [resolve(aspect=v, preset=preset, custom_mp=0.37) for v in (wide, tall)]
    assert (a.width, a.height) == (b.height, b.width)


@pytest.mark.parametrize("preset", (*PRESETS, r.H3_PRESET_CUSTOM))
def test_numeric_9_21_without_public_option(preset):
    assert "9:21" not in r.H3_V39_ASPECT_OPTIONS
    wide = resolve(aspect="21:9", preset=preset)
    tall = r.resolve_h3_numeric_resolution(aspect_ratio=9 / 21, aspect_source="test",
                                           preset=preset)
    assert (wide.width, wide.height) == (tall.height, tall.width)


@pytest.mark.parametrize("width,height", [(1200, 800), (800, 1200), (900, 900)])
@pytest.mark.parametrize("preset", PRESETS)
def test_auto_uses_image_ratio_not_native_dimensions(width, height, preset):
    frame = image(width, height)
    actual = resolve(aspect=r.H3_ASPECT_AUTO, preset=preset, first_frame=frame)
    assert actual == r.resolve_h3_size_source(size_source=r.H3_SIZE_SOURCE_FIRST_IMAGE,
                                             preset=preset, first_frame=frame)


@pytest.mark.parametrize("source", r.H3_SIZE_SOURCE_OPTIONS)
@pytest.mark.parametrize("frame", [None, image(1200, 800)])
def test_all_historical_paths_unchanged(source, frame):
    args = dict(size_source=source, aspect=r.H3_ASPECT_AUTO, preset=r.H3_PRESET_BALANCED,
                first_frame=frame, width=704, height=1024)
    assert r.resolve_h3_v39_size_source(**args) == r.resolve_h3_size_source(**args)


def test_missing_auto_retains_both_fallbacks():
    args = dict(aspect=r.H3_ASPECT_AUTO, preset=r.H3_PRESET_DRAFT, width=960, height=640)
    new = resolve(**args)
    old = r.resolve_h3_v39_size_source(size_source=r.H3_SIZE_SOURCE_LEGACY, **args)
    assert (new.width, new.height) == (960, 640)
    assert (old.width, old.height) == (736, 416)
    assert new.warnings and not old.warnings


@pytest.mark.parametrize("width,height", [(32, 16384), (960, 640), (703, 1024), (32, 16416), (32.5, 32)])
def test_manual_validation_unchanged(width, height):
    args = dict(size_source=r.H3_SIZE_SOURCE_MANUAL, width=width, height=height)
    try:
        expected = r.resolve_h3_size_source(**args)
    except (ValueError, TypeError) as exc:
        with pytest.raises(type(exc), match=str(exc)):
            r.resolve_h3_v39_size_source(**args)
    else:
        assert r.resolve_h3_v39_size_source(**args) == expected


def test_v38_schema_fallback_and_widget_order_unchanged():
    old, new = H3ContinuumSamplerV38.INPUT_TYPES(), H3ContinuumSamplerV39.INPUT_TYPES()
    assert list(old["required"]) == list(new["required"])
    assert old["required"]["aspect"][0] == r.H3_ASPECT_OPTIONS
    assert old["required"]["size_source"][0] == r.H3_SIZE_SOURCE_OPTIONS
    assert new["required"]["aspect"][0] == (r.H3_ASPECT_AUTO, *EXPECTED)
    assert new["required"]["size_source"][0] == (*r.H3_SIZE_SOURCE_OPTIONS, r.H3_V39_MP_MODE)
    assert H3ContinuumSamplerV38._resolve_resolution is r.resolve_h3_size_source
    for ratio in set(EXPECTED) - set(r.H3_ASPECT_OPTIONS):
        assert r.resolve_h3_aspect(ratio) == (16 / 9, "Landscape 16:9 fallback")


def test_frontend_backend_parity_and_serialization():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required; the fixture also runs independently on Linux")
    result = subprocess.run([node, str(ROOT / "tests/frontend_resolution_v39.cjs")],
                            check=True, capture_output=True, text=True)
    for row in json.loads(result.stdout):
        args = row["inputs"]
        if row["geometry"]:
            args["first_frame"] = image(*row["geometry"])
        plan = r.resolve_h3_v39_size_source(**args)
        preview = row["preview"]
        if preview["pending"]:
            assert row["geometry"] is None
            assert "width" not in preview and "actualMP" not in preview
        else:
            assert (preview["width"], preview["height"]) == (plan.width, plan.height)
            assert preview["actualMP"] == plan.actual_mp
