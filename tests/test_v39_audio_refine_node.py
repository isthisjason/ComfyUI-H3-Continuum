"""Exercise the public adapter through the existing group engine, without AudioRefine."""
import copy

import pytest
import torch

from ComfyUI_H3_Continuum_Join.v3 import audio_refine_nodes as node
from ComfyUI_H3_Continuum_Join.v3.targeted_second_pass import run_targeted_second_pass_groups
from .test_v38_audio_only_refine import _fixture


def _inputs(videos, audios, plan, denoise):
    return dict(model=[object()], clip=[object()], video_latents=videos,
                audio_latents=audios, assembly_plan=[plan], seed=[19], steps=[3],
                sampler_name=["euler"], scheduler=["normal"], audio_denoise=[denoise])


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("denoise", [0.0, 0.3])
def test_groups_geometry_and_passthrough(monkeypatch, count, denoise):
    groups, videos, audios, plan = _fixture()
    videos, audios = videos[:count], audios[:count]
    plan["second_pass_contract"]["physical_groups"] = groups[:count]
    # Final video is already x2; sampling must never substitute its source geometry.
    for video in videos:
        video["samples"] = video["samples"].repeat_interleave(2, -1).repeat_interleave(2, -2)
    before = copy.deepcopy(plan)
    calls = []

    def sample(**kwargs):
        calls.append(kwargs)
        return {"video": kwargs["latent"]["video"] + 999,
                "audio": kwargs["latent"]["audio"] + len(calls)}

    def run(**kwargs):
        return run_targeted_second_pass_groups(
            **kwargs, encode_prompt_fn=lambda _clip, prompt, **_kw: [prompt],
            latent_builder=lambda video, audio: {"video": video, "audio": audio},
            sample_fn=sample, stream_extractor=lambda value: (value["video"], value["audio"]),
            clone_model_fn=lambda model, **_kw: model, enable_preview=False)

    monkeypatch.setattr(node, "run_targeted_second_pass_groups", run)
    monkeypatch.setattr(node, "_audio_schedule", lambda *_args: (lambda name: name, torch.tensor([0.3, 0.2, 0.1, 0.])))
    out_v, out_a, updated, status = node.H3ContinuumAudioRefineV39().refine(**_inputs(videos, audios, plan, denoise))
    assert plan == before
    assert len(out_v) == len(out_a) == count
    assert all(out is original for out, original in zip(out_v, videos))
    assert updated["width"] == updated["height"] == 256
    assert [g["logical_chunks"] for g in updated["second_pass_contract"]["physical_groups"]] == [g["logical_chunks"] for g in groups[:count]]
    if denoise:
        assert [call["conditioning"] for call in calls] == [[f"group {i + 1}"] for i in range(count)]
        for i, (out, original) in enumerate(zip(out_a, audios)):
            assert torch.equal(out["samples"], original["samples"] + i + 1)
    else:
        assert not calls
        assert all(out is original for out, original in zip(out_a, audios))
    assert "video=bit-exact preserved" in status


def test_mismatched_groups_fail_even_for_noop():
    _, videos, audios, plan = _fixture()
    with pytest.raises(ValueError, match="audio latent group count"):
        node.H3ContinuumAudioRefineV39().refine(**_inputs(videos, audios[:1], plan, 0.0))


def test_schedule_matches_core_tail(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    core = ModuleType("comfy")
    sampling = ModuleType("comfy.samplers")
    calls = []
    def calculate(model_sampling, scheduler, steps):
        calls.append((model_sampling, scheduler, steps))
        return torch.arange(steps + 1).flip(0)
    sampling.calculate_sigmas = calculate
    sampling.sampler_object = object()
    core.samplers = sampling
    monkeypatch.setitem(sys.modules, "comfy", core)
    monkeypatch.setitem(sys.modules, "comfy.samplers", sampling)
    model = SimpleNamespace(get_model_object=lambda name: name)
    _, sigmas = node._audio_schedule(model, 3, "normal", 0.3)
    assert calls == [("model_sampling", "normal", 10)]
    assert sigmas.tolist() == [3, 2, 1, 0]


@pytest.mark.parametrize("with_context", [False, True])
def test_optional_context_and_vae_forwarded(monkeypatch, with_context):
    _, videos, audios, plan = _fixture()
    context, vae = object(), object()
    captured = {}
    def run(**kwargs):
        captured.update(kwargs)
        return videos, audios, plan, "context preparation delegated"
    monkeypatch.setattr(node, "run_targeted_second_pass_groups", run)
    monkeypatch.setattr(node, "_audio_schedule", lambda *_args: (lambda name: name, torch.tensor([0.3, 0.])))
    inputs = _inputs(videos, audios, plan, 0.3)
    if with_context:
        inputs.update(refine_context=[context], video_vae=[vae])
    node.H3ContinuumAudioRefineV39().refine(**inputs)
    assert captured["model"] is inputs["model"][0]
    assert captured["refine_context"] is (context if with_context else None)
    assert captured["video_vae"] is (vae if with_context else None)


def test_native_core_sampler_runtime_in_isolated_process():
    """Keep native Comfy imports isolated from fixtures that replace its modules."""
    import subprocess
    import sys
    from pathlib import Path

    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("audio_refine_runtime_smoke.py"))],
        capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"solver_steps": 6' in result.stdout
    assert '"video_preserved": true' in result.stdout


def test_recorded_second_pass_geometry_is_required_for_context_reuse():
    from .test_v35_refine_context import _group
    from ComfyUI_H3_Continuum_Join.v3 import refine_context as rc

    context = rc.make_refine_context([_group()], source_width=48, source_height=32,
                                    conditioning_mode="fl2va")
    group = dict(group_id=0, logical_chunks=[1], physical_frames=124,
                 prompt_policy="single", physical_prompt="prompt", trim_prefix_frames=0,
                 source_batch=1, latent_channels=24, source_latent_t=37,
                 source_latent_h=2, source_latent_w=3, source_width=48, source_height=32)
    plan = dict(width=96, height=64, second_pass_contract=dict(
        target_width=96, target_height=64, physical_groups=[group]))
    assert rc.validate_refine_context(context, plan) is context
    # An unrelated resize or a falsely recorded source must still fail clearly.
    for invalid in (dict(plan, width=128),
                    dict(plan, second_pass_contract=dict(plan["second_pass_contract"],
                         physical_groups=[dict(group, source_width=64)]))):
        with pytest.raises(rc.RefineContextError, match="dimensions differ"):
            rc.validate_refine_context(context, invalid)
