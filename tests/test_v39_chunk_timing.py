"""Requested custom timing through native sequencing and downstream refinement."""
from copy import deepcopy

import pytest
import torch

from ComfyUI_H3_Continuum_Join.v3.chunk_timing import resolve_chunk_timing, timing_from_contract
from ComfyUI_H3_Continuum_Join.v2.prompts import make_prompt_plan
from ComfyUI_H3_Continuum_Join.v2 import sequence
from ComfyUI_H3_Continuum_Join.v3.plan import prepare_physical_decode_entries, validate_assembly_plan
from ComfyUI_H3_Continuum_Join.v3.second_pass import run_second_pass_groups
from ComfyUI_H3_Continuum_Join.v3 import audio_refine_nodes
from ComfyUI_H3_Continuum_Join.v3.targeted_second_pass import run_targeted_second_pass_groups
from ComfyUI_H3_Continuum_Join.temporal import align_frame_count_up, make_extension_shape
from . import test_v38_review_execution_cap as cap
from .test_timeline_video import _Video, _VAE


def test_resolution_and_fractional_rounding():
    legacy = resolve_chunk_timing(3, 10)
    custom = resolve_chunk_timing(3, 10, "Custom Sequence", "10, 8, 10")
    assert legacy.logical_chunk_seconds == (10., 10., 10.)
    assert custom.cumulative_seconds == (0., 10., 18., 28.)
    assert custom.requested_frame_boundaries == (0, 240, 432, 672)
    assert custom.total_seconds == 28 and custom.target_frames == 672
    fractions = resolve_chunk_timing(3, 5, "Custom Sequence", "4.1,4.2,4.3")
    assert fractions.requested_frame_boundaries == tuple(round(s * 24) for s in fractions.cumulative_seconds)


@pytest.mark.parametrize("value", ["10,8", "10,8,10,5", "10,3,10", "10,31,10", "10,nan,10", "10,inf,10", "10,x,10", "10,,10"])
def test_invalid_custom_durations(value):
    with pytest.raises(ValueError, match="Custom|custom"):
        resolve_chunk_timing(3, 10, "Custom Sequence", value)


def test_absolute_timeline_and_chunk_prompts():
    timing = resolve_chunk_timing(3, 10, "Custom Sequence", "10,8,10")
    plan = make_prompt_plan(mode="Timeline", script="[0-10s]\none\n[10-18s]\ntwo\n[18-28s]\nthree", chunks=3, chunk_seconds=10, chunk_timing=timing)
    assert plan["prompts"] == ["one", "two", "three"]
    assert timing_from_contract(plan["chunk_timing"]) == timing
    legacy = make_prompt_plan(mode="Fixed", script="shot", chunks=3, chunk_seconds=10)
    assert "chunk_timing" not in legacy


def generated(monkeypatch, durations, *, limit=None, session=None):
    first = torch.zeros(1, 64, 96, 3)
    last = torch.ones_like(first)
    runtime = cap._install_fake_runtime(monkeypatch, first_frame=first, last_frame=last)
    timing = resolve_chunk_timing(3, 10, "Custom Sequence", durations)
    monkeypatch.setattr(cap, "make_prompt_plan", lambda **kw: make_prompt_plan(**kw, chunk_timing=timing))
    outputs = cap._run_sequence(runtime, chunks=3, chunk_seconds=10, first_frame=first, last_frame=last,
                                capture_refine_context=True, limit=limit, session=session)
    return runtime, timing, outputs


@pytest.mark.parametrize("durations,merged,group_count", [("10,8,10", False, 3), ("10,5,5", True, 2)])
def test_native_groups_refinement_and_assembly(monkeypatch, durations, merged, group_count):
    runtime, timing, outputs = generated(monkeypatch, durations)
    entries, _, session, status, context = outputs
    assert len(runtime.samples) == group_count
    assert entries[0]["plan"]["total_frames"] == align_frame_count_up(round(10 * 24))
    if not merged:
        retained = entries[0]["plan"]["net_frames"]
        for index, entry in enumerate(entries[1:], 1):
            expected = make_extension_shape(entry["plan"]["context_frames"], (timing.requested_frame_boundaries[index+1] - retained) / 24)
            assert entry["plan"]["total_frames"] == expected.total_frames
            retained += entry["plan"]["net_frames"]
    assert session["settings"]["chunk_timing"] == timing.contract()
    assert "Requested frame boundaries:" in status
    physical, plan = prepare_physical_decode_entries(entries, chunk_seconds=10, preserve_final_frame=True, terminal_merged=merged)
    assert len(physical) == group_count
    assert plan["target_frames"] == timing.target_frames
    assert plan["chunk_timing"] == timing.contract()
    membership = [group["logical_chunks"] for group in plan["second_pass_contract"]["physical_groups"]]
    assert membership == ([[1], [2, 3]] if merged else [[1], [2], [3]])
    assert [group["logical_chunk_seconds"] for group in plan["second_pass_contract"]["physical_groups"]] == ([[10], [5,5]] if merged else [[10], [8], [10]])
    assert [list(group["logical_chunks"]) for group in context["groups"]] == membership
    videos = [{"samples": entry["video"].repeat_interleave(2, -1).repeat_interleave(2, -2)} for entry in physical]
    audios = [{"samples": entry["audio"]} for entry in physical]
    hooks = dict(latent_builder=lambda v, a: {"video": v, "audio": a},
                 stream_extractor=lambda value: (value["video"], value["audio"]),
                 clone_model_fn=lambda model, **kw: model, enable_preview=False)
    # Both real orchestrators validate/reconstruct captured conditioning at final geometry.
    v, a, refined_plan, _ = run_second_pass_groups(model=runtime.model, clip=runtime.clip, sampler=object(), sigmas=torch.tensor([.3, 0.]),
        video_latents=videos, audio_latents=audios, assembly_plan=plan, refine_seed=42, refine_context=context,
        sample_fn=lambda **kw: kw["latent"], **hooks)
    assert refined_plan["chunk_timing"] == timing.contract()
    def run_audio(**kw):
        return run_targeted_second_pass_groups(**kw, sample_fn=lambda **call: {"video": call["latent"]["video"]+99, "audio": call["latent"]["audio"]+1}, **hooks)
    monkeypatch.setattr(audio_refine_nodes, "run_targeted_second_pass_groups", run_audio)
    monkeypatch.setattr(audio_refine_nodes, "_audio_schedule", lambda *args: (lambda name: name, torch.tensor([.3, 0.])))
    out_v, out_a, out_plan, _ = audio_refine_nodes.H3ContinuumAudioRefineV39().refine(model=[runtime.model], clip=[runtime.clip],
        video_latents=v, audio_latents=a, assembly_plan=[refined_plan], seed=[42], steps=[2], sampler_name=["euler"], scheduler=["normal"], audio_denoise=[.3], refine_context=[context])
    assert all(x is y for x, y in zip(out_v, v))
    assert len(out_v) == len(out_a) == group_count
    assert out_plan["chunk_timing"] == timing.contract()
    assert out_plan["target_frames"] == timing.target_frames
    validate_assembly_plan(out_plan)
    from ComfyUI_H3_Continuum_Join.v2.decoder import enforce_total_frames
    # The existing final correction follows requested duration, not natural physical length.
    images, audio, _ = enforce_total_frames(torch.zeros(700, 1, 1, 3), {"waveform": torch.zeros(1, 2, 1400), "sample_rate": 48},
                                            target_frames=out_plan["target_frames"], preserve_final_frame=True)
    assert images.shape[0] == timing.target_frames
    assert audio["waveform"].shape[-1] == timing.target_frames * 2


def test_resume_stops_at_changed_duration(monkeypatch):
    _, _, outputs = generated(monkeypatch, "10,8,10")
    session = outputs[2]
    runtime, timing, result = generated(monkeypatch, "10,9,10", session=session)
    assert result[0][0]["reused"] is True
    assert [entry["reused"] for entry in result[0]] == [True, False, False]
    assert len(runtime.samples) == 2
    assert result[2]["settings"]["chunk_timing"] == timing.contract()


def test_timeline_media_windows():
    from ComfyUI_H3_Continuum_Join.timeline_video import prepare_timeline_video_source, encode_timeline_video_chunk
    from ComfyUI_H3_Continuum_Join.video_reference_modes import prepare_follow_video_source, select_follow_window
    from ComfyUI_H3_Continuum_Join.driving_audio import slice_driving_audio_latent
    from types import SimpleNamespace
    timing = resolve_chunk_timing(3, 10, "Custom Sequence", "10,8,10")
    frames = torch.zeros(672, 32, 32, 3)
    video = _Video(frames, duration=28)
    source = prepare_timeline_video_source(video, chunks=3, chunk_seconds=10, output_width=32, output_height=32, size_mode="Match Output", chunk_timing=timing)
    for index in range(3):
        encode_timeline_video_chunk(_VAE(), source, index)
    assert [(start, duration) for start, duration, _ in video.calls] == [(0,10), (10,8), (18,10)]
    follow = prepare_follow_video_source(frames, chunk_seconds=10, output_width=32, output_height=32, chunk_timing=timing)
    audio = SimpleNamespace(audio_latent=torch.arange(1120).reshape(1, 1, 1, 1120))
    for start, stop in zip(timing.requested_frame_boundaries, timing.requested_frame_boundaries[1:]):
        window = select_follow_window(follow, start_frame=start, visible_frames=stop-start)
        assert (window.requested_start, window.requested_stop) == (start, stop)
        sliced = slice_driving_audio_latent(audio, cumulative_retained_before=start, total_frames=stop-start, trim_frames=0, fps=24)
        assert sliced.flatten().tolist() == list(range(round(start*40/24), round(stop*40/24)))


def test_uniform_sampler_does_not_forward_custom_fields(monkeypatch):
    from ComfyUI_H3_Continuum_Join.v3.driving_nodes import H3ContinuumSamplerV38, H3ContinuumSamplerV39
    seen = []
    monkeypatch.setattr(H3ContinuumSamplerV38, "run", lambda self, **kw: seen.append(kw) or "sentinel")
    sampler = H3ContinuumSamplerV39()
    sampler.run(chunks=3, chunk_seconds=10)
    sampler.run(chunks=3, chunk_seconds=10, chunk_timing="Uniform", custom_durations="invalid inactive text")
    assert seen[0] == seen[1]
    assert "_chunk_timing" not in seen[0]
    sampler.run(chunks=3, chunk_seconds=10, chunk_timing="Custom Sequence", custom_durations="10,8,10")
    assert seen[2]["_chunk_timing"].target_frames == 672
    schema = sampler.INPUT_TYPES()
    assert list(schema["optional"])[-2:] == ["chunk_timing", "custom_durations"]


def test_partial_review_keeps_fractional_frame_phase(monkeypatch):
    _, timing, result = generated(monkeypatch, "4.1,4.2,4.3")
    entries = result[0]
    _, full = prepare_physical_decode_entries(entries, chunk_seconds=10, preserve_final_frame=True, terminal_merged=False)
    _, scoped = prepare_physical_decode_entries(entries[1:2], chunk_seconds=10, preserve_final_frame=False, terminal_merged=False)
    assert scoped["target_frames"] == timing.requested_frame_boundaries[2] - timing.requested_frame_boundaries[1]
    assert scoped["chunk_timing"]["frame_origin_seconds"] == 4.1
    validate_assembly_plan(scoped)
    assert full["target_frames"] == timing.target_frames


def test_uniform_native_sequence_matches_pre_feature_engine(monkeypatch):
    import subprocess
    import sys
    from types import ModuleType
    name = "ComfyUI_H3_Continuum_Join.v2._uniform_sequence_baseline"
    old = ModuleType(name)
    old.__package__ = "ComfyUI_H3_Continuum_Join.v2"
    monkeypatch.setitem(sys.modules, name, old)
    exec(compile(subprocess.check_output(["git", "show", "HEAD:v2/sequence.py"]).decode(), "uniform_sequence_baseline.py", "exec"), old.__dict__)
    runtime = cap._install_fake_runtime(monkeypatch)
    for symbol in ("check_comfy_h3_runtime", "prepare_identity_assets", "encode_identity_latents", "empty_h3_latent", "accelerator_summary", "clone_model_for_chunk", "latent_from_cpu", "sample_chunk"):
        setattr(old, symbol, getattr(sequence, symbol))
    modern = cap._run_sequence(runtime, chunks=3, chunk_seconds=10)
    monkeypatch.setattr(cap, "sequence", old)
    legacy = cap._run_sequence(runtime, chunks=3, chunk_seconds=10)
    for current, baseline in zip(modern[0], legacy[0]):
        assert current["plan"] == baseline["plan"]
        assert current["seed"] == baseline["seed"]
        assert torch.equal(current["video"], baseline["video"])
        assert torch.equal(current["audio"], baseline["audio"])
    assert modern[2]["settings"] == legacy[2]["settings"]


@pytest.mark.parametrize("durations,expected", [("10,8,10", [[1],[2],[3]]), ("10,5,5", [[1],[2,3]])])
def test_public_v39_routes_timing_and_reference_groups(monkeypatch, durations, expected):
    from types import SimpleNamespace
    from ComfyUI_H3_Continuum_Join.v3.driving_nodes import H3ContinuumSamplerV39
    from .test_v38_review_run_storage import _production_kwargs
    first = torch.zeros(1, 64, 96, 3)
    last = torch.ones_like(first)
    runtime = cap._install_fake_runtime(monkeypatch, first_frame=first, last_frame=last)
    runtime.assets.identity_hash = "a" * 64
    monkeypatch.setattr(H3ContinuumSamplerV39, "_resolve_resolution", staticmethod(lambda **kw: SimpleNamespace(width=96, height=64)))
    inputs = _production_kwargs()
    inputs.update(model=runtime.model, clip=runtime.clip, chunks=3, chunk_seconds=10, run_storage="Off", generation_mode="Full Run",
                  first_frame=first, last_frame=last, chunk_timing="Custom Sequence", custom_durations=durations)
    outputs = H3ContinuumSamplerV39().run(**inputs)["result"]
    assert len(outputs[0]) == len(outputs[1]) == len(expected)
    assert [g["logical_chunks"] for g in outputs[2]["second_pass_contract"]["physical_groups"]] == expected
    assert outputs[2]["chunk_timing"]["logical_chunk_seconds"] == [float(v) for v in durations.split(",")]
    assert len(runtime.samples) == len(expected)
