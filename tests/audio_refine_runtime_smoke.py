"""Native Core scheduler/factory/solver smoke; synthetic denoiser, no checkpoint."""
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace, ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.environ.get("H3_COMFY_ROOT", str(ROOT.parent.parent)))
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))
import comfy.cli_args
comfy.cli_args.args.cpu = True
# Pin the local test package rather than another dependency's tests namespace.
tests_package = ModuleType("tests")
tests_package.__path__ = [str(ROOT / "tests")]
sys.modules["tests"] = tests_package
from tests import conftest  # Install the repository's normal package alias.
import torch
import comfy.samplers
import comfy.model_sampling
from ComfyUI_H3_Continuum_Join.v3 import audio_refine_nodes as node
from ComfyUI_H3_Continuum_Join.v3.targeted_second_pass import run_targeted_second_pass_groups
from tests.test_v38_audio_only_refine import _fixture


class Sampling(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
    pass


class Denoiser:
    """Analytic audio target exercises the actual CPU Euler integration loop."""
    def __init__(self, sampling):
        self.inner_model = SimpleNamespace(model_sampling=sampling)
        self.model_patcher = SimpleNamespace(model=self)
        self.cfg = 1.0

    def __call__(self, x, sigma, **kwargs):
        return torch.full_like(x, 0.25)


started = time.monotonic()
sampling = Sampling()
model = SimpleNamespace(get_model_object=lambda name: sampling)
groups, videos, audios, plan = _fixture()
solver_steps = []
received = []
original_runner = node.run_targeted_second_pass_groups


def sample(**kwargs):
    sampler, sigmas, latent = kwargs["sampler"], kwargs["sigmas"], kwargs["latent"]
    assert isinstance(sampler, comfy.samplers.KSAMPLER)
    assert sampler.sampler_function.__name__ == "sample_euler"
    assert sigmas.numel() == 4 and sigmas[-1] == 0
    received.append(kwargs["conditioning"])
    audio = latent["audio"]
    noise = torch.randn(audio.shape, generator=torch.Generator().manual_seed(kwargs["seed"]))
    refined = sampler.sample(Denoiser(sampling), sigmas, {"seed": kwargs["seed"]},
        lambda *args: solver_steps.append(args[0]), noise, latent_image=audio, disable_pbar=True)
    return {"video": latent["video"], "audio": refined}


def run(**kwargs):
    return original_runner(**kwargs, encode_prompt_fn=lambda clip, prompt, **kw: [prompt],
        latent_builder=lambda v, a: {"video": v, "audio": a}, sample_fn=sample,
        stream_extractor=lambda value: (value["video"], value["audio"]),
        clone_model_fn=lambda model, **kw: model, enable_preview=False)


# Only substitute checkpoint/CLIP execution: _audio_schedule and all Core sampler
# functions are native, including the public node's sampler_factory(name) call.
node.run_targeted_second_pass_groups = run
out_v, out_a, updated, status = node.H3ContinuumAudioRefineV39().refine(
    model=[model], clip=[object()], video_latents=videos, audio_latents=audios,
    assembly_plan=[plan], seed=[19], steps=[3], sampler_name=["euler"],
    scheduler=["normal"], audio_denoise=[0.3])
assert len(out_v) == len(out_a) == 2
assert all(out is original for out, original in zip(out_v, videos))
assert all(not torch.equal(out["samples"], original["samples"]) for out, original in zip(out_a, audios))
assert received == [["group 1"], ["group 2"]]
assert len(solver_steps) == 6
print(json.dumps({"workflow": "public Audio Refine -> native Core Euler/normal",
    "prompt": received, "media": "synthetic grouped AV tensors", "model": "analytic CPU denoiser",
    "loras": [], "sampler": "euler", "scheduler": "normal", "steps": 3, "seed": 19,
    "audio_denoise": 0.3, "dimensions": [plan["width"], plan["height"]],
    "logical_chunks": [g["logical_chunks"] for g in groups], "chunk_seconds": "synthetic topology fixture",
    "physical_groups": 2, "solver_steps": len(solver_steps), "video_preserved": True,
    "audio_changed": True, "elapsed_seconds": time.monotonic() - started}))
