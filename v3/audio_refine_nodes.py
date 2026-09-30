"""Public audio-only adapter over Continuum's physical-group refinement engine."""

from .refine_target import MODE_AUDIO_ONLY
from .second_pass import validate_second_pass_inputs, update_second_pass_geometry
from .second_pass_nodes import _single_list_input
from .targeted_second_pass import run_targeted_second_pass_groups


def _audio_schedule(model, steps, scheduler, denoise):
    # Match Core BasicScheduler's partial-denoise tail; MODEL patches stay intact.
    import comfy.samplers

    total_steps = steps if denoise >= 1.0 else int(steps / denoise)
    sigmas = comfy.samplers.calculate_sigmas(
        model.get_model_object("model_sampling"), scheduler, total_steps,
    )
    return comfy.samplers.sampler_object, sigmas[-(steps + 1):].cpu()


class H3ContinuumAudioRefineV39:
    """Refine all physical audio groups against their final, frozen video."""

    DESCRIPTION = (
        "Audio-only physical-group refinement using final video as frozen context. "
        "Connect the matching assembly plan and optional refine context. Uses Core "
        "BasicGuider (no extra CFG), and accepts upstream MODEL patches normally. "
        "Audio-targeted sampling remains Experimental / Production HOLD."
    )
    INPUT_IS_LIST = True
    RETURN_TYPES = ("LATENT", "LATENT", "H3_CONTINUUM_ASSEMBLY_PLAN", "STRING")
    RETURN_NAMES = ("video_latents", "refined_audio_latents", "updated_assembly_plan", "status")
    OUTPUT_IS_LIST = (True, True, False, False)
    FUNCTION = "refine"
    CATEGORY = "MiniMax H3/Continuum/Advanced"

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers

        return {
            "required": {
                "model": ("MODEL", {"tooltip": "H3 MODEL; upstream patches are retained."}),
                "clip": ("CLIP", {"tooltip": "Used for physical prompts when complete refine context is absent."}),
                "video_latents": ("LATENT", {"tooltip": "Final video groups, including external spatial upscale/detail refinement."}),
                "audio_latents": ("LATENT", {"tooltip": "Matching audio groups in original physical order."}),
                "assembly_plan": ("H3_CONTINUUM_ASSEMBLY_PLAN", {"tooltip": "Matching physical-group plan; temporal geometry is preserved."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                  "control_after_generate": True, "tooltip": "Base seed derived independently per physical group."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "Audio refinement sampling steps."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "Core sampler used for audio-only refinement."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "Core sigma scheduler; denoise selects its tail."}),
                "audio_denoise": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01,
                                            "tooltip": "Audio noise strength; zero returns exact video/audio passthrough without sampling."}),
            },
            "optional": {
                "refine_context": ("H3_CONTINUUM_REFINE_CONTEXT", {"tooltip": "Preserves each physical group's image/reference conditioning."}),
                "video_vae": ("VAE", {"tooltip": "Required when context reconstruction needs image encoding at final video geometry."}),
            },
        }

    def refine(self, model, clip, video_latents, audio_latents, assembly_plan,
               seed, steps, sampler_name, scheduler, audio_denoise,
               refine_context=None, video_vae=None):
        model = _single_list_input("model", model)
        plan = _single_list_input("assembly_plan", assembly_plan)
        steps = int(_single_list_input("steps", steps))
        denoise = float(_single_list_input("audio_denoise", audio_denoise))
        sampler_name = _single_list_input("sampler_name", sampler_name)
        scheduler = _single_list_input("scheduler", scheduler)
        validation = validate_second_pass_inputs(video_latents, audio_latents, plan)
        count = validation["physical_group_count"]
        detail = f"audio_denoise={denoise:g}; steps={steps}; sampler={sampler_name}; scheduler={scheduler}; video=bit-exact preserved"
        if denoise == 0:
            # Validate alignment even for bypass, but avoid conditioning/encoding/model work.
            return (list(video_latents), list(audio_latents),
                    update_second_pass_geometry(plan, video_latents),
                    f"Audio Refine: physical groups={count}; refined groups=none; sampling skipped; {detail}")
        sampler_factory, sigmas = _audio_schedule(model, steps, scheduler, denoise)
        videos, audios, updated, status = run_targeted_second_pass_groups(
            model=model, clip=_single_list_input("clip", clip),
            sampler=sampler_factory(sampler_name), sigmas=sigmas,
            video_latents=video_latents, audio_latents=audio_latents,
            assembly_plan=plan, refine_seed=int(_single_list_input("seed", seed)),
            refine_target=MODE_AUDIO_ONLY,
            refine_context=None if refine_context is None else _single_list_input("refine_context", refine_context),
            video_vae=None if video_vae is None else _single_list_input("video_vae", video_vae),
        )
        return videos, audios, updated, f"Audio Refine: physical groups={count}; refined groups={list(range(1, count + 1))}; {detail}\n{status}"
