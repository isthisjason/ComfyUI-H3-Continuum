"""Public socket adapter; media conversion stays in the reusable module."""

from ..fantastic_refs import adapt_fantastic_refs, VIDEO_SLOTS, DRIVING_AUDIO_SOURCES
from ..reference_audio import REFERENCE_AUDIOS_TYPE
from .reference_images_v39 import REFERENCE_IMAGES_V39_TYPE


class H3ContinuumFantasticMediaAdapter:
    CATEGORY = "MiniMax H3/Continuum/Input"
    DESCRIPTION = (
        "Route decoded Fantastic H3_REFS into existing Continuum V3.9 inputs. "
        "Images use All chunks. Video and driving audio require explicit selection."
    )
    RETURN_TYPES = (REFERENCE_IMAGES_V39_TYPE, REFERENCE_AUDIOS_TYPE, "IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("reference_images", "audio_references", "video_frames", "driving_audio", "diagnostic")
    FUNCTION = "adapt"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "references": ("H3_REFS", {"tooltip": "Decoded bundle from Fantastic Media Loader or Prompt Builder; items metadata is optional."}),
                "video_slot": (VIDEO_SLOTS, {"default": "None", "tooltip": "Select one effective Fantastic video for Video Guide Frames. None routes no video. Repeat/Follow remains a Sampler setting."}),
                "driving_audio_source": (DRIVING_AUDIO_SOURCES, {"default": "None", "tooltip": "Explicitly select an audio timeline. Driving Audio guides generation and supplies final audio; None routes no driving audio."}),
            },
            "optional": {
                "audio_vae": ("VAE", {"tooltip": "Continuum's reference-audio helper requires a Core Audio VAE when a contiguous standalone audio prefix is present. Driving Audio VAE connects separately to the Sampler."}),
            },
        }

    def adapt(self, references, video_slot="None", driving_audio_source="None", audio_vae=None):
        return adapt_fantastic_refs(references, audio_vae, video_slot, driving_audio_source)
