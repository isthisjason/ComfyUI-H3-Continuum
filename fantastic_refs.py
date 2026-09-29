"""Translate Fantastic's decoded, unversioned H3_REFS without importing it."""

from __future__ import annotations

from .reference_audio import prepare_reference_audio_bundle
from .v3.reference_images_v39 import ReferenceImagesV39Bundle, REFERENCE_USE_ALL


VIDEO_SLOTS = ("None", "Video 1", "Video 2", "Video 3")
DRIVING_AUDIO_SOURCES = (
    "None", "Audio 1", "Audio 2", "Audio 3",
    "Video 1 Audio", "Video 2 Audio", "Video 3 Audio",
)


def adapt_fantastic_refs(references, audio_vae=None, video_slot="None",
                        driving_audio_source="None"):
    """Return existing Continuum bundles and explicitly selected media.

    The effective lists are authoritative: loader metadata still includes disabled
    items and cannot reconstruct slot identity. Never modify upstream containers.
    """
    warnings = [
        "Fantastic H3_REFS has no schema version; interpreting the observed dictionary contract.",
        "Only one of the three Fantastic videos can currently be routed into public Continuum V3.9.",
    ]
    if not isinstance(references, dict):
        warnings.append("H3_REFS is not a dictionary; no media routed.")
        references = {}

    def slots(key, limit):
        values = references.get(key, [])
        if values is None:
            values = []
        if not isinstance(values, (list, tuple)):
            warnings.append(f"{key} is not a list or tuple; no slots routed from this field.")
            values = []
        count = sum(value is not None for value in values)
        if len(values) > limit:
            warnings.append(f"{key}: slots beyond {limit} are omitted.")
        return tuple(values[:limit]) + (None,) * max(0, limit - len(values)), count

    pictures, picture_count = slots("pictures", 9)
    videos, video_count = slots("videos", 3)
    audios, audio_count = slots("audios", 3)
    soundtracks, soundtrack_count = slots("video_audios", 3)
    images = ReferenceImagesV39Bundle(pictures, REFERENCE_USE_ALL, ("off",) * 9)

    # The existing audio helper requires contiguous native Audio numbering.
    # Stop at the first hole rather than moving later voices to different tags.
    prefix_length = next((i for i, value in enumerate(audios) if value is None), 3)
    omitted = [f"Audio {i + 1}" for i in range(prefix_length, 3) if audios[i] is not None]
    if omitted:
        warnings.append(
            f"Non-contiguous standalone audio: Audio {prefix_length + 1} is missing; "
            f"omitted {', '.join(omitted)} from reference audio. Numbering is not compacted."
        )
    safe_audios = audios[:prefix_length] + (None,) * (3 - prefix_length)
    audio_references = prepare_reference_audio_bundle(*safe_audios, audio_vae)

    def select(choice, choices, values, label):
        if choice not in choices:
            warnings.append(f"Invalid {label} selector {choice!r}; routed None.")
            return None, "None"
        if choice == "None":
            return None, "None"
        selected = values[choices.index(choice) - 1]
        if selected is None:
            warnings.append(f"Requested {label} {choice} is missing; routed None.")
            return None, f"{choice} (missing; routed None)"
        return selected, choice

    video, video_description = select(video_slot, VIDEO_SLOTS, videos, "video")
    driving, driving_description = select(
        driving_audio_source, DRIVING_AUDIO_SOURCES, audios + soundtracks, "driving audio",
    )
    report = [
        f"Effective Fantastic media: pictures={picture_count}, videos={video_count}, "
        f"audios={audio_count}, paired soundtracks={soundtrack_count}.",
        "Picture positions use effective Fantastic ordering after disabled-item filtering; "
        "original media-state numbering is not retained. Assignment: All chunks.",
        "Reference images: " + "; ".join(
            f"R{i} <- " + (f"effective Picture {i}" if value is not None else "None")
            for i, value in enumerate(pictures, 1)
        ),
        "Reference audio: " + "; ".join(
            f"Audio {i + 1} -> " + (f"<Audio {i + 1}>" if i < prefix_length else
                                    "omitted (gap)" if audios[i] is not None else "None")
            for i in range(3)
        ),
        f"Selected video: {video_description}.",
        f"Selected driving audio: {driving_description}.",
        "Paired soundtracks are never automatically routed as reference or driving audio.",
        *(f"Warning: {warning}" for warning in warnings),
    ]
    return images, audio_references, video, driving, "\n".join(report)
