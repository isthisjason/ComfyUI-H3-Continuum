"""Exercise the external dictionary boundary without importing Fantastic."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from ComfyUI_H3_Continuum_Join import nodes
from ComfyUI_H3_Continuum_Join.fantastic_refs import adapt_fantastic_refs
from ComfyUI_H3_Continuum_Join.reference_audio import prepare_reference_audio_bundle, ReferenceAudioError
from ComfyUI_H3_Continuum_Join.v3.fantastic_adapter_nodes import H3ContinuumFantasticMediaAdapter
from ComfyUI_H3_Continuum_Join.v3.reference_images_v39 import effective_reference_inputs


def image(value=0.5):
    return torch.full((1, 8, 8, 3), value)


def audio(rate=32000, value=0.1):
    return {"waveform": torch.full((1, 2, 320), value), "sample_rate": rate}


VAE = SimpleNamespace(audio_sample_rate=32000)


def test_empty_bundle_and_defaults():
    images, audios, video, driving, diagnostic = adapt_fantastic_refs({})
    assert images.images == (None,) * 9
    assert images.reference_use == "All chunks"
    assert (audios, video, driving) == (None, None, None)
    assert "pictures=0, videos=0, audios=0" in diagnostic
    assert "no schema version" in diagnostic
    assert "Only one of the three" in diagnostic
    for index in range(1, 10):
        assert f"R{index} <- None" in diagnostic


@pytest.mark.parametrize("count", [1, 9])
def test_picture_identity_padding_and_existing_runtime(count):
    pictures = [image(i / 10) for i in range(count)]
    bundle, _, _, _, report = adapt_fantastic_refs({"pictures": pictures})
    assert all(bundle.images[i] is pictures[i] for i in range(count))
    assert all(value is None for value in bundle.images[count:])
    direct, extra, selectors = effective_reference_inputs(bundle, chunks=3)
    assert direct[0] is pictures[0]
    assert selectors == {f"R{i}": "all" if i <= count else "off" for i in range(1, 10)}
    assert f"R{count} <- effective Picture {count}" in report


def test_disabled_items_metadata_does_not_reconstruct_original_slots():
    first, second = image(0.2), image(0.7)
    source = {"pictures": [first, second], "items": [
        {"kind": "picture", "enabled": False},
        {"kind": "picture", "enabled": True},
        {"kind": "picture", "enabled": True},
    ]}
    bundle, _, _, _, report = adapt_fantastic_refs(source)
    assert bundle.images[0] is first and bundle.images[1] is second
    assert "original media-state numbering is not retained" in report


def test_prompt_builder_shape_without_items_and_unknown_keys():
    picture = image()
    source = {"pictures": [None, picture] + [None] * 7,
              "videos": [None] * 3, "video_audios": [None] * 3,
              "audios": [None] * 3, "future_metadata": object()}
    bundle, *rest = adapt_fantastic_refs(source)
    assert bundle.images[0] is None and bundle.images[1] is picture
    assert rest[:3] == [None, None, None]


def test_three_contiguous_audios_match_existing_helper():
    values = [audio(value=i / 10) for i in range(1, 4)]
    _, bundle, _, driving, report = adapt_fantastic_refs({"audios": values}, VAE)
    expected = prepare_reference_audio_bundle(*values, VAE)
    assert bundle.contract == expected.contract
    assert bundle.reference_audio_vae is VAE
    assert driving is None
    for i in range(1, 4):
        assert f"Audio {i} -> <Audio {i}>" in report


@pytest.mark.parametrize("gap", [0, 1, 2])
def test_audio_gaps_do_not_compact_and_driving_selection_is_independent(gap):
    values = [audio(value=i / 10) for i in range(1, 4)]
    values[gap] = None
    _, bundle, _, driving, report = adapt_fantastic_refs(
        {"audios": values}, VAE, driving_audio_source="Audio 3")
    assert (bundle.count if bundle else 0) == gap
    assert driving is values[2]
    if gap < 2:
        assert "Non-contiguous standalone audio" in report
        assert "Audio 3 -> omitted (gap)" in report
    else:
        assert "Non-contiguous" not in report


@pytest.mark.parametrize("count", [1, 2, 3])
def test_explicit_video_selection_and_no_automatic_selection(count):
    videos = [image(i / 10).expand(5, -1, -1, -1) for i in range(count)]
    source = {"videos": videos}
    assert adapt_fantastic_refs(source)[2] is None
    for i in range(count):
        result = adapt_fantastic_refs(source, video_slot=f"Video {i + 1}")
        assert result[2] is videos[i]
        assert f"videos={count}" in result[4]


@pytest.mark.parametrize("selector", ["invalid", "Video 4", "", None])
def test_invalid_video_selector_is_diagnostic_only(selector):
    result = adapt_fantastic_refs({"videos": [image()]}, video_slot=selector)
    assert result[2] is None
    assert "Invalid video selector" in result[4]


def test_missing_video_and_invalid_driving_selector():
    result = adapt_fantastic_refs({}, video_slot="Video 2", driving_audio_source="invalid")
    assert result[2:4] == (None, None)
    assert "Video 2 is missing" in result[4]
    assert "Invalid driving audio selector" in result[4]


@pytest.mark.parametrize("index", [1, 2, 3])
def test_paired_soundtrack_only_routes_when_explicitly_selected(index):
    tracks = [audio(value=i / 10) for i in range(1, 4)]
    source = {"video_audios": tracks}
    assert adapt_fantastic_refs(source)[1:4] == (None, None, None)
    result = adapt_fantastic_refs(source, driving_audio_source=f"Video {index} Audio")
    assert result[1] is None
    assert result[3] is tracks[index - 1]


def test_missing_paired_soundtrack():
    result = adapt_fantastic_refs({"video_audios": [None]}, driving_audio_source="Video 1 Audio")
    assert result[3] is None
    assert "Video 1 Audio is missing" in result[4]


@pytest.mark.parametrize("rate", [16000, 44100, 48000])
def test_sample_rates_and_input_immutability(rate):
    # Exercise the real helper's resampling while ensuring upstream values survive.
    track, picture = audio(rate), image()
    original_waveform = track["waveform"]
    source = {"pictures": [picture], "audios": [track], "videos": [picture],
              "video_audios": [track], "items": [{"enabled": False}], "extra": {"x": 1}}
    before = deepcopy(source)
    result = adapt_fantastic_refs(source, VAE, "Video 1", "Audio 1")
    assert result[3] is track and result[2] is picture
    assert track["waveform"] is original_waveform
    assert result[1].sources[0].source_sample_rate == rate
    assert result[1].sources[0].resolved_vae_sample_rate == 32000
    assert track["sample_rate"] == rate
    assert set(source) == set(before)
    for key in ("pictures", "videos"):
        assert source[key][0] is picture
        assert torch.equal(source[key][0], before[key][0])
    for key in ("audios", "video_audios"):
        assert source[key][0] is track
        assert torch.equal(track["waveform"], before[key][0]["waveform"])
        assert set(track) == set(before[key][0])
    assert source["items"] == before["items"] and source["extra"] == before["extra"]


def test_existing_vae_requirement_is_preserved():
    with pytest.raises(ReferenceAudioError, match="reference_audio_vae is required"):
        adapt_fantastic_refs({"audios": [audio()]})
    # An empty safe prefix needs no VAE, even if a later slot is selected for driving.
    result = adapt_fantastic_refs({"audios": [None, audio()]}, driving_audio_source="Audio 2")
    assert result[1] is None and result[3] is not None


def test_registration_schema_and_node_execution():
    node_id = "H3ContinuumFantasticMediaAdapter"
    assert nodes.NODE_CLASS_MAPPINGS[node_id] is H3ContinuumFantasticMediaAdapter
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS[node_id] == "H3 Continuum Fantastic Media Adapter"
    schema = H3ContinuumFantasticMediaAdapter.INPUT_TYPES()
    assert schema["required"]["references"][0] == "H3_REFS"
    assert schema["optional"]["audio_vae"][0] == "VAE"
    for group in schema.values():
        assert all(definition[1]["tooltip"] for definition in group.values())
    assert H3ContinuumFantasticMediaAdapter.RETURN_TYPES == (
        "H3_CONTINUUM_REFERENCE_IMAGES_V39", "H3_CONTINUUM_AUDIO_REFERENCES", "IMAGE", "AUDIO", "STRING")
    assert H3ContinuumFantasticMediaAdapter().adapt({}) == adapt_fantastic_refs({})


def test_public_node_forwards_explicit_selections_and_audio_vae():
    picture = image(0.3)
    videos = [image(0.1).expand(5, -1, -1, -1), image(0.9).expand(5, -1, -1, -1)]
    standalone = audio(44100, value=0.2)
    soundtrack = audio(48000, value=0.8)
    audio_vae = SimpleNamespace(audio_sample_rate=44100)
    references = {
        "pictures": [picture], "videos": videos,
        "audios": [standalone], "video_audios": [None, soundtrack],
    }

    # Exercise the public wrapper and real conversion, with distinct media so
    # dropped/defaulted selectors or a substituted VAE cannot satisfy the checks.
    images, audios, video, driving, diagnostic = H3ContinuumFantasticMediaAdapter().adapt(
        references=references,
        video_slot="Video 2",
        driving_audio_source="Video 2 Audio",
        audio_vae=audio_vae,
    )

    assert images.images[0] is picture
    assert video is videos[1]
    assert driving is soundtrack
    assert audios.reference_audio_vae is audio_vae
    assert audios.count == 1
    assert audios.sources[0].resolved_vae_sample_rate == 44100
    assert torch.equal(audios.sources[0].waveform, standalone["waveform"])
    assert "Selected video: Video 2." in diagnostic
    assert "Selected driving audio: Video 2 Audio." in diagnostic
