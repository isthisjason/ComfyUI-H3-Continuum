# Fantastic Media Loader compatibility

Connect Fantastic's **MiniMaxH3MediaLoader** or Prompt Builder `references`
output to **H3 Continuum Fantastic Media Adapter** (`H3ContinuumFantasticMediaAdapter`).
Fantastic remains an external dependency for producing the bundle; Continuum does
not import it, decode files again, or use `items` metadata to rebuild media slots.

| Adapter output | Continuum V3.9 input |
| --- | --- |
| reference_images | Reference Images |
| audio_references | Audio References |
| video_frames | Video Guide Frames (`reference_video_1`) |
| driving_audio | Driving Audio |
| diagnostic | Optional STRING display node |

Pictures retain **effective Fantastic ordering**, after disabled items have been
removed by Fantastic. They map to R1–R9, padded with empty positions, and apply to
All chunks. Original media-state numbering does not survive filtering. These are
reference images, not automatic First/Last Images. Existing Continuum prompt
routing still determines native picture tags; the adapter never rewrites prompts.

Up to three standalone audios map to the same numbered reference-audio slots.
Connect an Audio VAE to the adapter's `audio_vae` **VAE** socket when any safe
contiguous prefix is present. This is the socket type used by Continuum's existing
reference-audio helper, not a new AUDIO_VAE type. A gap ends the routed prefix:
`[Audio 1, None, Audio 3]` routes Audio 1 and diagnoses omitted Audio 3. It never
renumbers Audio 3. Original sample rates and upstream AUDIO dictionaries remain
unchanged; the existing helper resamples its conditioning copy for the VAE.

Video selection defaults to None. Explicitly select Video 1, 2, or 3 to pass its
already-decoded 24 fps IMAGE batch into the Sampler. The Sampler retains its own
Repeat Reference / experimental Follow Timeline choice and preprocessing. Only
one video can be routed; videos are never merged. This does not expose the separate
internal Core VIDEO timeline path or provide lazy video loading.

Driving audio selection also defaults to None. Select a standalone audio or paired
video soundtrack explicitly, and connect the Driving Audio VAE to the Sampler.
Driving audio guides generation and supplies the effective source prefix for final
output. Standalone reference audio only conditions generation. A selected standalone
audio remains a reference too when it belongs to the contiguous prefix; selection
does not remove or renumber it. Paired soundtracks never become references automatically.
This adapter does not reproduce native paired video/audio reference conditioning.

The diagnostic reports counts, mappings, selections, missing slots, audio gaps,
and the unversioned H3_REFS contract. Loader dictionaries with `items` and Prompt
Builder dictionaries without it are supported; unknown keys are ignored. Missing
selections and invalid selector values route None with diagnostics. Existing
Continuum validation of required audio payloads and VAE availability still applies.
No prompt restrictions or model compatibility gates are added.

Existing workflows and Sampler schemas are unchanged. CPU tests verify conversion.

Manual ComfyUI acceptance was reported by the user for image-only, standalone
audio, and video routing; paired soundtrack selection; multiple image/audio
references; three-speaker voice/reference binding; and multiple camera cuts across
a long Continuum sequence. This supports the standalone feature milestone. Detailed
run settings and artifacts were not supplied for an independently reproducible GPU gate.
