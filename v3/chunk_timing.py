"""Requested logical timing, deliberately separate from native H3 group topology."""
from dataclasses import dataclass
import math

from ..constants import CHUNK_SECONDS_MIN, CHUNK_SECONDS_MAX, FPS


@dataclass(frozen=True)
class ChunkTiming:
    logical_chunk_seconds: tuple[float, ...]
    cumulative_seconds: tuple[float, ...]
    requested_frame_boundaries: tuple[int, ...]
    frame_origin_seconds: float = 0.0

    @property
    def total_seconds(self):
        return self.cumulative_seconds[-1]

    @property
    def target_frames(self):
        return self.requested_frame_boundaries[-1]

    def contract(self):
        result = {"version": 1, "logical_chunk_seconds": list(self.logical_chunk_seconds),
                  "requested_frame_boundaries": list(self.requested_frame_boundaries)}
        if self.frame_origin_seconds:
            result["frame_origin_seconds"] = self.frame_origin_seconds
        return result


def timing_from_seconds(values, *, frame_origin_seconds=0.0):
    values = tuple(float(value) for value in values)
    if not values or any(not math.isfinite(value) or not CHUNK_SECONDS_MIN <= value <= CHUNK_SECONDS_MAX for value in values):
        raise ValueError(f"Each custom duration must be finite and between {CHUNK_SECONDS_MIN:g} and {CHUNK_SECONDS_MAX:g} seconds")
    cumulative = [0.0]
    for value in values:
        cumulative.append(cumulative[-1] + value)
    # Keep the existing cumulative round(seconds * FPS) model, including fractions.
    origin = float(frame_origin_seconds)
    if not math.isfinite(origin) or origin < 0:
        raise ValueError("Invalid requested timing origin")
    boundaries = tuple(round((origin + value) * FPS) - round(origin * FPS) for value in cumulative)
    return ChunkTiming(values, tuple(cumulative), boundaries, origin)


def resolve_chunk_timing(chunks, chunk_seconds, mode="Uniform", custom=""):
    if mode == "Uniform":
        return timing_from_seconds([chunk_seconds] * int(chunks))
    if mode != "Custom Sequence":
        raise ValueError(f"Unknown Chunk Timing mode: {mode!r}")
    parts = str(custom).split(",")
    if len(parts) != int(chunks):
        raise ValueError(f"Custom Durations requires exactly {chunks} values; received {len(parts)}")
    try:
        return timing_from_seconds(parts)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid Custom Durations: {exc}") from exc


def timing_from_contract(contract, *, count=None):
    if not isinstance(contract, dict) or type(contract.get("version")) is not int or contract["version"] != 1:
        raise ValueError("Invalid chunk_timing contract version")
    timing = timing_from_seconds(contract.get("logical_chunk_seconds", ()), frame_origin_seconds=contract.get("frame_origin_seconds", 0))
    if count is not None and len(timing.logical_chunk_seconds) != count:
        raise ValueError("chunk_timing duration count does not match logical chunks")
    if contract.get("requested_frame_boundaries") != list(timing.requested_frame_boundaries):
        raise ValueError("chunk_timing requested frame boundaries are inconsistent")
    return timing


def timing_from_entries(entries):
    """Rebase requested timing for full, partial, or Review-scoped output plans."""
    if not entries or not any("logical_timing" in entry["plan"] for entry in entries):
        return None
    if not all("logical_timing" in entry["plan"] for entry in entries):
        raise ValueError("Mixed custom/uniform logical timing in stored entries")
    # Review views rebase frame endpoints, preserving the original fractional phase.
    return timing_from_seconds([entry["plan"]["logical_timing"]["seconds"] for entry in entries],
                               frame_origin_seconds=entries[0]["plan"]["logical_timing"].get("start_seconds", 0))


def timing_group_contracts(timing, *, terminal_merge):
    """Include the prefix timing so no dependent suffix can resume across a change."""
    result = []
    count = len(timing.logical_chunk_seconds)
    for index in range(count):
        start, stop = index, index + 1
        if terminal_merge and index >= count - 2:
            start, stop = count - 2, count
        result.append({"logical_chunks": list(range(start + 1, stop + 1)),
                       "prefix_seconds": list(timing.logical_chunk_seconds[:stop]),
                       "requested_frame_boundaries": list(timing.requested_frame_boundaries[:stop + 1])})
    return result
