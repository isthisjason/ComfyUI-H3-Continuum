"""Custom timing identity and certified-prefix reuse, without relaxed Take lineage."""
from copy import deepcopy
import subprocess
import sys
from types import ModuleType

import pytest

from ComfyUI_H3_Continuum_Join import run_storage as storage
from ComfyUI_H3_Continuum_Join.v3.chunk_timing import timing_from_seconds, timing_group_contracts
from . import test_v38_review_run_storage as fixtures
from . import test_v31_run_storage as contracts


def custom_contract(values, *, terminal=False):
    contract = fixtures._contract(chunks=len(values), terminal=terminal)
    timing = timing_from_seconds(values)
    contract["chunk_timing"] = timing.contract()
    contract["chunk_timing_groups"] = timing_group_contracts(timing, terminal_merge=terminal)
    contract["nonce_lineage_sha256"] = storage._nonce_lineage_hash(contract)
    return storage._apply_reroll_branch_contract(contract, boundary=0, requested_nonce=0, effective_nonce=0)


def test_distinct_full_identity_and_prefix_hashes():
    a = custom_contract([10,8,10])
    b = custom_contract([10,10,10])
    c = custom_contract([8,10,10])
    assert len({storage.revision_identity(value)[1] for value in (a,b,c)}) == 3
    assert a["chunk_contract_hashes"][0] == b["chunk_contract_hashes"][0]
    assert a["chunk_contract_hashes"][1:] != b["chunk_contract_hashes"][1:]
    assert all(x != y for x,y in zip(a["chunk_contract_hashes"], c["chunk_contract_hashes"]))
    assert "chunk_timing" in storage._review_contract_mismatches(a, b)[-1]


@pytest.mark.parametrize("saved", [fixtures._contract(chunks=3), custom_contract([5,5,5])])
def test_certified_import_stops_at_first_timing_change(tmp_path, saved):
    source = fixtures._persist(tmp_path, saved, prefix=3, review_unit=None, updated_utc="2026-09-29T00:00:00Z")
    current = custom_contract([5,4,5])
    controller = fixtures._controller(tmp_path, current)
    entries, records = controller._compatible_plan_prefix(source.manifest, current)
    assert len(entries) == len(records) == 1
    # A later 5s chunk cannot bypass the incompatible middle group.
    assert storage._timing_chunk_equivalent(saved, current, 0, requested_nonce=0)
    assert not storage._timing_chunk_equivalent(saved, current, 2, requested_nonce=0)


def test_terminal_pair_is_atomic_and_take_lineage_strict(tmp_path):
    saved = custom_contract([5,5,5], terminal=True)
    source = fixtures._persist(tmp_path, saved, prefix=3, review_unit=None, updated_utc="2026-09-29T00:00:00Z")
    changed = custom_contract([4,5,5], terminal=True)
    controller = fixtures._controller(tmp_path, changed)
    entries, records = controller._compatible_plan_prefix(source.manifest, changed)
    assert not entries and not records
    with pytest.raises(storage.RunStorageError, match="lineage"):
        controller._valid_provenance_prefix(source.manifest, current_contract=changed, enforce_sampling_contract=True)


def test_uniform_contract_matches_pre_feature_source(monkeypatch):
    # Use the same observable objects for old/new contracts; no new uniform fields.
    module_name = "ComfyUI_H3_Continuum_Join._uniform_storage_baseline"
    old = ModuleType(module_name)
    old.__package__ = "ComfyUI_H3_Continuum_Join"
    monkeypatch.setitem(sys.modules, module_name, old)
    source = subprocess.check_output(["git", "show", "HEAD:run_storage.py"]).decode()
    exec(compile(source, "uniform_storage_baseline.py", "exec"), old.__dict__)
    objects = dict(model=contracts._Model(), clip=contracts._Clip(), video_vae=contracts._VideoVAE(), sampler=contracts._Sampler())
    new, _, _ = contracts._sampling_contract(**objects)
    monkeypatch.setattr(contracts, "build_sampling_contract", old.build_sampling_contract)
    baseline, _, _ = contracts._sampling_contract(**objects)
    assert new == baseline
    assert storage.revision_identity(new) == old.revision_identity(baseline)
    assert "chunk_timing" not in new


def test_real_sampling_contract_includes_resolved_custom_timing(monkeypatch):
    build = contracts.build_sampling_contract
    def custom(**kw):
        kw["prompt_plan"] = dict(kw["prompt_plan"], chunk_timing=timing_from_seconds([10,8,10]).contract())
        return build(**kw)
    monkeypatch.setattr(contracts, "build_sampling_contract", custom)
    contract, _, _ = contracts._sampling_contract()
    assert contract["chunk_timing"]["requested_frame_boundaries"] == [0,240,432,672]
    assert len(contract["chunk_timing_groups"]) == 3
