"""W6 risk-profile draft surface + G6 research_paper bit-identity."""

from __future__ import annotations

from dataclasses import fields

import pytest

from prism.live.daily import DailyBookConfig
from prism.live.risk_profile import (
    CERTIFIED_B1_PAPER_CONFIG,
    PROFILE_FILE_KEYS,
    PROFILE_FILES,
    PROFILE_IDS,
    RATIFIED_DEGROSS_G,
    assert_declared_matches_resolved,
    assert_research_paper_bit_identity,
    book_config_matches_certified_paper,
    declared_profile,
    declared_profile_path,
    load_declared_profile,
    resolve_risk_profile,
    validate_profile_payload,
)


def test_unknown_profile_fails_loud():
    with pytest.raises(ValueError, match="unknown risk profile"):
        resolve_risk_profile("yolo")
    with pytest.raises(ValueError, match="unknown risk profile"):
        validate_profile_payload({"profile_id": "yolo"})


def test_research_paper_matches_certified_constants():
    profile = resolve_risk_profile("research_paper")
    assert profile.profile_id == "research_paper"
    assert book_config_matches_certified_paper(profile.book_config)
    assert_research_paper_bit_identity(profile.book_config)
    assert profile.hedge.de_gross_armed is False
    assert profile.hedge.crypto_book.enabled is False
    assert profile.hedge.trend_sleeve.enabled is False
    assert profile.hedge.equity_sleeve.enabled is True
    # Field-for-field equality with the pin constant (G6 surface).
    for f in fields(DailyBookConfig):
        assert getattr(profile.book_config, f.name) == getattr(CERTIFIED_B1_PAPER_CONFIG, f.name), f.name


def test_research_paper_rejects_de_gross_arm_and_gross_override():
    with pytest.raises(ValueError, match="de_gross"):
        resolve_risk_profile("research_paper", de_gross_armed=True)
    with pytest.raises(ValueError, match="max_gross"):
        resolve_risk_profile("research_paper", max_gross=0.5)


def test_profiles_may_only_tighten_gross():
    with pytest.raises(ValueError, match="tighten"):
        resolve_risk_profile("conservative", max_gross=1.5)
    c = resolve_risk_profile("conservative")
    assert c.book_config.max_gross == pytest.approx(0.5)
    assert c.book_config.max_gross <= CERTIFIED_B1_PAPER_CONFIG.max_gross
    b = resolve_risk_profile("balanced")
    assert b.book_config.max_gross == pytest.approx(1.0)
    a = resolve_risk_profile("assertive")
    assert a.hedge.crypto_book.enabled is False


def test_all_profile_ids_resolve_and_public_dict_is_draft():
    for pid in sorted(PROFILE_IDS):
        p = resolve_risk_profile(pid)
        assert p.profile_id == pid
        assert p.de_gross_g_ceiling == pytest.approx(0.5)
        public = p.to_public_dict()
        assert public["schema_status"] == "FROZEN"
        assert validate_profile_payload(public) == pid


def test_g6_assert_research_paper_bit_identity_fails_on_divergence():
    forked = DailyBookConfig(
        book="decile_neutral",
        decile=0.10,
        decision_every=21,
        max_gross=0.5,  # silent fork of the promotion instrument
        max_symbol_abs_weight=0.10,
        whole_shares=True,
    )
    with pytest.raises(AssertionError, match="research_paper"):
        assert_research_paper_bit_identity(forked)


# ---------------------------------------------------------------------------
# Declared profile files (§3.13): the four ids are the reference project's files
# ---------------------------------------------------------------------------


def test_every_id_names_a_file_and_the_file_agrees_with_what_the_code_resolves():
    # The registry stays the construction path and the file is the declaration of
    # record; this is what keeps the two from drifting apart silently.
    assert set(PROFILE_FILES) == set(PROFILE_IDS)
    for pid in sorted(PROFILE_IDS):
        declared = assert_declared_matches_resolved(pid)
        assert declared.name == pid
        assert declared.path.endswith(PROFILE_FILES[pid])
        assert resolve_risk_profile(pid).declared_file == PROFILE_FILES[pid]


def test_research_paper_maps_onto_the_existing_paper_file():
    # The old id stays accepted and names the file it maps to (Q7/Q19). paper.toml
    # already declared `name = "research_paper"`; a second file with that name would
    # give one profile two declarations.
    assert PROFILE_FILES["research_paper"] == "examples/reference/profiles/paper.toml"
    declared = declared_profile("research_paper")
    assert declared.limit_value("max_gross") == pytest.approx(CERTIFIED_B1_PAPER_CONFIG.max_gross)
    assert declared.limit_value("max_name_weight") == pytest.approx(CERTIFIED_B1_PAPER_CONFIG.max_symbol_abs_weight)
    # G6 bit-identity is unaffected by the move.
    assert_research_paper_bit_identity(resolve_risk_profile("research_paper").book_config)


def test_the_declared_files_carry_only_the_section_313_keys():
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for pid in sorted(PROFILE_IDS):
        payload = tomllib.loads((root / PROFILE_FILES[pid]).read_text(encoding="utf-8"))
        assert set(payload) <= PROFILE_FILE_KEYS, pid


def test_tighten_only_survives_the_move_to_files(tmp_path):
    # Gross above the certified pin, a per-name cap above it, and a regime-scale
    # floor above the ratified de-gross depth are each refused at load. The engine
    # holds no universal threshold; this is the reference project's rule, declared
    # in its files and enforced when they are read.
    def write(stem: str, body: str):
        path = tmp_path / f"{stem}.toml"
        path.write_text(body, encoding="utf-8")
        return path

    base = 'name = "conservative"\nmin_account_status = "undetermined"\n'
    loose_gross = write(
        "gross", base + '\n[[limits]]\nname = "max_gross"\nclass = "preference"\npriority = 1\nvalue = 1.5\n'
    )
    with pytest.raises(ValueError, match="tighten"):
        load_declared_profile(loose_gross)

    loose_name = write(
        "name",
        base + '\n[[limits]]\nname = "max_name_weight"\nclass = "preference"\npriority = 1\nvalue = 0.35\n',
    )
    with pytest.raises(ValueError, match="max_name_weight"):
        load_declared_profile(loose_name)

    loose_g = write("scale", base + "min_regime_scale = 0.9\n")
    with pytest.raises(ValueError, match="min_regime_scale"):
        load_declared_profile(loose_g)

    # A project with its own pins loads with the discipline off and enforces its own.
    assert load_declared_profile(loose_gross, tighten_only=False).limit_value("max_gross") == 1.5


def test_a_mis_spelled_key_is_refused_rather_than_silently_defaulted(tmp_path):
    path = tmp_path / "p.toml"
    path.write_text(
        'name = "conservative"\nmin_account_status = "undetermined"\nmin_regime_scal = 0.5\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unknown top-level keys"):
        load_declared_profile(path)


def test_a_capital_suitability_claim_is_not_inferred_by_the_loader(tmp_path):
    path = tmp_path / "p.toml"
    path.write_text('name = "conservative"\nmin_account_status = "measured"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="min_account_usd"):
        load_declared_profile(path)


def test_a_missing_file_names_the_id_and_the_path_it_looked_for(tmp_path):
    with pytest.raises(FileNotFoundError, match="conservative"):
        declared_profile_path("conservative", root=tmp_path)


def test_every_control_is_either_declared_or_reported_as_inexpressible():
    """No control disappears in the move to files.

    Each field of the two resolved objects is either something a declared file *can*
    say (a §3.7 limit or a §3.13 key) or is named in ``INEXPRESSIBLE_CONTROLS`` with
    the reason. A future field that is neither fails here rather than quietly ceasing
    to bind.
    """
    from dataclasses import fields as dataclass_fields

    from prism.live.risk_profile import HedgePolicy, inexpressible_controls

    reported = inexpressible_controls()
    expressible_book_fields = {
        "max_gross",  # the max_gross limit
        "max_symbol_abs_weight",  # the max_name_weight limit
        "max_participation",  # the participation_per_name limit
    }
    for f in dataclass_fields(DailyBookConfig):
        if f.name in expressible_book_fields:
            continue
        assert f"book_config.{f.name}" in reported, f.name
    for f in dataclass_fields(HedgePolicy):
        assert f"hedge.{f.name}" in reported, f.name
    # de_gross_g_ceiling is the one that *did* find a home: §3.7 puts the de-gross
    # bound on regime_scale, which §3.13 declares as the top-level min_regime_scale.
    assert declared_profile("conservative").min_regime_scale == pytest.approx(RATIFIED_DEGROSS_G)


def test_the_sleeve_mix_is_what_separates_balanced_from_assertive():
    # The two files are identical in every limit they can express, which is the
    # point of the report above: what distinguishes them is the sleeve mix, and no
    # limit class of §3.7 can carry it.
    balanced, assertive = declared_profile("balanced"), declared_profile("assertive")
    assert balanced.limit_value("max_gross") == assertive.limit_value("max_gross")
    assert balanced.limit_value("max_name_weight") == assertive.limit_value("max_name_weight")
    resolved_balanced = resolve_risk_profile("balanced").hedge
    resolved_assertive = resolve_risk_profile("assertive").hedge
    assert resolved_balanced.equity_sleeve != resolved_assertive.equity_sleeve


def test_declared_limits_load_through_the_section_37_loader():
    from prism.portfolio.limits import LimitClass, reduction_order

    declared = declared_profile("conservative")
    order = reduction_order(declared.limits)
    assert [limit.name for limit in order] == ["max_gross", "max_name_weight"]
    assert all(limit.cls is LimitClass.PREFERENCE for limit in order)
    assert all(limit.source.endswith("conservative.toml") for limit in declared.limits)
