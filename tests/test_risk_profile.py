"""W6 risk-profile draft surface + G6 research_paper bit-identity."""

from __future__ import annotations

from dataclasses import fields

import pytest

from prism.live.daily import DailyBookConfig
from prism.live.risk_profile import (
    CERTIFIED_B1_PAPER_CONFIG,
    PROFILE_IDS,
    assert_research_paper_bit_identity,
    book_config_matches_certified_paper,
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
        assert getattr(profile.book_config, f.name) == getattr(
            CERTIFIED_B1_PAPER_CONFIG, f.name
        ), f.name


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
