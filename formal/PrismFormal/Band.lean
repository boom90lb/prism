/-!
# §7.3 — the online no-trade band

Model of `prism.portfolio.construct.step_no_trade_band`: hold the previous
weight when the proposed move is within the band, else move to the target.
Integer micro-weights; `band` is a non-negative width.

The last theorem formally pins the batch-replay defect: folding the band from
`held = 0` (what a batch `apply_no_trade_band` does at the head of every
window) is **not** the online evolution from the true held state — witnessed
concretely, checked by `decide`. This is the reason the online form exists
and why the batch form must never be used to continue a live book.
-/

namespace PrismFormal

def stepBand (prev target : Int) (band : Nat) : Int :=
  if (target - prev).natAbs ≤ band then prev else target

/-- The band never invents a third value: hold or move, nothing else. -/
theorem stepBand_cases (prev target : Int) (band : Nat) :
    stepBand prev target band = prev ∨ stepBand prev target band = target := by
  unfold stepBand
  split
  · exact Or.inl rfl
  · exact Or.inr rfl

/-- Re-applying the band against the same target is a no-op: hysteresis is
stable, no oscillation from repeated evaluation at one decision point. -/
theorem stepBand_idem (prev target : Int) (band : Nat) :
    stepBand (stepBand prev target band) target band = stepBand prev target band := by
  by_cases h : (target - prev).natAbs ≤ band
  · simp [stepBand, h]
  · simp [stepBand, h]

/-- A trade happens only strictly past the band width. -/
theorem stepBand_trades_only_past_band (prev target : Int) (band : Nat)
    (h : stepBand prev target band ≠ prev) :
    band < (target - prev).natAbs := by
  unfold stepBand at h
  by_cases hc : (target - prev).natAbs ≤ band
  · simp [hc] at h
  · omega

/-- Online evolution: fold the band step through a target sequence. -/
def onlineFold (held : Int) (band : Nat) : List Int → Int
  | [] => held
  | t :: ts => onlineFold (stepBand held t band) band ts

/-- **The batch-replay defect, formally.** Replaying the band from `held = 0`
is not the online evolution from the true held state. Concrete witness:
held 5, band 3, one target 4 — online holds 5 (move of 1 is inside the band),
batch-from-zero trades to 4 (move of 4 is outside). -/
theorem batch_replay_from_zero_diverges :
    ∃ (held : Int) (band : Nat) (targets : List Int),
      onlineFold held band targets ≠ onlineFold 0 band targets := by
  refine ⟨5, 3, [4], ?_⟩
  decide

end PrismFormal
