/-!
# §7.4 — the participation gate

Model of `prism.execution.participation`: a per-name-day trade delta capped
to a participation limit. The gate must be a pure attenuator — never exceed
the cap, never exceed the requested trade, never flip its direction.
-/

namespace PrismFormal

def capTrade (delta : Int) (cap : Nat) : Int :=
  if delta.natAbs ≤ cap then delta
  else if 0 ≤ delta then (cap : Int) else -(cap : Int)

/-- The gated trade never exceeds the cap. -/
theorem capTrade_le_cap (delta : Int) (cap : Nat) :
    (capTrade delta cap).natAbs ≤ cap := by
  unfold capTrade
  split
  · omega
  · split <;> omega

/-- Down-only: gating never increases the requested trade size. -/
theorem capTrade_down_only (delta : Int) (cap : Nat) :
    (capTrade delta cap).natAbs ≤ delta.natAbs := by
  unfold capTrade
  split
  · omega
  · split <;> omega

/-- Direction-preserving, buy side: a non-negative delta stays non-negative. -/
theorem capTrade_sign_nonneg (delta : Int) (cap : Nat) (h : 0 ≤ delta) :
    0 ≤ capTrade delta cap := by
  unfold capTrade
  split
  · exact h
  · first
    | (split <;> omega)
    | omega

/-- Direction-preserving, sell side: a non-positive delta stays non-positive. -/
theorem capTrade_sign_nonpos (delta : Int) (cap : Nat) (h : delta ≤ 0) :
    capTrade delta cap ≤ 0 := by
  unfold capTrade
  split
  · exact h
  · first
    | (split <;> omega)
    | omega

end PrismFormal
