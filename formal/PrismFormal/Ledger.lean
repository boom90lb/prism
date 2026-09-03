/-!
# N4 — the ledger conserves capital

Model: exact integer arithmetic (micro-dollars / micro-shares). The Python
implementation (`src/prism/execution/target_weights.py`) computes in float64;
these theorems machine-check the *accounting algebra* itself. The bridge to
the float implementation is the property test
(`tests/test_ledger_conservation.py`), which asserts the float ledger tracks
this algebra within tolerance. Division of labor: Lean proves the algorithm
cannot create or destroy cash; pytest checks the float code implements the
algorithm.
-/

namespace PrismFormal

/-- Dot product over integer lists (position quantities × prices). Ragged
tails are ignored — the panel contract upstream guarantees equal length. -/
def dot : List Int → List Int → Int
  | [], _ => 0
  | _, [] => 0
  | a :: as, b :: bs => a * b + dot as bs

/-- A book: cash plus per-name signed position quantities. -/
structure Book where
  cash : Int
  qty  : List Int

/-- Mark-to-market wealth at `prices`. -/
def wealth (b : Book) (prices : List Int) : Int :=
  b.cash + dot b.qty prices

/-- Rebalance to `newQty` at `prices`, paying `costs`: cash absorbs the traded
notional and the charged costs; nothing else moves. -/
def rebalance (b : Book) (newQty prices : List Int) (costs : Int) : Book :=
  { cash := b.cash - (dot newQty prices - dot b.qty prices) - costs
    qty  := newQty }

/-- **N4, single step.** A rebalance at fixed prices changes wealth by exactly
the charged costs — trading itself creates and destroys no cash. -/
theorem rebalance_conserves (b : Book) (newQty prices : List Int) (costs : Int) :
    wealth (rebalance b newQty prices costs) prices = wealth b prices - costs := by
  simp only [wealth, rebalance]
  omega

/-- Between rebalances, wealth moves only by position × price change: the
mark-to-market PnL is attributable entirely to held quantities. -/
theorem mark_to_market_pnl (b : Book) (p p' : List Int) :
    wealth b p' - wealth b p = dot b.qty p' - dot b.qty p := by
  simp only [wealth]
  omega

/-- One accounting period: new prices arrive, the book rebalances at them. -/
structure Period where
  prices : List Int
  newQty : List Int
  costs  : Int

/-- Run the book through a sequence of periods; returns the final book and
the final prices it is marked at. -/
def run : Book → List Int → List Period → Book × List Int
  | b, p, [] => (b, p)
  | b, _, per :: rest =>
      run (rebalance b per.newQty per.prices per.costs) per.prices rest

/-- Total mark-to-market PnL accrued across the periods. -/
def totalPnl : Book → List Int → List Period → Int
  | _, _, [] => 0
  | b, p, per :: rest =>
      (dot b.qty per.prices - dot b.qty p)
        + totalPnl (rebalance b per.newQty per.prices per.costs) per.prices rest

/-- Total costs charged across the periods. -/
def totalCosts : List Period → Int
  | [] => 0
  | per :: rest => per.costs + totalCosts rest

/-- **N4, multi-step.** Over any period sequence, final wealth equals initial
wealth plus accrued PnL minus charged costs — equity moves by realized PnL
minus costs and by nothing else, across arbitrarily many rebalances. -/
theorem run_conserves (ps : List Period) (b : Book) (p : List Int) :
    wealth (run b p ps).1 (run b p ps).2
      = wealth b p + totalPnl b p ps - totalCosts ps := by
  induction ps generalizing b p with
  | nil => simp [run, totalPnl, totalCosts]
  | cons per rest ih =>
      simp only [run, totalPnl, totalCosts]
      rw [ih]
      simp only [wealth, rebalance]
      omega

end PrismFormal
