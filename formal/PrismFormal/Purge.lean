/-!
# I-1 — purge and embargo geometry

Model of the index arithmetic in
`prism.validation.walk_forward.PurgedWalkForward`: a purged training row's
entire forward-label window closes strictly before the test slice, and an
embargoed row is excluded from successor-fold training. What Lean checks here
is the *geometry* of the rules; that the Python splitter implements these
rules is pinned by `tests/test_walk_forward.py`.
-/

namespace PrismFormal

/-- A fold's test slice: the half-open index range `[testStart, testEnd)`. -/
structure Fold where
  testStart : Nat
  testEnd   : Nat

/-- The purge rule: train row `i` with forward-label horizon `h` survives only
if its label window `[i, i + h]` closes strictly before the test slice. -/
def purgedKeep (f : Fold) (h i : Nat) : Prop := i + h < f.testStart

instance (f : Fold) (h i : Nat) : Decidable (purgedKeep f h i) := by
  unfold purgedKeep
  infer_instance

/-- **No label-window overlap.** Every bar a purged training row's label can
touch lies strictly before the test slice. -/
theorem purge_label_disjoint (f : Fold) (h i j : Nat)
    (hk : purgedKeep f h i) (_hj : i ≤ j) (hj' : j ≤ i + h) :
    j < f.testStart := by
  unfold purgedKeep at hk
  omega

/-- In particular, the purged training row itself precedes the test slice:
train and test index sets are disjoint. -/
theorem purge_train_before_test (f : Fold) (h i : Nat)
    (hk : purgedKeep f h i) : i < f.testStart := by
  unfold purgedKeep at hk
  omega

/-- The embargo rule: after fold `f`, rows in `[testEnd, testEnd + e)` are
banned from later training. -/
def embargoed (f : Fold) (e i : Nat) : Prop :=
  f.testEnd ≤ i ∧ i < f.testEnd + e

/-- A successor fold's training keep-rule: passes its own purge and is not
embargoed by the predecessor. -/
def trainKeep (pred succ : Fold) (h e i : Nat) : Prop :=
  purgedKeep succ h i ∧ ¬ embargoed pred e i

/-- **Embargo exclusion.** A row kept for successor training lies entirely
outside the predecessor's embargo window. -/
theorem train_avoids_embargo (pred succ : Fold) (h e i : Nat)
    (hk : trainKeep pred succ h e i) :
    i < pred.testEnd ∨ pred.testEnd + e ≤ i := by
  have hne := hk.2
  unfold embargoed at hne
  omega

end PrismFormal
