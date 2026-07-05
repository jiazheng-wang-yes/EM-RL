namespace LeanProverV1Smoke

theorem smoke_true : True := by
  exact True.intro

theorem smoke_and_intro : And True True := by
  exact And.intro True.intro True.intro

theorem smoke_prop_id (p : Prop) (hp : p) : p := by
  exact hp

theorem smoke_and_comm (p q : Prop) (hp : p) (hq : q) : And q p := by
  exact And.intro hq hp

theorem smoke_nat_refl (n : Nat) : n = n := by
  rfl

end LeanProverV1Smoke
