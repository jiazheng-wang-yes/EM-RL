import LeanProverV1Smoke.Basic

namespace LeanProverV1Smoke

theorem smoke_import_true : True := by
  exact smoke_true

theorem smoke_import_prop (p : Prop) (hp : p) : p := by
  exact smoke_prop_id p hp

end LeanProverV1Smoke
