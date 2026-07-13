import Mathlib.Data.Bool.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Option.Basic
import Mathlib.Data.Set.Basic

open Lean Meta

private def allowedNamePrefix (name : Name) : Bool :=
  let text := name.toString
  #["Nat.", "Int.", "List.", "Finset.", "Set.", "Bool.", "Option.", "Function.",
    "Equiv.", "Multiset.", "Array.", "Map.", "Rat."].any (fun candidate => text.startsWith candidate)

private def declarationModule? (env : Environment) (name : Name) : Option Name := do
  let moduleIdx <- env.getModuleIdxFor? name
  env.allImportedModuleNames[moduleIdx.toNat]?

private def readLimit : IO Nat := do
  let value := (← IO.getEnv "RLLM_MATHLIB_EXPORT_MAX").getD "4096"
  return value.toNat?.getD 4096

run_cmd do
  let outputPath := (← IO.getEnv "RLLM_MATHLIB_EXPORT_OUTPUT").getD "mathlib_declarations.jsonl"
  let limit ← readLimit
  let rows : Array Json ← Lean.Elab.Command.liftCoreM <| MetaM.run' do
    let env ← getEnv
    let mut declarations : Array (Name × ConstantInfo × Name) := #[]
    for (name, info) in env.constants.map₁ do
      if name.isInternalDetail || name.isAnonymous || !allowedNamePrefix name then
        continue
      let some moduleName := declarationModule? env name | continue
      unless moduleName.toString.startsWith "Mathlib." do
        continue
      match info with
      | .thmInfo _ => declarations := declarations.push (name, info, moduleName)
      | _ => pure ()

    let sortedDeclarations := declarations.qsort fun left right => Name.quickLt left.1 right.1
    let mut rows : Array Json := #[]
    for (name, info, moduleName) in sortedDeclarations do
      if rows.size >= limit then
        break
      let typeFormat ← ppExpr info.type
      let typeText := typeFormat.pretty
      if typeText.length < 12 || typeText.length > 1200 then
        continue
      rows := rows.push <| Json.mkObj [
        ("name", toJson name.toString),
        ("module", toJson moduleName.toString),
        ("type", toJson typeText)
      ]
    return rows

  let lines := (Array.toList rows).map Json.compress
  IO.FS.writeFile outputPath (String.intercalate "\n" lines ++ "\n")
  logInfo m!"Exported {rows.size} Mathlib theorem declarations to {outputPath}"
