"""19ij — the canonical blocked key, its equivalence proof, and all-or-nothing arming.

The rule must hold at FIVE water-fills or GA-fitness stops matching delivered. Two of them
cannot express a gatewayFid at all, so every site keys on (bank, vampMid, currency) instead
— which is an identity for (bank, gatewayFid) on the fids a run can route to. These checks
are what stop that identity being assumed: the FULL mid list is genuinely ambiguous (five
cross-brand collisions), and it is only scoping to the run's own gateways that makes it
true.
"""
import importlib.util, pathlib, sys, csv, io
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
_sp = importlib.util.spec_from_file_location(
    "bf", str(ROOT / "src/routing_optimiser/s4_search/blocked_fill.py"))
bf = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(bf)

FAIL=[]
def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ")+n+(("   "+d) if d else ""))
    if not ok: FAIL.append(n)

MID = list(csv.DictReader(io.open(ROOT/"data/mappings/Master_MID_List.csv", encoding="utf-8-sig")))

# ── the canonical key, on the REAL mid list ─────────────────────────────────────────────
pairs = {("403163", "woodforest-usd-tav"), ("473702", "adyen-usd-tav-avonline")}
keys = bf.canonical_keys(pairs, MID)
check("canonical_keys maps a blocked fid to (bank, vampMid, currency)",
      keys == {("403163", "woodforest - total av", "usd"),
               ("473702", "adyen_totalantivirusonline", "usd")}, str(sorted(keys)))

# UNSCOPED: the whole mid list IS ambiguous, and the report must say so rather than hide it.
rep_all = bf.equivalence_report(pairs, MID)
check("unscoped, the FULL mid list is ambiguous (5 cross-brand collisions) and it is reported",
      len(rep_all["ambiguous"]) == 5 and rep_all["scoped"] is False,
      f"{len(rep_all['ambiguous'])} of {rep_all['groups']} groups")
check("...but no blocked pair lands on one, so it is still safe",
      rep_all["safe"] is True)

# SCOPED to the run's own gateways — the only question that matters.
TAV = [r["gatewayFid"].strip().lower() for r in MID
       if str(r.get("IsActive", "")).strip().upper() == "TRUE"
       and str(r.get("brand", "")).strip().lower() in ("totalav", "total av")]
rep = bf.equivalence_report(pairs, MID, in_scope_fids=TAV)
check("scoped to the run's 37 TotalAV fids, the identity HOLDS — zero ambiguous groups",
      not rep["ambiguous"] and rep["scoped"] is True and rep["n_scope"] == len(TAV),
      f"{rep['groups']} groups over {rep['n_scope']} in-scope fid(s)")
check("and the cross-brand collisions really do vanish under scoping",
      len(rep_all["ambiguous"]) > 0 and len(rep["ambiguous"]) == 0)
check("and no blocked pair lands on an ambiguous group", not rep["ambiguous_hit"])
check("every blocked fid is mapped", not rep["unmapped"], str(rep["unmapped"]))
check("=> safe to arm on this mid list", rep["safe"] is True)

# an UNMAPPED pair must be caught, not silently dropped
rep2 = bf.equivalence_report(pairs | {("403163", "not-a-real-fid")}, MID)
check("an unmapped blocked fid makes it UNSAFE",
      rep2["safe"] is False and rep2["unmapped"] == [("403163", "not-a-real-fid")])

# a synthetic AMBIGUOUS mid list must be caught
FAKE = [{"gatewayFid": "gw-a", "vampMid": "VM", "currency": "usd", "IsActive": "TRUE"},
        {"gatewayFid": "gw-b", "vampMid": "VM", "currency": "usd", "IsActive": "TRUE"}]
rep3 = bf.equivalence_report({("b1", "gw-a")}, FAKE)
check("two active fids under one (vampMid,currency) makes it UNSAFE",
      rep3["safe"] is False and rep3["ambiguous"] == [("vm", "usd")]
      and rep3["ambiguous_hit"] == [("vm", "usd")])
check("an INACTIVE sibling does not create ambiguity",
      bf.equivalence_report({("b1", "gw-a")},
                            FAKE[:1] + [dict(FAKE[1], IsActive="FALSE")])["safe"] is True)

# ── arming is all-or-nothing ──────────────────────────────────────────────────────────
armed, msg = bf.arming_verdict(False)
check("unrequested => off, and says so", armed is False and "rule OFF" in msg)
armed, msg = bf.arming_verdict(True)
check("requested with NOTHING wired => REFUSED, not half-applied",
      armed is False and "REFUSED" in msg and "5 of 5" in msg)
for s in bf.SITES[:-1]:
    bf.register(s)
armed, msg = bf.arming_verdict(True)
check("requested with 4 of 5 wired => still REFUSED, and names the gap",
      armed is False and "1 of 5" in msg and "band_kernel_flat" in msg)
bf.register(bf.SITES[-1])
armed, msg = bf.arming_verdict(True)
check("all 5 wired => ARMED", armed is True and "RULE ON at all 5" in msg)
try:
    bf.register("not_a_site"); ok = False
except ValueError:
    ok = True
check("an unknown site cannot register itself into the gate", ok)

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
