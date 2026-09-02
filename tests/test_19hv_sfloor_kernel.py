import sys, numpy as np, importlib.util
def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp); sys.modules[name] = m; sp.loader.exec_module(m); return m
sys.path.insert(0, "/tmp/pwF/src")
new = load("bp_new", "/tmp/pwF/src/routing_optimiser/s4_search/band_projection.py")
old = load("bp_old", "/tmp/ref/bp_head.py")

rng = np.random.default_rng(7)
P, K = 3, 8
nprofile = 2
# rows: profile 0 -> rows 0,1,2 ; profile 1 -> rows 3,4,5
nR = 6
propidx_c = np.array([0,1,2,3,4,5], np.int64)
pw_c   = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0])     # row 2 masked -> ef 0
base_c = np.array([0.5, 0.5, 0.0, 0.2, 0.0, 0.8])
mv_c   = np.array([0.4, 0.4, 0.4, 0.6, 0.6, 0.6])
vcpos_c= np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
profiles = np.array([0,1], np.int64); cstart = np.array([0,3], np.int64); ccnt = np.array([3,3], np.int64)
nC = 4
cap_rowc = np.array([0,1,3,5], np.int64); cap_band = np.array([0,0,1,1], np.int64)
cap_c = np.array([0,0,1,1], np.int64); cap_ctot = np.array([100.,200.,300.,400.]); cap_base = np.array([.5,.5,.2,.8])
nA = 3
pc_orgc = np.array([0,3,5], np.int64); pc_vc = np.array([1.,2.,3.]); pc_pool = np.array([10.,20.,30.])
pc_band = np.array([0,1,1], np.int64); pc_heldfac = np.array([.3,.4,.5])
pc_gc = np.array([0,1,1], np.int64); pc_gkc = np.array([0,1,1], np.int64)
vconst = np.zeros(2)
prop_raw = rng.random((P, K)) * np.array([1,1,1,1,1,1,0,0])
cap, nlane = 0.6, P
ef_c = (pw_c > 0).astype(float)

def buffers(lanes, B=2):
    return (np.zeros((P,B)), np.zeros((P,B)), np.zeros((lanes,nprofile)), np.zeros((lanes,nprofile)),
            np.zeros((lanes,nprofile)), np.zeros((lanes,nR)), np.zeros((lanes,nR)),
            np.zeros(lanes, np.int64), np.zeros((lanes, 2)))
A = (propidx_c, pw_c, base_c, mv_c, vcpos_c, profiles, cstart, ccnt,
     cap_rowc, cap_band, cap_c, cap_ctot, cap_base,
     pc_orgc, pc_vc, pc_pool, pc_band, pc_heldfac, pc_gc, pc_gkc, vconst)

# --- 1. OFF must be bit-identical to HEAD ---
b1 = buffers(P); b2 = buffers(P)
v_o, t_o = old._cb_kernel_impl(prop_raw, *A, cap, nlane, b1[0],b1[1],b1[2],b1[3],b1[4],b1[5],b1[6],b1[7],b1[8])
pshf = np.zeros((1,1))
v_n, t_n = new._cb_kernel_impl(prop_raw, *A, cap, nlane, b2[0],b2[1],b2[2],b2[3],b2[4],b2[5],b2[6],b2[7],b2[8],
                               ef_c, 0.0, 0, pshf)
print("OFF vamp bit-identical:", np.array_equal(v_o.view(np.int64), v_n.view(np.int64)))
print("OFF txn  bit-identical:", np.array_equal(t_o.view(np.int64), t_n.view(np.int64)))

# --- 2. ON must equal delivery's rule, hand-computed ---
EF = 0.25
b3 = buffers(P); pshf3 = np.zeros((P, nR))
v_f, t_f = new._cb_kernel_impl(prop_raw, *A, cap, nlane, b3[0],b3[1],b3[2],b3[3],b3[4],b3[5],b3[6],b3[7],b3[8],
                              ef_c, EF, 1, pshf3)
ok = True
for p in range(P):
    for (s,e) in ((0,3),(3,6)):
        pr = prop_raw[p, propidx_c[s:e]] * pw_c[s:e]
        ps = pr.sum()
        if ps <= 0:
            exp = base_c[s:e].copy()
        else:
            elig = (ef_c[s:e] > 0) & ((base_c[s:e] > 0) | (pr > 0))
            nef = elig.sum()
            flc = min(EF, 1.0/nef) if nef > 0 else 0.0
            sh = pr / ps
            sh = np.where(elig, np.maximum(sh, flc), sh)
            exp = sh / sh.sum() if sh.sum() > 0 else sh
        got = pshf3[p, s:e]
        if not np.allclose(got, exp, rtol=0, atol=1e-15):
            ok = False; print("  MISMATCH p", p, s, got, exp)
print("ON pshf == delivery's floor+renorm rule:", ok)
# and the VAMP path must be untouched by arming
print("ON vamp == OFF vamp (VAMP is unfloored):",
      np.array_equal(v_n.view(np.int64), v_f.view(np.int64)))
print("ON txn  != OFF txn (TXN moved):", not np.array_equal(t_n.view(np.int64), t_f.view(np.int64)))
