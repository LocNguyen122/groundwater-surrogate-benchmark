"""Independence of Pool T fields from the original corpus and between Pool T cells (deviation D6).

Reports: max |r| of each Pool T field against original fields of IDENTICAL covariance (a shared random stream gives
1.000), the largest |r| against any original field, and the twin structure inside Pool T from the QA fields table.
r = Pearson correlation of centred natural-log K over the full grid. Usage:
  python pool_independence_v7.py <pool_T FLIPPED root> <master FLIPPED root> <fresh registry csv>
         <registry_v7 csv> <qa pool_T_fields.csv> <out.json>
"""
import csv, json, os, sys, collections
import numpy as np

ROOT_T, ROOT_M, FRESH, REG7, QAF, OUT = sys.argv[1:7]


def u(f):
    z = np.log(np.load(f)["K"].astype(np.float64)).ravel()
    z -= z.mean()
    return z / np.linalg.norm(z)


sim = list(csv.DictReader(open(REG7)))
mf = collections.defaultdict(list)
allm = []
for r in sim:
    if r["long_dispersivity"] != "0.62" or r["trans_ratio"] != "0.1":
        continue
    for q in range(1, 6):
        f = f"{ROOT_M}/{r['param_folder']}/real_{q:03d}.npz"
        if os.path.exists(f):
            v = u(f)
            mf[(float(r["correlation_length"]), float(r["anisotropy"]))].append(v)
            allm.append(v)
M = np.stack(allm)
reg = [r for r in csv.DictReader(open(FRESH)) if r["pool"] == "T" and r["transport_id"] == "0"]
eq, anyr = [], []
for r in reg:
    for q in range(1, 6):
        f = f"{ROOT_T}/param_{r['param_id']}/real_{q:03d}.npz"
        if os.path.exists(f):
            v = u(f)
            eq.append(max(abs(float(v @ m)) for m in mf[(float(r["lambda"]), float(r["anisotropy"]))]))
            anyr.append(float(np.abs(M @ v).max()))
qa = list(csv.DictReader(open(QAF)))
out = {"n_fields": len(eq), "max_abs_r_vs_equal_covariance_master": max(eq),
       "max_abs_r_vs_any_master": max(anyr), "median_of_per_field_max_vs_any_master": float(np.median(anyr)),
       "n_master_fields_compared": int(M.shape[0]),
       "pool_T_fields_with_twin_inside_pool": sum(float(x["max_abs_r_other_new_cells"]) > 0.5 for x in qa)}
json.dump(out, open(OUT, "w"), indent=1)
print(out)
