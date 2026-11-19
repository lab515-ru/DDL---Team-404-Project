from types import SimpleNamespace as NS
import math
from domains import build_domains, MultiDomainBatchSampler

P = "/pscratch/sd/l/leana_ab/perlmutter-class/project/data"
a = NS(domains="all", mmp_cache=f"{P}/mmptrack/cache",
       wt_root=f"{P}/wildtrack/Wildtrack_dataset", wt_cache=f"{P}/wildtrack/wt_270x480.npy",
       mx_root=f"{P}/multiviewx/MultiviewX", mx_cache=f"{P}/multiviewx/mx_270x480.npy", dom_holdout=40)
doms = build_domains(a)
for d in doms:
    print(d.name, "views", d.views, "grid", d.grid, "geo", tuple(d.geo[0].shape),
          "train", len(d.train), "val", len(d.val), "test", None if d.test is None else len(d.test))
sizes = [len(d.train) for d in doms]
w = [math.sqrt(s) for s in sizes]
print("sampling share", [round(x / sum(w), 3) for x in w])
ss = [list(MultiDomainBatchSampler(sizes, w, 8, r, 4, 50, 0, 0)) for r in range(4)]
print("same domain on all ranks:", all(len({ss[r][s][0][0] for r in range(4)}) == 1 for s in range(50)))
print("indices disjoint across ranks:", all(len({i for r in range(4) for _, i in ss[r][s]}) == 32 for s in range(50)))
late = list(MultiDomainBatchSampler(sizes, w, 8, 1, 4, 50, 20, 0))
print("resume matches:", late == ss[1][20:])
x, y, d0 = __import__("domains").MultiDS(doms)[(0, 0)]
print("sample", tuple(x.shape), tuple(y.shape), d0)
