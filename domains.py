import glob, os, random
import torch
from torch.utils.data import Dataset, Subset

IMG, FEAT = (270, 480), (16, 30)
SCENES = ["cafe_shop", "industry_safety", "lobby", "office", "retail"]


class Domain:
    def __init__(self, name, views, grid, geo, train, val, test):
        self.name, self.views, self.grid, self.geo = name, views, grid, geo
        self.train, self.val, self.test = train, val, test


def _mmp(a, scene):
    from mmptrack import MMPDataset, load_scene, make_geometry_mmp
    root = a.mmp_cache
    _, cams, grid = load_scene(root, scene)

    def stems(split):
        return sorted(os.path.basename(f)[:-5] for f in glob.glob(f"{root}/{split}__*__{scene}_*.json"))

    tr, va = stems("train"), stems("validation")
    mk = lambda s: MMPDataset(root, scene, s, IMG) if s else None
    geo = make_geometry_mmp(root, scene, IMG, grid, FEAT)
    # validation recordings split in half by recording: even -> val, odd -> test
    return Domain(f"mmp_{scene}", len(cams), grid, geo, mk(tr), mk(va[::2]), mk(va[1::2]))


def _frames(full, hold, gap=5):
    n = len(full)
    return Subset(full, range(0, n - hold - gap)), Subset(full, range(n - hold, n))


def _wt(a):
    from wildtrack import WildtrackDataset
    from geometry import make_geometry
    g = (360, 120)
    full = WildtrackDataset(a.wt_root, IMG, g, cache=a.wt_cache)
    tr, va = _frames(full, a.dom_holdout)
    return Domain("wildtrack", 7, g, make_geometry("wildtrack", a.wt_root, IMG, g, FEAT), tr, va, None)


def _mx(a):
    from multiviewx import MultiviewXDataset
    from geometry import make_geometry
    g = (160, 250)
    full = MultiviewXDataset(a.mx_root, IMG, g, cache=a.mx_cache)
    tr, va = _frames(full, a.dom_holdout)
    return Domain("multiviewx", 6, g, make_geometry("multiviewx", a.mx_root, IMG, g, FEAT), tr, va, None)


def build_domains(a):
    names = SCENES and (a.domains.split(",") if a.domains != "all" else
                        [f"mmp:{s}" for s in SCENES] + ["wildtrack", "multiviewx"])
    out = []
    for n in names:
        out.append(_mmp(a, n.split(":")[1]) if n.startswith("mmp:") else _wt(a) if n == "wildtrack" else _mx(a))
    return out


class MultiDS(Dataset):
    """Index is (domain, i). Returns views, target, domain id."""
    def __init__(self, doms):
        self.parts = [d.train for d in doms]

    def __len__(self):
        return sum(len(p) for p in self.parts)

    def __getitem__(self, k):
        d, i = k
        v, t = self.parts[d][i]
        return v, t, d


class MultiDomainBatchSampler:
    """Same domain sequence on every rank (seeded); each rank takes its own slice of the indices."""
    def __init__(self, sizes, weights, bs, rank, world, steps, start, seed):
        self.sizes, self.weights, self.bs = sizes, weights, bs
        self.rank, self.world, self.steps, self.start, self.seed = rank, world, steps, start, seed

    def __len__(self):
        return max(self.steps - self.start, 0)

    def __iter__(self):
        rng = random.Random(self.seed)
        bufs = [[] for _ in self.sizes]
        need = self.bs * self.world
        for s in range(self.steps):
            d = rng.choices(range(len(self.sizes)), weights=self.weights)[0]
            while len(bufs[d]) < need:
                p = list(range(self.sizes[d]))
                rng.shuffle(p)
                bufs[d] += p
            idx, bufs[d] = bufs[d][:need], bufs[d][need:]
            if s >= self.start:
                yield [(d, i) for i in idx[self.rank::self.world]]
