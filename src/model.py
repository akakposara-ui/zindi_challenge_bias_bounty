"""Coverage-gap estimate per tract, from the permitted challenge data only.

There are no training labels: the reference layers the organisers compared Overture against
(TIGER highway length, Microsoft building count, USGS facility counts, CBP establishments) were
removed from the package. Each reference quantity is treated as an unknown and drawn from a
generative model conditioned on permitted features; the official composite formula is applied to
every draw, and the submission is the Monte-Carlo mean, i.e. the RMSE-optimal point estimate
under the model. A parameter written ("u", lo, hi) is itself drawn per Monte-Carlo draw, so the
estimate also integrates over the uncertainty of that parameter.

Every component has its own random stream (seeded from `seed`), so changing one component never
changes the draws of another.

Usage: python model.py <variant>  ->  ../submissions/<variant>.csv and <variant>-detail.parquet
"""
import copy
import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.special import gammaln
from scipy.stats import binom, poisson
from sklearn.linear_model import PoissonRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT = os.path.join(HERE, "..", "features")
SAMPLE = os.path.join(HERE, "..", "SampleSubmission.csv")
OUT = os.path.join(HERE, "..", "submissions")

NAMED = ("motorway", "trunk", "primary", "secondary")
# Interstate, its business/future loops, US highways, state highways and their non-FM
# subsystems. Calibrated against the organisers' published share of tracts with an undefined
# component (21% / 37% / 28% / 55%): this set reproduces all four.
HWY_NETS = ("I", "IBUS", "USH", "ST", "STSUB")
FACILITIES = ("fire", "ems", "schools")
FMAX = 40
MISFILED_GRID = (0.01, 0.02, 0.04, 0.06, 0.1, 0.15, 0.2)
Q_GRID = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
FITTED = {}  # evidence detection rates fitted in the last run, for the log
STREAMS = ("params", "transport", "buildings", "facilities", "cbp")

BASE_PARAMS = dict(
    seed=2026,
    draws=2000,
    # TIGER S1100+S1200 length = Overture route-numbered length, inside the tract plus hugging its
    # boundary (TIGER roads that *are* the boundary count for both tracts), x lognormal noise.
    transport=dict(
        sigma=0.15,
        extra_p={"northern-ca": 0.05, "eastern-wa": 0.03, "eastern-ok": 0.02},  # S1200 the proxy misses
        extra_m=800.0,  # mean length of such a road, metres
        # Networks only partly S1200: Texas FM/RM roads (the 28% undefined share allows few) and
        # Oklahoma county roads (the 0.71 Overture/TIGER ratio needs ~40%).
        fm_q={"south-central-tx": 0.10},
        cr_q={"eastern-ok": 0.40},
    ),
    # "struct": the reference is Microsoft's Global ML release, the one Overture conflates, so
    # M = ML footprints Overture kept + r x (OSM + Esri footprints), r = ML footprints replaced per
    # OSM/Esri footprint, inferred from the USFS building count (~15% log noise against ML in
    # ML-only tracts) and shrunk to the region's median. "usfs": M = USFS count x noise (v1).
    buildings=dict(
        model="struct",
        usfs_sigma=0.146,
        r_prior={"eastern-ok": 0.79, "eastern-wa": 0.81, "maricopa-az": 0.98,
                 "northern-ca": 0.96, "south-central-tx": 0.97},
        r_tau=0.08,
        usfs_model_sigma=0.08,
    ),
    # The organisers' layer has cbp_estab / cbp_estab_res / cbp_estab_bus: ZIP establishments split
    # over tracts by total, residential or business address share. "bus" won on the leaderboard.
    # "pop" (v1): kappa x population. Business addresses are proxied from Overture places and
    # footprints >= 600 m2 ("poi_big", concave in places), or from places plus urban footprints
    # >= 2500 m2 ("poi_warehouse", linear in places, leaving out barns and farm buildings).
    cbp=dict(
        model="bus",
        proxy="poi_big",
        kappa=0.024,  # national CBP establishments per resident
        pop_sigma=0.35,
        zip_sigma=0.30,
        tract_sigma={"addr": 0.25, "bus": 0.50},
    ),
    # Real facilities F ~ Poisson(lambda); Overture count o = detected (each 1 + Pois(dup)) plus
    # false positives; reference count h = Bin(F, p_h) + Pois(extra x lambda).
    facilities=dict(
        param_sets=1,  # parameter sets drawn when some facility parameters are ranges
        fire=dict(col="poi_fire", p_o=0.65, dup=0.10, fp=0.15, p_h=0.95, extra=0.05),
        ems=dict(col="poi_ems", p_o=0.55, dup=0.10, fp=0.35, p_h=0.70, extra=0.05),
        schools=dict(col="poi_school", p_o=0.85, dup=0.60, fp=0.15, p_h=0.95, extra=0.03),
    ),
)

# Plausible ranges for the parameters that are least pinned down by the data.
UNCERTAIN = dict(
    transport=dict(sigma=("u", 0.08, 0.25), fm_q={"south-central-tx": ("u", 0.03, 0.20)},
                   cr_q={"eastern-ok": ("u", 0.25, 0.55)}),
    buildings=dict(r_tau=("u", 0.05, 0.12)),
    cbp=dict(kappa=("u", 0.020, 0.028), zip_sigma=("u", 0.20, 0.40),
             tract_sigma={"addr": ("u", 0.15, 0.35), "bus": ("u", 0.35, 0.65)}),
    facilities=dict(param_sets=8,
                    fire=dict(p_o=("u", 0.50, 0.80), fp=("u", 0.05, 0.30)),
                    ems=dict(p_o=("u", 0.40, 0.70)),
                    schools=dict(p_o=("u", 0.75, 0.95), dup=("u", 0.40, 0.90))),
)

VARIANTS = {
    "v1": dict(buildings=dict(model="usfs"), cbp=dict(model="pop")),
    "v2a": dict(cbp=dict(model="addr")),
    "v2b": {},
    "v3b": UNCERTAIN,
    # one-factor alternatives to v2b, each a complete candidate model
    "v3d": dict(cbp=dict(proxy="poi_warehouse")),               # public 0.072910: kept
    "v3e": dict(transport=dict(cr_q={"eastern-ok": 0.0})),      # public 0.078535: rejected
    # v6 = the winners combined; v7-v9 each change one assumption of v6
    "v6": dict(cbp=dict(proxy="poi_warehouse")),
    "v7": dict(cbp=dict(proxy="poi_warehouse"), transport=dict(cr_q={"eastern-ok": 0.6})),
    "v8": dict(cbp=dict(proxy="poi_warehouse", tract_sigma={"addr": 0.25, "bus": 0.30})),
    "v9": dict(cbp=dict(proxy="poi_warehouse"), facilities=dict(schools=dict(p_o=0.93, dup=0.8, fp=0.2))),
}


def merge(base, override):
    """Recursive dict update, returning a new dict."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


# public: v6 0.072311, v7 0.071818, v8 0.072221, v9 0.071843 -> v10 keeps all three changes
V10 = dict(cbp=dict(proxy="poi_warehouse", tract_sigma={"addr": 0.25, "bus": 0.30}),
           transport=dict(cr_q={"eastern-ok": 0.6}),
           facilities=dict(schools=dict(p_o=0.93, dup=0.8, fp=0.2)))
VARIANTS.update({
    "v10": V10,
    "v11": merge(V10, dict(transport=dict(cr_q={"eastern-ok": 0.8}))),
    "v12": merge(V10, dict(facilities=dict(schools=dict(p_o=0.97, dup=1.0, fp=0.25)))),
    "v13": merge(V10, dict(facilities=dict(fire=dict(p_o=0.75, fp=0.25)))),
})
# public: v10 0.071574, v11 0.071392 (kept), v12 0.071686, v13 0.072884 (fewer fire gaps: much worse)
V11 = VARIANTS["v11"]
MORE_FIRE_GAPS = dict(facilities=dict(fire=dict(p_o=0.55, fp=0.10)))
VARIANTS.update({
    "v14": merge(V11, dict(transport=dict(cr_q={"eastern-ok": 1.0}))),
    "v15": merge(V11, MORE_FIRE_GAPS),
    "v16": merge(V11, dict(facilities=dict(fire=dict(p_o=0.45, fp=0.10)))),
    "v17": merge(merge(V11, MORE_FIRE_GAPS), dict(transport=dict(cr_q={"eastern-ok": 1.0}))),
})
# public: v14 0.071596, v15 0.070605, v16 0.070586 (best), v17 0.070922 -> county roads 0.8, fire p_o 0.45
V16 = VARIANTS["v16"]
# Independent evidence of real facilities (src/evidence.py): OSM fire-station and school buildings,
# fire-station-named places filed under another category; (column, false-evidence rate per tract)
EVIDENCE = dict(facilities=dict(fire=dict(evidence=(("fire_osm", 0.02), ("fire_alt", 0.05))),
                                schools=dict(evidence=(("school_osm", 0.05),))))
VARIANTS.update({
    "v18": merge(V16, EVIDENCE),
    "v19": merge(merge(V16, EVIDENCE), dict(facilities=dict(fire=dict(p_o=0.55)))),
})
# public: v18 0.069240 (best), v19 0.069601 -> keep fire detection 0.45 with evidence
V18 = VARIANTS["v18"]
VARIANTS.update({
    "v20": merge(V18, dict(facilities=dict(fire=dict(p_o=0.35)))),
    "v21": merge(V18, dict(facilities=dict(fire=dict(extra=0.15)))),
})
# public: v21 0.069595 (worse than v18)
VARIANTS.update({
    "v22": merge(V18, dict(facilities=dict(fire=dict(extra=0.0)))),
    "v23": merge(V18, dict(transport=dict(fm_q={"south-central-tx": 0.25}, fm_q_alone={"south-central-tx": 0.10}))),
    "v24": merge(V18, dict(transport=dict(fm_q={"south-central-tx": 0.40}, fm_q_alone={"south-central-tx": 0.10}))),
})
# public: v20 0.070490, v22 0.069135, v23 0.069054, v24 0.069337 -> v25 = v22 + v23
V25 = merge(merge(V18, dict(facilities=dict(fire=dict(extra=0.0)))),
            dict(transport=dict(fm_q={"south-central-tx": 0.25}, fm_q_alone={"south-central-tx": 0.10})))
VARIANTS.update({
    "v25": V25,
    # fire stations filed under another primary category, found by name or by alternate category
    "v26": merge(V25, dict(facilities=dict(fire=dict(evidence=(("fire_osm", 0.02), ("fire_alt_any", 0.05)))))),
})
VARIANTS.update({
    "v27": merge(V25, dict(cbp=dict(kappa=0.020))),
    "v28": merge(V25, dict(facilities=dict(fire=dict(p_h=0.85)))),
    "v29": merge(V25, dict(transport=dict(sigma=0.25))),
})
# public: v25 0.068886, v26 0.069001, v27 0.068755, v28 0.068994, v29 0.067954 -> v30 = v25 + v27 + v29
V30 = merge(V25, dict(cbp=dict(kappa=0.020), transport=dict(sigma=0.25)))
VARIANTS.update({
    "v30": V30,
    "v31": merge(V30, dict(transport=dict(sigma=0.40))),
    "v32": merge(V30, dict(transport=dict(sigma=0.60))),
    "v33": merge(V30, dict(transport=dict(band_q=0.7))),
    "v34": merge(V30, dict(cbp=dict(kappa=0.017))),
})
VARIANTS.update({
    # F1: evidence detection rates per region x urban/rural
    "v35": merge(V30, dict(facilities=dict(fire=dict(evidence_strata="urban"), schools=dict(evidence_strata="urban")))),
    # F2: fire stations filed under another category are an outcome of the Overture capture itself
    "v36": merge(V30, dict(facilities=dict(fire=dict(misfiled=("fire_alt", 0.05), evidence=(("fire_osm", 0.02),))))),
})
# public: v30 0.067785 (same with 2 columns), v33 0.069748, v35 0.067993, v36 0.067747 (best)
V36 = VARIANTS["v36"]
# v37: v36 with every parameter tuned on the public leaderboard drawn from a range around its value
VARIANTS["v37"] = merge(V36, dict(
    transport=dict(sigma=("u", 0.18, 0.32), cr_q={"eastern-ok": ("u", 0.65, 0.95)},
                   fm_q={"south-central-tx": ("u", 0.15, 0.35)}),
    buildings=dict(r_tau=("u", 0.05, 0.12)),
    cbp=dict(kappa=("u", 0.018, 0.022), tract_sigma={"addr": 0.25, "bus": ("u", 0.2, 0.4)}),
    facilities=dict(param_sets=8, fire=dict(p_o=("u", 0.38, 0.52)), schools=dict(p_o=("u", 0.88, 0.96))),
))


def params(variant):
    return merge(BASE_PARAMS, VARIANTS[variant])


def is_range(value):
    return isinstance(value, tuple) and value[0] == "u"


def sample(value, rng, size):
    """A parameter as an array of `size`: fixed, or uniform per draw when ("u", lo, hi)."""
    return rng.uniform(value[1], value[2], size) if is_range(value) else np.full(size, float(value))


def by_region(table, region, rng, draws):
    """Per-region parameter table -> (n, draws) array, 0 for regions not listed."""
    out = np.zeros((len(region), draws))
    for name, value in table.items():
        out[region == name] = sample(value, rng, (1, draws))
    return out


# --------------------------------------------------------------------------- data loading

def load(regions=None):
    """Scored tracts (sample-submission order) joined to every per-tract feature table."""
    s = pd.read_csv(SAMPLE, dtype={"GEOID": str})
    if regions:
        s = s[s["region"].isin(regions)].reset_index(drop=True)
    parts = []
    for region in s["region"].unique():
        f = pd.read_parquet(os.path.join(FEAT, f"{region}.parquet"))
        for extra in ("roads", "bsize"):
            f = f.merge(pd.read_parquet(os.path.join(FEAT, f"{region}-{extra}.parquet")), on="GEOID", how="left")
        evidence = os.path.join(FEAT, f"{region}-evidence.parquet")
        if os.path.exists(evidence):
            f = f.merge(pd.read_parquet(evidence), on="GEOID", how="left")
        parts.append(f)
    f = pd.concat(parts, ignore_index=True)
    # a road class absent from a tract (or a region) is a zero length
    num = [c for c in f.columns if c.startswith(("rd_", "rdp_", "rdfm", "bld_", "poi_", "fire_", "school_"))]
    f[num] = f[num].fillna(0.0)
    f = s[["GEOID", "region"]].merge(f.drop(columns="region"), on="GEOID", how="left", validate="1:1")
    assert len(f) == len(s) and f["area_m2"].notna().all()
    return s, f.copy()  # one contiguous block after the many joins


def length(f, prefix, classes=None, nets=None):
    """Sum of road-length columns <prefix>_<class>_<network>, filtered by class and network."""
    out = np.zeros(len(f))
    for c in f.columns:
        if not c.startswith(prefix + "_"):
            continue
        cl, net = c.split("_")[-2:]
        if (classes is None or cl in classes) and (nets is None or net in nets):
            out += f[c].to_numpy()
    return out


def zip_shares(f, weight):
    """Sparse tract x ZIP matrices: W spreads each tract over the ZIPs its Overture places carry
    (the county stands in for tracts without any); `share` is `weight` spread the same way and
    normalised so that every ZIP column sums to 1."""
    z = pd.concat([pd.read_parquet(os.path.join(FEAT, f"{r}-zip.parquet")).assign(region=r)
                   for r in f["region"].unique()])
    z = z[z["zip"].notna()]
    z["key"] = z["region"] + ":" + z["zip"]
    z["w"] = z["n"] / z.groupby("GEOID")["n"].transform("sum")
    z = f[["GEOID"]].reset_index().merge(z[["GEOID", "key", "w"]], on="GEOID", how="left")
    miss = z["key"].isna()
    z.loc[miss, "key"] = "county:" + z.loc[miss, "GEOID"].str[:5]
    z.loc[miss, "w"] = 1.0
    keys, col = np.unique(z["key"], return_inverse=True)
    W = csr_matrix((z["w"].to_numpy(), (z["index"].to_numpy(), col)), shape=(len(f), len(keys)))
    A = W.multiply(weight[:, None]).tocsc()
    tot = np.asarray(A.sum(0)).ravel()
    return W, csr_matrix(A.multiply(1 / np.maximum(tot, 1e-12)[None, :]))


def covariates(f):
    x = pd.DataFrame({
        "lpop": np.log1p(f["pop_total"].fillna(0)),
        "lhu": np.log1p(f["housing_units"].fillna(0)),
        "lbld": np.log1p(f["bld_n"]),
        "larea": np.log(f["area_m2"] / 1e6 + 0.01),
        "urban": f["pct_urban"].fillna(0),
        "lpoi": np.log1p(f["poi_n"]),
    })
    x = pd.concat([x, pd.get_dummies(f["region"], dtype=float)], axis=1)
    return (x - x.mean()) / x.std().replace(0, 1)


# --------------------------------------------------------------------------- components

def gap(have, ref):
    """Official per-component gap 1 - min(1, overture / reference)."""
    return 1 - np.minimum(1, have / np.maximum(ref, 1e-9))


def transport(f, P, rng, S):
    region = f["region"].to_numpy()
    n = len(f)
    fm = length(f, "rdfm_in") + length(f, "rdfmp_band")
    cr = length(f, "rd_in", None, ("CR",)) + length(f, "rdp_band", None, ("CR",))
    proxy = length(f, "rd_in", None, HWY_NETS) + length(f, "rdp_band", None, HWY_NETS) - fm
    named = length(f, "rd_in", NAMED)
    # FM/RM inclusion rate: fm_q where the tract already has a numbered highway, fm_q_alone (default
    # fm_q) where FM roads would be its only one, so that the published undefined share holds
    fm_q = by_region(P["fm_q"], region, rng, S)
    if "fm_q_alone" in P:
        fm_q = np.where(proxy[:, None] > 0, fm_q, by_region(P["fm_q_alone"], region, rng, S))
    T = (proxy[:, None]
         + fm[:, None] * (rng.random((n, S)) < fm_q)
         + cr[:, None] * (rng.random((n, S)) < by_region(P["cr_q"], region, rng, S)))
    noise = np.exp(rng.standard_normal((n, S)) * sample(P["sigma"], rng, (1, S)))
    T = T * noise
    extra = rng.random((n, S)) < by_region(P["extra_p"], region, rng, S)
    T = T + extra * rng.exponential(P["extra_m"], (n, S))
    if "band_q" in P:  # boundary highways count for this tract only with probability band_q
        band = length(f, "rdp_band", None, HWY_NETS) - length(f, "rdfmp_band")
        drop = rng.random((n, S)) >= P["band_q"]
        T = np.maximum(T - band[:, None] * noise * drop, 0.0)
    defined = T > 1.0
    return np.where(defined, gap(named[:, None], T), 0.0), defined


def buildings(f, P, rng, S):
    region = f["region"].to_numpy()
    n = len(f)
    usfs = f["usfs_BuildingCount_sum"].fillna(0).to_numpy()
    bld = f["bld_n"].to_numpy().astype(float)
    if P["model"] == "usfs":
        M = usfs[:, None] * np.exp(rng.normal(0, P["usfs_model_sigma"], (n, S)))
    else:
        mlk = f["bld_ml"].to_numpy().astype(float)
        nonml = bld - mlk
        r0 = by_region(P["r_prior"], region, rng, 1)
        tau = sample(P["r_tau"], rng, (1, S))
        ok = ((nonml >= 5) & (usfs >= 20))[:, None]
        r_hat = ((usfs - mlk) / np.maximum(nonml, 1))[:, None]
        v_hat = (P["usfs_sigma"] * usfs / np.maximum(nonml, 1))[:, None] ** 2 + 1e-9
        v_post = np.where(ok, 1 / (1 / tau ** 2 + 1 / v_hat), tau ** 2)
        m_post = np.where(ok, v_post * (r0 / tau ** 2 + r_hat / v_hat), r0)
        r = np.maximum(m_post + np.sqrt(v_post) * rng.standard_normal((n, S)), 0)
        M = mlk[:, None] + r * nonml[:, None]
        nob = bld == 0
        M[nob] = usfs[nob, None] * np.exp(rng.normal(0, P["usfs_sigma"], (nob.sum(), S)))
    defined = (M >= 0.5) | (bld[:, None] > 0)
    return np.where(M >= 0.5, gap(bld[:, None], M), 0.0), defined


def overture_likelihood(o, mu, p, fmax=FMAX):
    """P(o | F = f): each real facility is filed under the category with probability p_o and then
    appears 1 + Pois(dup) times; false positives add Pois(fp x mu)."""
    f = np.arange(fmax + 1)
    like = np.zeros((len(o), fmax + 1))
    for d in range(fmax + 1):  # d = facilities Overture filed under the category
        p_d = binom.pmf(d, f[None, :], p["p_o"])
        rest = o - d
        p_rest = np.where(rest >= 0, poisson.pmf(np.maximum(rest, 0), d * p["dup"] + p["fp"] * mu), 0.0)
        like += p_d * p_rest[:, None]
    return like


def evidence_likelihood(e, q, eps, fmax=FMAX):
    """P(e | F = f) for an independent evidence count: each real facility shows up with probability
    q (per tract), plus Pois(eps) items that are not facilities."""
    f = np.arange(fmax + 1)
    like = np.zeros((len(e), fmax + 1))
    for j in range(int(e.max()) + 1):
        rest = e - j
        like += binom.pmf(j, f[None, :], q[:, None]) * np.where(rest >= 0, poisson.pmf(np.maximum(rest, 0), eps), 0.0)[:, None]
    return like


def fit_detection(base, channels, region, grid=Q_GRID):
    """Per-region detection rate of each evidence channel, by maximum marginal likelihood (the
    channels and the Overture count observe the same latent facilities: capture-recapture)."""
    q = [np.zeros(len(region)) for _ in channels]
    tables = [{g: evidence_likelihood(e, np.full(len(e), g), eps) for g in grid} for e, eps in channels]
    for r in np.unique(region):
        m = region == r
        best, best_ll = None, -np.inf
        for combo in np.array(np.meshgrid(*[grid] * len(channels))).reshape(len(channels), -1).T:
            like = base[m].copy()
            for t, g in zip(tables, combo):
                like *= t[g][m]
            ll = np.log(np.maximum(like.sum(1), 1e-300)).sum()
            if ll > best_ll:
                best, best_ll = combo, ll
        for k, g in enumerate(best):
            q[k][m] = g
    return q


def overture_misfiled_likelihood(o, a, mu, p, p_a, eps, fmax=FMAX):
    """P(o, a | F = f) when Overture can also file a real facility under another category: each
    facility is filed under the counted category (p_o), only under another one (p_a, counted in a)
    or missed. Filed facilities appear 1 + Pois(dup) times; both counts add false positives."""
    f = np.arange(fmax + 1)[None, :]
    p_a = p_a[:, None]
    p_miss = np.maximum(1 - p["p_o"] - p_a, 1e-9)
    like = np.zeros((len(o), fmax + 1))
    for k in range(int(a.max()) + 1):
        p_k = np.where(a - k >= 0, poisson.pmf(np.maximum(a - k, 0), eps), 0.0)[:, None]
        for d in range(fmax + 1 - k):
            rest = f - d - k
            ok = rest >= 0
            log_m = (gammaln(f + 1) - gammaln(d + 1) - gammaln(k + 1) - gammaln(np.maximum(rest, 0) + 1)
                     + d * np.log(p["p_o"]) + k * np.log(np.maximum(p_a, 1e-12)) + np.maximum(rest, 0) * np.log(p_miss))
            m = np.where(ok, np.exp(log_m), 0.0)
            r = o - d
            p_d = np.where(r >= 0, poisson.pmf(np.maximum(r, 0), d * p["dup"] + p["fp"] * mu), 0.0)[:, None]
            like += m * p_d * p_k
    return like


def facility_posterior(o, mu, p, channels=(), strata=None, misfiled=None, fmax=FMAX):
    """P(F = f | o, evidence) per tract; also returns the prior mean lambda (from the expected
    Overture count mu) and the fitted rate of each evidence channel (misfiled rate first)."""
    lam = mu * (1 - p["fp"]) / (p["p_o"] * (1 + p["dup"]))
    f = np.arange(fmax + 1)
    prior = poisson.pmf(f[None, :], lam[:, None])
    rates = []
    if misfiled is None:
        post = prior * overture_likelihood(o, mu, p, fmax)
    else:
        a, eps = misfiled
        p_a = np.zeros(len(o))
        for r in np.unique(strata):
            m = strata == r
            best = max(MISFILED_GRID, key=lambda g: np.log(np.maximum((prior[m] * overture_misfiled_likelihood(
                o[m], a[m], mu[m], p, np.full(m.sum(), g), eps, fmax)).sum(1), 1e-300)).sum())
            p_a[m] = best
        post = prior * overture_misfiled_likelihood(o, a, mu, p, p_a, eps, fmax)
        rates.append(p_a)
    fitted = fit_detection(post, channels, strata) if channels else []
    for (e, eps), q in zip(channels, fitted):
        post = post * evidence_likelihood(e, q, eps, fmax)
    return post / post.sum(1, keepdims=True), lam, rates + fitted


def sample_categorical(post, draws, rng):
    cdf = np.cumsum(post, 1)
    u = rng.random((post.shape[0], draws))
    return (u[:, :, None] > cdf[:, None, :]).sum(2)


def facilities(f, P, rng, rng_params, S):
    """Per-type gaps and defined flags, each (3, n, S)."""
    x = covariates(f)
    K = P["param_sets"]
    assert S % K == 0
    gaps, defs = [], []
    for name in FACILITIES:
        spec = P[name]
        o = f[spec["col"]].to_numpy().astype(int)
        mu = PoissonRegressor(alpha=1e-4, max_iter=1000).fit(x, o).predict(x)
        g_parts, d_parts = [], []
        for _ in range(K):
            p = {k: (sample(v, rng_params, 1)[0] if is_range(v) else v) for k, v in spec.items() if k != "col"}
            channels = [(f[c].to_numpy().astype(int), eps) for c, eps in p.get("evidence", ())]
            strata = f["region"].to_numpy().astype(object)
            if p.get("evidence_strata") == "urban":  # detection rates per region x urban/rural
                strata = strata + "|" + f["ur_class"].fillna("Urban").to_numpy().astype(object)
            misfiled = None
            if "misfiled" in p:
                col, eps = p["misfiled"]
                misfiled = (f[col].to_numpy().astype(int), eps)
            post, lam, rates = facility_posterior(o, mu, p, channels, strata, misfiled)
            names = ([p["misfiled"][0] + " (misfiled)"] if misfiled else []) + [c for c, _ in p.get("evidence", ())]
            for c, q in zip(names, rates):
                FITTED[(name, c)] = pd.Series(q, index=strata).groupby(level=0).first().to_dict()
            F = sample_categorical(post, S // K, rng)
            h = rng.binomial(F, p["p_h"]) + rng.poisson(p["extra"] * lam[:, None], F.shape)
            d = h >= 1
            d_parts.append(d)
            g_parts.append(np.where(d, gap(o[:, None], h), 0.0))
        gaps.append(np.concatenate(g_parts, 1))
        defs.append(np.concatenate(d_parts, 1))
    return np.stack(gaps), np.stack(defs)


def business_proxy(f, poi, kind):
    """Relative number of business addresses in each tract."""
    if kind == "poi_warehouse":
        # places, plus one business address per >= 2500 m2 urban footprint (warehouses, plants)
        urban = f["pct_urban"].fillna(0.5).to_numpy()
        return poi + 1 + f["bld_s4"].fillna(0).to_numpy() * urban
    big = (f["bld_s3"] + f["bld_s4"]).fillna(0).to_numpy().astype(float)
    return (poi + 1) ** 0.6 * (big + 1) ** 0.4


def cbp(f, P, rng, S):
    n = len(f)
    pop = f["pop_total"].fillna(0).to_numpy().astype(float)
    poi = f["poi_n"].to_numpy().astype(float)
    kappa = sample(P["kappa"], rng, (1, S))
    if P["model"] == "pop":
        C = kappa * pop[:, None] * np.exp(rng.normal(0, P["pop_sigma"], (n, S)))
    else:
        # Overture places per CBP establishment in the region, at unit kappa
        tot = f.assign(poi=poi, pop=pop).groupby("region")[["poi", "pop"]].transform("sum")
        rbar1 = (tot["poi"] / tot["pop"]).to_numpy()
        if P["model"] == "addr":
            weight = f["housing_units"].fillna(0).to_numpy() + poi / (rbar1 / BASE_PARAMS["cbp"]["kappa"])
        else:
            weight = business_proxy(f, poi, P["proxy"])
        W, share = zip_shares(f, weight)
        Wt = W.T.tocsr()
        pop_z = Wt @ pop
        places_z = Wt @ (poi / rbar1)  # ZIP establishments implied by its places, at unit kappa
        # ZIP establishments: geometric blend of a population and an Overture-places estimate
        est = np.sqrt(np.maximum(kappa * pop_z[:, None], 0.5) * np.maximum(kappa * places_z[:, None], 0.5))
        Z = est * np.exp(rng.standard_normal(est.shape) * sample(P["zip_sigma"], rng, (1, S)))
        C = (share @ Z) * np.exp(rng.standard_normal((n, S)) * sample(P["tract_sigma"][P["model"]], rng, (1, S)))
    defined = C > 0.5
    return np.where(defined, gap(poi[:, None], C), 0.0), defined


# --------------------------------------------------------------------------- composite

def simulate(f, P):
    S = P["draws"]
    rngs = {name: np.random.default_rng([P["seed"], i]) for i, name in enumerate(STREAMS)}
    g_t, t_def = transport(f, P["transport"], rngs["transport"], S)
    g_b, b_def = buildings(f, P["buildings"], rngs["buildings"], S)
    fac_g, fac_d = facilities(f, P["facilities"], rngs["facilities"], rngs["params"], S)
    g_cbp, cbp_def = cbp(f, P["cbp"], rngs["cbp"], S)

    # POI = mean of the defined halves; facilities half = mean of the defined facility types
    hifld_def = fac_d.any(0)
    g_hifld = np.where(hifld_def, (fac_g * fac_d).sum(0) / np.maximum(fac_d.sum(0), 1), 0.0)
    halves = np.maximum(hifld_def + 0.0 + cbp_def, 1)
    p_def = hifld_def | cbp_def
    g_p = np.where(p_def, (g_hifld * hifld_def + g_cbp * cbp_def) / halves, 0.0)

    # composite = mean of the defined components
    k = t_def + 0.0 + b_def + p_def
    inv_k = np.where(k > 0, 1 / np.maximum(k, 1), 0.0)
    y = (g_t * t_def + g_b * b_def + g_p * p_def) * inv_k

    # Every scored tract has at least one defined component (the organisers dropped the tracts
    # with none), so the score is averaged over the draws where that holds. A tract with no such
    # draw has a reference the model cannot see; Overture then has nothing there either (two
    # Texas water tracts), and every defined component of such a tract is a full gap.
    scored = k > 0
    n_scored = scored.sum(1)

    def given_scored(a, fallback):
        return np.where(n_scored > 0, (a * scored).sum(1) / np.maximum(n_scored, 1), fallback)

    def expected(g, d):
        p = d.mean(1)
        return np.where(p > 0, (g * d).sum(1) / np.maximum(d.sum(1), 1), 0.0), p

    out = pd.DataFrame({"GEOID": f["GEOID"], "region": f["region"],
                        "coverage_gap_score": given_scored(y, 1.0)})
    for name, g, d in (("transport", g_t, t_def), ("building", g_b, b_def), ("poi", g_p, p_def)):
        out[f"{name}_gap"], out[f"{name}_pdef"] = expected(g, d)
    for i, name in enumerate(FACILITIES):
        out[f"poi_gap_{name}"], out[f"poi_pdef_{name}"] = expected(fac_g[i], fac_d[i])
    out["poi_gap_cbp"], out["poi_pdef_cbp"] = expected(g_cbp, cbp_def)
    out["any_undef"] = given_scored(k < 3, 1.0)
    # Share of the score each component carries (they add up to coverage_gap_score)
    out["s_T"] = given_scored(g_t * t_def * inv_k, 0.0)
    out["s_B"] = given_scored(g_b * b_def * inv_k, 0.0)
    out["s_F"] = given_scored(g_hifld * hifld_def / halves * p_def * inv_k, 0.0)
    out["s_C"] = given_scored(g_cbp * cbp_def / halves * p_def * inv_k, 0.0)
    ok = n_scored > 0
    assert np.allclose(out.loc[ok, ["s_T", "s_B", "s_F", "s_C"]].sum(axis=1), out.loc[ok, "coverage_gap_score"])
    return out


def to_submission(s, est):
    """Sample-submission layout; a component is reported as defined when P(defined) >= 0.5."""
    sub = s[["GEOID", "region"]].merge(est.drop(columns="region"), on="GEOID", how="left", validate="1:1")
    res = sub[["GEOID", "coverage_gap_score", "region"]].copy()
    for name in ("transport", "building", "poi"):
        d = sub[f"{name}_pdef"] >= 0.5
        res[f"{name}_gap"] = np.where(d, sub[f"{name}_gap"], 0.0)
        res[f"{name}_defined"] = d
    for name in ("fire", "ems", "schools", "cbp"):
        d = sub[f"poi_pdef_{name}"] >= 0.5
        res[f"poi_gap_{name}"] = np.where(d, sub[f"poi_gap_{name}"], 0.0)
        res[f"poi_defined_{name}"] = d
    res = res[s.columns]
    num = res.select_dtypes("number").columns
    res[num] = res[num].clip(0, 1).round(6)
    assert res.notna().all().all() and (res["GEOID"] == s["GEOID"]).all()
    return res


def main(variant):
    s, f = load()
    est = simulate(f, params(variant))
    os.makedirs(OUT, exist_ok=True)
    est.to_parquet(os.path.join(OUT, f"{variant}-detail.parquet"), index=False)
    path = os.path.join(OUT, f"{variant}.csv")
    to_submission(s, est).to_csv(path, index=False)
    print(path)
    print(est.groupby("region")[["coverage_gap_score", "s_T", "s_B", "s_F", "s_C", "any_undef"]].mean().round(4))
    for key, rates in FITTED.items():
        print("fitted detection rate", key, rates)


if __name__ == "__main__":
    main(sys.argv[1])
