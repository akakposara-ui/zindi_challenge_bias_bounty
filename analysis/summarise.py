"""Bias discovery, step 2: the tables quoted in the write-up, from analysis/out/*.parquet and the
per-tract feature tables (which carry the strata columns).

Groups
  tribal  "reservation/trust": >= 50 % of the tract's area in a legal AIANNH area (strata
          `tribal_legal_pct`); "OTSA/statistical": in a tribal statistical area; else "non-tribal"
  rural   USDA RUCA 2020 primary code: metro 1-3, micropolitan 4-6, small town 7-9, rural 10

Shares of footprints are pooled over the tract samples (each sampled footprint counts once).

Usage (from the repository root): python analysis/summarise.py  ->  analysis/out/tables.md
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT = os.path.join(HERE, "..", "features")
OUT = os.path.join(HERE, "out")
REGIONS = ["maricopa-az", "eastern-wa", "eastern-ok", "northern-ca", "south-central-tx"]
SHARES = ["share_no_scored_100", "share_track_only_100", "share_no_scored_200", "share_track_only_200"]
CONTEXT = {"svi_overall": "SVI", "cvi_overall": "CVI", "epht_heat_days_summer": "heat days (summer)",
           "usfs_WHP_mean": "wildfire hazard (WHP)"}


def highway_ratio(region):
    """Overture named-highway length over its route-numbered highway length (the proxy for the
    TIGER S1100/S1200 network the automated road metric scores against), per tract."""
    sys.path.insert(0, os.path.join(HERE, "..", "src"))
    from model import HWY_NETS, NAMED, length

    roads = pd.read_parquet(os.path.join(FEAT, f"{region}-roads.parquet")).fillna(0.0)
    named = length(roads, "rd_in", NAMED)
    numbered = length(roads, "rd_in", None, HWY_NETS) - length(roads, "rdfm_in")
    return pd.DataFrame({"GEOID": roads["GEOID"], "named_m": named, "numbered_m": numbered})


def load():
    rows = []
    for r in REGIONS:
        path = os.path.join(OUT, f"road_access_{r}.parquet")
        if not os.path.exists(path):
            continue
        f = pd.read_parquet(os.path.join(FEAT, f"{r}.parquet"))
        cols = ["GEOID", "region", "pop_total", "bld_n", "ruca_primary", "tribal_legal_pct",
                "tribal_statistical", "aiannh_name", *CONTEXT]
        rows.append(f[cols].merge(pd.read_parquet(path), on="GEOID", how="left")
                    .merge(highway_ratio(r), on="GEOID", how="left"))
    x = pd.concat(rows, ignore_index=True)
    x["tribal"] = np.where(x["tribal_legal_pct"].fillna(0) >= 0.5, "reservation/trust",
                           np.where(x["tribal_statistical"].fillna(False), "OTSA/statistical", "non-tribal"))
    x["rurality"] = pd.cut(x["ruca_primary"], [0, 3, 6, 9, 10], labels=["metro", "micro", "small town", "rural"])
    for c in ("len_scored_m", "len_track_m", "len_other_m"):
        x[c] = x[c].fillna(0.0)
    return x


def pooled(d):
    w = d["sampled_buildings"].fillna(0)
    out = {"tracts": len(d), "population": int(d["pop_total"].sum()),
           "track share of road km": d["len_track_m"].sum() / (d["len_track_m"] + d["len_scored_m"]).sum()}
    for s in SHARES:
        out[s] = np.average(d[s].fillna(0), weights=w) if w.sum() else np.nan
    out["est. structures track-only (100 m)"] = (d["share_track_only_100"].fillna(0) * d["bld_n"]).sum()
    numbered = d["numbered_m"].sum()
    out["named / numbered highway km"] = d["named_m"].sum() / numbered if numbered >= 500 else np.nan
    return pd.Series(out, dtype=object)


COUNTS = ("tracts", "population", "est. structures track-only (100 m)", "pop_total", "sampled_buildings")
KM_COLS = set()  # filled with the road-class columns of table 3


def table(df, digits=3):
    """Markdown table (index columns first), without extra dependencies; counts as integers,
    kilometre columns with one decimal, shares with `digits`."""
    df = df.reset_index()

    def fmt(col, v):
        if pd.isna(v):
            return ""
        if col in COUNTS:
            return f"{int(round(float(v))):,}"
        if isinstance(v, (float, np.floating, int, np.integer)):
            return f"{float(v):.1f}" if col.endswith("_km") or col in KM_COLS else f"{float(v):.{digits}f}"
        return str(v)

    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(c, v) for c, v in zip(df.columns, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)



def main():
    x = load()
    parts = ["# Bias discovery tables\n"]
    by_group = x.groupby(["region", "rurality", "tribal"], observed=True).apply(pooled, include_groups=False)
    parts += ["## 1. By region, rurality and tribal status\n", table(by_group), ""]
    res = x[x["tribal"] == "reservation/trust"]
    per_res = res.groupby(["region", "aiannh_name"]).apply(pooled, include_groups=False)
    ctx = res.groupby(["region", "aiannh_name"])[list(CONTEXT)].median().rename(columns=CONTEXT)
    per_res = per_res.join(ctx).sort_values("population", ascending=False)
    parts += ["## 2. Reservations and trust land (tract medians for the context columns)\n", table(per_res), ""]
    tracts = res.set_index(["region", "aiannh_name", "GEOID"])[["pop_total", "sampled_buildings", *SHARES]]
    tracts = tracts.sort_values("share_track_only_100", ascending=False)
    parts += ["## 2b. Reservation tracts\n", table(tracts), ""]
    routes = []
    for r in REGIONS:
        path = os.path.join(OUT, f"tribal_routes_{r}.parquet")
        if os.path.exists(path):
            routes.append(pd.read_parquet(path).merge(x[["GEOID", "region", "tribal", "aiannh_name"]], on="GEOID"))
    routes = pd.concat(routes)
    routes = routes[routes["tribal"] == "reservation/trust"]
    km = routes.pivot_table(index=["region", "aiannh_name"], columns="class", values="m", aggfunc="sum", fill_value=0) / 1000
    KM_COLS.update(km.columns)
    km = km.assign(total_km=km.sum(axis=1))
    km["track share"] = km.get("track", 0) / km["total_km"]
    km = km[km["total_km"] >= 5].sort_values("total_km", ascending=False)
    parts += ["## 3. Named tribal routes (Indian Route / BIA / Navajo Route + number) by Overture class, km\n",
              table(km), ""]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "tables.md"), "w") as fh:
        fh.write("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    sys.exit(main())
