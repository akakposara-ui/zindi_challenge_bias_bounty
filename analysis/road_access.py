"""Bias discovery, step 1: which roads reach the homes, and which of them the escape-route layer
keeps.

The challenge's scored road layer (`<region>-overture-roads.parquet`) keeps the drivable classes
motorway, trunk, primary, secondary, tertiary, residential and unclassified ("escape-route
framing"). The unfiltered extract (`<region>-overture-roads-unfiltered.parquet`, same release)
shows what it drops: `track`, `service`, `living_street`, `unknown` and non-vehicle classes.

Per tract this script writes
  len_scored_m, len_track_m, len_other_m   vehicle-road length clipped to the tract, by group
  sampled_buildings                        footprints in a deterministic 10 % sample
  share_no_scored_<d>                      sampled footprints with no scored road within d m
  share_track_only_<d>                     ... and a `track` within d m (reached only by tracks)
  share_other_only_<d>                     ... no track either, but a service/other road
and, for named tribal routes (Indian Route / Indian Service Route / BIA / Navajo Route + number),
their length by Overture class.

Only challenge data is read, so every number in the write-up is reproducible from the bucket.

Usage (from the repository root):
  python analysis/road_access.py [region ...]
      -> analysis/out/road_access_<region>.parquet, analysis/out/tribal_routes_<region>.parquet
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from db import REGIONS, TO_5070, connect, ref, tracts  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
SCORED = ("motorway", "trunk", "primary", "secondary", "tertiary", "residential", "unclassified")
EXCLUDED_VEHICLE = ("track", "service", "living_street", "unknown")
DISTANCES = (100, 200)
SAMPLE_MOD = 10  # keep footprints whose id hashes to 0 mod 10
TRIBAL_ROUTE = r"(?i)^(indian (service )?route|bia( route)?|navajo route)\s*-?\s*[0-9]+"


def sql_list(values):
    return ", ".join(f"'{v}'" for v in values)


def load_layers(con, region):
    """Tables `r` (vehicle roads with their group, EPSG:5070) and `b` (sampled footprint centres)."""
    con.sql(f"""
        CREATE OR REPLACE TABLE r AS
        SELECT CASE WHEN class IN ({sql_list(SCORED)}) THEN 'scored'
                    WHEN class = 'track' THEN 'track' ELSE 'other' END AS grp,
               class, names.primary AS name,
               ST_Transform(geometry::GEOMETRY, {TO_5070}) AS g
        FROM '{ref(region, 'overture-roads-unfiltered.parquet')}'
        WHERE class IN ({sql_list(SCORED + EXCLUDED_VEHICLE)})""")
    con.sql(f"""
        CREATE OR REPLACE TABLE b AS
        SELECT row_number() OVER () AS bid, t.GEOID, ST_Transform(c.centre, {TO_5070}) AS p
        FROM (SELECT ST_Point((bbox.xmin + bbox.xmax) / 2, (bbox.ymin + bbox.ymax) / 2) AS centre
              FROM '{ref(region, 'overture-buildings.parquet')}'
              WHERE hash(id) % {SAMPLE_MOD} = 0) c
        JOIN tr t ON ST_Contains(t.geom, c.centre)""")


def road_lengths(con):
    return con.sql("""
        SELECT t.GEOID, r.grp, sum(ST_Length(ST_Intersection(r.g, t.g5070))) AS m
        FROM r JOIN tr t ON ST_Intersects(r.g, t.g5070)
        GROUP BY ALL""").df().pivot_table(index="GEOID", columns="grp", values="m", fill_value=0.0)


def building_access(con):
    near = con.sql(f"""
        SELECT b.bid, r.grp, min(ST_Distance(b.p, r.g)) AS d
        FROM b JOIN r ON ST_DWithin(b.p, r.g, {max(DISTANCES)})
        GROUP BY ALL""").df().pivot_table(index="bid", columns="grp", values="d", aggfunc="min")
    homes = con.sql("SELECT bid, GEOID FROM b").df().join(near, on="bid")
    for grp in ("scored", "track", "other"):
        if grp not in homes:
            homes[grp] = float("nan")
    flags = {}
    for d in DISTANCES:
        no_scored = ~(homes["scored"] <= d)
        has_track = homes["track"] <= d
        flags[f"share_no_scored_{d}"] = no_scored
        flags[f"share_track_only_{d}"] = no_scored & has_track
        flags[f"share_other_only_{d}"] = no_scored & ~has_track & (homes["other"] <= d)
    homes = homes.assign(**flags)
    return homes.groupby("GEOID").agg(sampled_buildings=("bid", "size"), **{k: (k, "mean") for k in flags})


def tribal_routes(con):
    return con.sql(f"""
        SELECT t.GEOID, r.class, sum(ST_Length(ST_Intersection(r.g, t.g5070))) AS m
        FROM (SELECT * FROM r WHERE regexp_matches(coalesce(name, ''), '{TRIBAL_ROUTE}')) r
        JOIN tr t ON ST_Intersects(r.g, t.g5070)
        GROUP BY ALL""").df()


def build(region):
    con = connect()
    tracts(con, region)
    load_layers(con, region)
    lengths = road_lengths(con).rename(columns=lambda c: f"len_{c}_m")
    access = building_access(con)
    out = lengths.join(access, how="outer").reset_index()
    os.makedirs(OUT, exist_ok=True)
    out.to_parquet(os.path.join(OUT, f"road_access_{region}.parquet"), index=False)
    tribal_routes(con).to_parquet(os.path.join(OUT, f"tribal_routes_{region}.parquet"), index=False)
    print(f"  {region}: {int(out['sampled_buildings'].sum())} sampled footprints, {len(out)} tracts", flush=True)


if __name__ == "__main__":
    for r in sys.argv[1:] or REGIONS:
        build(r)
