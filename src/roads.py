"""Per-tract road lengths (metres, EPSG:5070) by Overture class and route network.

  rd_in_<class>_<net>        clipped inside the tract (every class of the roads extract)
  rdp_band_<class>_<net>     named-class or route-numbered roads running along the outside of the
                             tract boundary: pieces longer than MIN_PIECE_M inside a BAND_M band,
                             so that roads merely crossing the boundary do not count
  rdfm_in_<class>,
  rdfmp_band_<class>         the same two for Texas Farm/Ranch-to-Market roads, which are also
                             part of the STSUB network columns

Usage: python roads.py [region ...]  ->  ../features/<region>-roads.parquet
"""
import sys

from db import REGIONS, TO_5070, connect, ref, save, tracts

NAMED = ("motorway", "trunk", "primary", "secondary")
BAND_M = 25.0
MIN_PIECE_M = 60.0

# Route network of a segment = the highest-ranked network among its routes.
NETCAT_SQL = """
CASE
  WHEN n = 'US:I' THEN 1
  WHEN n LIKE 'US:I:%' THEN 2
  WHEN n = 'US:US' OR n IN ('US:US:Business', 'US:US:Alternate', 'US:US:Bypass', 'US:US:Truck', 'US:US:Spur') THEN 3
  WHEN regexp_matches(n, '^US:[A-Z]{2}$') THEN 4
  WHEN regexp_matches(n, '^US:[A-Z]{2}:(FM|RM|Loop|Spur|Business|Beltway|Park|Toll|NTTA|Trunk|Turnpike|Alternate|Bypass|Truck|Scenic|PR)$')
       OR n LIKE 'US:TX:Harris:HCTRA%' THEN 5
  WHEN regexp_matches(n, '^US:[A-Z]{2}:') THEN 6
  ELSE 9
END"""
NETNAMES = {1: "I", 2: "IBUS", 3: "USH", 4: "ST", 5: "STSUB", 6: "CR", 9: "NONE"}
FM_NETWORKS = "['US:TX:FM', 'US:TX:RM']"


def load_roads(con, region):
    """Table `rd`: class, network category, FM/RM flag and EPSG:5070 geometry per segment;
    table `band`: the BAND_M ring outside each tract."""
    con.sql(f"""
        CREATE OR REPLACE TABLE rd AS
        SELECT class,
               coalesce((SELECT min({NETCAT_SQL}) FROM (SELECT unnest(routes).network AS n)), 9) AS netcat,
               coalesce(list_has_any(list_transform(routes, x -> x.network), {FM_NETWORKS}), false) AS fm,
               ST_Transform(geometry::GEOMETRY, {TO_5070}) AS g
        FROM '{ref(region, 'overture-roads.parquet')}'""")
    con.sql(f"""
        CREATE OR REPLACE TABLE band AS
        SELECT GEOID, ST_Difference(ST_Buffer(g5070, {BAND_M}), g5070) AS g5070 FROM tr""")


def inside(con):
    return con.sql("""
        SELECT t.GEOID, r.class, r.netcat, r.fm, sum(ST_Length(ST_Intersection(r.g, t.g5070))) AS m
        FROM rd r JOIN tr t ON ST_Intersects(r.g, t.g5070)
        GROUP BY ALL""").df()


def along_boundary(con):
    named = ", ".join(f"'{c}'" for c in NAMED)
    return con.sql(f"""
        SELECT GEOID, class, netcat, fm, sum(m) AS m FROM (
            SELECT b.GEOID, r.class, r.netcat, r.fm, ST_Length(ST_Intersection(r.g, b.g5070)) AS m
            FROM (SELECT * FROM rd WHERE class IN ({named}) OR netcat < 9) r
            JOIN band b ON ST_Intersects(r.g, b.g5070))
        WHERE m > {MIN_PIECE_M}
        GROUP BY ALL""").df()


def wide(lengths, prefix, by_network=True):
    """Long (GEOID, class, netcat, m) -> one column per class (and network)."""
    keys = ["class", "net"] if by_network else ["class"]
    lengths = lengths.assign(net=lengths["netcat"].map(NETNAMES))
    table = lengths.pivot_table(index="GEOID", columns=keys, values="m", aggfunc="sum", fill_value=0.0)
    table.columns = ["_".join([prefix, *(c if by_network else [c])]) for c in table.columns]
    return table


def build(region):
    con = connect()
    tracts(con, region)
    load_roads(con, region)
    parts = []
    for lengths, tag in ((inside(con), "in"), (along_boundary(con), "band")):
        fm = lengths[(lengths["netcat"] == 5) & lengths["fm"]]
        parts.append(wide(lengths, "rd_in" if tag == "in" else "rdp_band"))
        parts.append(wide(fm, "rdfm_in" if tag == "in" else "rdfmp_band", by_network=False))
    out = parts[0].join(parts[1:], how="outer").fillna(0.0).reset_index()
    save(out, region, "-roads")
    print(f"  {region}: roads {out.shape}", flush=True)


if __name__ == "__main__":
    for r in sys.argv[1:] or REGIONS:
        build(r)
