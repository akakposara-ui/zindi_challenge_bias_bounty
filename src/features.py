"""Per-tract buildings, places, ACS housing units and strata, from the permitted data only.
No TIGER / Microsoft / HIFLD / CBP data is read anywhere.

Points and footprints are assigned to the tract containing their bbox centre.

Usage: python features.py [region ...]   ->  ../features/<region>.parquet
"""
import sys

from db import BASE, REGIONS, connect, ref, save, tracts

# Overture categories the organisers matched to each USGS facility layer
FIRE = "fire_department"
EMS = "ambulance_and_ems_services"
SCHOOLS = ("elementary_school", "middle_school", "high_school", "school", "private_school", "public_school")


def buildings(con, region):
    """Overture footprints per tract, split by the upstream source that supplied them."""
    return con.sql(f"""
        WITH b AS (
            SELECT ST_Point((bbox.xmin + bbox.xmax) / 2, (bbox.ymin + bbox.ymax) / 2) AS centre,
                   sources[1].dataset AS dataset
            FROM '{ref(region, 'overture-buildings.parquet')}')
        SELECT t.GEOID,
               count(*) AS bld_n,
               count(*) FILTER (WHERE dataset = 'OpenStreetMap') AS bld_osm,
               count(*) FILTER (WHERE dataset = 'Microsoft ML Buildings') AS bld_ml,
               count(*) FILTER (WHERE dataset = 'Esri Community Maps') AS bld_esri
        FROM b JOIN tr t ON ST_Contains(t.geom, b.centre)
        GROUP BY 1""").df()


def places(con, region):
    """All Overture places per tract, and those in each facility category."""
    schools = ", ".join(f"'{s}'" for s in SCHOOLS)
    return con.sql(f"""
        WITH p AS (
            SELECT ST_Point((bbox.xmin + bbox.xmax) / 2, (bbox.ymin + bbox.ymax) / 2) AS centre,
                   categories.primary AS category
            FROM '{ref(region, 'overture-pois.parquet')}')
        SELECT t.GEOID,
               count(*) AS poi_n,
               count(*) FILTER (WHERE category = '{FIRE}') AS poi_fire,
               count(*) FILTER (WHERE category = '{EMS}') AS poi_ems,
               count(*) FILTER (WHERE category IN ({schools})) AS poi_school
        FROM p JOIN tr t ON ST_Contains(t.geom, p.centre)
        GROUP BY 1""").df()


def build(region):
    con = connect()
    tracts(con, region)
    df = con.sql("SELECT GEOID, ST_Area(g5070) AS area_m2 FROM tr").df()
    acs = con.sql(f"SELECT GEOID, housing_units FROM '{ref(region, 'census-acs-housing.parquet')}'").df()
    strata = con.sql(f"SELECT * FROM '{BASE}/strata/{region}/{region}-strata-tract-table.parquet'").df()
    df = df.merge(acs, on="GEOID", how="left").merge(strata, on="GEOID", how="left")
    for layer in (buildings, places):
        print(f"  {region}: {layer.__name__}", flush=True)
        df = df.merge(layer(con, region), on="GEOID", how="left")
    df.insert(1, "region", region)
    save(df, region)
    print(f"  {region}: {df.shape}", flush=True)


if __name__ == "__main__":
    for r in sys.argv[1:] or REGIONS:
        build(r)
