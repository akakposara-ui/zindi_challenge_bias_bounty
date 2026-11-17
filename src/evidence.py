"""Independent evidence that a facility really exists in a tract, beyond the Overture category the
organisers counted:

  fire_osm     OpenStreetMap fire-station buildings in the Overture buildings layer (class
               `fire_station`, or a civic building with a fire-station name), one per ~100 m cell
  fire_alt     Overture places with a fire-station name filed under another category (for example
               "... Fire Protection District" as `fire_protection_service`), more than 150 m from
               any `fire_department` place
  fire_alt_any the same, also counting places whose alternate categories include
               `fire_department`
  school_osm   OpenStreetMap school buildings (class `school`, or education buildings with a school
               name), one per ~200 m cell (a campus)

Usage: python evidence.py [region ...]  ->  ../features/<region>-evidence.parquet
"""
import sys

from db import REGIONS, TO_5070, connect, ref, save, tracts

FIRE_NAME = (r"(?i)(fire (station|dept|department|district|rescue|company|protection district|hall)"
             r"|volunteer fire|\bvfd\b|\besd\b|fire (&|and) rescue)")
SCHOOL_NAME = r"(?i)(school|elementary|middle|high|academy)"


def osm_buildings(con, region):
    return con.sql(f"""
        WITH b AS (
            SELECT (bbox.xmin + bbox.xmax) / 2 AS x, (bbox.ymin + bbox.ymax) / 2 AS y,
                   class = 'fire_station' OR (subtype = 'civic' AND regexp_matches(coalesce(names.primary, ''), '{FIRE_NAME}')) AS fire,
                   class = 'school' OR (subtype = 'education' AND regexp_matches(coalesce(names.primary, ''), '{SCHOOL_NAME}')) AS school
            FROM '{ref(region, 'overture-buildings.parquet')}'
            WHERE sources[1].dataset = 'OpenStreetMap'
              AND (class IN ('fire_station', 'school') OR subtype IN ('civic', 'education')))
        SELECT t.GEOID,
               count(DISTINCT (round(x, 3), round(y, 3))) FILTER (WHERE fire) AS fire_osm,
               count(DISTINCT (round(x * 500) / 500, round(y * 500) / 500)) FILTER (WHERE school) AS school_osm
        FROM b JOIN tr t ON ST_Contains(t.geom, ST_Point(b.x, b.y))
        WHERE fire OR school
        GROUP BY 1""").df()


def alternative_fire_places(con, region):
    """Per tract: fire stations Overture filed under another primary category, more than 150 m from
    any `fire_department` place (`fire_alt`: by name; `fire_alt_any`: by name or alternate category)."""
    con.sql(f"""
        CREATE OR REPLACE TABLE p AS
        SELECT categories.primary AS category,
               regexp_matches(coalesce(names.primary, ''), '{FIRE_NAME}') AS fire_name,
               coalesce(list_contains(categories.alternate, 'fire_department'), false) AS fire_alternate,
               ST_Point((bbox.xmin + bbox.xmax) / 2, (bbox.ymin + bbox.ymax) / 2) AS centre
        FROM '{ref(region, 'overture-pois.parquet')}'
        WHERE categories.primary = 'fire_department'
           OR regexp_matches(coalesce(names.primary, ''), '{FIRE_NAME}')
           OR list_contains(categories.alternate, 'fire_department')""")
    return con.sql(f"""
        WITH alt AS (SELECT row_number() OVER () AS id, centre, ST_Transform(centre, {TO_5070}) AS g,
                            fire_name AND coalesce(category, '') <> 'ambulance_and_ems_services' AS by_name
                     FROM p WHERE coalesce(category, '') <> 'fire_department'
                       AND (fire_name OR fire_alternate)),
             dept AS (SELECT ST_Transform(centre, {TO_5070}) AS g FROM p WHERE category = 'fire_department'),
             near AS (SELECT DISTINCT a.id FROM alt a JOIN dept d ON ST_DWithin(a.g, d.g, 150))
        SELECT t.GEOID, count(*) FILTER (WHERE by_name) AS fire_alt, count(*) AS fire_alt_any
        FROM alt a JOIN tr t ON ST_Contains(t.geom, a.centre)
        WHERE a.id NOT IN (SELECT id FROM near)
        GROUP BY 1""").df()


def build(region):
    con = connect()
    tracts(con, region)
    out = con.sql("SELECT GEOID FROM tr").df()
    out = out.merge(osm_buildings(con, region), on="GEOID", how="left")
    out = out.merge(alternative_fire_places(con, region), on="GEOID", how="left").fillna(0)
    save(out, region, "-evidence")
    print(f"  {region}: evidence {out[['fire_osm', 'fire_alt', 'fire_alt_any', 'school_osm']].sum().to_dict()}", flush=True)


if __name__ == "__main__":
    for r in sys.argv[1:] or REGIONS:
        build(r)
