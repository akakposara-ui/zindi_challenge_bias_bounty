"""Per tract: Overture places by 5-digit postcode (to rebuild ZIP membership) and Overture
buildings by footprint size class (large footprints stand in for business addresses).

Usage: python zips.py [region ...]
       -> ../features/<region>-zip.parquet   (GEOID, zip, n)
       -> ../features/<region>-bsize.parquet (GEOID, bld_s1..bld_s4: bbox area < 150, < 600,
                                              < 2500, >= 2500 m2)
"""
import sys

from db import REGIONS, connect, ref, save, tracts


def postcodes(con, region):
    return con.sql(f"""
        WITH p AS (
            SELECT ST_Point((bbox.xmin + bbox.xmax) / 2, (bbox.ymin + bbox.ymax) / 2) AS centre,
                   regexp_extract(addresses[1].postcode, '^[0-9]{{5}}') AS zip
            FROM '{ref(region, 'overture-pois.parquet')}')
        SELECT t.GEOID, nullif(p.zip, '') AS zip, count(*) AS n
        FROM p JOIN tr t ON ST_Contains(t.geom, p.centre)
        GROUP BY ALL""").df()


def footprint_sizes(con, region):
    # bbox area in m2, equirectangular at the footprint's latitude
    return con.sql(f"""
        WITH b AS (
            SELECT ST_Point((bbox.xmin + bbox.xmax) / 2, (bbox.ymin + bbox.ymax) / 2) AS centre,
                   (bbox.xmax - bbox.xmin) * 111320 * cos(radians((bbox.ymin + bbox.ymax) / 2))
                   * (bbox.ymax - bbox.ymin) * 110540 AS area
            FROM '{ref(region, 'overture-buildings.parquet')}')
        SELECT t.GEOID,
               count(*) FILTER (WHERE area < 150) AS bld_s1,
               count(*) FILTER (WHERE area >= 150 AND area < 600) AS bld_s2,
               count(*) FILTER (WHERE area >= 600 AND area < 2500) AS bld_s3,
               count(*) FILTER (WHERE area >= 2500) AS bld_s4
        FROM b JOIN tr t ON ST_Contains(t.geom, b.centre)
        GROUP BY 1""").df()


def build(region):
    con = connect()
    tracts(con, region)
    zips = postcodes(con, region)
    save(zips, region, "-zip")
    sizes = footprint_sizes(con, region)
    save(sizes, region, "-bsize")
    print(f"  {region}: zip {zips.shape} bsize {sizes.shape}", flush=True)


if __name__ == "__main__":
    for r in sys.argv[1:] or REGIONS:
        build(r)
