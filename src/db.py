"""Shared settings for the feature scripts: bucket address, DuckDB connection, tract table.

Every layer is GeoParquet in OGC:CRS84 (lon/lat). DuckDB follows the EPSG axis order (lat/lon)
when told 'EPSG:4326', so every transform passes always_xy := true.
"""
import os

import duckdb

BASE = "s3://us-west-2.opendata.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge"
REGIONS = ["eastern-ok", "maricopa-az", "northern-ca", "south-central-tx", "eastern-wa"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "features")
TO_5070 = "'EPSG:4326', 'EPSG:5070', always_xy := true"  # CONUS Albers, metres


def connect():
    """DuckDB with httpfs + spatial, configured for the public bucket (no credentials)."""
    con = duckdb.connect()
    con.sql("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial;")
    # the bucket name contains dots, so path-style addressing is required
    con.sql("SET s3_region='us-west-2'; SET s3_url_style='path';")
    con.sql("SET memory_limit='6GB'; SET preserve_insertion_order=false;")
    return con


def ref(region, layer):
    return f"{BASE}/reference/{region}/{region}-{layer}"


def tracts(con, region):
    """Table `tr`: the region's scored tracts, in lon/lat (`geom`) and EPSG:5070 (`g5070`)."""
    con.sql(f"""
        CREATE OR REPLACE TABLE tr AS
        SELECT GEOID, geometry::GEOMETRY AS geom,
               ST_Transform(geometry::GEOMETRY, {TO_5070}) AS g5070
        FROM '{BASE}/strata/{region}/{region}-census-tracts.parquet'""")


def save(df, region, suffix=""):
    os.makedirs(OUT, exist_ok=True)
    df.to_parquet(os.path.join(OUT, f"{region}{suffix}.parquet"), index=False)
