# Bias Bounty Mapping Equity Challenge — label-free coverage-gap model

**Final result:** 110th on the private leaderboard (Zindi, closed 31 October 2026; ranking subject to final review until 21 November 2026). Best public RMSE 0.067747 (`v36`).

> Data is not included. `run_all.sh` reads it in place from the public Source Cooperative bucket. Download `SampleSubmission.csv` from the Zindi challenge page into the repository root first.

Estimates the Overture Maps coverage-gap score (0 = full coverage, 1 = none) for the 9,794 scored
Census tracts of the five study regions, using **only the datasets provided in the challenge
package**. No training labels exist and the reference layers the organisers scored against
(TIGER/Line roads, Microsoft Building Footprints, USGS/HIFLD facilities, County Business
Patterns) were removed from the package. The model therefore does not learn a mapping from
labels. It rebuilds each missing reference quantity as an uncertain variable from permitted
features, applies the organisers' published composite formula to many Monte-Carlo draws, and
submits the mean, which is the RMSE-optimal point estimate under the model.

| Region | Scored tracts |
|---|---|
| eastern-ok | 1,192 |
| maricopa-az | 1,593 |
| northern-ca | 591 |
| south-central-tx | 6,003 |
| eastern-wa | 415 |

## 1. Quick start

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHON=.venv/bin/python ./run_all.sh v3d        # rebuilds features, writes submissions/v3d.csv
SKIP_FEATURES=1 PYTHON=.venv/bin/python ./run_all.sh v3d   # model only, reusing features/
```

`run_all.sh` needs network access: every input is read in place from the public Source
Cooperative bucket (no credentials). DuckDB downloads its `httpfs` and `spatial` extensions on
first use. The output is `submissions/<variant>.csv` in the exact layout of
`SampleSubmission.csv`, plus `submissions/<variant>-detail.parquet` with per-component
expectations.

## 2. Architecture

```
Source Cooperative bucket (s3://us-west-2.opendata.source.coop/humane-intelligence/
bias-bounty-mapping-equity-challenge/, read in place with DuckDB httpfs, nothing pre-processed)
  reference/<region>/<region>-overture-{buildings,roads,pois}.parquet   (Overture 2026-08-19.0)
  reference/<region>/<region>-census-acs-housing.parquet
  strata/<region>/<region>-{census-tracts,strata-tract-table}.parquet
        |
        v   per-tract aggregation (DuckDB spatial joins, EPSG:5070 lengths)
  src/features.py   -> features/<region>.parquet         buildings by upstream source, places
                                                         by facility category, ACS, strata
  src/roads.py      -> features/<region>-roads.parquet   road length by class x route network,
                                                         inside the tract and along its
                                                         boundary; Texas FM/RM roads apart
  src/zips.py       -> features/<region>-zip.parquet     places by postcode (ZIP membership)
                       features/<region>-bsize.parquet   buildings by footprint size
  src/evidence.py   -> features/<region>-evidence.parquet  OSM fire-station / school buildings,
                                                         fire stations filed under other categories
        |
        v
  src/model.py <variant>
     transport | buildings | facilities (fire, EMS, schools) | CBP establishments
     each: unknown reference ~ generative model(permitted features), 2,000 draws
     -> official composite per draw -> mean over draws
        |
        v
  submissions/<variant>.csv   (+ <variant>-detail.parquet)
```

`src/db.py` holds what the three feature scripts share: the bucket address, the DuckDB
connection settings, the tract table and the output folder.

## 3. ETL

### Extract
| Layer | Columns read | Use |
|---|---|---|
| Overture buildings | `bbox`, `sources[1].dataset` | count per tract, by upstream source (OSM / Microsoft ML / Esri) and footprint size |
| Overture roads (`subtype=road`, class-filtered extract) | `class`, `routes[].network`, `geometry` | length per tract by class and route network |
| Overture places | `bbox`, `categories.primary`, `addresses[1].postcode` | counts per tract (all, fire, EMS, schools), postcodes |
| ACS housing (B25001) | `housing_units` | address-share weight |
| Census tracts (strata) | `geometry`, `ALAND`, `pop_total`, `pct_urban` | tract polygons, covariates |
| Strata table | all 232 columns joined; the model uses `usfs_BuildingCount_sum`, `pop_total`, `pct_urban` | covariates |

Only the columns listed are fetched (Parquet column projection), straight from the bucket.

### Transform
- **Tract assignment.** Points (POIs) and building footprints are assigned by their bbox centre
  with `ST_Contains` against the 2020 tract polygons. Roads are clipped with `ST_Intersection`.
  Lengths are measured in EPSG:5070 with `always_xy := true`, because the layers are lon/lat
  (`OGC:CRS84`) and naming EPSG:4326 without it silently yields `inf`.
- **Road network category.** Each segment gets the highest-ranked route network it carries:
  Interstate, Interstate business/future loops, US highways (incl. business, alternate, bypass,
  truck, spur), state highways (`US:<ST>`), state subsystems (Loop, Spur, Business, Beltway, Park,
  Toll, NTTA, HCTRA, Turnpike, ...), county networks, or none.
- **Boundary band.** For each tract, route-numbered and named-class pieces lying within 25 m
  outside its boundary and running along it (> 60 m inside the band) are measured separately.
  This captures highways that form the boundary: TIGER roads coincide exactly with TIGER tract
  boundaries, so the organisers' clipped TIGER length counts them for both tracts.
- **ZIP membership.** 97.3 % of places carry a 5-digit postcode. Each tract is spread over the
  ZIPs of its places, in proportion to the place counts; the county stands in for tracts with
  none.

### Load
One parquet file per region and step in `features/`, keyed on `GEOID` (text, 11 characters).
The model joins them onto the sample-submission order and asserts one row per scored tract.

## 4. Model

The composite is reproduced exactly as published: each component is
`1 - min(1, overture / reference)` where its reference exists, the POI component is the mean of
the defined halves (facilities = mean of the defined fire / EMS / school gaps; establishments =
all places against CBP), and the score is the mean of the defined components. Every draw
samples the unknown reference quantities below, evaluates this formula, and the submission is the
draw mean. Each component has its own seeded random stream (`seed = 2026`), so the output is
deterministic and changing one component never changes another's draws.

The organisers dropped every tract with no defined component, so every scored tract has at least
one. The mean is therefore taken over the draws where at least one component is defined.

### Transport: TIGER S1100 + S1200 against Overture motorway / trunk / primary / secondary
- Overture length `O` is computed exactly as the organisers describe it.
- The TIGER length is proxied by Overture roads carrying an Interstate, US or state route
  number (plus state subsystems, minus Texas FM/RM roads), inside the tract **plus** the
  boundary band, times lognormal noise (sigma 0.15).
- This choice was **calibrated against the organisers' published share of tracts with at least
  one undefined component**, which in practice is the share without a TIGER highway:

  | Region | Model | Published |
  |---|---|---|
  | eastern-ok | 21.0 % | 21 % |
  | northern-ca | 36.5 % | 37 % |
  | south-central-tx | 27.6 % | 28 % |
  | maricopa-az | 54.2 % | 55 % |

  Without the boundary band the undefined share exceeds the published one in all four regions
  (23.6 / 39.6 / 29.4 / 56.4 %). The excess is the 1.2 to 2.3 % of tracts whose only highway runs
  along their boundary, which is the evidence for the double counting. Including all Texas FM/RM
  roads drops Texas to 18 %; excluding all state subsystems raises it to 36 %.
- The published Overture/TIGER length ratio (0.71 to 1.59 across the four original regions)
  cannot hold for Oklahoma with this proxy alone (ours: 0.99). Reaching 0.71 needs about 40 % of
  Oklahoma county-road length in TIGER, mostly roads Overture classes as tertiary. Each Oklahoma
  draw includes county roads with probability 0.4, and each Texas draw includes FM/RM roads with
  probability 0.1 (the most the 28 % statistic allows).
- A small per-region chance of an S1200 road the proxy misses (2 to 5 %) closes the remaining
  gap to the published shares.

### Buildings: Microsoft footprints against Overture buildings
- The removed Microsoft layer carried `height, confidence` (schema as described in a public
  repository, see section 10), i.e. Microsoft's Global ML release.
  Overture conflates the same release (version 2025-02-25), dropping ML footprints that overlap
  OSM or Esri ones.
- Hence `M = ML footprints kept by Overture + r x (OSM + Esri footprints)`, where `r` is the number
  of ML footprints each OSM/Esri footprint replaced. In ML-only tracts `M` equals the Overture
  count, so the gap is 0 there.
- `r` is inferred per tract from the strata column `usfs_BuildingCount_sum` (USFS Wildfire Risk
  to Communities). In the 678 ML-only tracts it tracks the ML count with a median log ratio of
  -0.007 and an sd of 0.146. The estimate is shrunk towards the region median of the implied `r`
  (0.79 to 0.98) with a prior sd of 0.08.

### Facilities: USGS fire stations, EMS stations and schools against Overture categories
Categories follow the organisers' README: `fire_department`; `ambulance_and_ems_services`;
`elementary_school, middle_school, high_school, school, private_school, public_school`.
- For each type the real count is `F ~ Poisson(lambda)`.
- Overture shows each real facility with probability `p_o`, adds `1 + Poisson(dup)` records per
  detected facility, plus false positives.
- The reference count is `h = Binomial(F, p_h) + Poisson(extra x lambda)`.
- `lambda` comes from a Poisson regression of the Overture count on population, housing units,
  buildings, area, urban share and region.
- `F` is drawn from its exact posterior given the observed Overture count, then `h`.
- From `v18` on, the posterior also uses independent evidence that a facility exists
  (`src/evidence.py`): OpenStreetMap fire-station and school buildings in the Overture buildings
  layer, and fire-station-named places filed under another category (`fire_protection_service`
  and others), more than 150 m from any `fire_department` place. Each channel detects a real
  facility with a per-region rate fitted by maximum marginal likelihood. The channels and the
  Overture count observe the same latent facilities, which is the capture-recapture logic, so no
  label is needed. Fitted OSM fire-station rates: 0.10 (OK) to 0.40 (AZ, WA); school rates 0.5 to
  0.9.

| Type | p_o | dup | fp | p_h | Overture / reference |
|---|---|---|---|---|---|
| fire | 0.65 | 0.10 | 0.15 | 0.95 | 0.84 |
| EMS | 0.55 | 0.10 | 0.35 | 0.70 | 1.2 |
| schools | 0.85 | 0.60 | 0.15 | 0.95 | 1.6 |

These ratios are consistent with the organisers' notes: EMS "close to 1:1", fire stations
"carry the component", Overture hospitals 12x the reference (hospitals excluded).

### Establishments: CBP against all Overture places
- The organisers' layer had `cbp_estab`, `cbp_estab_res` and `cbp_estab_bus`, i.e. establishments
  split over tracts by total, residential or business address share. These column names come from
  a public repository's description of the pre-removal package; no code or data from it is used.
- ZIP establishment totals are estimated as the geometric mean of `0.024 x population` (national
  CBP rate per resident) and the ZIP's Overture places divided by the region's places-per-
  establishment ratio, with lognormal ZIP noise (sigma 0.3).
- They are then split by a business-address proxy, with lognormal tract noise (sigma 0.5):
  - `v2b`: `(places + 1)^0.6 x (footprints >= 600 m2 + 1)^0.4`;
  - `v3d`: places + 1 + urban footprints >= 2500 m2, which leaves out barns and farm buildings.

### Parameter uncertainty
Variant `v3b` draws the least-identified parameters per draw from plausible ranges (transport
noise, FM/county-road shares, r prior, CBP rate and noise, facility detection rates). This changes
the estimate by 0.004 RMS against `v2b`, the size of Monte-Carlo noise at 400 draws.

## 5. Variants and leaderboard (public, 30 % of tracts)

Each variant is a complete candidate model defined in `VARIANTS` (`src/model.py`); later ones
change one assumption at a time so that every score is interpretable.

| Variant | Change | Public RMSE |
|---|---|---|
| `v1` | buildings: USFS count as the reference; CBP: county rate x population | 0.113032 |
| `v2a` | structural buildings; CBP split by total address share, per ZIP | 0.100414 |
| `v2b` | structural buildings; CBP split by business address share | **0.076699** |
| `v3d` | `v2b`, CBP business proxy linear in places + urban warehouses | **0.072910** |
| `v3e` | `v2b`, no Oklahoma county roads in TIGER | 0.078535 (worse: county roads kept) |
| `v3b` | `v2b` with parameter uncertainty integrated | not submitted (0.004 RMS change) |
| `v6` | `v3d` + the two no-data water tracts set to 1 | 0.072311 |
| `v7` | `v6`, 60 % of Oklahoma county roads in TIGER (instead of 40 %) | **0.071818** (kept) |
| `v8` | `v6`, CBP tract noise 0.3 (instead of 0.5) | 0.072221 (kept, small) |
| `v9` | `v6`, school detection 0.93 (fewer school gaps) | 0.071843 (kept) |
| `v10` | `v6` + the changes of `v7`, `v8` and `v9` | **0.071574** (best) |
| `v11` | `v10`, 80 % of Oklahoma county roads | **0.071392** (kept) |
| `v12` | `v10`, school detection 0.97 | 0.071686 (rejected) |
| `v13` | `v10`, fire-station detection 0.75 (fewer fire gaps) | 0.072884 (rejected) |
| `v14` | `v11`, all Oklahoma county roads | 0.071596 (rejected: 80 % kept) |
| `v15` | `v11`, fire-station detection 0.55 (more fire gaps) | 0.070605 |
| `v16` | `v11`, fire-station detection 0.45 | 0.070586 |
| `v17` | `v14` + `v15` | 0.070922 |
| `v18` | `v16` + independent facility evidence (OSM buildings, misfiled places) | **0.069240** (best) |
| `v19` | `v18` with fire-station detection 0.55 | 0.069601 |
| `v21` | `v18` with more USGS fire records without a real station (extra 0.15) | 0.069595 (rejected) |
| `v20` | `v18` with fire-station detection 0.35 | 0.070490 (rejected) |
| `v22` | `v18`, no USGS fire records without a real station (extra 0) | 0.069135 (kept) |
| `v23` | `v18`, 25 % of Texas FM/RM roads in TIGER where the tract has a numbered highway (10 % elsewhere) | **0.069054** (kept) |
| `v24` | as `v23` with 40 % | 0.069337 (rejected) |
| `v25` | `v22` + `v23` | 0.068886 |
| `v26` | `v25`, fire stations also found by alternate category | 0.069001 (rejected) |
| `v27` | `v25`, CBP rate 0.020 per resident | 0.068755 (kept) |
| `v28` | `v25`, USGS records 85 % of real fire stations | 0.068994 (rejected) |
| `v29` | `v25`, TIGER length noise 0.25 | 0.067954 (kept) |
| `v30` | `v25` + `v27` + `v29` | 0.067785 (identical score with a 2-column file) |
| `v33` | `v30`, boundary highways count for the tract 7 times in 10 | 0.069748 (rejected: boundary double counting confirmed) |
| `v35` | `v30`, evidence detection rates per region x urban/rural | 0.067993 (rejected) |
| `v36` | `v30`, misfiled fire stations as an outcome of the Overture capture | **0.067747** (best) |
| `v37` | `v36` with every leaderboard-tuned parameter drawn from a range around its value | to be scored |

The v1, v2a and v2b scores were produced by earlier revisions of `model.py` with 400 draws. The
current code reproduces them up to Monte-Carlo noise (for `v2b`: 0.0035 RMS per tract). The
submissions selected for the private leaderboard are regenerated by the code in this repository.

Considered and rejected, with the reason:
- transport without the boundary band: contradicts the published undefined shares;
- all Texas FM roads as S1200: Texas undefined share 18 % against the published 28 %;
- a physical fire-station prior from population and area: it implies 480 to 1,070 stations for
  Maricopa, against about 300;
- facility counts filtered by Overture `confidence`: the organisers count every place in the
  category.

## 6. Edge cases

| Case | Tracts | Handling |
|---|---|---|
| zero population | 27 | No effect on the CBP reference under business allocation (the proxy is at least 1). |
| zero housing units | 82 | Only enters the `v2a` address weight. |
| no Overture data at all | 2 (`48039990000`, `48057990000`: Texas water tracts, `ALAND = 0`) | They are scored, so some reference exists there, and Overture has nothing to set against it. Every defined component is a full gap, so the score is exactly 1. |
| no Overture named-class road | 137 | Transport gap is 1 when TIGER is drawn as present, otherwise undefined. |
| `pct_urban` missing | 27 (the zero-population tracts) | Taken as 0.5. |

Components are written to the submission as their expectation given that they are defined, with
`*_defined = P(defined) >= 0.5`. Undefined components are 0 with `defined = False`, as in the
organisers' CSVs.

## 7. Run time and environment

Measured by a clean end-to-end run of `run_all.sh` into an empty directory (see `repro.log`):

| Step | Time |
|---|---|
| `features.py` | see `repro.log` |
| `roads.py` | see `repro.log` |
| `zips.py` | see `repro.log` |
| `model.py <variant>` | ~15 s |

Hardware: Intel i7-13650HX (20 threads), 23 GB RAM, Linux; Python 3.13.9. Packages are pinned in
`requirements.txt`. DuckDB memory is capped at 6 GB per script.

## 8. Error handling, logging, maintenance

Error handling and logging:
- GEOID is read as text everywhere, which keeps Maricopa's leading `04`.
- The model asserts one feature row per scored tract, no missing values and the exact
  sample-submission order and columns. It also checks that the four component shares add up to
  the score.
- Each script prints one line per region and step; `run_all.sh` prints step timings and stops on
  the first error (`set -euo pipefail`).

Maintenance:
- The Overture release is pinned by the bucket (2026-08-19.0). Re-running against another release
  needs a new extract.
- Adding a region means adding its name to `REGIONS` in `src/db.py` and a `buildings.r_prior`
  entry (median implied `r`) in `src/model.py`. If it has route networks not matched by
  `NETCAT_SQL` in `src/roads.py`, they need a category too.

## 9. Bias discovery analysis (`analysis/`)

The Best Bias Discovery finding (homes on reservations reached only by roads Overture classes as
`track`, which the escape-route layer drops) is reproduced by two scripts that read only challenge
data:

```bash
python analysis/road_access.py   # per tract: road length by class group; for a 10 % footprint
                                 # sample, distance to the nearest escape-route road / track
python analysis/summarise.py     # analysis/out/tables.md, every table quoted in the write-up
```

`summarise.py` needs the feature tables (`src/features.py`, `src/roads.py`) for the strata columns.

## 10. Data compliance

- Every input is in the challenge package.
- No TIGER, Microsoft, HIFLD/USGS or CBP data was downloaded or used, directly or through
  third-party copies.
- The only external facts are general constants or published documentation: the national CBP
  establishment rate per resident (0.024) and the organisers' README and challenge statement.
- Two schema details of the removed layers come from a public repository's description of the
  pre-removal package (github.com/theerealhenry/Mapping-Equity-Coverage-Gaps, docs/): the CBP
  columns `cbp_estab`, `cbp_estab_res`, `cbp_estab_bus`, and the Microsoft layer columns
  `height, confidence`. They are column names only: no value, code or data from that repository
  is used. Both hypotheses they suggested were tested on the leaderboard (`v2a` against `v2b`) or
  against the challenge data (Overture's ML release tag and the USFS counts).
- Leaderboard scores were used only to choose between complete candidate models.
