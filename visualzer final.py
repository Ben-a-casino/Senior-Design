from pathlib import Path
from typing import Callable, Optional, Tuple, List, Dict

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import LineString, Polygon, Point
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Optional internet lookup support
try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    requests = None
    BeautifulSoup = None


# =========================================================
# CONFIG
# =========================================================

PROJECTED_CRS = "EPSG:5070"
INPUT_CRS = "EPSG:4326"

# Downtown Baltimore reference point
DOWNTOWN_BALTIMORE_LON = -76.6122
DOWNTOWN_BALTIMORE_LAT = 39.2904

# Optional target tract workflow:
# Set this to a full 11-digit tract GEOID if you want a specific tract and its neighbors exported.
# Example format only: "24510123456"
# For Charles Village, this must be a BALTIMORE CITY tract GEOID, so your tract file must include Baltimore City.
TARGET_TRACT_ID = None

TOP_5_HIGH_SCHOOLS = [
    {
        "rank": 1,
        "school_name": "Eastern Technical High School",
        "address": "1100 Mace Avenue, Baltimore, MD 21221",
        "lon": -76.4598,
        "lat": 39.3088,
    },
    {
        "rank": 2,
        "school_name": "Western School of Technology & Environmental Science",
        "address": "100 Kenwood Avenue, Catonsville, MD 21228",
        "lon": -76.7332,
        "lat": 39.2698,
    },
    {
        "rank": 3,
        "school_name": "George W. Carver Center for Arts & Technology",
        "address": "938 York Road, Towson, MD 21204",
        "lon": -76.6046,
        "lat": 39.4105,
    },
    {
        "rank": 4,
        "school_name": "Towson High School",
        "address": "69 Cedar Avenue, Towson, MD 21286",
        "lon": -76.6018,
        "lat": 39.4003,
    },
    {
        "rank": 5,
        "school_name": "Hereford High School",
        "address": "17301 York Road, Parkton, MD 21120",
        "lon": -76.6500,
        "lat": 39.6416,
    },
]


# =========================================================
# HELPERS
# =========================================================

def meters_to_miles(meters: float) -> float:
    return meters / 1609.344


# =========================================================
# PATH RESOLUTION
# =========================================================

def resolve_tract_path(
    preferred_d_drive_roots: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    allowed_extensions: Optional[List[str]] = None,
    max_depth: int = 6
) -> str:
    """
    Search for a Baltimore County / Baltimore-area tract file.
    Strongly prefers Baltimore matches and rejects obvious wrong geographies.
    """
    if preferred_d_drive_roots is None:
        preferred_d_drive_roots = [
            r"D:\\",
            r"D:\\Senior-Design",
            r"D:\\TransitNightmare",
            r"D:\\BusPosition",
            r"D:\\data",
            str(Path(__file__).resolve().parent),
            str(Path.cwd()),
        ]

    if keywords is None:
        keywords = [
            "baltimore county",
            "baltimore_county",
            "baltimorecounty",
            "baltimore city",
            "baltimore_city",
            "baltimorecity",
            "tract",
            "tracts",
            "census tract",
            "census_tract",
            "bus position",
            "transit nightmare",
        ]

    if allowed_extensions is None:
        allowed_extensions = [".shp", ".geojson", ".gpkg", ".json"]

    keywords = [k.lower() for k in keywords]
    allowed_extensions = [ext.lower() for ext in allowed_extensions]
    banned_terms = ["maricopa", "arizona", "phoenix"]

    search_roots = [Path(p) for p in preferred_d_drive_roots]
    candidates = []

    def score_path(path: Path) -> int:
        path_str = str(path).lower()
        name_str = path.name.lower()
        parent_str = str(path.parent).lower()

        if any(term in path_str for term in banned_terms):
            return -10000

        score = 0
        if "baltimore" in path_str:
            score += 50
        if "county" in path_str or "city" in path_str:
            score += 20
        if "tract" in name_str:
            score += 15

        for kw in keywords:
            if kw in name_str:
                score += 12
            if kw in parent_str:
                score += 8
            if kw in path_str:
                score += 5

        if path.suffix.lower() == ".shp":
            score += 5
        elif path.suffix.lower() in [".geojson", ".gpkg"]:
            score += 3

        return score

    def safe_walk(root: Path, current_depth: int = 0):
        if current_depth > max_depth:
            return
        if not root.exists() or not root.is_dir():
            return

        try:
            for entry in root.iterdir():
                try:
                    if entry.is_file() and entry.suffix.lower() in allowed_extensions:
                        score = score_path(entry)
                        if score > 0:
                            candidates.append((score, entry))
                    elif entry.is_dir():
                        safe_walk(entry, current_depth + 1)
                except (PermissionError, OSError):
                    continue
        except (PermissionError, OSError):
            return

    for root in search_roots:
        safe_walk(root, 0)

    if not candidates:
        raise FileNotFoundError("Could not locate a Baltimore-area tract geospatial file.")

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_path = candidates[0]
    print(f"Resolved tract path: {best_path} (score={best_score})")
    return str(best_path)


# =========================================================
# SYNTHETIC BACKUP
# =========================================================

def make_synthetic_baltimore_county_mask() -> Polygon:
    base = Polygon([
        (0, 0), (10, 0), (11, 2), (12, 4), (12, 6), (11, 8), (10, 10),
        (8, 11), (6, 11.5), (4, 11), (2.5, 10), (1, 8.5), (0.3, 6.5),
        (-0.2, 4.5), (0, 2)
    ])

    notch1 = Point(1.0, 10.5).buffer(1.2, resolution=32)
    notch2 = Point(11.0, 1.0).buffer(1.0, resolution=32)
    notch3 = Point(10.8, 8.8).buffer(0.9, resolution=32)

    return base.difference(notch1.union(notch2).union(notch3))


def generate_synthetic_baltimore_county_tracts(
    rows: int = 14,
    cols: int = 14,
    cell_size: float = 6000.0,
    crs: str = PROJECTED_CRS
) -> gpd.GeoDataFrame:
    mask_unit = make_synthetic_baltimore_county_mask()
    scaled_coords = [(x * cell_size, y * cell_size) for x, y in mask_unit.exterior.coords]
    county_mask = Polygon(scaled_coords)

    polygons = []
    geoids = []
    counter = 1

    for r in range(rows):
        for c in range(cols):
            x0 = c * cell_size
            y0 = r * cell_size
            cell = Polygon([
                (x0, y0),
                (x0 + cell_size, y0),
                (x0 + cell_size, y0 + cell_size),
                (x0, y0 + cell_size)
            ])

            inter = cell.intersection(county_mask)
            if inter.is_empty:
                continue
            if inter.area < 0.20 * cell.area:
                continue

            polygons.append(inter)
            geoids.append(f"BCO_{counter:03d}")
            counter += 1

    return gpd.GeoDataFrame(
        {"GEOID": geoids, "geometry": polygons},
        geometry="geometry",
        crs=crs
    )


def generate_synthetic_high_schools_for_backup(tracts_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    bounds = tracts_gdf.total_bounds
    minx, miny, maxx, maxy = bounds

    synthetic_points = [
        Point(minx + 0.18 * (maxx - minx), maxy - 0.08 * (maxy - miny)),
        Point(minx + 0.33 * (maxx - minx), maxy - 0.12 * (maxy - miny)),
        Point(minx + 0.48 * (maxx - minx), maxy - 0.10 * (maxy - miny)),
        Point(minx + 0.62 * (maxx - minx), maxy - 0.14 * (maxy - miny)),
        Point(minx + 0.78 * (maxx - minx), maxy - 0.18 * (maxy - miny)),
    ]

    return gpd.GeoDataFrame(
        {
            "rank": [1, 2, 3, 4, 5],
            "school_name": [f"Synthetic Top School {i}" for i in range(1, 6)],
            "address": ["synthetic backup"] * 5,
            "geometry": synthetic_points
        },
        geometry="geometry",
        crs=tracts_gdf.crs
    )


# =========================================================
# DATA LOADING / CLEANING
# =========================================================

def load_tracts(path: str, crs_expected: Optional[str] = INPUT_CRS) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)

    if gdf.empty:
        raise ValueError("Input tract data is empty.")
    if gdf.crs is None:
        raise ValueError("Input tract data has no CRS.")

    if crs_expected and gdf.crs.to_string() != crs_expected:
        gdf = gdf.to_crs(crs_expected)

    return gdf


def validate_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf = gdf[~gdf.geometry.isna()]
    gdf = gdf[~gdf.geometry.is_empty]

    invalid_mask = ~gdf.geometry.is_valid
    if invalid_mask.any():
        gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].buffer(0)

    gdf = gdf[gdf.geometry.is_valid]

    if gdf.empty:
        raise ValueError("No valid geometries remain after cleaning.")

    return gdf


def detect_id_column(gdf: gpd.GeoDataFrame, preferred: str = "GEOID") -> str:
    if preferred in gdf.columns:
        return preferred

    candidates = [
        "geoid", "GEOID10", "GEOID20", "TRACTCE", "tractce", "TRACT", "tract",
        "OBJECTID", "objectid", "NAME", "name"
    ]

    lower_map = {c.lower(): c for c in gdf.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    for col in gdf.columns:
        if col != gdf.geometry.name:
            return col

    raise ValueError("Could not detect a tract ID column.")


def project_for_distance(gdf: gpd.GeoDataFrame, target_crs: str = PROJECTED_CRS) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS.")
    return gdf.to_crs(target_crs)


def compute_centroids(gdf: gpd.GeoDataFrame, projected_crs: str = PROJECTED_CRS) -> gpd.GeoDataFrame:
    gdf_proj = project_for_distance(gdf, projected_crs).copy()
    gdf_proj["centroid"] = gdf_proj.geometry.centroid
    gdf_proj["centroid_x"] = gdf_proj["centroid"].x
    gdf_proj["centroid_y"] = gdf_proj["centroid"].y
    return gdf_proj


# =========================================================
# LABELS + INTERNET LOOKUP
# =========================================================

def build_polygon_label_column(
    gdf: gpd.GeoDataFrame,
    id_col: str,
    preferred_name_cols: Optional[List[str]] = None,
    output_col: str = "polygon_label"
) -> gpd.GeoDataFrame:
    gdf = gdf.copy()

    if preferred_name_cols is None:
        preferred_name_cols = [
            "NAME", "Name", "name",
            "TRACT_NAME", "tract_name",
            "LABEL", "label",
            "NAMELSAD", "namelsad"
        ]

    chosen_name_col = None
    for col in preferred_name_cols:
        if col in gdf.columns:
            non_null = gdf[col].astype(str).str.strip().replace("nan", "")
            if (non_null != "").any():
                chosen_name_col = col
                break

    if chosen_name_col is not None:
        gdf[output_col] = (
            gdf[chosen_name_col].astype(str).str.strip()
            + " | "
            + gdf[id_col].astype(str)
        )
    else:
        gdf[output_col] = "Tract " + gdf[id_col].astype(str)

    return gdf


def tract_lookup_urls_from_geoid(geoid: str) -> dict:
    """
    Build tract lookup URLs from a full 11-digit tract GEOID.
    Works best for standard census tract GEOIDs.
    """
    geoid = str(geoid)

    if len(geoid) != 11 or not geoid.isdigit():
        return {
            "geoid": geoid,
            "census_reporter_url": None,
            "geocodio_url": None
        }

    state_fips = geoid[:2]
    county_fips = geoid[2:5]
    tract_code = geoid[5:]

    census_reporter_url = f"https://censusreporter.org/profiles/14000US{geoid}/"
    geocodio_url = f"https://www.geocod.io/geoids/maryland/baltimore-county-{state_fips}{county_fips}/{tract_code}/"

    return {
        "geoid": geoid,
        "census_reporter_url": census_reporter_url,
        "geocodio_url": geocodio_url
    }


def lookup_tract_name_online(geoid: str, timeout: int = 8) -> Optional[str]:
    """
    Optionally try to fetch a tract page title from Census Reporter.
    Returns None if internet, requests, or parsing is unavailable.
    """
    if requests is None or BeautifulSoup is None:
        return None

    urls = tract_lookup_urls_from_geoid(geoid)
    url = urls["census_reporter_url"]
    if not url:
        return None

    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else None
        return title
    except Exception:
        return None


def build_tract_lookup_table(gdf: gpd.GeoDataFrame, id_col: str, label_col: str = "polygon_label") -> pd.DataFrame:
    rows = []
    seen = set()

    for _, row in gdf.iterrows():
        geoid = str(row[id_col])
        if geoid in seen:
            continue
        seen.add(geoid)

        urls = tract_lookup_urls_from_geoid(geoid)
        online_name = lookup_tract_name_online(geoid)

        rows.append({
            "tract_id": geoid,
            "tract_label": str(row[label_col]) if label_col in gdf.columns else f"Tract {geoid}",
            "online_tract_name": online_name,
            "census_reporter_url": urls["census_reporter_url"],
            "geocodio_url": urls["geocodio_url"]
        })

    return pd.DataFrame(rows)


# =========================================================
# ADJACENCY / EDGE NETWORK
# =========================================================

def get_adjacent_pairs(
    gdf: gpd.GeoDataFrame,
    id_col: Optional[str] = None,
    require_shared_boundary: bool = True
) -> pd.DataFrame:
    gdf = gdf.copy()

    if id_col is None:
        gdf["tract_id"] = gdf.index.astype(str)
        id_col = "tract_id"

    sindex = gdf.sindex
    rows = []

    for idx, poly in gdf.geometry.items():
        possible = list(sindex.intersection(poly.bounds))

        for j in possible:
            if idx >= j:
                continue

            other = gdf.geometry[j]

            if not poly.touches(other):
                continue

            shared_len = poly.boundary.intersection(other.boundary).length

            if require_shared_boundary and shared_len <= 0:
                continue

            rows.append({
                "i": idx,
                "j": j,
                "tract_i": str(gdf.at[idx, id_col]),
                "tract_j": str(gdf.at[j, id_col]),
                "shared_boundary_length_m": float(shared_len)
            })

    return pd.DataFrame(rows)


def build_centroid_edge_network(
    tracts_gdf: gpd.GeoDataFrame,
    projected_crs: str = PROJECTED_CRS,
    id_col: Optional[str] = None,
    require_shared_boundary: bool = True,
    label_col: str = "polygon_label"
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    tr_proj = compute_centroids(tracts_gdf, projected_crs)

    if label_col not in tr_proj.columns:
        tr_proj = build_polygon_label_column(tr_proj, id_col=id_col, output_col=label_col)

    adj_df = get_adjacent_pairs(
        tr_proj,
        id_col=id_col,
        require_shared_boundary=require_shared_boundary
    )

    edge_rows = []

    for _, r in adj_df.iterrows():
        i, j = r["i"], r["j"]
        c_i = tr_proj.at[i, "centroid"]
        c_j = tr_proj.at[j, "centroid"]

        edge_rows.append({
            "i": i,
            "j": j,
            "tract_i": str(r["tract_i"]),
            "tract_j": str(r["tract_j"]),
            "tract_i_label": str(tr_proj.at[i, label_col]),
            "tract_j_label": str(tr_proj.at[j, label_col]),
            "centroid_i": c_i,
            "centroid_j": c_j,
            "source_centroid_x": c_i.x,
            "source_centroid_y": c_i.y,
            "target_centroid_x": c_j.x,
            "target_centroid_y": c_j.y,
            "shared_boundary_length_m": r["shared_boundary_length_m"],
            "edge_length_m": float(c_i.distance(c_j)),
            "is_adjacent": True,
            "geometry": LineString([c_i, c_j]),
        })

    return tr_proj, gpd.GeoDataFrame(edge_rows, geometry="geometry", crs=projected_crs)


# =========================================================
# DISTANCES
# =========================================================

def fallback_walk_distance(edge_gdf: gpd.GeoDataFrame, multiplier: float = 1.2) -> gpd.GeoDataFrame:
    edge_gdf = edge_gdf.copy()
    edge_gdf["walk_distance_m"] = edge_gdf.geometry.length * multiplier
    edge_gdf["walk_distance_type"] = "planar_approx"
    return edge_gdf


def fallback_drive_distance(edge_gdf: gpd.GeoDataFrame, multiplier: float = 1.35) -> gpd.GeoDataFrame:
    edge_gdf = edge_gdf.copy()
    edge_gdf["drive_distance_m"] = edge_gdf.geometry.length * multiplier
    edge_gdf["drive_distance_type"] = "planar_approx"
    return edge_gdf


def calculate_pair_distances(
    edge_gdf: gpd.GeoDataFrame,
    walk_router: Optional[Callable] = None,
    drive_router: Optional[Callable] = None
) -> gpd.GeoDataFrame:
    res = edge_gdf.copy()

    if walk_router is not None:
        res["walk_distance_m"] = res.apply(
            lambda r: walk_router(r["centroid_i"], r["centroid_j"]),
            axis=1
        )
        res["walk_distance_type"] = "network_router"
    else:
        res = fallback_walk_distance(res)

    if drive_router is not None:
        res["drive_distance_m"] = res.apply(
            lambda r: drive_router(r["centroid_i"], r["centroid_j"]),
            axis=1
        )
        res["drive_distance_type"] = "network_router"
    else:
        res = fallback_drive_distance(res)

    return res


# =========================================================
# SCHOOLS / DOWNTOWN TARGET
# =========================================================

def build_school_points(real_data: bool, tracts_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if not real_data:
        return generate_synthetic_high_schools_for_backup(tracts_gdf)

    school_df = pd.DataFrame(TOP_5_HIGH_SCHOOLS).copy()

    school_gdf = gpd.GeoDataFrame(
        school_df,
        geometry=gpd.points_from_xy(school_df["lon"], school_df["lat"]),
        crs=INPUT_CRS
    )
    return school_gdf.to_crs(tracts_gdf.crs)


def assign_schools_to_origin_tracts(
    schools_gdf: gpd.GeoDataFrame,
    tracts_gdf: gpd.GeoDataFrame,
    id_col: str
) -> gpd.GeoDataFrame:
    schools = schools_gdf.copy()

    if "school_name" not in schools.columns:
        if "name" in schools.columns:
            schools = schools.rename(columns={"name": "school_name"})
        else:
            schools["school_name"] = [f"School {i+1}" for i in range(len(schools))]

    if "address" not in schools.columns:
        schools["address"] = ""

    if "rank" not in schools.columns:
        schools["rank"] = range(1, len(schools) + 1)

    joined = gpd.sjoin(
        schools,
        tracts_gdf[[id_col, "geometry"]],
        how="left",
        predicate="within"
    ).drop(columns=["index_right"], errors="ignore")

    missing = joined[id_col].isna()

    if missing.any():
        tract_centroids = tracts_gdf.copy()
        tract_centroids["geometry"] = tract_centroids.geometry.centroid

        nearest_input = joined.loc[missing].copy()
        nearest_input = nearest_input[["rank", "school_name", "address", "geometry"]]

        nearest = gpd.sjoin_nearest(
            nearest_input,
            tract_centroids[[id_col, "geometry"]],
            how="left",
            distance_col="nearest_tract_dist_m"
        ).drop(columns=["index_right"], errors="ignore")

        joined.loc[missing, id_col] = nearest[id_col].values

    joined = joined.rename(columns={id_col: "origin_tract_id"})
    return joined


def get_downtown_target_tract(tracts_gdf: gpd.GeoDataFrame, id_col: str) -> str:
    downtown_point = gpd.GeoDataFrame(
        {"name": ["Downtown Baltimore"]},
        geometry=[Point(DOWNTOWN_BALTIMORE_LON, DOWNTOWN_BALTIMORE_LAT)],
        crs=INPUT_CRS
    ).to_crs(tracts_gdf.crs)

    tract_centroids = tracts_gdf.copy()
    tract_centroids["geometry"] = tract_centroids.geometry.centroid

    nearest = gpd.sjoin_nearest(
        downtown_point,
        tract_centroids[[id_col, "geometry"]],
        how="left",
        distance_col="dist_to_downtown_m"
    ).drop(columns=["index_right"], errors="ignore")

    return str(nearest.iloc[0][id_col])


# =========================================================
# GRAPH / MATRICES / NEIGHBORS
# =========================================================

def build_graph(edge_gdf: gpd.GeoDataFrame, weight_col: str = "drive_distance_m") -> nx.Graph:
    G = nx.Graph()
    for _, row in edge_gdf.iterrows():
        G.add_edge(
            row["tract_i"],
            row["tract_j"],
            weight=float(row[weight_col]),
            geometry=row["geometry"]
        )
    return G


def build_distance_matrix(
    edge_gdf: gpd.GeoDataFrame,
    weight_col: str = "drive_distance_m",
    node_label_mode: str = "label"
) -> pd.DataFrame:
    if node_label_mode == "label":
        left_col = "tract_i_label"
        right_col = "tract_j_label"
    else:
        left_col = "tract_i"
        right_col = "tract_j"

    nodes = sorted(set(edge_gdf[left_col]).union(set(edge_gdf[right_col])))
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    mat = np.full((len(nodes), len(nodes)), np.inf)
    np.fill_diagonal(mat, 0.0)

    for _, row in edge_gdf.iterrows():
        a = row[left_col]
        b = row[right_col]
        w = float(row[weight_col])

        i = node_to_idx[a]
        j = node_to_idx[b]
        mat[i, j] = w
        mat[j, i] = w

    return pd.DataFrame(mat, index=nodes, columns=nodes)


def build_neighbor_distance_table(edge_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    rows = []

    for _, row in edge_gdf.iterrows():
        base = {
            "walk_distance_m": float(row["walk_distance_m"]),
            "walk_distance_miles": meters_to_miles(float(row["walk_distance_m"])),
            "drive_distance_m": float(row["drive_distance_m"]),
            "drive_distance_miles": meters_to_miles(float(row["drive_distance_m"])),
            "edge_length_m": float(row["edge_length_m"]),
            "shared_boundary_length_m": float(row["shared_boundary_length_m"]),
        }

        rows.append({
            "tract_id": str(row["tract_i"]),
            "tract_label": str(row["tract_i_label"]),
            "neighbor_tract_id": str(row["tract_j"]),
            "neighbor_tract_label": str(row["tract_j_label"]),
            **base
        })
        rows.append({
            "tract_id": str(row["tract_j"]),
            "tract_label": str(row["tract_j_label"]),
            "neighbor_tract_id": str(row["tract_i"]),
            "neighbor_tract_label": str(row["tract_i_label"]),
            **base
        })

    return pd.DataFrame(rows).sort_values(
        ["tract_label", "neighbor_tract_label"]
    ).reset_index(drop=True)


def get_target_tract_and_neighbors(
    tracts_gdf: gpd.GeoDataFrame,
    edge_gdf: gpd.GeoDataFrame,
    id_col: str,
    target_tract_id: str
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    target_tract_id = str(target_tract_id)

    target_row = tracts_gdf[tracts_gdf[id_col].astype(str) == target_tract_id]
    if target_row.empty:
        return pd.DataFrame(), None

    target_series = target_row.iloc[0]

    neighbors = edge_gdf[
        (edge_gdf["tract_i"].astype(str) == target_tract_id) |
        (edge_gdf["tract_j"].astype(str) == target_tract_id)
    ].copy()

    neighbor_rows = []
    for _, row in neighbors.iterrows():
        if str(row["tract_i"]) == target_tract_id:
            neighbor_rows.append({
                "target_tract_id": target_tract_id,
                "target_tract_label": str(row["tract_i_label"]),
                "neighbor_tract_id": str(row["tract_j"]),
                "neighbor_tract_label": str(row["tract_j_label"]),
                "walk_distance_m": float(row["walk_distance_m"]),
                "walk_distance_miles": meters_to_miles(float(row["walk_distance_m"])),
                "drive_distance_m": float(row["drive_distance_m"]),
                "drive_distance_miles": meters_to_miles(float(row["drive_distance_m"])),
            })
        else:
            neighbor_rows.append({
                "target_tract_id": target_tract_id,
                "target_tract_label": str(row["tract_j_label"]),
                "neighbor_tract_id": str(row["tract_i"]),
                "neighbor_tract_label": str(row["tract_i_label"]),
                "walk_distance_m": float(row["walk_distance_m"]),
                "walk_distance_miles": meters_to_miles(float(row["walk_distance_m"])),
                "drive_distance_m": float(row["drive_distance_m"]),
                "drive_distance_miles": meters_to_miles(float(row["drive_distance_m"])),
            })

    return pd.DataFrame(neighbor_rows), target_series


def shortest_path_tracts(
    edge_gdf: gpd.GeoDataFrame,
    start_tract: str,
    end_tract: str,
    weight_col: str = "drive_distance_m"
):
    G = build_graph(edge_gdf, weight_col=weight_col)

    start_tract = str(start_tract)
    end_tract = str(end_tract)

    if start_tract not in G:
        raise ValueError(f"Start tract '{start_tract}' not found in graph.")
    if end_tract not in G:
        raise ValueError(f"End tract '{end_tract}' not found in graph.")

    path_nodes = nx.shortest_path(G, source=start_tract, target=end_tract, weight="weight")
    total_cost = nx.shortest_path_length(G, source=start_tract, target=end_tract, weight="weight")

    path_pairs = {tuple(sorted((a, b))) for a, b in zip(path_nodes[:-1], path_nodes[1:])}

    mask = edge_gdf.apply(
        lambda r: tuple(sorted((str(r["tract_i"]), str(r["tract_j"])))) in path_pairs,
        axis=1
    )
    path_edges_gdf = edge_gdf[mask].copy()

    return path_nodes, path_edges_gdf, total_cost


def summarize_route_distances(edge_gdf: gpd.GeoDataFrame, path_nodes: List[str]) -> dict:
    path_pairs = {tuple(sorted((a, b))) for a, b in zip(path_nodes[:-1], path_nodes[1:])}

    path_edge_rows = edge_gdf[
        edge_gdf.apply(
            lambda r: tuple(sorted((str(r["tract_i"]), str(r["tract_j"])))) in path_pairs,
            axis=1
        )
    ].copy()

    total_walk_m = float(path_edge_rows["walk_distance_m"].sum())
    total_drive_m = float(path_edge_rows["drive_distance_m"].sum())
    total_edge_m = float(path_edge_rows["edge_length_m"].sum())

    return {
        "path_edge_rows": path_edge_rows,
        "total_walk_m": total_walk_m,
        "total_drive_m": total_drive_m,
        "total_edge_m": total_edge_m
    }


# =========================================================
# PLOTTING
# =========================================================

def get_route_bounds(routes_gdf: gpd.GeoDataFrame, pad_fraction: float = 0.18):
    minx, miny, maxx, maxy = routes_gdf.total_bounds
    dx = maxx - minx
    dy = maxy - miny

    if dx == 0:
        dx = 1000
    if dy == 0:
        dy = 1000

    return (
        minx - pad_fraction * dx,
        maxx + pad_fraction * dx,
        miny - pad_fraction * dy,
        maxy + pad_fraction * dy
    )


def plot_top5_school_routes_dual_view(
    tracts_gdf: gpd.GeoDataFrame,
    edge_gdf: gpd.GeoDataFrame,
    school_routes: pd.DataFrame,
    schools_with_origins: gpd.GeoDataFrame,
    downtown_target_tract: str,
    id_col: str,
    output_path: str
):
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    ax1, ax2 = axes

    for ax in axes:
        tracts_gdf.plot(
            ax=ax,
            facecolor="#e0e0e0",
            edgecolor="black",
            linewidth=0.4,
            alpha=0.8
        )

    edge_gdf.plot(ax=ax1, color="#aaaaaa", linewidth=0.5, alpha=0.25)
    edge_gdf.plot(ax=ax2, color="#cccccc", linewidth=0.4, alpha=0.18)

    colors = ["red", "blue", "green", "purple", "orange"]
    route_lines_all = []

    for idx, (_, route_row) in enumerate(school_routes.iterrows()):
        color = colors[idx % len(colors)]
        route_edges = route_row["route_edges_gdf"]

        if route_edges is not None and len(route_edges) > 0:
            route_edges.plot(ax=ax1, color=color, linewidth=2.8, alpha=0.95)
            route_edges.plot(ax=ax2, color=color, linewidth=3.4, alpha=0.95)
            route_lines_all.append(route_edges)

    schools_with_origins.plot(
        ax=ax1,
        color="gold",
        edgecolor="black",
        markersize=90,
        marker="^",
        zorder=5
    )
    schools_with_origins.plot(
        ax=ax2,
        color="gold",
        edgecolor="black",
        markersize=110,
        marker="^",
        zorder=5
    )

    for _, row in schools_with_origins.iterrows():
        x, y = row.geometry.x, row.geometry.y
        label = f"{int(row['rank'])}. {row['school_name']}"
        ax1.text(x, y, label, fontsize=7, ha="left", va="bottom")
        ax2.text(x, y, label, fontsize=8, ha="left", va="bottom")

    downtown_poly = tracts_gdf[tracts_gdf[id_col].astype(str) == str(downtown_target_tract)]
    if not downtown_poly.empty:
        downtown_poly.plot(ax=ax1, facecolor="cyan", edgecolor="black", alpha=0.55)
        downtown_poly.plot(ax=ax2, facecolor="cyan", edgecolor="black", alpha=0.65)

        downtown_centroid = downtown_poly.geometry.centroid.iloc[0]
        ax1.scatter(downtown_centroid.x, downtown_centroid.y, s=80, c="black", zorder=6)
        ax2.scatter(downtown_centroid.x, downtown_centroid.y, s=95, c="black", zorder=6)
        ax1.text(downtown_centroid.x, downtown_centroid.y, "Downtown target tract", fontsize=8, ha="left", va="bottom")
        ax2.text(downtown_centroid.x, downtown_centroid.y, "Downtown target tract", fontsize=9, ha="left", va="bottom")

    ax1.set_title("Top 5 Baltimore County high school routes to downtown target tract\nCountywide view")
    ax1.set_axis_off()

    if route_lines_all:
        all_route_lines = pd.concat(route_lines_all, ignore_index=True)
        all_route_lines = gpd.GeoDataFrame(all_route_lines, geometry="geometry", crs=edge_gdf.crs)
        xmin, xmax, ymin, ymax = get_route_bounds(all_route_lines, pad_fraction=0.20)
        ax2.set_xlim(xmin, xmax)
        ax2.set_ylim(ymin, ymax)

    ax2.set_title("Top 5 school routes\nDowntown zoom")
    ax2.set_axis_off()

    legend_elements = [
        Patch(facecolor="#e0e0e0", edgecolor="black", label="Tracts"),
        Line2D([0], [0], color="#aaaaaa", lw=1.5, label="Adjacent centroid edges"),
        Line2D([0], [0], color="red", lw=3, label="Shortest-path routes"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gold",
               markeredgecolor="black", markersize=10, label="Top high schools"),
        Patch(facecolor="cyan", edgecolor="black", label="Downtown target tract"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="jpg")
    plt.show()


# =========================================================
# PIPELINE
# =========================================================

def run_pipeline(
    path: Optional[str],
    id_col: Optional[str],
    input_crs: str = INPUT_CRS,
    projected_crs: str = PROJECTED_CRS,
    require_shared_boundary: bool = True,
    walk_router: Optional[Callable] = None,
    drive_router: Optional[Callable] = None,
    synthetic_backup: bool = True
):
    using_synthetic = False

    if path is None:
        if not synthetic_backup:
            raise FileNotFoundError("No real tract path provided and synthetic backup disabled.")
        tracts = generate_synthetic_baltimore_county_tracts(crs=projected_crs)
        using_synthetic = True
        if id_col is None or id_col not in tracts.columns:
            id_col = "GEOID"
    else:
        try:
            tracts = load_tracts(path, crs_expected=input_crs)
            tracts = validate_geometries(tracts)
            if id_col is None or id_col not in tracts.columns:
                id_col = detect_id_column(tracts, preferred="GEOID")
        except Exception as exc:
            if not synthetic_backup:
                raise
            print(f"Falling back to synthetic county-shaped tract data because real data failed: {exc}")
            tracts = generate_synthetic_baltimore_county_tracts(crs=projected_crs)
            using_synthetic = True
            id_col = "GEOID"

    tracts = build_polygon_label_column(tracts, id_col=id_col, output_col="polygon_label")

    tr_proj, edges = build_centroid_edge_network(
        tracts,
        projected_crs=projected_crs,
        id_col=id_col,
        require_shared_boundary=require_shared_boundary,
        label_col="polygon_label"
    )

    edges = calculate_pair_distances(
        edges,
        walk_router=walk_router,
        drive_router=drive_router
    )

    return tr_proj, edges, using_synthetic, id_col


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    ID_COL = "GEOID"

    OUTPUT_JPG = "top5_high_schools_to_downtown_routes.jpg"
    OUTPUT_ROUTE_SUMMARY_CSV = "top5_high_school_route_summary.csv"
    OUTPUT_ROUTE_EDGES_CSV = "top5_high_school_route_edges.csv"
    OUTPUT_NEIGHBOR_CSV = "tract_neighbor_distances.csv"
    OUTPUT_DRIVE_MATRIX_CSV = "tract_drive_distance_matrix_labeled.csv"
    OUTPUT_WALK_MATRIX_CSV = "tract_walk_distance_matrix_labeled.csv"
    OUTPUT_LOOKUP_CSV = "tract_lookup_table.csv"
    OUTPUT_TARGET_NEIGHBORS_CSV = "target_tract_neighbors.csv"

    tract_path = None
    try:
        tract_path = resolve_tract_path(
            preferred_d_drive_roots=[
                r"D:\\",
                r"D:\\Senior-Design",
                r"D:\\TransitNightmare",
                r"D:\\BusPosition",
                r"D:\\data",
                str(Path(__file__).resolve().parent),
                str(Path.cwd()),
            ],
            keywords=[
                "baltimore county",
                "baltimore_county",
                "baltimorecounty",
                "baltimore city",
                "baltimore_city",
                "baltimorecity",
                "tract",
                "tracts",
                "census tract",
                "census_tract",
                "bus position",
                "transit nightmare",
            ],
            allowed_extensions=[".shp", ".geojson", ".gpkg", ".json"],
            max_depth=6
        )
        print(f"Using tract file: {tract_path}")
    except Exception as exc:
        print(f"No real Baltimore-area tract file found. Using synthetic tract backup. Reason: {exc}")
        tract_path = None

    tracts_projected, edge_network, using_synthetic, actual_id_col = run_pipeline(
        path=tract_path,
        id_col=ID_COL,
        input_crs=INPUT_CRS,
        projected_crs=PROJECTED_CRS,
        require_shared_boundary=True,
        walk_router=None,
        drive_router=None,
        synthetic_backup=True
    )

    if len(tracts_projected) < 2:
        raise ValueError("Need at least 2 tracts to compute routes.")

    # Labeled matrices + lookup table
    drive_distance_matrix = build_distance_matrix(edge_network, weight_col="drive_distance_m", node_label_mode="label")
    walk_distance_matrix = build_distance_matrix(edge_network, weight_col="walk_distance_m", node_label_mode="label")
    neighbor_distance_table = build_neighbor_distance_table(edge_network)
    tract_lookup_table = build_tract_lookup_table(tracts_projected, actual_id_col, label_col="polygon_label")

    drive_distance_matrix.to_csv(OUTPUT_DRIVE_MATRIX_CSV)
    walk_distance_matrix.to_csv(OUTPUT_WALK_MATRIX_CSV)
    neighbor_distance_table.to_csv(OUTPUT_NEIGHBOR_CSV, index=False)
    tract_lookup_table.to_csv(OUTPUT_LOOKUP_CSV, index=False)

    # Optional target tract and neighbors export
    if TARGET_TRACT_ID is not None:
        target_neighbors_df, target_row = get_target_tract_and_neighbors(
            tracts_projected, edge_network, actual_id_col, TARGET_TRACT_ID
        )
        if len(target_neighbors_df) > 0:
            target_neighbors_df.to_csv(OUTPUT_TARGET_NEIGHBORS_CSV, index=False)
            print(f"\nTarget tract neighbors saved to: {OUTPUT_TARGET_NEIGHBORS_CSV}")
            if target_row is not None:
                print("Target tract found:")
                print({
                    "tract_id": str(target_row[actual_id_col]),
                    "tract_label": str(target_row["polygon_label"]),
                })
        else:
            print(f"\nTARGET_TRACT_ID={TARGET_TRACT_ID} was not found in the loaded tract layer.")

    # School routes
    schools_gdf = build_school_points(real_data=not using_synthetic, tracts_gdf=tracts_projected)
    schools_with_origins = assign_schools_to_origin_tracts(schools_gdf, tracts_projected, actual_id_col)
    downtown_target_tract = get_downtown_target_tract(tracts_projected, actual_id_col)

    route_summary_rows = []
    route_edge_rows = []

    for _, school in schools_with_origins.sort_values("rank").iterrows():
        school_name = school["school_name"]
        origin_tract = str(school["origin_tract_id"])

        path_nodes, route_edges_gdf, _ = shortest_path_tracts(
            edge_network,
            start_tract=origin_tract,
            end_tract=downtown_target_tract,
            weight_col="drive_distance_m"
        )

        route_summary = summarize_route_distances(edge_network, path_nodes)

        total_walk_m = route_summary["total_walk_m"]
        total_drive_m = route_summary["total_drive_m"]

        route_summary_rows.append({
            "rank": int(school["rank"]),
            "school_name": school_name,
            "address": school["address"],
            "origin_tract_id": origin_tract,
            "destination_tract_id": downtown_target_tract,
            "num_tracts_in_path": len(path_nodes),
            "path_sequence": " -> ".join(path_nodes),
            "total_walk_m": total_walk_m,
            "total_walk_miles": meters_to_miles(total_walk_m),
            "total_drive_m": total_drive_m,
            "total_drive_miles": meters_to_miles(total_drive_m),
            "route_edges_gdf": route_edges_gdf
        })

        route_edges_export = route_summary["path_edge_rows"].copy()
        if len(route_edges_export) > 0:
            route_edges_export["school_rank"] = int(school["rank"])
            route_edges_export["school_name"] = school_name
            route_edges_export["destination_tract_id"] = downtown_target_tract
            route_edges_export["walk_distance_miles"] = route_edges_export["walk_distance_m"].apply(meters_to_miles)
            route_edges_export["drive_distance_miles"] = route_edges_export["drive_distance_m"].apply(meters_to_miles)
            route_edge_rows.append(route_edges_export)

    school_routes_df = pd.DataFrame(route_summary_rows)

    plot_top5_school_routes_dual_view(
        tracts_gdf=tracts_projected,
        edge_gdf=edge_network,
        school_routes=school_routes_df,
        schools_with_origins=schools_with_origins,
        downtown_target_tract=downtown_target_tract,
        id_col=actual_id_col,
        output_path=OUTPUT_JPG
    )

    school_routes_export = school_routes_df.drop(columns=["route_edges_gdf"], errors="ignore").copy()
    school_routes_export.to_csv(OUTPUT_ROUTE_SUMMARY_CSV, index=False)

    if route_edge_rows:
        route_edges_all = pd.concat(route_edge_rows, ignore_index=True)
        route_edges_all = route_edges_all.drop(columns=["geometry", "centroid_i", "centroid_j"], errors="ignore")
        route_edges_all.to_csv(OUTPUT_ROUTE_EDGES_CSV, index=False)

    print("\nUsed synthetic backup:")
    print(using_synthetic)

    print("\nActual tract ID column used:")
    print(actual_id_col)

    print("\nDowntown target tract:")
    print(downtown_target_tract)

    print("\nTop 5 school route summary:")
    display_cols = [
        "rank", "school_name", "origin_tract_id", "destination_tract_id",
        "total_walk_miles", "total_drive_miles", "num_tracts_in_path"
    ]
    print(school_routes_export[display_cols])

    print("\nDrive distance matrix preview:")
    print(drive_distance_matrix.iloc[:10, :10])

    print("\nWalk distance matrix preview:")
    print(walk_distance_matrix.iloc[:10, :10])

    print("\nTract lookup table preview:")
    print(tract_lookup_table.head(10))

    print(f"\nJPG saved to: {OUTPUT_JPG}")
    print(f"Route summary CSV saved to: {OUTPUT_ROUTE_SUMMARY_CSV}")
    print(f"Route edges CSV saved to: {OUTPUT_ROUTE_EDGES_CSV}")
    print(f"Neighbor distances CSV saved to: {OUTPUT_NEIGHBOR_CSV}")
    print(f"Drive distance matrix CSV saved to: {OUTPUT_DRIVE_MATRIX_CSV}")
    print(f"Walk distance matrix CSV saved to: {OUTPUT_WALK_MATRIX_CSV}")
    print(f"Tract lookup CSV saved to: {OUTPUT_LOOKUP_CSV}")