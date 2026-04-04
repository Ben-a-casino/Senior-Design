#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 11:31:49 2026

@author: charmid
"""


# editted the code to just output adjacency matrix since the original outputted a weighted one 
# which overrode the normal one

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
import logging

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# =========================================================
# CONSTANTS / CONFIG
# =========================================================

Mode = Literal["walk", "drive"]
INVALID_STATUS = "INVALID_NON_NEIGHBOR"
INVALID_MATRIX_VALUE = "INVALID"

DOWNTOWN_BALTIMORE_LON = -76.6122
DOWNTOWN_BALTIMORE_LAT = 39.2904

TOP_5_HIGH_SCHOOLS = [
    {"rank": 1, "school_name": "Eastern Technical High School", "address": "1100 Mace Avenue, Baltimore, MD 21221", "lon": -76.4598, "lat": 39.3088},
    {"rank": 2, "school_name": "Western School of Technology & Environmental Science", "address": "100 Kenwood Avenue, Catonsville, MD 21228", "lon": -76.7332, "lat": 39.2698},
    {"rank": 3, "school_name": "George W. Carver Center for Arts & Technology", "address": "938 York Road, Towson, MD 21204", "lon": -76.6046, "lat": 39.4105},
    {"rank": 4, "school_name": "Towson High School", "address": "69 Cedar Avenue, Towson, MD 21286", "lon": -76.6018, "lat": 39.4003},
    {"rank": 5, "school_name": "Hereford High School", "address": "17301 York Road, Parkton, MD 21120", "lon": -76.6500, "lat": 39.6416},
]

# Top 10 high schools (deterministic coordinates)
TOP_10_HIGH_SCHOOLS = [
    ("Eastern Technical High School", -76.4598, 39.3088),
    ("Western School of Technology", -76.7332, 39.2698),
    ("Carver Center", -76.6046, 39.4105),
    ("Towson High School", -76.6018, 39.4003),
    ("Hereford High School", -76.6500, 39.6416),
    ("Dulaney High School", -76.5765, 39.3930),
    ("Perry Hall High School", -76.4430, 39.3549),
    ("Franklin High School", -76.7500, 39.3500),
    ("Parkville High School", -76.5976, 39.3690),
    ("Catonsville High School", -76.7121, 39.2718),
]


@dataclass(frozen=True)
class RouteResult:
    distance_m: float
    travel_time_min: float
    mode: str
    time_window: Optional[str]
    source: str


@dataclass(frozen=True)
class ProjectConfig:
    input_crs: str = "EPSG:4326"
    projected_crs: str = "EPSG:5070"
    default_id_col: str = "GEOID"
    require_shared_boundary: bool = True
    use_synthetic_backup: bool = True
    create_school_routes: bool = True
    search_roots: tuple[str, ...] = (
        r"D:\\",
        r"D:\\Senior-Design",
        r"D:\\TransitNightmare",
        r"D:\\BusPosition",
        r"D:\\data",
    )
    search_keywords: tuple[str, ...] = (
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
    )
    allowed_extensions: tuple[str, ...] = (".shp", ".geojson", ".gpkg", ".json")
    max_search_depth: int = 6
    walk_distance_factor: float = 1.20
    drive_distance_factor: float = 1.35
    walk_speed_m_per_min: float = 80.4672
    drive_speeds_m_per_min: dict[str, float] = field(
        default_factory=lambda: {
            "OFFPEAK": 536.448,
            "AM_PEAK_7_8": 402.336,
            "PM_PEAK_3_4": 375.7824,
        }
    )


@dataclass(frozen=True)
class OutputFiles:
    neighbor_csv: str = "tract_neighbor_distances.csv"
    walk_distance_matrix_csv: str = "tract_walk_distance_matrix_adjacent_only.csv"
    walk_time_matrix_csv: str = "tract_walk_time_matrix_adjacent_only.csv"
    drive_distance_matrix_csv: str = "tract_drive_distance_matrix_adjacent_only.csv"
    drive_time_offpeak_matrix_csv: str = "tract_drive_time_matrix_offpeak_adjacent_only.csv"
    drive_time_am_matrix_csv: str = "tract_drive_time_matrix_am_7_8_adjacent_only.csv"
    drive_time_pm_matrix_csv: str = "tract_drive_time_matrix_pm_3_4_adjacent_only.csv"
    query_results_csv: str = "tract_pair_query_results.csv"
    adjacency_map_png: str = "tract_adjacency_only_map.png"
    distance_heatmap_png: str = "adjacent_only_distance_heatmaps.png"
    school_route_map_png: str = "top5_high_schools_to_downtown_routes.png"
    school_route_summary_csv: str = "top5_high_school_route_summary.csv"
    school_route_edges_csv: str = "top5_high_school_route_edges.csv"


# =========================================================
# BASIC HELPERS
# =========================================================


def meters_to_miles(meters: float) -> float:
    return meters / 1609.344


# =========================================================
# FILE DISCOVERY / INPUT PREP
# =========================================================


def resolve_tract_path(config: ProjectConfig) -> Optional[str]:
    """Search likely folders for a Baltimore-area tract file."""
    banned_terms = {"maricopa", "arizona", "phoenix"}
    search_roots = [Path(root) for root in (*config.search_roots, str(Path.cwd()), str(Path(__file__).resolve().parent))]

    def score_path(path: Path) -> int:
        path_text = str(path).lower()
        name_text = path.name.lower()
        parent_text = str(path.parent).lower()

        if any(term in path_text for term in banned_terms):
            return -10_000

        score = 0
        if "baltimore" in path_text:
            score += 50
        if "county" in path_text or "city" in path_text:
            score += 20
        if "tract" in name_text:
            score += 15

        for keyword in config.search_keywords:
            if keyword in name_text:
                score += 12
            if keyword in parent_text:
                score += 8
            if keyword in path_text:
                score += 5

        if path.suffix.lower() == ".shp":
            score += 5
        elif path.suffix.lower() in {".geojson", ".gpkg"}:
            score += 3
        return score

    candidates: list[tuple[int, Path]] = []

    def walk_directory(root: Path, depth: int = 0) -> None:
        if depth > config.max_search_depth or not root.exists() or not root.is_dir():
            return
        try:
            for entry in root.iterdir():
                try:
                    if entry.is_file() and entry.suffix.lower() in config.allowed_extensions:
                        score = score_path(entry)
                        if score > 0:
                            candidates.append((score, entry))
                    elif entry.is_dir():
                        walk_directory(entry, depth + 1)
                except (PermissionError, OSError):
                    continue
        except (PermissionError, OSError):
            return

    for root in search_roots:
        walk_directory(root)

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_path = candidates[0]
    logger.info("Resolved tract path: %s (score=%s)", best_path, best_score)
    return str(best_path)


# =========================================================
# SYNTHETIC BACKUP DATA
# =========================================================


def make_synthetic_baltimore_county_mask() -> Polygon:
    base = Polygon([
        (0, 0), (10, 0), (11, 2), (12, 4), (12, 6), (11, 8), (10, 10),
        (8, 11), (6, 11.5), (4, 11), (2.5, 10), (1, 8.5), (0.3, 6.5),
        (-0.2, 4.5), (0, 2),
    ])
    notch1 = Point(1.0, 10.5).buffer(1.2, resolution=32)
    notch2 = Point(11.0, 1.0).buffer(1.0, resolution=32)
    notch3 = Point(10.8, 8.8).buffer(0.9, resolution=32)
    return base.difference(notch1.union(notch2).union(notch3))



def generate_synthetic_tracts(projected_crs: str) -> gpd.GeoDataFrame:
    rows, cols, cell_size = 14, 14, 6000.0
    county_mask = make_synthetic_baltimore_county_mask()
    county_mask = Polygon([(x * cell_size, y * cell_size) for x, y in county_mask.exterior.coords])

    polygons: list[Polygon] = []
    geoids: list[str] = []
    counter = 1

    for row in range(rows):
        for col in range(cols):
            x0 = col * cell_size
            y0 = row * cell_size
            cell = Polygon([
                (x0, y0),
                (x0 + cell_size, y0),
                (x0 + cell_size, y0 + cell_size),
                (x0, y0 + cell_size),
            ])
            intersection = cell.intersection(county_mask)
            if intersection.is_empty or intersection.area < 0.20 * cell.area:
                continue
            polygons.append(intersection)
            geoids.append(f"BCO_{counter:03d}")
            counter += 1

    return gpd.GeoDataFrame({"GEOID": geoids, "geometry": polygons}, geometry="geometry", crs=projected_crs)



def generate_synthetic_high_schools(tracts_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    bounds = tracts_gdf.total_bounds
    minx, miny, maxx, maxy = bounds
    points = [
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
            "geometry": points,
        },
        geometry="geometry",
        crs=tracts_gdf.crs,
    )


# =========================================================
# TRACT LOADING AND CLEANING
# =========================================================


def load_tracts(path: str, input_crs: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError("Input tract file is empty.")
    if gdf.crs is None:
        raise ValueError("Input tract file has no CRS.")
    if gdf.crs.to_string() != input_crs:
        gdf = gdf.to_crs(input_crs)
    return gdf



def validate_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    cleaned = gdf.copy()
    cleaned = cleaned[~cleaned.geometry.isna()]
    cleaned = cleaned[~cleaned.geometry.is_empty]

    invalid = ~cleaned.geometry.is_valid
    if invalid.any():
        cleaned.loc[invalid, "geometry"] = cleaned.loc[invalid, "geometry"].buffer(0)

    cleaned = cleaned[cleaned.geometry.is_valid]
    if cleaned.empty:
        raise ValueError("No valid tract geometries remain after cleaning.")
    return cleaned



def detect_id_column(gdf: gpd.GeoDataFrame, preferred: str) -> str:
    if preferred in gdf.columns:
        return preferred

    candidates = [
        "geoid", "GEOID10", "GEOID20", "TRACTCE", "tractce", "TRACT", "tract",
        "OBJECTID", "objectid", "NAME", "name",
    ]
    lower_lookup = {column.lower(): column for column in gdf.columns}

    for candidate in candidates:
        if candidate.lower() in lower_lookup:
            return lower_lookup[candidate.lower()]

    for column in gdf.columns:
        if column != gdf.geometry.name:
            return column

    raise ValueError("Could not detect a tract ID column.")



def add_polygon_labels(gdf: gpd.GeoDataFrame, id_col: str, output_col: str = "polygon_label") -> gpd.GeoDataFrame:
    name_candidates = ["NAME", "Name", "name", "TRACT_NAME", "tract_name", "LABEL", "label", "NAMELSAD", "namelsad"]
    labeled = gdf.copy()
    chosen_name_col = None

    for column in name_candidates:
        if column in labeled.columns:
            values = labeled[column].astype(str).str.strip().replace("nan", "")
            if (values != "").any():
                chosen_name_col = column
                break

    if chosen_name_col:
        labeled[output_col] = labeled[chosen_name_col].astype(str).str.strip() + " | " + labeled[id_col].astype(str)
    else:
        labeled[output_col] = "Tract " + labeled[id_col].astype(str)
    return labeled



def project_and_add_centroids(gdf: gpd.GeoDataFrame, projected_crs: str) -> gpd.GeoDataFrame:
    projected = gdf.to_crs(projected_crs).copy()
    projected["centroid"] = projected.geometry.centroid
    projected["centroid_x"] = projected["centroid"].x
    projected["centroid_y"] = projected["centroid"].y
    return projected


# =========================================================
# ADJACENCY NETWORK
# =========================================================


def get_adjacent_pairs(tracts_gdf: gpd.GeoDataFrame, id_col: str, require_shared_boundary: bool) -> pd.DataFrame:
    sindex = tracts_gdf.sindex
    rows: list[dict[str, object]] = []

    for idx, geometry in tracts_gdf.geometry.items():
        for other_idx in sindex.intersection(geometry.bounds):
            if idx >= other_idx:
                continue

            other = tracts_gdf.geometry[other_idx]
            if not geometry.touches(other):
                continue

            shared_boundary_length = geometry.boundary.intersection(other.boundary).length
            if require_shared_boundary and shared_boundary_length <= 0:
                continue

            rows.append(
                {
                    "i": idx,
                    "j": other_idx,
                    "tract_i": str(tracts_gdf.at[idx, id_col]),
                    "tract_j": str(tracts_gdf.at[other_idx, id_col]),
                    "shared_boundary_length_m": float(shared_boundary_length),
                }
            )

    return pd.DataFrame(rows)



def build_edge_network(
    tracts_gdf: gpd.GeoDataFrame,
    id_col: str,
    projected_crs: str,
    require_shared_boundary: bool,
    label_col: str = "polygon_label",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    tracts = project_and_add_centroids(tracts_gdf, projected_crs)
    adjacency = get_adjacent_pairs(tracts, id_col=id_col, require_shared_boundary=require_shared_boundary)

    edge_rows: list[dict[str, object]] = []
    for _, pair in adjacency.iterrows():
        i, j = pair["i"], pair["j"]
        centroid_i = tracts.at[i, "centroid"]
        centroid_j = tracts.at[j, "centroid"]

        edge_rows.append(
            {
                "i": i,
                "j": j,
                "tract_i": str(pair["tract_i"]),
                "tract_j": str(pair["tract_j"]),
                "tract_i_label": str(tracts.at[i, label_col]),
                "tract_j_label": str(tracts.at[j, label_col]),
                "centroid_i": centroid_i,
                "centroid_j": centroid_j,
                "shared_boundary_length_m": float(pair["shared_boundary_length_m"]),
                "edge_length_m": float(centroid_i.distance(centroid_j)),
                "is_adjacent": True,
                "geometry": LineString([centroid_i, centroid_j]),
            }
        )

    edge_gdf = gpd.GeoDataFrame(edge_rows, geometry="geometry", crs=projected_crs)
    return tracts, edge_gdf

def build_binary_adjacency_matrix(
    edge_gdf: gpd.GeoDataFrame,
    node_label_mode: str = "id",
) -> pd.DataFrame:
    if node_label_mode == "label":
        left_col, right_col = "tract_i_label", "tract_j_label"
    else:
        left_col, right_col = "tract_i", "tract_j"

    nodes = sorted(set(edge_gdf[left_col]).union(set(edge_gdf[right_col])))
    matrix = pd.DataFrame(0, index=nodes, columns=nodes, dtype=int)

    for _, row in edge_gdf.iterrows():
        matrix.loc[row[left_col], row[right_col]] = 1
        matrix.loc[row[right_col], row[left_col]] = 1

    return matrix
# =========================================================
# EDGE METRICS / MATRICES
# =========================================================


def estimate_route(origin: Point, destination: Point, mode: Mode, config: ProjectConfig, time_window: Optional[str] = None) -> RouteResult:
    straight_line_m = float(origin.distance(destination))

    if mode == "walk":
        distance_m = straight_line_m * config.walk_distance_factor
        return RouteResult(
            distance_m=distance_m,
            travel_time_min=distance_m / config.walk_speed_m_per_min,
            mode="walk",
            time_window=time_window,
            source="fallback_planar_walk",
        )

    if mode == "drive":
        distance_m = straight_line_m * config.drive_distance_factor
        speed = config.drive_speeds_m_per_min.get(time_window or "OFFPEAK", config.drive_speeds_m_per_min["OFFPEAK"])
        return RouteResult(
            distance_m=distance_m,
            travel_time_min=distance_m / speed,
            mode="drive",
            time_window=time_window,
            source="fallback_planar_drive",
        )

    raise ValueError(f"Unsupported mode: {mode}")



def calculate_pair_metrics(edge_gdf: gpd.GeoDataFrame, config: ProjectConfig) -> gpd.GeoDataFrame:
    metrics = edge_gdf.copy()
    walk_results = [estimate_route(row["centroid_i"], row["centroid_j"], "walk", config) for _, row in metrics.iterrows()]
    drive_offpeak = [estimate_route(row["centroid_i"], row["centroid_j"], "drive", config, "OFFPEAK") for _, row in metrics.iterrows()]
    drive_am = [estimate_route(row["centroid_i"], row["centroid_j"], "drive", config, "AM_PEAK_7_8") for _, row in metrics.iterrows()]
    drive_pm = [estimate_route(row["centroid_i"], row["centroid_j"], "drive", config, "PM_PEAK_3_4") for _, row in metrics.iterrows()]

    metrics["walk_distance_m"] = [result.distance_m for result in walk_results]
    metrics["walk_time_min"] = [result.travel_time_min for result in walk_results]
    metrics["walk_distance_type"] = [result.source for result in walk_results]
    metrics["walk_distance_miles"] = metrics["walk_distance_m"].apply(meters_to_miles)

    metrics["drive_distance_m"] = [result.distance_m for result in drive_offpeak]
    metrics["drive_distance_type"] = [result.source for result in drive_offpeak]
    metrics["drive_distance_miles"] = metrics["drive_distance_m"].apply(meters_to_miles)
    metrics["drive_time_min_offpeak"] = [result.travel_time_min for result in drive_offpeak]
    metrics["drive_time_min_am_peak_7_8"] = [result.travel_time_min for result in drive_am]
    metrics["drive_time_min_pm_peak_3_4"] = [result.travel_time_min for result in drive_pm]

    return metrics


'''
def build_adjacency_matrix(
    edge_gdf: gpd.GeoDataFrame,
    weight_col: str,
    node_label_mode: str = "label",
    invalid_value: object = INVALID_MATRIX_VALUE,
) -> pd.DataFrame:
    left_col, right_col = ("tract_i_label", "tract_j_label") if node_label_mode == "label" else ("tract_i", "tract_j")
    nodes = sorted(set(edge_gdf[left_col]).union(set(edge_gdf[right_col])))
    matrix = pd.DataFrame(invalid_value, index=nodes, columns=nodes, dtype=object)

    for node in nodes:
        matrix.loc[node, node] = 0.0

    for _, row in edge_gdf.iterrows():
        value = float(row[weight_col])
        matrix.loc[row[left_col], row[right_col]] = value
        matrix.loc[row[right_col], row[left_col]] = value

    return matrix
'''

def build_graph_from_edges(edge_gdf: gpd.GeoDataFrame, weight_col: str) -> nx.Graph:
    graph = nx.Graph()
    for _, row in edge_gdf.iterrows():
        graph.add_edge(str(row["tract_i"]), str(row["tract_j"]), weight=float(row[weight_col]))
    return graph



def build_all_pairs_shortest_path_matrix(edge_gdf: gpd.GeoDataFrame, weight_col: str) -> pd.DataFrame:
    graph = build_graph_from_edges(edge_gdf, weight_col)
    nodes = sorted(graph.nodes())
    matrix = pd.DataFrame(np.inf, index=nodes, columns=nodes)
    for node in nodes:
        matrix.loc[node, node] = 0.0

    for source, lengths in nx.all_pairs_dijkstra_path_length(graph, weight="weight"):
        for target, value in lengths.items():
            matrix.loc[source, target] = value
    return matrix


# =========================================================
# NEIGHBOR LOOKUP / QUERY HELPERS
# =========================================================


def _normalize_pair(tract_a: str, tract_b: str) -> tuple[str, str]:
    return tuple(sorted((str(tract_a), str(tract_b))))



def build_neighbor_lookup(edge_gdf: gpd.GeoDataFrame) -> dict[tuple[str, str], pd.Series]:
    return {_normalize_pair(row["tract_i"], row["tract_j"]): row for _, row in edge_gdf.iterrows()}



def get_direct_neighbor_metrics(edge_gdf: gpd.GeoDataFrame, tract_a: str, tract_b: str) -> dict[str, object]:
    lookup = build_neighbor_lookup(edge_gdf)
    pair = _normalize_pair(tract_a, tract_b)

    if pair not in lookup:
        return {
            "tract_a": str(tract_a),
            "tract_b": str(tract_b),
            "status": INVALID_STATUS,
            "message": "These tracts do not share a boundary, so no direct tract-to-tract distance is reported.",
        }

    row = lookup[pair]
    return {
        "tract_a": str(tract_a),
        "tract_b": str(tract_b),
        "status": "VALID_NEIGHBOR_PAIR",
        "tract_i": str(row["tract_i"]),
        "tract_j": str(row["tract_j"]),
        "tract_i_label": str(row["tract_i_label"]),
        "tract_j_label": str(row["tract_j_label"]),
        "walk_distance_m": float(row["walk_distance_m"]),
        "walk_distance_miles": float(row["walk_distance_miles"]),
        "walk_time_min": float(row["walk_time_min"]),
        "drive_distance_m": float(row["drive_distance_m"]),
        "drive_distance_miles": float(row["drive_distance_miles"]),
        "drive_time_min_offpeak": float(row["drive_time_min_offpeak"]),
        "drive_time_min_am_peak_7_8": float(row["drive_time_min_am_peak_7_8"]),
        "drive_time_min_pm_peak_3_4": float(row["drive_time_min_pm_peak_3_4"]),
        "shared_boundary_length_m": float(row["shared_boundary_length_m"]),
    }



def build_query_results_table(edge_gdf: gpd.GeoDataFrame, query_pairs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame([get_direct_neighbor_metrics(edge_gdf, tract_a, tract_b) for tract_a, tract_b in query_pairs])



def build_neighbor_distance_table(edge_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in edge_gdf.iterrows():
        shared = {
            "walk_distance_m": float(row["walk_distance_m"]),
            "walk_distance_miles": float(row["walk_distance_miles"]),
            "walk_time_min": float(row["walk_time_min"]),
            "drive_distance_m": float(row["drive_distance_m"]),
            "drive_distance_miles": float(row["drive_distance_miles"]),
            "drive_time_min_offpeak": float(row["drive_time_min_offpeak"]),
            "drive_time_min_am_peak_7_8": float(row["drive_time_min_am_peak_7_8"]),
            "drive_time_min_pm_peak_3_4": float(row["drive_time_min_pm_peak_3_4"]),
            "edge_length_m": float(row["edge_length_m"]),
            "shared_boundary_length_m": float(row["shared_boundary_length_m"]),
            "status": "VALID_NEIGHBOR_PAIR",
        }
        rows.extend([
            {
                "tract_id": str(row["tract_i"]),
                "tract_label": str(row["tract_i_label"]),
                "neighbor_tract_id": str(row["tract_j"]),
                "neighbor_tract_label": str(row["tract_j_label"]),
                **shared,
            },
            {
                "tract_id": str(row["tract_j"]),
                "tract_label": str(row["tract_j_label"]),
                "neighbor_tract_id": str(row["tract_i"]),
                "neighbor_tract_label": str(row["tract_i_label"]),
                **shared,
            },
        ])

    return pd.DataFrame(rows).sort_values(["tract_label", "neighbor_tract_label"]).reset_index(drop=True)



def build_output_tables(edge_gdf: gpd.GeoDataFrame) -> dict[str, pd.DataFrame]:
    return {
        "binary_adjacency_matrix": build_binary_adjacency_matrix(edge_gdf, node_label_mode="id"),
        "binary_adjacency_matrix_labeled": build_binary_adjacency_matrix(edge_gdf, node_label_mode="label"),
        "neighbor_table": build_neighbor_distance_table(edge_gdf),
    }



def export_core_outputs(outputs: dict[str, pd.DataFrame], files: OutputFiles) -> None:
    outputs["neighbor_table"].to_csv(files.neighbor_csv, index=False)
    outputs["binary_adjacency_matrix"].to_csv("tract_binary_adjacency_matrix.csv")
    outputs["binary_adjacency_matrix_labeled"].to_csv("tract_binary_adjacency_matrix_labeled.csv")

# =========================================================
# SCHOOL ROUTES
# =========================================================


def build_school_points(real_data: bool, tracts_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if not real_data:
        return generate_synthetic_high_schools(tracts_gdf)

    school_df = pd.DataFrame(TOP_5_HIGH_SCHOOLS)
    schools = gpd.GeoDataFrame(
        school_df,
        geometry=gpd.points_from_xy(school_df["lon"], school_df["lat"]),
        crs="EPSG:4326",
    )
    return schools.to_crs(tracts_gdf.crs)



def assign_schools_to_origin_tracts(schools_gdf: gpd.GeoDataFrame, tracts_gdf: gpd.GeoDataFrame, id_col: str) -> gpd.GeoDataFrame:
    joined = gpd.sjoin(
        schools_gdf,
        tracts_gdf[[id_col, "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")

    missing = joined[id_col].isna()
    if missing.any():
        tract_centroids = tracts_gdf.copy()
        tract_centroids["geometry"] = tract_centroids.geometry.centroid
        nearest = gpd.sjoin_nearest(
            joined.loc[missing, ["rank", "school_name", "address", "geometry"]],
            tract_centroids[[id_col, "geometry"]],
            how="left",
            distance_col="nearest_tract_dist_m",
        ).drop(columns=["index_right"], errors="ignore")
        joined.loc[missing, id_col] = nearest[id_col].values

    return joined.rename(columns={id_col: "origin_tract_id"})



def get_downtown_target_tract(tracts_gdf: gpd.GeoDataFrame, id_col: str) -> str:
    downtown = gpd.GeoDataFrame(
        {"name": ["Downtown Baltimore"]},
        geometry=[Point(DOWNTOWN_BALTIMORE_LON, DOWNTOWN_BALTIMORE_LAT)],
        crs="EPSG:4326",
    ).to_crs(tracts_gdf.crs)

    centroids = tracts_gdf.copy()
    centroids["geometry"] = centroids.geometry.centroid
    nearest = gpd.sjoin_nearest(
        downtown,
        centroids[[id_col, "geometry"]],
        how="left",
        distance_col="dist_to_downtown_m",
    ).drop(columns=["index_right"], errors="ignore")
    return str(nearest.iloc[0][id_col])



def shortest_path_tracts(edge_gdf: gpd.GeoDataFrame, start_tract: str, end_tract: str, weight_col: str) -> tuple[list[str], gpd.GeoDataFrame, float]:
    graph = nx.Graph()
    for _, row in edge_gdf.iterrows():
        graph.add_edge(
            str(row["tract_i"]),
            str(row["tract_j"]),
            weight=float(row[weight_col]),
        )

    path_nodes = nx.shortest_path(graph, source=str(start_tract), target=str(end_tract), weight="weight")
    total_cost = nx.shortest_path_length(graph, source=str(start_tract), target=str(end_tract), weight="weight")

    lookup = build_neighbor_lookup(edge_gdf)
    path_rows = []
    for left, right in zip(path_nodes[:-1], path_nodes[1:]):
        pair = _normalize_pair(left, right)
        if pair in lookup:
            path_rows.append(lookup[pair])

    route_edges = gpd.GeoDataFrame(path_rows, geometry="geometry", crs=edge_gdf.crs)
    return path_nodes, route_edges, float(total_cost)



def summarize_route_distances(path_edges: gpd.GeoDataFrame) -> dict[str, object]:
    if path_edges.empty:
        return {"total_walk_m": 0.0, "total_drive_m": 0.0, "path_edge_rows": pd.DataFrame()}

    return {
        "total_walk_m": float(path_edges["walk_distance_m"].sum()),
        "total_drive_m": float(path_edges["drive_distance_m"].sum()),
        "path_edge_rows": path_edges.copy(),
    }



def plot_school_routes(
    tracts_gdf: gpd.GeoDataFrame,
    edge_gdf: gpd.GeoDataFrame,
    school_routes: pd.DataFrame,
    schools_with_origins: gpd.GeoDataFrame,
    downtown_target_tract: str,
    id_col: str,
    output_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    county_ax, zoom_ax = axes

    for axis in axes:
        tracts_gdf.plot(ax=axis, facecolor="#e0e0e0", edgecolor="black", linewidth=0.4, alpha=0.8)

    edge_gdf.plot(ax=county_ax, color="#aaaaaa", linewidth=0.5, alpha=0.25)
    edge_gdf.plot(ax=zoom_ax, color="#cccccc", linewidth=0.4, alpha=0.18)

    colors = ["red", "blue", "green", "purple", "orange"]
    for idx, (_, route_row) in enumerate(school_routes.iterrows()):
        route_edges = route_row.get("route_edges_gdf")
        if route_edges is None or len(route_edges) == 0:
            continue
        color = colors[idx % len(colors)]
        route_edges.plot(ax=county_ax, color=color, linewidth=2.8, alpha=0.95)
        route_edges.plot(ax=zoom_ax, color=color, linewidth=3.4, alpha=0.95)

    schools_with_origins.plot(ax=county_ax, color="gold", edgecolor="black", markersize=90, marker="^")
    schools_with_origins.plot(ax=zoom_ax, color="gold", edgecolor="black", markersize=110, marker="^")

    for _, school in schools_with_origins.iterrows():
        x, y = school.geometry.x, school.geometry.y
        label = f"{int(school['rank'])}. {school['school_name']}"
        county_ax.text(x, y, label, fontsize=7, ha="left", va="bottom")
        zoom_ax.text(x, y, label, fontsize=8, ha="left", va="bottom")

    downtown_polygon = tracts_gdf[tracts_gdf[id_col].astype(str) == str(downtown_target_tract)]
    if not downtown_polygon.empty:
        downtown_polygon.plot(ax=county_ax, facecolor="cyan", edgecolor="black", alpha=0.55)
        downtown_polygon.plot(ax=zoom_ax, facecolor="cyan", edgecolor="black", alpha=0.65)
        centroid = downtown_polygon.geometry.centroid.iloc[0]
        county_ax.scatter(centroid.x, centroid.y, s=80, c="black")
        zoom_ax.scatter(centroid.x, centroid.y, s=95, c="black")

    centroid = tracts_gdf.geometry.centroid
    minx, miny, maxx, maxy = centroid.x.quantile(0.15), centroid.y.quantile(0.15), centroid.x.quantile(0.85), centroid.y.quantile(0.85)
    zoom_ax.set_xlim(minx, maxx)
    zoom_ax.set_ylim(miny, maxy)

    county_ax.set_title("Top Baltimore County high school routes to downtown\nCountywide view")
    zoom_ax.set_title("Top Baltimore County high school routes to downtown\nZoomed view")
    county_ax.set_axis_off()
    zoom_ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def build_school_route_outputs(
    tracts_gdf: gpd.GeoDataFrame,
    edge_gdf: gpd.GeoDataFrame,
    id_col: str,
    using_synthetic: bool,
    files: OutputFiles,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    schools_gdf = build_school_points(real_data=not using_synthetic, tracts_gdf=tracts_gdf)
    schools_with_origins = assign_schools_to_origin_tracts(schools_gdf, tracts_gdf, id_col)
    downtown_target_tract = get_downtown_target_tract(tracts_gdf, id_col)

    route_summary_rows: list[dict[str, object]] = []
    route_edge_rows: list[pd.DataFrame] = []

    for _, school in schools_with_origins.sort_values("rank").iterrows():
        path_nodes, route_edges_gdf, _ = shortest_path_tracts(
            edge_gdf,
            start_tract=str(school["origin_tract_id"]),
            end_tract=downtown_target_tract,
            weight_col="drive_distance_m",
        )
        route_summary = summarize_route_distances(route_edges_gdf)

        route_summary_rows.append(
            {
                "rank": int(school["rank"]),
                "school_name": school["school_name"],
                "address": school["address"],
                "origin_tract_id": str(school["origin_tract_id"]),
                "destination_tract_id": downtown_target_tract,
                "num_tracts_in_path": len(path_nodes),
                "path_sequence": " -> ".join(path_nodes),
                "total_walk_m": route_summary["total_walk_m"],
                "total_walk_miles": meters_to_miles(route_summary["total_walk_m"]),
                "total_drive_m": route_summary["total_drive_m"],
                "total_drive_miles": meters_to_miles(route_summary["total_drive_m"]),
                "route_edges_gdf": route_edges_gdf,
            }
        )

        route_edges_export = route_summary["path_edge_rows"].copy()
        if not route_edges_export.empty:
            route_edges_export["school_rank"] = int(school["rank"])
            route_edges_export["school_name"] = school["school_name"]
            route_edges_export["destination_tract_id"] = downtown_target_tract
            route_edges_export["walk_distance_miles"] = route_edges_export["walk_distance_m"].apply(meters_to_miles)
            route_edges_export["drive_distance_miles"] = route_edges_export["drive_distance_m"].apply(meters_to_miles)
            route_edge_rows.append(route_edges_export)

    school_routes_df = pd.DataFrame(route_summary_rows)
    school_routes_df.drop(columns=["route_edges_gdf"], errors="ignore").to_csv(files.school_route_summary_csv, index=False)

    if route_edge_rows:
        pd.concat(route_edge_rows, ignore_index=True).to_csv(files.school_route_edges_csv, index=False)
    else:
        pd.DataFrame().to_csv(files.school_route_edges_csv, index=False)

    plot_school_routes(
        tracts_gdf=tracts_gdf,
        edge_gdf=edge_gdf,
        school_routes=school_routes_df,
        schools_with_origins=schools_with_origins,
        downtown_target_tract=downtown_target_tract,
        id_col=id_col,
        output_path=files.school_route_map_png,
    )
    return school_routes_df, schools_with_origins


# =========================================================
# VISUALIZATION HELPERS
# =========================================================


def matrix_to_numeric(matrix: pd.DataFrame) -> pd.DataFrame:
    numeric = matrix.where(matrix != INVALID_MATRIX_VALUE, np.nan)
    return numeric.apply(pd.to_numeric, errors="coerce")



def plot_adjacency_map(tract_nodes: gpd.GeoDataFrame, edge_network: gpd.GeoDataFrame, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 11))
    tract_nodes.plot(ax=ax, facecolor="#e8ecef", edgecolor="black", linewidth=0.4, alpha=0.9)
    edge_network.plot(ax=ax, color="#d1495b", linewidth=0.8, alpha=0.7)
    gpd.GeoDataFrame(geometry=tract_nodes["centroid"], crs=tract_nodes.crs).plot(ax=ax, color="#1f77b4", markersize=6, alpha=0.9)
    ax.set_title("Adjacent-Only Tract Network")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def plot_distance_heatmaps(walk_distance_matrix: pd.DataFrame, drive_distance_matrix: pd.DataFrame, output_path: str) -> None:
    walk_numeric = matrix_to_numeric(walk_distance_matrix)
    drive_numeric = matrix_to_numeric(drive_distance_matrix)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    walk_im = axes[0].imshow(walk_numeric.values, aspect="auto")
    drive_im = axes[1].imshow(drive_numeric.values, aspect="auto")

    axes[0].set_title("Adjacent-Only Walk Distance Matrix")
    axes[1].set_title("Adjacent-Only Drive Distance Matrix")
    for axis in axes:
        axis.set_xlabel("Tracts")
        axis.set_ylabel("Tracts")

    fig.colorbar(walk_im, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(drive_im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_top10_routes(
    tracts_gdf: gpd.GeoDataFrame,
    edge_gdf: gpd.GeoDataFrame,
    schools_gdf: gpd.GeoDataFrame,
    dest_tract_id: str,
    route_geoms: dict[str, list[LineString]],
    id_col: str,
    out_path: str = "top10_routes.png",
) -> None:
    """
    High-quality single-panel plot for top 10 routes.
    """
    plot_tracts = tracts_gdf.to_crs(PROJECTED_CRS).copy()
    plot_edges = edge_gdf.to_crs(PROJECTED_CRS).copy()
    plot_schools = schools_gdf.to_crs(PROJECTED_CRS).copy()

    fig, ax = plt.subplots(1, 1, figsize=(14, 12))

    # Base layer: tract polygons
    plot_tracts.plot(ax=ax, facecolor="#f3f3f3", edgecolor="#bfc0c2", linewidth=0.3)

    # adjacency edges
    if not plot_edges.empty:
        plot_edges.plot(ax=ax, color="#cfcfcf", linewidth=0.6, alpha=0.5)

    cmap = plt.cm.get_cmap("tab10")
    for idx, (school_name, geoms) in enumerate(route_geoms.items()):
        color = cmap(idx % 10)
        for geom in geoms:
            if geom is None or geom.is_empty:
                continue
            xs, ys = geom.xy
            ax.plot(xs, ys, color=color, linewidth=3.2, solid_capstyle="round", zorder=3)

    # Plot schools as yellow triangles
    plot_schools.plot(ax=ax, color="gold", edgecolor="black", marker="^", markersize=110, zorder=6)
    for i, row in plot_schools.reset_index(drop=True).iterrows():
        x, y = row.geometry.x, row.geometry.y
        label = f"{i+1}. {row['school_name']}"
        ax.text(x, y, label, fontsize=8, ha="left", va="bottom", zorder=7)

    # Highlight downtown tract
    downtown_poly = plot_tracts[plot_tracts[id_col].astype(str) == str(dest_tract_id)]
    if not downtown_poly.empty:
        downtown_poly.plot(ax=ax, facecolor="cyan", edgecolor="black", alpha=0.6, zorder=5)
        centroid = downtown_poly.geometry.centroid.iloc[0]
        ax.scatter(centroid.x, centroid.y, s=140, c="black", zorder=8)

    ax.set_title("Top 10 High School Routes to Downtown Baltimore", fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_top10_routes(
    tracts_gdf: gpd.GeoDataFrame,
    edge_gdf: gpd.GeoDataFrame,
    id_col: str,
    out_map: str = "top10_routes.png",
    out_csv: str = "top10_route_summary.csv",
) -> pd.DataFrame:
    """
    Build school points, assign to tracts, compute shortest paths to downtown, plot and save outputs.
    Returns summary DataFrame.
    """
    # build schools gdf
    schools_df = pd.DataFrame(TOP_10_HIGH_SCHOOLS, columns=["school_name", "lon", "lat"]).reset_index()
    schools_df = schools_df.rename(columns={"index": "rank"})
    schools_gdf = gpd.GeoDataFrame(
        schools_df,
        geometry=gpd.points_from_xy(schools_df.lon, schools_df.lat),
        crs="EPSG:4326",
    ).to_crs(tracts_gdf.crs)

    # assign to origin tracts
    schools_with_origins = assign_schools_to_origin_tracts(schools_gdf, tracts_gdf, id_col)

    # downtown target
    downtown_target = get_downtown_target_tract(tracts_gdf, id_col)

    summary_rows: list[dict[str, object]] = []
    route_geoms: dict[str, list[LineString]] = {}

    for _, school in schools_with_origins.sort_values("rank").iterrows():
        start = str(school["origin_tract_id"])
        school_name = school["school_name"]
        try:
            path_nodes, route_edges_gdf, total_cost = shortest_path_tracts(edge_gdf, start, downtown_target, weight_col="drive_distance_m")
        except Exception:
            path_nodes, route_edges_gdf, total_cost = [], gpd.GeoDataFrame(), float("nan")

        total_drive = float(total_cost) if not pd.isna(total_cost) else float("nan")
        summary_rows.append(
            {
                "school_name": school_name,
                "origin_tract": start,
                "destination_tract": downtown_target,
                "total_distance_m": total_drive,
                "total_distance_miles": meters_to_miles(total_drive) if not pd.isna(total_drive) else float("nan"),
                "num_tracts_in_path": len(path_nodes) if path_nodes else 0,
                "path_sequence": " -> ".join(path_nodes) if path_nodes else "",
            }
        )

        geoms = []
        if not route_edges_gdf.empty:
            for _, r in route_edges_gdf.iterrows():
                geoms.append(r.geometry)
        route_geoms[school_name] = geoms

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_csv, index=False)
    logger.info("Saved top10 route summary to %s", out_csv)

    plot_top10_routes(
        tracts_gdf=tracts_gdf,
        edge_gdf=edge_gdf,
        schools_gdf=schools_with_origins,
        dest_tract_id=downtown_target,
        route_geoms=route_geoms,
        id_col=id_col,
        out_path=out_map,
    )
    logger.info("Saved top10 route map to %s", out_map)
    return summary_df


# =========================================================
# PIPELINE
# =========================================================


def prepare_tract_data(config: ProjectConfig) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, bool, str]:
    tract_path = resolve_tract_path(config)
    using_synthetic = False

    if tract_path is None:
        if not config.use_synthetic_backup:
            raise FileNotFoundError("No real Baltimore-area tract file was found.")
        logger.warning("No real tract file found. Using synthetic backup tract data.")
        tracts = generate_synthetic_tracts(projected_crs=config.projected_crs)
        using_synthetic = True
        id_col = config.default_id_col
    else:
        try:
            tracts = load_tracts(tract_path, input_crs=config.input_crs)
            tracts = validate_geometries(tracts)
            id_col = detect_id_column(tracts, preferred=config.default_id_col)
        except Exception as exc:
            if not config.use_synthetic_backup:
                raise
            logger.warning("Failed to use real tract file (%s). Falling back to synthetic data.", exc)
            tracts = generate_synthetic_tracts(projected_crs=config.projected_crs)
            using_synthetic = True
            id_col = config.default_id_col

    tracts = add_polygon_labels(tracts, id_col=id_col)
    tract_nodes, edge_network = build_edge_network(
        tracts_gdf=tracts,
        id_col=id_col,
        projected_crs=config.projected_crs,
        require_shared_boundary=config.require_shared_boundary,
    )
    edge_network = calculate_pair_metrics(edge_network, config)
    return tract_nodes, edge_network, using_synthetic, id_col


# =========================================================
# MAIN
# =========================================================


def main() -> None:
    config = ProjectConfig()
    files = OutputFiles()

    tract_nodes, edge_network, using_synthetic, id_col = prepare_tract_data(config)
    outputs = build_output_tables(edge_network)
    export_core_outputs(outputs, files)

    plot_adjacency_map(tract_nodes, edge_network, files.adjacency_map_png)

    example_query_pairs: list[tuple[str, str]] = []
    if len(tract_nodes) >= 2:
        tract_ids = tract_nodes[id_col].astype(str).tolist()
        example_query_pairs.append((tract_ids[0], tract_ids[1]))
        example_query_pairs.append((tract_ids[0], tract_ids[-1]))

    if example_query_pairs:
        query_results = build_query_results_table(edge_network, example_query_pairs)
        query_results.to_csv(files.query_results_csv, index=False)

    logger.info("Used synthetic backup: %s", using_synthetic)
    logger.info("Actual tract ID column used: %s", id_col)
    logger.info("Number of tracts: %s", len(tract_nodes))
    logger.info("Number of direct neighbor edges: %s", len(edge_network))
    logger.info("Saved binary adjacency matrix to tract_binary_adjacency_matrix.csv")
    logger.info("Saved labeled binary adjacency matrix to tract_binary_adjacency_matrix_labeled.csv")

    print("\nBinary adjacency matrix:")
    print(outputs["binary_adjacency_matrix"])

if __name__ == "__main__":
    main()