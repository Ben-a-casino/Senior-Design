from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

Mode = Literal["walk", "drive"]
INVALID_STATUS = "INVALID_NON_NEIGHBOR"


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
    walk_distance_matrix_csv: str = "tract_walk_distance_matrix_labeled.csv"
    walk_time_matrix_csv: str = "tract_walk_time_matrix_labeled.csv"
    drive_distance_matrix_csv: str = "tract_drive_distance_matrix_labeled.csv"
    drive_time_offpeak_matrix_csv: str = "tract_drive_time_matrix_offpeak_labeled.csv"
    drive_time_am_matrix_csv: str = "tract_drive_time_matrix_am_7_8_labeled.csv"
    drive_time_pm_matrix_csv: str = "tract_drive_time_matrix_pm_3_4_labeled.csv"
    query_results_csv: str = "tract_pair_query_results.csv"


# =========================================================
# BASIC HELPERS
# =========================================================


def meters_to_miles(meters: float) -> float:
    return meters / 1609.344


# =========================================================
# FILE DISCOVERY
# =========================================================


def resolve_tract_path(config: ProjectConfig) -> Optional[str]:
    """Search for a Baltimore-area tract file on likely project drives."""
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


# =========================================================
# ROUTING METRICS FOR DIRECT NEIGHBORS ONLY
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


# =========================================================
# NEIGHBOR VALIDATION AND QUERY UTILITIES
# =========================================================


def _normalize_pair(tract_a: str, tract_b: str) -> tuple[str, str]:
    return tuple(sorted((str(tract_a), str(tract_b))))



def build_neighbor_lookup(edge_gdf: gpd.GeoDataFrame) -> dict[tuple[str, str], pd.Series]:
    lookup: dict[tuple[str, str], pd.Series] = {}
    for _, row in edge_gdf.iterrows():
        lookup[_normalize_pair(row["tract_i"], row["tract_j"])] = row
    return lookup



def are_neighbors(edge_gdf: gpd.GeoDataFrame, tract_a: str, tract_b: str) -> bool:
    lookup = build_neighbor_lookup(edge_gdf)
    return _normalize_pair(tract_a, tract_b) in lookup



def get_direct_neighbor_metrics(edge_gdf: gpd.GeoDataFrame, tract_a: str, tract_b: str) -> dict[str, object]:
    pair = _normalize_pair(tract_a, tract_b)
    lookup = build_neighbor_lookup(edge_gdf)

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


# =========================================================
# OUTPUT TABLES
# =========================================================


def build_adjacency_matrix(
    edge_gdf: gpd.GeoDataFrame,
    weight_col: str,
    node_label_mode: str = "label",
    invalid_value: object = "INVALID",
) -> pd.DataFrame:
    """
    Build an adjacency-only matrix.

    Neighbor pairs receive a direct edge value.
    Non-neighbor pairs stay marked as INVALID.
    """
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



def build_neighbor_distance_table(edge_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for _, row in edge_gdf.iterrows():
        shared_metrics = {
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

        rows.extend(
            [
                {
                    "tract_id": str(row["tract_i"]),
                    "tract_label": str(row["tract_i_label"]),
                    "neighbor_tract_id": str(row["tract_j"]),
                    "neighbor_tract_label": str(row["tract_j_label"]),
                    **shared_metrics,
                },
                {
                    "tract_id": str(row["tract_j"]),
                    "tract_label": str(row["tract_j_label"]),
                    "neighbor_tract_id": str(row["tract_i"]),
                    "neighbor_tract_label": str(row["tract_i_label"]),
                    **shared_metrics,
                },
            ]
        )

    return pd.DataFrame(rows).sort_values(["tract_label", "neighbor_tract_label"]).reset_index(drop=True)


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



def build_output_tables(edge_network: gpd.GeoDataFrame) -> dict[str, pd.DataFrame]:
    return {
        "neighbor_table": build_neighbor_distance_table(edge_network),
        "walk_distance_matrix": build_adjacency_matrix(edge_network, "walk_distance_m", node_label_mode="label"),
        "walk_time_matrix": build_adjacency_matrix(edge_network, "walk_time_min", node_label_mode="label"),
        "drive_distance_matrix": build_adjacency_matrix(edge_network, "drive_distance_m", node_label_mode="label"),
        "drive_time_offpeak_matrix": build_adjacency_matrix(edge_network, "drive_time_min_offpeak", node_label_mode="label"),
        "drive_time_am_matrix": build_adjacency_matrix(edge_network, "drive_time_min_am_peak_7_8", node_label_mode="label"),
        "drive_time_pm_matrix": build_adjacency_matrix(edge_network, "drive_time_min_pm_peak_3_4", node_label_mode="label"),
    }



def export_outputs(outputs: dict[str, pd.DataFrame], files: OutputFiles) -> None:
    outputs["neighbor_table"].to_csv(files.neighbor_csv, index=False)
    outputs["walk_distance_matrix"].to_csv(files.walk_distance_matrix_csv)
    outputs["walk_time_matrix"].to_csv(files.walk_time_matrix_csv)
    outputs["drive_distance_matrix"].to_csv(files.drive_distance_matrix_csv)
    outputs["drive_time_offpeak_matrix"].to_csv(files.drive_time_offpeak_matrix_csv)
    outputs["drive_time_am_matrix"].to_csv(files.drive_time_am_matrix_csv)
    outputs["drive_time_pm_matrix"].to_csv(files.drive_time_pm_matrix_csv)


# =========================================================
# MAIN
# =========================================================


def main() -> None:
    config = ProjectConfig()
    files = OutputFiles()

    tract_nodes, edge_network, using_synthetic, id_col = prepare_tract_data(config)
    outputs = build_output_tables(edge_network)
    export_outputs(outputs, files)

    # Example pair checks.
    # Replace these tract IDs with real tract IDs from your dataset when testing.
    example_query_pairs: list[tuple[str, str]] = []
    if len(tract_nodes) >= 2:
        tract_ids = tract_nodes[id_col].astype(str).tolist()
        example_query_pairs.append((tract_ids[0], tract_ids[1]))
        example_query_pairs.append((tract_ids[0], tract_ids[-1]))

    if example_query_pairs:
        query_results = build_query_results_table(edge_network, example_query_pairs)
        query_results.to_csv(files.query_results_csv, index=False)
        logger.info("Query result preview:\n%s", query_results)

    logger.info("Used synthetic backup: %s", using_synthetic)
    logger.info("Actual tract ID column used: %s", id_col)
    logger.info("Number of tracts: %s", len(tract_nodes))
    logger.info("Number of direct neighbor edges: %s", len(edge_network))
    logger.info("Neighbor table saved to: %s", files.neighbor_csv)
    logger.info("Walk distance adjacency matrix saved to: %s", files.walk_distance_matrix_csv)
    logger.info("Walk time adjacency matrix saved to: %s", files.walk_time_matrix_csv)
    logger.info("Drive distance adjacency matrix saved to: %s", files.drive_distance_matrix_csv)
    logger.info("Drive offpeak adjacency matrix saved to: %s", files.drive_time_offpeak_matrix_csv)
    logger.info("Drive AM adjacency matrix saved to: %s", files.drive_time_am_matrix_csv)
    logger.info("Drive PM adjacency matrix saved to: %s", files.drive_time_pm_matrix_csv)

    if example_query_pairs:
        logger.info("Pair query results saved to: %s", files.query_results_csv)


if __name__ == "__main__":
    main()
