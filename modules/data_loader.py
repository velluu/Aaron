"""
data_loader.py — Loads all input JSON files from the root directory.

Handles both map1 and map2 naming conventions automatically.
"""

import json
import os
import glob


def _find_json(root: str, patterns: list[str]) -> str:
    """Find the first matching JSON file from a list of candidate patterns."""
    for pattern in patterns:
        matches = glob.glob(os.path.join(root, pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find any of {patterns} in {root}")


def load_map(root: str) -> dict:
    """Load the map file (N, T, adjacency matrix, road_weights)."""
    path = _find_json(root, ["public_map.json", "public_map_*.json"])
    with open(path, "r") as f:
        return json.load(f)


def load_sensor_data(root: str) -> dict:
    """Load sensor data (rainfall, wind, visibility, earth_shock)."""
    path = _find_json(root, ["sensor_data.json", "sensor_data_*.json"])
    with open(path, "r") as f:
        return json.load(f)


def load_objectives(root: str) -> dict:
    """Load objectives (start_node, trucks, drones, objectives, penalties)."""
    path = _find_json(root, ["objectives.json", "objectives_*.json"])
    with open(path, "r") as f:
        return json.load(f)


def load_all(root: str) -> tuple[dict, dict, dict]:
    """Load map, sensor_data, objectives from root dir."""
    return load_map(root), load_sensor_data(root), load_objectives(root)
