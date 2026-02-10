"""
solution_writer.py — Write, validate, and score solutions.
"""

import json
import os


def write_solution(solution: dict[str, list[int]], root: str) -> str:
    """Write solution dict to solution.json in root directory."""
    path = os.path.join(root, "solution.json")
    with open(path, "w") as f:
        json.dump(solution, f, indent=2)
    return path


def validate_solution(solution: dict[str, list[int]], map_data: dict,
                      objectives_data: dict) -> list[str]:
    """
    Validate solution for correctness.
    Returns list of error strings (empty = valid).
    """
    errors = []
    T = map_data["T"]
    adj = map_data["map"]
    N = map_data["N"]

    expected_start = objectives_data["start_node"]
    num_trucks = objectives_data["trucks"]
    num_drones = objectives_data["drones"]

    # Check vehicle count
    truck_ids = [k for k in solution if k.startswith("truck")]
    drone_ids = [k for k in solution if k.startswith("drone")]
    if len(truck_ids) != num_trucks:
        errors.append(f"Expected {num_trucks} trucks, got {len(truck_ids)}")
    if len(drone_ids) != num_drones:
        errors.append(f"Expected {num_drones} drones, got {len(drone_ids)}")

    for vid, path in solution.items():
        vtype = "truck" if vid.startswith("truck") else "drone"

        if len(path) != T:
            errors.append(f"{vid}: path length {len(path)} != T={T}")
            continue

        if path[0] != expected_start:
            errors.append(f"{vid}: starts at {path[0]}, expected {expected_start}")

        for t in range(len(path) - 1):
            curr = path[t] - 1
            nxt = path[t + 1] - 1

            if curr < 0 or curr >= N or nxt < 0 or nxt >= N:
                errors.append(f"{vid} t={t}: node out of range")
                continue

            if curr == nxt:
                continue

            rtype = adj[curr][nxt]
            if rtype < 0:
                errors.append(f"{vid} t={t}: no edge {curr+1}->{nxt+1}")
            if vtype == "truck" and rtype == 0:
                errors.append(f"{vid} t={t}: truck on airspace {curr+1}->{nxt+1}")

    return errors


def score_solution(solution: dict[str, list[int]], map_data: dict,
                   sensor_data: dict, objectives_data: dict,
                   cost_engine) -> dict:
    """
    Compute total score for a solution.
    Returns dict with score breakdown.
    """
    T = map_data["T"]
    objectives = objectives_data["objectives"]
    penalty = objectives_data["late_penalty_per_step"]

    # Travel costs
    travel_costs = {}
    for vid, path in solution.items():
        vtype = "truck" if vid.startswith("truck") else "drone"
        total = 0.0
        for t in range(len(path) - 1):
            curr = path[t] - 1
            nxt = path[t + 1] - 1
            if curr == nxt:
                continue
            cost = cost_engine.get_cost(curr, nxt, t, vtype)
            if cost < float('inf'):
                total += cost
        travel_costs[vid] = total

    # Objective scoring
    obj_scores = {}
    obj_details = {}
    completed = set()

    for obj in objectives:
        oid = obj["id"]
        target = obj["node"]
        release = obj["release"]
        deadline = obj["deadline"]
        pmax = obj["points"]

        best_score = 0.0
        best_vid = None
        best_arrival = None

        for vid, path in solution.items():
            for t in range(release, min(deadline + 1, len(path))):
                if path[t] == target and oid not in completed:
                    lateness = t - release
                    score = max(0.0, pmax - penalty * lateness)
                    if score > best_score:
                        best_score = score
                        best_vid = vid
                        best_arrival = t

        if best_vid is not None:
            completed.add(oid)

        obj_scores[oid] = best_score
        obj_details[oid] = {
            "target": target,
            "score": best_score,
            "vehicle": best_vid,
            "arrival": best_arrival,
            "max_points": pmax,
        }

    total_obj = sum(obj_scores.values())
    total_travel = sum(travel_costs.values())

    return {
        "total_score": total_obj - total_travel,
        "total_objective_score": total_obj,
        "total_travel_cost": total_travel,
        "travel_costs": travel_costs,
        "objective_scores": obj_scores,
        "objective_details": obj_details,
        "objectives_completed": len(completed),
        "objectives_total": len(objectives),
    }
