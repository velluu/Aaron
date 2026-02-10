"""
main.py — Disaster Logistics Optimizer

Reads input files from the current directory, computes optimal vehicle
routes through a disaster-affected network, and writes solution.json.

Usage:
    python main.py
"""

import os
import sys
import json
import time as timer

from modules.data_loader import load_all
from modules.weather import precompute_blocking
from modules.cost import CostEngine
from modules.optimizer import optimize
from modules.solution_writer import write_solution, validate_solution, score_solution


def main():
    start = timer.time()
    root = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("  Disaster Logistics Optimizer v2.0")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────
    print("\n[1/5] Loading input data...")
    map_data, sensor_data, objectives_data = load_all(root)
    N = map_data["N"]
    T = map_data["T"]
    trucks = objectives_data["trucks"]
    drones = objectives_data["drones"]
    num_obj = len(objectives_data["objectives"])
    total_pts = sum(o["points"] for o in objectives_data["objectives"])
    print(f"  Nodes: {N}, Time steps: {T}")
    print(f"  Fleet: {trucks} truck(s), {drones} drone(s)")
    print(f"  Objectives: {num_obj} ({total_pts} pts available)")

    # ── Precompute blocking ──────────────────────────────────────
    print("\n[2/5] Computing weather blocking...")
    blocking = precompute_blocking(map_data, sensor_data)
    blocked_count = sum(1 for v in blocking.values() if v)
    print(f"  {blocked_count} blocked (road_type, time, vehicle) combos")

    # ── Build cost engine ────────────────────────────────────────
    print("\n[3/5] Building cost engine...")
    cost_engine = CostEngine(map_data, sensor_data, blocking)
    print("  Ready")

    # ── Optimize ─────────────────────────────────────────────────
    print("\n[4/5] Running optimizer...")
    solution = optimize(map_data, sensor_data, objectives_data,
                        cost_engine, blocking)
    opt_time = timer.time() - start
    print(f"\n  Optimization time: {opt_time:.2f}s")

    # ── Validate & Score ─────────────────────────────────────────
    print("\n[5/5] Validating and scoring...")
    errors = validate_solution(solution, map_data, objectives_data)
    if errors:
        print("  VALIDATION ERRORS:")
        for e in errors:
            print(f"    - {e}")
        sys.exit(1)
    else:
        print("  Solution is VALID")

    result = score_solution(solution, map_data, sensor_data,
                            objectives_data, cost_engine)

    # ── Results ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Objectives completed: {result['objectives_completed']}/{result['objectives_total']}")
    print(f"  Objective score:      {result['total_objective_score']:.1f}")
    print(f"  Travel cost:          {result['total_travel_cost']:.1f}")
    print(f"  {'─'*40}")
    print(f"  TOTAL SCORE:          {result['total_score']:.1f}")

    print(f"\n  Travel costs:")
    for vid, cost in result["travel_costs"].items():
        print(f"    {vid}: {cost:.1f}")

    print(f"\n  Objectives:")
    for oid, d in sorted(result["objective_details"].items()):
        if d["vehicle"]:
            print(f"    Obj {oid:>2}: node {d['target']:>2} | "
                  f"{d['score']:>6.0f}/{d['max_points']:>4} pts | "
                  f"{d['vehicle']} @ t={d['arrival']} | DONE")
        else:
            print(f"    Obj {oid:>2}: node {d['target']:>2} | "
                  f"    0/{d['max_points']:>4} pts | MISSED")

    # ── Write output ─────────────────────────────────────────────
    output_path = write_solution(solution, root)
    total_time = timer.time() - start
    print(f"\n  Output: {output_path}")
    print(f"  Total time: {total_time:.2f}s")


if __name__ == "__main__":
    main()
