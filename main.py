"""
main.py — Disaster Logistics Optimizer

Reads input files from the current directory, computes optimal vehicle
routes through a disaster-affected network, and writes solution.json.

If a problemstatement/ folder exists with map1/map2 subfolders, it will
automatically solve both maps and save solutions to a solutions/ folder.

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


def solve(root: str, label: str = "") -> tuple[dict, dict]:
    """Run the full pipeline on whatever input files are in root. Returns score result."""
    start = timer.time()
    tag = f" [{label}]" if label else ""

    print(f"\n{'=' * 60}")
    print(f"  Disaster Logistics Optimizer v2.0{tag}")
    print(f"{'=' * 60}")

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
    else:
        print("  Solution is VALID")

    result = score_solution(solution, map_data, sensor_data,
                            objectives_data, cost_engine)

    # ── Results ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS{tag}")
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

    return result, solution


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    ps = os.path.join(root, "problemstatement")

    # Detect map folders
    map_dirs = []
    if os.path.isdir(ps):
        for name in sorted(os.listdir(ps)):
            sub = os.path.join(ps, name)
            if os.path.isdir(sub):
                # Check for nested folder (map1/map1/)
                inner = os.path.join(sub, name)
                if os.path.isdir(inner):
                    map_dirs.append((name, inner))
                else:
                    map_dirs.append((name, sub))

    if len(map_dirs) >= 2:
        # Multiple maps found — solve each directly from their folders
        solutions_dir = os.path.join(root, "solutions")
        os.makedirs(solutions_dir, exist_ok=True)
        summary = []

        for map_name, map_dir in map_dirs:
            result, solution = solve(map_dir, label=map_name)

            # Save to solutions/
            out = os.path.join(solutions_dir, f"solution_{map_name}.json")
            with open(out, "w") as fh:
                json.dump(solution, fh, indent=2)
            print(f"  Saved: {out}")

            summary.append((map_name, result))

        # Print summary
        print(f"\n{'=' * 60}")
        print(f"  SUMMARY")
        print(f"{'=' * 60}")
        for map_name, result in summary:
            print(f"  {map_name}: {result['objectives_completed']}/{result['objectives_total']} objectives, "
                  f"score = {result['total_score']:.0f}")

    else:
        # Single map — just solve whatever's in root
        solve(root)


if __name__ == "__main__":
    main()
