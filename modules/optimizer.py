"""
optimizer.py — Elite multi-strategy optimizer with local search improvement.

Architecture:
  1. Phase 1: Multi-strategy greedy construction (8+ strategies)
  2. Phase 2: Insertion — try to fit missed objectives into gaps
  3. Phase 3: Swap improvement — move objectives between vehicles
  4. Phase 4: 2-opt within each vehicle — reorder objective sequence
  5. Phase 5: Re-insertion — final pass to fill any remaining gaps
"""

from modules.router import find_path, find_paths_to_targets, PathResult


# ---------------------------------------------------------------------------
#  Vehicle state
# ---------------------------------------------------------------------------

class Vehicle:
    """Tracks a vehicle's route, cost, and assigned objectives."""
    __slots__ = ('vid', 'vtype', 'node', 'time', 'segments', 'total_cost',
                 'total_score', 'assigned', 'start_node')

    def __init__(self, vid: str, vtype: str, start_node: int):
        self.vid = vid
        self.vtype = vtype
        self.start_node = start_node
        self.node = start_node
        self.time = 0
        self.segments = []       # list of (path_nodes, obj_id_or_None)
        self.total_cost = 0.0
        self.total_score = 0.0
        self.assigned = []       # objective ids in order

    def clone(self):
        v = Vehicle(self.vid, self.vtype, self.start_node)
        v.node = self.node
        v.time = self.time
        v.segments = [s for s in self.segments]
        v.total_cost = self.total_cost
        v.total_score = self.total_score
        v.assigned = self.assigned.copy()
        return v


# ---------------------------------------------------------------------------
#  Scoring
# ---------------------------------------------------------------------------

def obj_score(arrival: int, obj: dict, penalty: float) -> float:
    """Compute objective score given arrival time."""
    if arrival < obj["release"] or arrival > obj["deadline"]:
        return 0.0
    lateness = arrival - obj["release"]
    return max(0.0, obj["points"] - penalty * lateness)


# ---------------------------------------------------------------------------
#  Assignment helpers
# ---------------------------------------------------------------------------

def try_assign(vehicle: Vehicle, obj: dict, cost_engine, T: int,
               penalty: float) -> tuple | None:
    """
    Evaluate assigning obj to vehicle. Returns (profit, path_result,
    effective_arrival, score, travel_cost) or None.
    """
    target = obj["node"] - 1
    deadline = obj["deadline"]

    if vehicle.time > deadline:
        return None

    result = find_path(
        cost_engine, vehicle.node, vehicle.time,
        target, vehicle.vtype, T, deadline=deadline
    )
    if result is None:
        return None

    effective = max(result.arrival_time, obj["release"])
    if effective > deadline:
        return None

    sc = obj_score(effective, obj, penalty)
    profit = sc - result.total_cost
    return profit, result, effective, sc, result.total_cost


def do_assign(vehicle: Vehicle, obj: dict, path_result: PathResult,
              effective_arrival: int, sc: float) -> None:
    """Apply assignment to vehicle (mutates)."""
    target = obj["node"] - 1
    route = path_result.path[1:]  # skip current node (already in path)
    arrival = path_result.arrival_time
    wait = max(0, obj["release"] - arrival)
    wait_nodes = [target] * wait

    vehicle.segments.append((route + wait_nodes, obj["id"]))
    vehicle.total_cost += path_result.total_cost
    vehicle.total_score += sc
    vehicle.time = effective_arrival
    vehicle.node = target
    vehicle.assigned.append(obj["id"])


# ---------------------------------------------------------------------------
#  Phase 1: Greedy construction
# ---------------------------------------------------------------------------

def greedy_construct(vehicles: list[Vehicle], objectives: list[dict],
                     cost_engine, T: int, penalty: float,
                     sort_key=None, mode="global") -> list[Vehicle]:
    """
    Greedy construction with different modes:
      "global"   — always pick globally-best (vehicle, obj) pair
      "sorted"   — iterate objectives in sort order, assign first profitable
      "vehicle"  — round-robin: each vehicle picks its best objective in turn
    """
    remaining = {o["id"]: o for o in objectives}
    vs = [v.clone() for v in vehicles]

    if mode == "sorted" and sort_key:
        return _greedy_sorted(vs, remaining, cost_engine, T, penalty, sort_key)
    elif mode == "vehicle":
        return _greedy_vehicle_round_robin(vs, remaining, cost_engine, T, penalty)
    else:
        return _greedy_global(vs, remaining, cost_engine, T, penalty)


def _greedy_global(vs, remaining, cost_engine, T, penalty):
    """Global greedy: always pick the best (vehicle, obj) pair."""
    while remaining:
        best_metric = -float('inf')
        best_vi = None
        best_oid = None
        best_data = None

        for vi, v in enumerate(vs):
            if v.time >= T:
                continue
            for oid, obj in remaining.items():
                result = try_assign(v, obj, cost_engine, T, penalty)
                if result is None:
                    continue
                profit, pr, eff, sc, tc = result
                if sc <= 0:
                    continue

                time_used = max(1, eff - v.time)
                efficiency = profit / time_used
                metric = profit + efficiency * 2.0 + sc * 0.1

                if metric > best_metric:
                    best_metric = metric
                    best_vi = vi
                    best_oid = oid
                    best_data = (pr, eff, sc)

        if best_vi is None:
            break

        pr, eff, sc = best_data
        do_assign(vs[best_vi], remaining[best_oid], pr, eff, sc)
        del remaining[best_oid]

    return vs


def _greedy_sorted(vs, remaining, cost_engine, T, penalty, sort_key):
    """Process objectives in sorted order; assign each to best vehicle."""
    sorted_oids = sorted(remaining.keys(),
                         key=lambda oid: sort_key(remaining[oid]),
                         reverse=True)

    for oid in sorted_oids:
        if oid not in remaining:
            continue
        obj = remaining[oid]

        best_profit = -float('inf')
        best_vi = None
        best_data = None

        for vi, v in enumerate(vs):
            if v.time >= T:
                continue
            result = try_assign(v, obj, cost_engine, T, penalty)
            if result is None:
                continue
            profit, pr, eff, sc, tc = result
            if sc <= 0:
                continue
            if profit > best_profit:
                best_profit = profit
                best_vi = vi
                best_data = (pr, eff, sc)

        if best_vi is not None and best_profit > -float('inf'):
            pr, eff, sc = best_data
            do_assign(vs[best_vi], obj, pr, eff, sc)
            del remaining[oid]

    return vs


def _greedy_vehicle_round_robin(vs, remaining, cost_engine, T, penalty):
    """Round-robin: each vehicle picks its best objective in turn."""
    changed = True
    while remaining and changed:
        changed = False
        for vi, v in enumerate(vs):
            if v.time >= T or not remaining:
                continue

            best_metric = -float('inf')
            best_oid = None
            best_data = None

            for oid, obj in remaining.items():
                result = try_assign(v, obj, cost_engine, T, penalty)
                if result is None:
                    continue
                profit, pr, eff, sc, tc = result
                if sc <= 0:
                    continue
                time_used = max(1, eff - v.time)
                metric = profit + profit / time_used * 2.0
                if metric > best_metric:
                    best_metric = metric
                    best_oid = oid
                    best_data = (pr, eff, sc)

            if best_oid is not None:
                pr, eff, sc = best_data
                do_assign(v, remaining[best_oid], pr, eff, sc)
                del remaining[best_oid]
                changed = True

    return vs


# ---------------------------------------------------------------------------
#  Phase 2: Insertion — try to squeeze missed objectives into gaps
# ---------------------------------------------------------------------------

def insertion_pass(vehicles: list[Vehicle], all_objectives: list[dict],
                   cost_engine, T: int, penalty: float) -> list[Vehicle]:
    """Try to insert unassigned objectives into vehicles that have time left."""
    assigned_ids = set()
    for v in vehicles:
        assigned_ids.update(v.assigned)

    unassigned = [o for o in all_objectives if o["id"] not in assigned_ids]
    if not unassigned:
        return vehicles

    # Sort unassigned by points (descending) — try high-value first
    unassigned.sort(key=lambda o: o["points"], reverse=True)

    improved = True
    while improved:
        improved = False
        for obj in list(unassigned):
            best_profit = -float('inf')
            best_vi = None
            best_data = None

            for vi, v in enumerate(vehicles):
                if v.time >= T:
                    continue
                result = try_assign(v, obj, cost_engine, T, penalty)
                if result is None:
                    continue
                profit, pr, eff, sc, tc = result
                if profit <= 0 and sc <= 0:
                    continue
                if profit > best_profit:
                    best_profit = profit
                    best_vi = vi
                    best_data = (pr, eff, sc)

            if best_vi is not None:
                pr, eff, sc = best_data
                do_assign(vehicles[best_vi], obj, pr, eff, sc)
                unassigned.remove(obj)
                improved = True

    return vehicles


# ---------------------------------------------------------------------------
#  Phase 3: Swap improvement — try reassigning objectives between vehicles
# ---------------------------------------------------------------------------

def rebuild_vehicle(vehicle: Vehicle, obj_sequence: list[dict],
                    cost_engine, T: int, penalty: float,
                    start_node: int) -> Vehicle | None:
    """
    Rebuild a vehicle from scratch with a specific objective sequence.
    Returns new Vehicle or None if infeasible.
    """
    v = Vehicle(vehicle.vid, vehicle.vtype, start_node)
    for obj in obj_sequence:
        result = try_assign(v, obj, cost_engine, T, penalty)
        if result is None:
            return None
        profit, pr, eff, sc, tc = result
        do_assign(v, obj, pr, eff, sc)
    return v


def swap_improvement(vehicles: list[Vehicle], objectives_map: dict,
                     cost_engine, T: int, penalty: float) -> list[Vehicle]:
    """
    Try moving and swapping objectives between vehicle pairs.
    Accept changes that improve total profit.
    """
    improved = True
    rounds = 0
    max_rounds = 10

    while improved and rounds < max_rounds:
        improved = False
        rounds += 1

        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                vi = vehicles[i]
                vj = vehicles[j]

                current_profit = (vi.total_score - vi.total_cost +
                                  vj.total_score - vj.total_cost)

                # Try moving each objective from vi to vj
                for oid in list(vi.assigned):
                    obj = objectives_map[oid]

                    new_seq_i = [objectives_map[x] for x in vi.assigned if x != oid]
                    new_seq_j = [objectives_map[x] for x in vj.assigned] + [obj]
                    new_seq_j.sort(key=lambda o: o["release"])

                    new_vi = rebuild_vehicle(vi, new_seq_i, cost_engine, T,
                                            penalty, vi.start_node)
                    new_vj = rebuild_vehicle(vj, new_seq_j, cost_engine, T,
                                            penalty, vj.start_node)

                    if new_vi is None or new_vj is None:
                        continue

                    new_profit = (new_vi.total_score - new_vi.total_cost +
                                  new_vj.total_score - new_vj.total_cost)

                    if new_profit > current_profit + 0.1:
                        vehicles[i] = new_vi
                        vehicles[j] = new_vj
                        improved = True
                        break

                if improved:
                    break

                # Try swapping one objective from each vehicle
                for oid_i in list(vi.assigned):
                    for oid_j in list(vj.assigned):
                        obj_i = objectives_map[oid_i]
                        obj_j = objectives_map[oid_j]

                        new_seq_i = [objectives_map[x] for x in vi.assigned
                                     if x != oid_i] + [obj_j]
                        new_seq_j = [objectives_map[x] for x in vj.assigned
                                     if x != oid_j] + [obj_i]
                        new_seq_i.sort(key=lambda o: o["release"])
                        new_seq_j.sort(key=lambda o: o["release"])

                        new_vi = rebuild_vehicle(vi, new_seq_i, cost_engine, T,
                                                penalty, vi.start_node)
                        new_vj = rebuild_vehicle(vj, new_seq_j, cost_engine, T,
                                                penalty, vj.start_node)

                        if new_vi is None or new_vj is None:
                            continue

                        new_profit = (new_vi.total_score - new_vi.total_cost +
                                      new_vj.total_score - new_vj.total_cost)

                        if new_profit > current_profit + 0.1:
                            vehicles[i] = new_vi
                            vehicles[j] = new_vj
                            improved = True
                            break
                    if improved:
                        break

            if improved:
                break

    return vehicles


# ---------------------------------------------------------------------------
#  Phase 4: Reorder within vehicle — try permutations of objective sequence
# ---------------------------------------------------------------------------

def reorder_improvement(vehicles: list[Vehicle], objectives_map: dict,
                        cost_engine, T: int, penalty: float) -> list[Vehicle]:
    """
    For each vehicle, try reordering via pairwise swaps and sort heuristics.
    """
    for vi_idx in range(len(vehicles)):
        v = vehicles[vi_idx]
        if len(v.assigned) <= 1:
            continue

        best_profit = v.total_score - v.total_cost
        best_v = v
        obj_list = [objectives_map[oid] for oid in v.assigned]

        # Pairwise swaps
        seq = list(v.assigned)
        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                new_seq = seq.copy()
                new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                obj_seq = [objectives_map[oid] for oid in new_seq]
                new_v = rebuild_vehicle(v, obj_seq, cost_engine, T,
                                        penalty, v.start_node)
                if new_v is None:
                    continue
                new_profit = new_v.total_score - new_v.total_cost
                if new_profit > best_profit + 0.1:
                    best_profit = new_profit
                    best_v = new_v

        # Sort heuristics
        for sort_fn in [
            lambda o: o["release"],
            lambda o: o["deadline"],
            lambda o: -o["points"],
        ]:
            sorted_seq = sorted(obj_list, key=sort_fn)
            new_v = rebuild_vehicle(v, sorted_seq, cost_engine, T,
                                    penalty, v.start_node)
            if new_v is not None:
                new_profit = new_v.total_score - new_v.total_cost
                if new_profit > best_profit + 0.1:
                    best_profit = new_profit
                    best_v = new_v

        vehicles[vi_idx] = best_v

    return vehicles


# ---------------------------------------------------------------------------
#  Phase 5: Aggressive insertion — relax profit threshold
# ---------------------------------------------------------------------------

def aggressive_insertion(vehicles: list[Vehicle], all_objectives: list[dict],
                         cost_engine, T: int, penalty: float) -> list[Vehicle]:
    """
    Final pass: insert remaining objectives even if marginal profit,
    as long as score > 0 (some points are better than none).
    """
    assigned_ids = set()
    for v in vehicles:
        assigned_ids.update(v.assigned)

    unassigned = [o for o in all_objectives if o["id"] not in assigned_ids]
    unassigned.sort(key=lambda o: o["points"], reverse=True)

    for obj in unassigned:
        best_score = 0.0
        best_vi = None
        best_data = None

        for vi, v in enumerate(vehicles):
            if v.time >= T:
                continue
            result = try_assign(v, obj, cost_engine, T, penalty)
            if result is None:
                continue
            profit, pr, eff, sc, tc = result
            # Accept if we gain ANY net positive score
            if sc > tc and sc > best_score:
                best_score = sc
                best_vi = vi
                best_data = (pr, eff, sc)

        if best_vi is not None:
            pr, eff, sc = best_data
            do_assign(vehicles[best_vi], obj, pr, eff, sc)

    return vehicles


# ---------------------------------------------------------------------------
#  Build final path from segments
# ---------------------------------------------------------------------------

def build_path(vehicle: Vehicle, T: int) -> list[int]:
    """Convert vehicle segments into a flat path of length T (0-indexed)."""
    path = [vehicle.start_node]  # t=0

    for route_nodes, _ in vehicle.segments:
        path.extend(route_nodes)

    # Pad with waits
    while len(path) < T:
        path.append(path[-1])

    return path[:T]


# ---------------------------------------------------------------------------
#  Compute solution score (for comparing strategies)
# ---------------------------------------------------------------------------

def compute_total_profit(vehicles: list[Vehicle]) -> float:
    """Total profit = sum(score) - sum(cost) across all vehicles."""
    return sum(v.total_score - v.total_cost for v in vehicles)


def count_completed(vehicles: list[Vehicle]) -> int:
    """Total objectives completed across all vehicles."""
    return sum(len(v.assigned) for v in vehicles)


# ---------------------------------------------------------------------------
#  Main optimizer
# ---------------------------------------------------------------------------

def optimize(map_data: dict, sensor_data: dict, objectives_data: dict,
             cost_engine, blocking: dict) -> dict[str, list[int]]:
    """
    Main optimization pipeline:
    1. Try many greedy strategies → keep best
    2. Insertion of missed objectives
    3. Swap improvement between vehicles
    4. Reorder within vehicles
    5. Aggressive final insertion
    """
    T = map_data["T"]
    start_node = objectives_data["start_node"] - 1
    num_trucks = objectives_data["trucks"]
    num_drones = objectives_data["drones"]
    penalty = objectives_data["late_penalty_per_step"]
    objectives = objectives_data["objectives"]
    obj_map = {o["id"]: o for o in objectives}

    def make_vehicles():
        vs = []
        for i in range(1, num_trucks + 1):
            vs.append(Vehicle(f"truck{i}", "truck", start_node))
        for i in range(1, num_drones + 1):
            vs.append(Vehicle(f"drone{i}", "drone", start_node))
        return vs

    # ── Phase 1: Multi-strategy greedy construction ──────────────────────

    strategies = [
        # Global greedy (all pick same globally-best pair)
        ("global",               None, "global"),
        # Sorted: process objectives in a specific order
        ("sorted_max_pts",       lambda o: o["points"], "sorted"),
        ("sorted_deadline",      lambda o: -o["deadline"], "sorted"),
        ("sorted_density",       lambda o: o["points"] / max(1, o["deadline"] - o["release"]), "sorted"),
        ("sorted_release",       lambda o: -o["release"], "sorted"),
        ("sorted_tight_window",  lambda o: -(o["deadline"] - o["release"]), "sorted"),
        ("sorted_late_deadline", lambda o: o["deadline"], "sorted"),
        ("sorted_wide_window",   lambda o: o["deadline"] - o["release"], "sorted"),
        # Vehicle round-robin
        ("round_robin",          None, "vehicle"),
    ]

    best_vehicles = None
    best_profit = -float('inf')
    best_name = ""

    for name, sort_key, mode in strategies:
        vs = make_vehicles()
        result_vs = greedy_construct(vs, objectives, cost_engine, T,
                                     penalty, sort_key, mode)
        profit = compute_total_profit(result_vs)
        completed = count_completed(result_vs)
        print(f"    Strategy '{name}': "
              f"profit={profit:.0f}, completed={completed}/{len(objectives)}")

        if profit > best_profit:
            best_profit = profit
            best_vehicles = result_vs
            best_name = name

    print(f"    >>> Phase 1 best: '{best_name}' profit={best_profit:.0f}")

    # ── Phase 2: Insertion ───────────────────────────────────────────────

    print("    Running insertion pass...")
    best_vehicles = insertion_pass(best_vehicles, objectives, cost_engine,
                                   T, penalty)
    p2_profit = compute_total_profit(best_vehicles)
    p2_count = count_completed(best_vehicles)
    print(f"    >>> Phase 2: profit={p2_profit:.0f}, completed={p2_count}/{len(objectives)}")

    # ── Phase 3: Swap improvement ────────────────────────────────────────

    print("    Running swap improvement...")
    best_vehicles = swap_improvement(best_vehicles, obj_map, cost_engine,
                                      T, penalty)
    p3_profit = compute_total_profit(best_vehicles)
    p3_count = count_completed(best_vehicles)
    print(f"    >>> Phase 3: profit={p3_profit:.0f}, completed={p3_count}/{len(objectives)}")

    # ── Phase 4: Reorder within vehicles ─────────────────────────────────

    print("    Running reorder improvement...")
    best_vehicles = reorder_improvement(best_vehicles, obj_map, cost_engine,
                                         T, penalty)
    p4_profit = compute_total_profit(best_vehicles)
    p4_count = count_completed(best_vehicles)
    print(f"    >>> Phase 4: profit={p4_profit:.0f}, completed={p4_count}/{len(objectives)}")

    # ── Phase 2b: Re-insert after reordering ─────────────────────────────

    print("    Running post-reorder insertion...")
    best_vehicles = insertion_pass(best_vehicles, objectives, cost_engine,
                                   T, penalty)
    p4b_profit = compute_total_profit(best_vehicles)
    p4b_count = count_completed(best_vehicles)
    print(f"    >>> Phase 4b: profit={p4b_profit:.0f}, completed={p4b_count}/{len(objectives)}")

    # ── Phase 5: Aggressive final insertion ──────────────────────────────

    print("    Running aggressive insertion...")
    best_vehicles = aggressive_insertion(best_vehicles, objectives,
                                          cost_engine, T, penalty)
    p5_profit = compute_total_profit(best_vehicles)
    p5_count = count_completed(best_vehicles)
    print(f"    >>> Phase 5 (final): profit={p5_profit:.0f}, completed={p5_count}/{len(objectives)}")

    # ── Iterative global improvement loop ────────────────────────────────

    print("    Running iterative improvement loop...")
    for iteration in range(10):
        prev_profit = compute_total_profit(best_vehicles)

        best_vehicles = swap_improvement(best_vehicles, obj_map, cost_engine,
                                          T, penalty)
        best_vehicles = reorder_improvement(best_vehicles, obj_map, cost_engine,
                                             T, penalty)
        best_vehicles = insertion_pass(best_vehicles, objectives, cost_engine,
                                       T, penalty)
        best_vehicles = aggressive_insertion(best_vehicles, objectives,
                                              cost_engine, T, penalty)

        new_profit = compute_total_profit(best_vehicles)
        delta = new_profit - prev_profit
        print(f"      Iteration {iteration + 1}: profit={new_profit:.0f} (delta={delta:+.0f})")
        if delta < 1.0:
            break

    # ── Build final solution ─────────────────────────────────────────────

    print("\n    Vehicle summary:")
    for v in best_vehicles:
        print(f"      {v.vid}: {len(v.assigned)} objectives, "
              f"score={v.total_score:.0f}, cost={v.total_cost:.0f}, "
              f"profit={v.total_score - v.total_cost:.0f}")

    solution = {}
    for v in best_vehicles:
        path_0indexed = build_path(v, T)
        solution[v.vid] = [node + 1 for node in path_0indexed]

    return solution
