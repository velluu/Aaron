"""
router.py — Time-dependent Dijkstra pathfinding on a time-expanded graph.

State = (node, time), cost = cumulative traversal cost.
Supports single-target and multi-target queries with deadlines.
"""

import heapq


class PathResult:
    """Result of a shortest-path query."""
    __slots__ = ('target', 'arrival_time', 'total_cost', 'path')

    def __init__(self, target: int, arrival_time: int,
                 total_cost: float, path: list[int]):
        self.target = target
        self.arrival_time = arrival_time
        self.total_cost = total_cost
        self.path = path


def time_dijkstra(cost_engine, start_node: int, start_time: int,
                  vehicle_type: str, T: int,
                  targets: set[int] = None,
                  deadline: int = None) -> dict[int, PathResult]:
    """
    Time-dependent Dijkstra from (start_node, start_time).

    Returns dict[target_node] -> PathResult for all reachable targets.
    """
    max_time = min(T, deadline + 1) if deadline is not None else T
    dist = {}
    prev = {}
    best = {}  # target -> (cost, time)

    start_state = (start_node, start_time)
    dist[start_state] = 0.0
    pq = [(0.0, start_node, start_time)]

    found_all = False

    while pq:
        cost, node, t = heapq.heappop(pq)

        if cost > dist.get((node, t), float('inf')):
            continue

        # Record if target
        if targets is None or node in targets:
            if node not in best or cost < best[node][0]:
                best[node] = (cost, t)

        # Early exit if all targets found
        if targets is not None and len(best) == len(targets):
            found_all = True

        if t >= max_time - 1:
            continue

        for neighbor, edge_cost in cost_engine.get_neighbors(node, t, vehicle_type):
            new_t = t + 1
            if new_t >= max_time:
                continue
            new_cost = cost + edge_cost
            if new_cost < dist.get((neighbor, new_t), float('inf')):
                dist[(neighbor, new_t)] = new_cost
                prev[(neighbor, new_t)] = (node, t)
                heapq.heappush(pq, (new_cost, neighbor, new_t))

    # Reconstruct paths
    results = {}
    for target, (cost, arrival_time) in best.items():
        path = []
        state = (target, arrival_time)
        while state != start_state:
            path.append(state[0])
            if state not in prev:
                break
            state = prev[state]
        path.append(start_node)
        path.reverse()
        results[target] = PathResult(target, arrival_time, cost, path)

    return results


def find_path(cost_engine, start_node: int, start_time: int,
              target_node: int, vehicle_type: str, T: int,
              deadline: int = None) -> PathResult | None:
    """Find cheapest path from start to a single target. Returns None if unreachable."""
    results = time_dijkstra(
        cost_engine, start_node, start_time, vehicle_type, T,
        targets={target_node}, deadline=deadline
    )
    return results.get(target_node)


def find_paths_to_targets(cost_engine, start_node: int, start_time: int,
                          targets: set[int], vehicle_type: str, T: int,
                          deadline: int = None) -> dict[int, PathResult]:
    """Find cheapest paths from start to multiple targets."""
    return time_dijkstra(
        cost_engine, start_node, start_time, vehicle_type, T,
        targets=targets, deadline=deadline
    )
