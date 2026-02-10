"""
cost.py — Traversal cost computation and efficient neighbor lookup.

Cost formulas:
  Airspace (type 0): cost = 0
  Safe road:         cost = W_base(type, t) * road_type
  Blocked road:      cost = W_base(type, t) * road_type * 5
  Wait:              cost = 0
"""


class CostEngine:
    """Efficient cost/neighbor lookup using precomputed blocking data."""

    __slots__ = ('adj', 'road_weights', 'N', 'T', 'blocking',
                 '_neighbor_cache', '_edge_types')

    def __init__(self, map_data: dict, sensor_data: dict, blocking: dict):
        self.adj = map_data["map"]
        self.road_weights = map_data["road_weights"]
        self.N = map_data["N"]
        self.T = map_data["T"]
        self.blocking = blocking

        # Precompute adjacency lists per node for speed
        self._edge_types = {}  # (i, j) -> road_type
        self._neighbor_cache = {}  # (node, vtype) -> list of (neighbor, road_type)

        for i in range(self.N):
            for vtype in ("truck", "drone"):
                neighbors = []
                for j in range(self.N):
                    if i == j:
                        continue
                    rtype = self.adj[i][j]
                    if rtype < 0:
                        continue
                    if vtype == "truck" and rtype == 0:
                        continue
                    neighbors.append((j, rtype))
                    self._edge_types[(i, j)] = rtype
                self._neighbor_cache[(i, vtype)] = neighbors

    def get_cost(self, i: int, j: int, t: int, vehicle_type: str) -> float:
        """Get traversal cost from node i to j at time t. Returns float('inf') if invalid."""
        if i == j:
            return 0.0
        rtype = self._edge_types.get((i, j))
        if rtype is None:
            return float('inf')
        if vehicle_type == "truck" and rtype == 0:
            return float('inf')
        if rtype == 0:
            return 0.0
        wb = self.road_weights[str(rtype)][t]
        if self.blocking.get((rtype, t, vehicle_type), False):
            return wb * rtype * 5
        return wb * rtype

    def get_neighbors(self, node: int, t: int, vehicle_type: str) -> list:
        """
        Get valid (neighbor, cost) pairs from node at time t.
        Includes staying in place (cost=0).
        """
        result = [(node, 0.0)]
        for j, rtype in self._neighbor_cache[(node, vehicle_type)]:
            if rtype == 0:
                result.append((j, 0.0))
            else:
                wb = self.road_weights[str(rtype)][t]
                if self.blocking.get((rtype, t, vehicle_type), False):
                    result.append((j, wb * rtype * 5))
                else:
                    result.append((j, wb * rtype))
        return result
