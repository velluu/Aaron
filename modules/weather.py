"""
weather.py — Weather blocking logic.

Blocking Rules:
  Trucks: blocked if  W_base * S_earth > 10  AND  W_base * S_rain > 30
  Drones: blocked if  W_base * S_wind  > 60  AND  W_base * S_vis  < 6
  Airspace (type 0): never blocked.
"""


def precompute_blocking(map_data: dict, sensor_data: dict) -> dict:
    """
    Precompute blocking status for every (road_type, t, vehicle_type).

    Returns:
        dict[(road_type, t, vehicle_type)] -> bool
    """
    T = map_data["T"]
    road_weights = map_data["road_weights"]
    blocked = {}

    rain = sensor_data["rainfall"]
    wind = sensor_data["wind"]
    vis = sensor_data["visibility"]
    earth = sensor_data["earth_shock"]

    for t in range(T):
        blocked[(0, t, "truck")] = False
        blocked[(0, t, "drone")] = False

        for rtype_str, weights in road_weights.items():
            rtype = int(rtype_str)
            wb = weights[t]

            # Truck blocking: both conditions must hold
            blocked[(rtype, t, "truck")] = (wb * earth[t] > 10) and (wb * rain[t] > 30)
            # Drone blocking: both conditions must hold
            blocked[(rtype, t, "drone")] = (wb * wind[t] > 60) and (wb * vis[t] < 6)

    return blocked
