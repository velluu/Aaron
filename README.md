# Disaster Logistics Optimizer

Optimal vehicle routing through a disaster-affected network under dynamic weather conditions.

## How It Works

Guides a fleet of trucks and drones through a city with time-varying road conditions, delivering supplies to maximize total score while minimizing travel cost.

### Architecture

```
main.py                  # Entry point — run this
requirements.txt         # No external dependencies
modules/
  data_loader.py         # Loads map, sensor, and objective JSON files
  weather.py             # Computes road blocking from sensor data
  cost.py                # Traversal cost engine with precomputed lookups
  router.py              # Time-dependent Dijkstra pathfinding
  optimizer.py           # Multi-strategy greedy + local search optimizer
  solution_writer.py     # Validation, scoring, and JSON output
```

### Optimizer Pipeline

1. **Multi-strategy greedy construction** — 9 strategies (sorted by deadline, release, density, etc.)  
2. **Insertion pass** — squeeze missed objectives into remaining vehicle time  
3. **Swap improvement** — move/swap objectives between vehicles  
4. **Reorder improvement** — try better orderings within each vehicle  
5. **Aggressive insertion** — final pass accepting any net-positive objectives  
6. **Iterative refinement** — repeat phases 3–5 until no improvement  

### Usage

```bash
python main.py
```

Input files (`public_map.json`, `sensor_data.json`, `objectives.json`) must be in the same directory as `main.py`. Outputs `solution.json`.

## Results

| Map | Objectives | Score |
|-----|-----------|-------|
| Map 1 | 30/30 | 16,176 |
| Map 2 | 25/25 | 10,225 |
