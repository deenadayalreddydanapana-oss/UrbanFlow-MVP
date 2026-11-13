"""Small A* implementation usable as a Part 3 routing adapter.

Graph format:
{
  "A": {"B": 4, "C": 2},
  "B": {"D": 5},
  ...
}
Weights must be non-negative. Heuristics are optional; without them A* behaves
like Dijkstra's algorithm and remains optimal.
"""
from __future__ import annotations

import heapq
from typing import Dict, Mapping, Optional, Tuple, List


class AStarRouter:
    def __init__(
        self,
        graph: Mapping[str, Mapping[str, float]],
        heuristic: Optional[Mapping[str, float]] = None,
    ):
        self.graph = graph
        self.heuristic = heuristic or {}
        for node, neighbours in graph.items():
            for neighbour, weight in neighbours.items():
                if weight < 0:
                    raise ValueError(f"Negative road cost: {node} -> {neighbour}")

    def find_route(self, start: str, goal: str) -> dict:
        if start not in self.graph or goal not in self.graph:
            raise ValueError("Start and goal must exist as graph nodes.")

        queue: List[Tuple[float, float, str]] = [
            (self.heuristic.get(start, 0.0), 0.0, start)
        ]
        came_from = {}
        cost_so_far = {start: 0.0}

        while queue:
            _, current_cost, current = heapq.heappop(queue)
            if current == goal:
                path = [goal]
                while path[-1] != start:
                    path.append(came_from[path[-1]])
                path.reverse()
                return {
                    "route": path,
                    "total_cost": round(cost_so_far[goal], 3),
                    "algorithm": "A*",
                }

            if current_cost > cost_so_far.get(current, float("inf")):
                continue

            for neighbour, edge_cost in self.graph.get(current, {}).items():
                new_cost = current_cost + float(edge_cost)
                if new_cost < cost_so_far.get(neighbour, float("inf")):
                    cost_so_far[neighbour] = new_cost
                    came_from[neighbour] = current
                    priority = new_cost + float(self.heuristic.get(neighbour, 0.0))
                    heapq.heappush(queue, (priority, new_cost, neighbour))

        return {"route": [], "total_cost": None, "algorithm": "A*", "error": "No route found."}
