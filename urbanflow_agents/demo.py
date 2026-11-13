"""Runnable demonstration with replaceable stand-in model components.

Run from the project root:
    python -m urbanflow_agents.demo

The detector and forecaster below are deterministic stand-ins, NOT trained SVM
or Decision Tree models. Replace them with your Part 2 and Part 1 model adapters.
"""
from .agents import Monitor, Analyzer, Dispatcher, UrbanFlowCoordinator
from .astar_router import AStarRouter


# Demo road network: edge costs represent travel time in minutes.
ROAD_GRAPH = {
    "A": {"B": 5, "C": 2},
    "B": {"A": 5, "D": 5},
    "C": {"A": 2, "D": 3, "E": 8},
    "D": {"B": 5, "C": 3, "E": 2},
    "E": {"C": 8, "D": 2},
}


class DemoSVMDetector:
    """Placeholder only: replace with the actual trained SVM inference code."""
    def __call__(self, sensor_data):
        crash_signal = sensor_data.get("camera_crash_signal", False)
        return {
            "incident_detected": bool(crash_signal),
            "incident_type": "crash" if crash_signal else "none",
            "confidence": 0.94 if crash_signal else 0.08,
            "details": "Demo camera signal indicates a possible collision."
                       if crash_signal else "No abnormal incident signal.",
        }


class DemoDecisionTreeForecaster:
    """Placeholder only: replace with the actual Part 1 Decision Tree model."""
    def __call__(self, event_payload):
        congestion = event_payload["sensor_data"].get("congestion_level", "low")
        return {
            "congestion_level": congestion,
            "forecast_horizon_minutes": 15,
            "predicted_travel_time_minutes": 12 if congestion == "high" else 6,
        }


class DemoRoutePlanner:
    """Adapter between Dispatcher and the Part 3 A* router."""
    def __init__(self):
        self.router = AStarRouter(ROAD_GRAPH)

    def __call__(self, event_payload):
        location = event_payload.get("location", {})
        start = location.get("current_node", "A")
        goal = location.get("destination_node", "E")
        result = self.router.find_route(start, goal)
        result["priority_action"] = (
            "Reserve green-light corridor for emergency vehicle."
            if event_payload["incident"]["detected"]
            else "Recommend alternate route for regular traffic."
        )
        return result


def main():
    coordinator = UrbanFlowCoordinator(
        monitor=Monitor(DemoSVMDetector()),
        analyzer=Analyzer(DemoDecisionTreeForecaster()),
        dispatcher=Dispatcher(DemoRoutePlanner()),
    )

    sample_sensor_data = {
        "event_id": "UF-1001",
        "location": {"current_node": "A", "destination_node": "E", "road": "Main Street"},
        "camera_crash_signal": True,
        "congestion_level": "high",
        "vehicle_count": 48,
        "average_speed_kmph": 9,
    }

    import json
    result = coordinator.run(sample_sensor_data)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
