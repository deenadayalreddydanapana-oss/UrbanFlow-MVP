"""Three lightweight cooperating agents for UrbanFlow AI.

All agents exchange JSON-serializable dictionaries. Model implementations are
injected, so this module does not depend on a particular ML library or model file.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional


JsonDict = Dict[str, Any]


def _json_copy(payload: JsonDict) -> JsonDict:
    """Validate JSON compatibility and make an isolated copy."""
    try:
        return json.loads(json.dumps(payload))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Agent payload must contain only JSON-compatible values: {exc}") from exc


class Monitor:
    """Runs the Part 2 incident detector (for example, the trained SVM)."""

    def __init__(self, incident_detector: Callable[[JsonDict], JsonDict]):
        self.incident_detector = incident_detector

    def process(self, sensor_data: JsonDict) -> JsonDict:
        detection = self.incident_detector(_json_copy(sensor_data))
        if not isinstance(detection, dict):
            raise TypeError("incident_detector must return a dictionary")

        detected = bool(detection.get("incident_detected", False))
        payload = {
            "event_id": sensor_data.get("event_id", f"event-{int(datetime.now().timestamp())}"),
            "timestamp": sensor_data.get(
                "timestamp", datetime.now(timezone.utc).isoformat()
            ),
            "location": _json_copy(sensor_data.get("location", {})),
            "sensor_data": _json_copy(sensor_data),
            "incident": {
                "detected": detected,
                "type": detection.get("incident_type", "unknown" if detected else "none"),
                "confidence": float(detection.get("confidence", 0.0)),
                "details": detection.get("details", ""),
            },
            "pipeline": ["Monitor"],
            "status": "incident_detected" if detected else "normal",
        }
        return _json_copy(payload)


class Analyzer:
    """Combines the incident report with the Part 1 traffic forecast."""

    def __init__(self, traffic_forecaster: Callable[[JsonDict], JsonDict]):
        self.traffic_forecaster = traffic_forecaster

    def process(self, monitor_payload: JsonDict) -> JsonDict:
        payload = _json_copy(monitor_payload)
        forecast = self.traffic_forecaster(payload)
        if not isinstance(forecast, dict):
            raise TypeError("traffic_forecaster must return a dictionary")

        payload["forecast"] = _json_copy(forecast)
        payload["pipeline"].append("Analyzer")

        # An incident or a high-congestion forecast can request rerouting.
        congestion = str(forecast.get("congestion_level", "unknown")).lower()
        high_congestion = congestion in {"high", "severe", "critical"}
        payload["response_required"] = bool(payload["incident"]["detected"] or high_congestion)
        payload["status"] = (
            "response_required" if payload["response_required"] else "monitoring"
        )
        return _json_copy(payload)


class Dispatcher:
    """Calls the Part 3 route planner and assigns emergency-vehicle priority."""

    def __init__(self, route_planner: Callable[[JsonDict], JsonDict]):
        self.route_planner = route_planner

    def process(self, analyzer_payload: JsonDict) -> JsonDict:
        payload = _json_copy(analyzer_payload)
        if not payload.get("response_required", False):
            payload["dispatch"] = {
                "dispatched": False,
                "reason": "No incident or high-congestion condition requires rerouting.",
            }
        else:
            route_result = self.route_planner(_json_copy(payload))
            if not isinstance(route_result, dict):
                raise TypeError("route_planner must return a dictionary")

            emergency = payload["incident"]["detected"] and (
                payload["incident"]["type"] in {"crash", "accident", "road_blockage"}
            )
            payload["dispatch"] = {
                "dispatched": True,
                "vehicle_priority": "EMERGENCY" if emergency else "NORMAL",
                "emergency_corridor_requested": bool(emergency),
                "route_result": _json_copy(route_result),
            }

        payload["pipeline"].append("Dispatcher")
        payload["status"] = (
            "dispatched" if payload["dispatch"]["dispatched"] else "no_action_required"
        )
        return _json_copy(payload)


class UrbanFlowCoordinator:
    """Runs Monitor -> Analyzer -> Dispatcher in sequence."""

    def __init__(self, monitor: Monitor, analyzer: Analyzer, dispatcher: Dispatcher):
        self.monitor = monitor
        self.analyzer = analyzer
        self.dispatcher = dispatcher

    def run(self, sensor_data: JsonDict) -> JsonDict:
        monitor_output = self.monitor.process(sensor_data)
        analyzer_output = self.analyzer.process(monitor_output)
        return self.dispatcher.process(analyzer_output)
