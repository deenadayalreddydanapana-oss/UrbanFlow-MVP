import unittest

from urbanflow_agents.agents import Monitor, Analyzer, Dispatcher, UrbanFlowCoordinator


class TestUrbanFlowAgents(unittest.TestCase):
    def setUp(self):
        detector = lambda data: {
            "incident_detected": data.get("crash", False),
            "incident_type": "crash" if data.get("crash", False) else "none",
            "confidence": 0.9,
        }
        forecaster = lambda payload: {"congestion_level": "high"}
        router = lambda payload: {"route": ["A", "C", "D"], "total_cost": 5}
        self.system = UrbanFlowCoordinator(
            Monitor(detector), Analyzer(forecaster), Dispatcher(router)
        )

    def test_crash_triggers_emergency_dispatch(self):
        result = self.system.run({"crash": True, "location": {}})
        self.assertEqual(result["pipeline"], ["Monitor", "Analyzer", "Dispatcher"])
        self.assertEqual(result["dispatch"]["vehicle_priority"], "EMERGENCY")
        self.assertTrue(result["dispatch"]["dispatched"])

    def test_normal_low_congestion_does_not_dispatch(self):
        system = UrbanFlowCoordinator(
            Monitor(lambda _: {"incident_detected": False}),
            Analyzer(lambda _: {"congestion_level": "low"}),
            Dispatcher(lambda _: self.fail("Router should not be called")),
        )
        result = system.run({"location": {}})
        self.assertFalse(result["dispatch"]["dispatched"])

    def test_payload_is_json_serializable(self):
        import json
        result = self.system.run({"crash": True, "location": {}})
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
