# UrbanFlow AI — Part 4: Cooperating Agents

A lightweight, standard-library-only implementation of the **Monitor → Analyzer → Dispatcher** workflow.

## Files

- `agents.py` — the three cooperating agent classes and the coordinator.
- `astar_router.py` — small A* router that can be used as a Part 3 adapter or replaced by your team's router.
- `demo.py` — runnable end-to-end demonstration.
- `sample_input.json` — example sensor/event payload.
- `tests/test_agents.py` — basic tests.

## Run

From the directory containing `urbanflow_agents/`:

```bash
python -m urbanflow_agents.demo
```

Run tests:

```bash
python -m unittest discover -s urbanflow_agents/tests -v
```

No external packages are required.

## Agent responsibilities

1. **Monitor** calls the incident detector from Part 2 (your trained SVM) and creates a JSON-compatible event payload.
2. **Analyzer** calls the traffic forecaster from Part 1 (your trained Decision Tree), combines its forecast with the incident data, and decides whether a response is required.
3. **Dispatcher** calls the route planner from Part 3 (your A* implementation) and marks qualifying crash/accident/road-blockage responses as `EMERGENCY`.

The coordinator passes a dictionary from one agent to the next. Each hand-off is copied through JSON serialization, which checks that the payload contains JSON-compatible values.

## Connect your actual Parts 1–3

The demo model classes are deliberately simple stand-ins. They do **not** load or represent trained SVM/Decision Tree models.

Replace the three demo adapters in `demo.py` with wrappers around your existing code. Each adapter must be a callable that accepts one dictionary and returns one dictionary:

### Part 2 — SVM detector contract

```python
def svm_detector(sensor_data):
    # Extract the exact features used during SVM training.
    # Example only: prediction = svm_model.predict(feature_row)[0]
    return {
        "incident_detected": bool(prediction),
        "incident_type": "crash" if prediction else "none",
        "confidence": float(confidence),
        "details": "Short explanation",
    }
```

If your SVM uses a feature scaler, apply the same fitted scaler used during training. Do not fit a new scaler on live input.

### Part 1 — Decision Tree forecast contract

```python
def decision_tree_forecaster(event_payload):
    # Extract the same input features and ordering used during training.
    # Example only: prediction = decision_tree_model.predict(feature_row)[0]
    return {
        "congestion_level": "high",  # low / medium / high / severe
        "forecast_horizon_minutes": 15,
        "predicted_travel_time_minutes": 12.5,
    }
```

### Part 3 — A* route planner contract

```python
def astar_route_planner(event_payload):
    # Read start/destination from event_payload["location"].
    # Call your Part 3 A* implementation.
    return {
        "route": ["Junction-A", "Junction-C", "Junction-D"],
        "total_cost": 8.4,
        "algorithm": "A*",
    }
```

Then inject the adapters:

```python
coordinator = UrbanFlowCoordinator(
    Monitor(svm_detector),
    Analyzer(decision_tree_forecaster),
    Dispatcher(astar_route_planner),
)
result = coordinator.run(sensor_data)
```

## Output structure

The final result retains the original sensor data and adds:

- `incident` — detector result and confidence.
- `forecast` — predicted congestion and travel-time fields.
- `response_required` — whether an incident or high congestion requires action.
- `dispatch` — whether routing was triggered, vehicle priority, and route result.
- `pipeline` — confirms the order in which agents ran.

## Important integration notes

- Agree on feature names, units, and ordering with the owners of Parts 1 and 2.
- The SVM's output must distinguish a true incident from normal traffic. A high congestion score alone should not be labelled as a crash.
- The current emergency-priority rule applies to detected `crash`, `accident`, or `road_blockage` incidents. Adjust the rule if your project defines emergency categories differently.
- The A* demo uses travel-time edge costs. For a real emergency corridor, the routing module must also account for blocked roads and any permitted emergency-only road rules.
- This module simulates agent collaboration inside one Python process. It is not distributed multi-agent infrastructure.
- Do not describe the demo stand-ins as trained ML models in your project presentation.
