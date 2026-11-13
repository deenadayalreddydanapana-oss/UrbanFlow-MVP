"""
traffic_engine.py  -  ML models and graph logic for UrbanFlow AI (classical ML only, no deep learning).

    PART 1: Traffic congestion forecasting (Decision Tree model, traffic_model.joblib)
    PART 2: Anomaly / incident detection   (Random Forest by default, SVM optional; detector.joblib)
    PART 3: Graph routing with NetworkX    (teammate: paste here)

Public interface (used by agents.py and app.py):
    predict_congestion(time_of_day, weather_severity, current_vehicle_count) -> float   [Part 1]
    get_detector()                      -> shared IncidentDetector                      [Part 2]
    detector.update(road_id, speed, density, ts) -> {'status': 'Normal'|'Alert', 'score', 'reason', ...}
    detector.reset()                    -> clear history (e.g. when the simulation restarts)

Run `python traffic_engine.py` once to train the Part 2 detector (saves detector.joblib).
Part 1 needs traffic_model.joblib in the same folder (trained and saved by its owner).
"""
from collections import defaultdict, deque
from pathlib import Path
import os
import threading

import math
import networkx as nx
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_curve, average_precision_score)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "detector.joblib"

# =====================================================================
# PART 1: TRAFFIC CONGESTION FORECASTING
# =====================================================================
_forecasting_model = None
_forecasting_model_path = BASE_DIR / "traffic_model.joblib"

def load_forecasting_model():
    """Loads the trained traffic congestion regression model."""
    global _forecasting_model
    if _forecasting_model is None:
        if _forecasting_model_path.exists():
            _forecasting_model = joblib.load(_forecasting_model_path)
        else:
            raise FileNotFoundError(f"Forecasting model not found at {_forecasting_model_path}")
    return _forecasting_model

def predict_congestion(time_of_day, weather_severity, current_vehicle_count):
    """
    Predicts traffic congestion score using the trained Decision Tree model.
    Inputs:
        - time_of_day (float/int): Hour or time encoding
        - weather_severity (float/int): Severity metric
        - current_vehicle_count (int/float): Number of vehicles
    Returns:
        - float: Predicted congestion level
    """
    model = load_forecasting_model()
    input_data = pd.DataFrame(
        [[time_of_day, weather_severity, current_vehicle_count]], 
        columns=['time_of_day', 'weather_severity', 'current_vehicle_count']
    )
    prediction = model.predict(input_data)
    return float(prediction[0])


# =====================================================================
# PART 2: ANOMALY / INCIDENT DETECTION
# =====================================================================
MODEL_TYPE = "random_forest"   # "random_forest" (default, more robust) or "svm"
TICKS_PER_DAY = 288            # one tick = 5 minutes of simulated time
HISTORY = 6                    # ticks of past speeds used for features
SMOOTH_WINDOW, SMOOTH_MIN_FLAGS = 3, 2    # alert only if 2 of the last 3 ticks are flagged
FEATS = ["speed", "density", "d1", "d3", "dev", "roll_std", "flow"]


def _free_flow(hour):
    """Normal speed (km/h) by hour of day, with morning and evening rush-hour dips."""
    return 50 - 18 * np.exp(-((hour - 8.5) ** 2) / 2) - 20 * np.exp(-((hour - 18) ** 2) / 3)


def simulate_sensor_data(n_roads=10, days=14, seed=42):
    """Create fake sensor readings with injected incidents (label=1 during an incident)."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_roads):
        n = days * TICKS_PER_DAY
        t = np.arange(n)
        hour = (t % TICKS_PER_DAY) * 24 / TICKS_PER_DAY
        speed = _free_flow(hour) * rng.uniform(0.8, 1.1) + rng.normal(0, 2.5, n)
        density = np.clip(80 - speed * 1.1 + rng.normal(0, 5, n), 5, None)
        label = np.zeros(n, int)
        for _ in range(rng.integers(8, 14)):
            s, d = rng.integers(50, n - 40), rng.integers(6, 18)
            speed[s:s + d] *= rng.uniform(0.0, 0.4)
            density[s:s + d] += rng.uniform(30, 60)
            label[s:s + d] = 1
        speed = np.clip(speed, 0, None)
        rows.append(pd.DataFrame({"road_id": r, "tick": t, "speed": speed,
                                  "density": density, "label": label}))
    return pd.concat(rows, ignore_index=True)


def add_features(df):
    """Training-time features from PAST values only (the current tick is excluded from rolling stats)."""
    df = df.sort_values(["road_id", "tick"]).copy()
    g = df.groupby("road_id")["speed"]
    df["d1"] = g.diff(1)
    df["d3"] = g.diff(3)
    prev = g.shift(1)
    df["roll_mean"] = prev.groupby(df["road_id"]).transform(
        lambda s: s.rolling(HISTORY, min_periods=HISTORY).mean())
    df["roll_std"] = prev.groupby(df["road_id"]).transform(
        lambda s: s.rolling(HISTORY, min_periods=HISTORY).std())
    df["dev"] = df["speed"] - df["roll_mean"]
    df["flow"] = df["speed"] * df["density"]
    return df.dropna()


def _live_features(history, speed, density):
    """One-row version of add_features() for real-time use. `history` = previous HISTORY speeds, oldest first."""
    h = list(history)
    return pd.DataFrame([{"speed": speed, "density": density,
                          "d1": speed - h[-1], "d3": speed - h[-3],
                          "dev": speed - np.mean(h), "roll_std": np.std(h, ddof=1),
                          "flow": speed * density}])[FEATS]


def _make_model(model_type=None):
    model_type = model_type or MODEL_TYPE
    if model_type == "svm":
        return make_pipeline(StandardScaler(),
                             SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"))
    if model_type == "random_forest":
        return RandomForestClassifier(n_estimators=150, min_samples_leaf=2,
                                      class_weight="balanced", random_state=0, n_jobs=-1)
    raise ValueError(f"Unknown MODEL_TYPE: {model_type!r} (use 'random_forest' or 'svm')")


def _score(model, X):
    """Anomaly score: higher = more likely an incident (works for SVM and Random Forest)."""
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def _detection_latency(d, col="alert"):
    """Average ticks between an incident starting and the first alert; also the missed share."""
    lat = []
    for _, g in d.groupby("road_id"):
        g = g.sort_values("tick")
        lab, al = g.label.values, g[col].values
        i = 0
        while i < len(lab):
            if lab[i] == 1 and (i == 0 or lab[i - 1] == 0):
                j = i
                while j < len(lab) and lab[j] == 1:
                    j += 1
                hit = np.where(al[i:j] == 1)[0]
                lat.append(hit[0] if len(hit) else np.nan)
                i = j
            else:
                i += 1
    lat = np.array(lat, float)
    return np.nanmean(lat), np.isnan(lat).mean(), len(lat)


def train_detector(n_roads=10, days=14, seed=42, evaluate=True, save=True, model_type=None):
    """Simulate data, train the model, tune the alert threshold, optionally evaluate and save."""
    model_type = model_type or MODEL_TYPE
    df = add_features(simulate_sensor_data(n_roads, days, seed))

    # Chronological split (never shuffle time series): 70% train / 15% validation / 15% test
    max_t = df.tick.max()
    train = df[df.tick <= 0.70 * max_t]
    val = df[(df.tick > 0.70 * max_t) & (df.tick <= 0.85 * max_t)]
    test = df[df.tick > 0.85 * max_t].copy()

    n_norm = min(12000, int((train.label == 0).sum()))
    normals = train[train.label == 0].sample(n_norm, random_state=0)
    train_s = pd.concat([normals, train[train.label == 1]])

    model = _make_model(model_type)
    model.fit(train_s[FEATS], train_s.label)
    if hasattr(model, "n_jobs"):
        model.set_params(n_jobs=1)    # single-row live predictions are faster without parallel overhead

    # Tune the alert threshold on the validation set only
    val_scores = _score(model, val[FEATS])
    p, r, th = precision_recall_curve(val.label, val_scores)
    f1 = 2 * p * r / (p + r + 1e-9)
    thr = float(th[np.argmax(f1[:-1])])

    if evaluate:
        test["score"] = _score(model, test[FEATS])
        test["raw"] = (test.score > thr).astype(int)
        test["alert"] = (test.groupby("road_id")["raw"].transform(
            lambda s: (s.rolling(SMOOTH_WINDOW, min_periods=1).sum() >= SMOOTH_MIN_FLAGS).astype(int)))
        print(f"model: {model_type} | threshold: {thr:.3f}")
        print(f"\n=== {model_type} (smoothed) ===")
        print(classification_report(test.label, test.alert, target_names=["Normal", "Alert"]))
        print(confusion_matrix(test.label, test.alert))
        print("test PR-AUC:", round(average_precision_score(test.label, test.score), 3))
        lat, missed, n_inc = _detection_latency(test)
        print(f"incidents: {n_inc} | mean latency: {lat:.1f} ticks | missed: {missed:.0%}")

        test["rule"] = (test.speed < 10).astype(int)
        iso = IsolationForest(contamination=0.03, random_state=0).fit(train[FEATS])
        test["iso"] = (iso.predict(test[FEATS]) == -1).astype(int)
        for name in ("rule", "iso"):
            print(f"\n=== baseline: {name} ===")
            print(classification_report(test.label, test[name], digits=3))

    bundle = {"model": model, "thr": thr, "feats": FEATS, "model_type": model_type}
    if save:
        joblib.dump(bundle, MODEL_PATH, compress=3)
        print("saved ->", MODEL_PATH)
    return bundle


class IncidentDetector:
    """Real-time detector: feed one sensor reading at a time, get Normal/Alert back."""

    def __init__(self, model, thr):
        self.m, self.thr = model, thr
        self._lock = threading.Lock()     # Flask may call from several threads
        self.reset()

    def reset(self):
        self.hist = defaultdict(lambda: deque(maxlen=HISTORY))        # recent speeds per road
        self.flags = defaultdict(lambda: deque(maxlen=SMOOTH_WINDOW))  # recent raw flags per road

    def update(self, road_id, speed, density, ts=None):
        """Returns {'timestamp','road_id','status','score','reason'}; status is 'Normal' or 'Alert'.
        Call once per road per tick, in time order. Needs HISTORY ticks of history before it can alert."""
        with self._lock:
            h = list(self.hist[road_id])
            status, score, reason = "Normal", 0.0, "normal flow"
            if len(h) == HISTORY:
                score = float(_score(self.m, _live_features(h, speed, density))[0])
                self.flags[road_id].append(score > self.thr)
                if sum(self.flags[road_id]) >= SMOOTH_MIN_FLAGS:
                    status = "Alert"
                    reason = (f"speed {speed:.0f} km/h is {np.mean(h) - speed:.0f} km/h below "
                              f"the recent average, density {density:.0f} veh/km")
            self.hist[road_id].append(speed)
            return {"timestamp": ts, "road_id": road_id, "status": status,
                    "score": round(score, 2), "reason": reason}


_detector = None
_detector_lock = threading.Lock()


def get_detector():
    """Shared detector for the whole app. Loads detector.joblib, or trains one if it is missing."""
    global _detector
    with _detector_lock:
        if _detector is None:
            if MODEL_PATH.exists():
                bundle = joblib.load(MODEL_PATH)
            else:
                bundle = train_detector(evaluate=False)
            _detector = IncidentDetector(bundle["model"], bundle["thr"])
        return _detector


# =====================================================================
# PART 3: GRAPH ROUTING WITH NETWORKX  (teammate's code goes here)
# =====================================================================
# =====================================================================
# PART 3: GRAPH ROUTING WITH NETWORKX
# =====================================================================

import math
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any

POSITIONS: Dict[str, Tuple[float, float]] = {
    "A": (0, 0), "B": (1, 1), "C": (1, -1), "D": (2, 2), "E": (2, 0),
    "F": (2, -2), "G": (3, 1), "H": (3, -1), "I": (4, 0), "J": (5, -1),
}

ROADS: List[Tuple[str, str, float]] = [
    ("A", "B", 4), ("A", "C", 7), ("B", "D", 5), ("B", "E", 6),
    ("C", "E", 2), ("C", "F", 5), ("D", "G", 4), ("E", "G", 3),
    ("E", "H", 6), ("F", "H", 2), ("G", "I", 4), ("H", "I", 3),
    ("H", "J", 5), ("I", "J", 2),
]

EMERGENCY_PRIORITY_ROADS = {
    frozenset(("A", "B")), frozenset(("B", "E")),
    frozenset(("E", "H")), frozenset(("H", "J")),
}

EMERGENCY_PRIORITY_FACTOR = 0.5
DEFAULT_ACCIDENT_PENALTY = 1000.0

def build_city_graph() -> nx.Graph:
    graph = nx.Graph()
    for node, position in POSITIONS.items():
        graph.add_node(node, pos=position)
    for u, v, base_time in ROADS:
        graph.add_edge(
            u, v,
            base_time=float(base_time),
            travel_time=float(base_time),
            emergency_time=float(base_time),
            accident=False,
        )
    for u, v in graph.edges:
        if frozenset((u, v)) in EMERGENCY_PRIORITY_ROADS:
            graph[u][v]["emergency_time"] = graph[u][v]["base_time"] * EMERGENCY_PRIORITY_FACTOR
    return graph

G = build_city_graph()

def heuristic(node: str, destination: str) -> float:
    x1, y1 = POSITIONS[node]
    x2, y2 = POSITIONS[destination]
    return math.dist((x1, y1), (x2, y2)) * 0.2

def find_route(source: str, destination: str, weight_type: str = "travel_time", graph: Optional[nx.Graph] = None) -> Dict[str, Any]:
    graph = graph if graph is not None else G
    if source not in graph: raise ValueError(f"Unknown source: {source}")
    if destination not in graph: raise ValueError(f"Unknown destination: {destination}")
    
    route = nx.astar_path(graph, source, destination, heuristic=lambda n, t: heuristic(n, t), weight=weight_type)
    cost = nx.path_weight(graph, route, weight=weight_type)
    return {"route": route, "cost": float(cost), "weight_type": weight_type, "cost_unit": "simulated travel-time units"}

def reset_road_costs(graph: Optional[nx.Graph] = None) -> nx.Graph:
    graph = graph if graph is not None else G
    for u, v, data in graph.edges(data=True):
        data["travel_time"] = data["base_time"]
        data["emergency_time"] = data["base_time"]
        data["accident"] = False
        if frozenset((u, v)) in EMERGENCY_PRIORITY_ROADS:
            data["emergency_time"] = data["base_time"] * EMERGENCY_PRIORITY_FACTOR
    return graph

def simulate_accident(u: str, v: str, penalty: float = DEFAULT_ACCIDENT_PENALTY, graph: Optional[nx.Graph] = None) -> nx.Graph:
    graph = graph if graph is not None else G
    graph[u][v]["travel_time"] = float(penalty)
    graph[u][v]["emergency_time"] = float(penalty)
    graph[u][v]["accident"] = True
    return graph

def calculate_routes(source: str = "A", destination: str = "J", accident_edge: Tuple[str, str] = ("E", "G"), accident_penalty: float = DEFAULT_ACCIDENT_PENALTY) -> Dict[str, Any]:
    reset_road_costs(G)
    baseline = find_route(source, destination, "travel_time", G)
    incident = None
    if accident_edge is not None:
        u, v = accident_edge
        simulate_accident(u, v, accident_penalty, G)
        incident = {"from": u, "to": v, "penalty": float(accident_penalty)}
    regular = find_route(source, destination, "travel_time", G)
    emergency = find_route(source, destination, "emergency_time", G)
    return {
        "source": source, "destination": destination, "incident": incident,
        "baseline_route": baseline["route"], "baseline_cost": baseline["cost"],
        "regular_route": regular["route"], "regular_cost": regular["cost"],
        "emergency_route": emergency["route"], "emergency_cost": emergency["cost"],
        "routes_are_different": regular["route"] != emergency["route"],
        "cost_unit": "simulated travel-time units"
    }

def get_graph_data(result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    regular_edges, emergency_edges = set(), set()
    if result:
        regular_edges = {frozenset((a, b)) for a, b in zip(result["regular_route"], result["regular_route"][1:])}
        emergency_edges = {frozenset((a, b)) for a, b in zip(result["emergency_route"], result["emergency_route"][1:])}
    nodes = [{"id": node, "x": POSITIONS[node][0], "y": POSITIONS[node][1]} for node in G.nodes]
    edges = []
    for u, v, data in G.edges(data=True):
        key = frozenset((u, v))
        edges.append({
            "source": u, "target": v, "base_time": data["base_time"],
            "travel_time": data["travel_time"], "emergency_time": data["emergency_time"],
            "accident": bool(data["accident"]), "on_regular_route": key in regular_edges,
            "on_emergency_route": key in emergency_edges,
        })
    return {"nodes": nodes, "edges": edges}

if __name__ == "__main__":
    # --- Part 1 check ---
    if _forecasting_model_path.exists():
        print("Part 1: traffic_model.joblib found. Example prediction:",
              round(predict_congestion(8, 1, 120), 3))
    else:
        print("Part 1: traffic_model.joblib NOT found next to this file.")

    # --- Part 2 check ---
    train_detector()
    det = get_detector()
    print("\nLive detector demo (road 0):")
    for i, s in enumerate([42, 41, 43, 40, 42, 41, 40, 12, 5, 4, 4]):
        print(det.update(0, s, 90 if s < 20 else 40, ts=i))

    # --- Part 3 check (Your routing engine test) ---
    print("\n--- Part 3 Graph Routing Test ---")
    output = calculate_routes()
    print("Baseline route: ", " -> ".join(output["baseline_route"]))
    print("Regular route:  ", " -> ".join(output["regular_route"]))
    print("Emergency route:", " -> ".join(output["emergency_route"]))
    print("Regular cost:   ", output["regular_cost"])
    print("Emergency cost: ", output["emergency_cost"])
