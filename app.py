from flask import Flask, render_template, jsonify, request
import json
import math
import ssl
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

# Import your team's ML logic
from traffic_engine import get_detector, predict_congestion

# Import your cooperating agents
from urbanflow_agents import UrbanFlowCoordinator, Monitor, Analyzer, Dispatcher

app = Flask(__name__)

# =====================================================================
# 0. REAL HYDERABAD MAP DATA
#    Map tiles + road network come from OpenStreetMap. Road routing uses the
#    public OSRM server (OpenStreetMap roads), so routes follow real streets.
#    The computer running Flask needs an internet connection.
# =====================================================================

OSRM_SERVERS = [
    "https://router.project-osrm.org",                 # public OSRM demo server
    "https://routing.openstreetmap.de/routed-car",     # backup (FOSSGIS), same API
]
INCIDENT_CLEARANCE_M = 150      # a route must pass at least this far from the crash
INCIDENT_POSITION = 0.45        # crash is placed 45% of the way along the fastest route

# Real places in Hyderabad. The dashboard builds its dropdowns from this list.
LANDMARKS = {
    "A": {"name": "Secunderabad Railway Station", "lat": 17.4337, "lng": 78.5014},
    "B": {"name": "KIMS Hospital (Secunderabad)",  "lat": 17.4413, "lng": 78.4983},
    "C": {"name": "Begumpet",                      "lat": 17.4440, "lng": 78.4670},
    "D": {"name": "Ameerpet",                      "lat": 17.4375, "lng": 78.4483},
    "E": {"name": "NIMS Hospital (Punjagutta)",    "lat": 17.4234, "lng": 78.4510},
    "F": {"name": "Apollo Hospital (Jubilee Hills)", "lat": 17.4156, "lng": 78.4106},
    "G": {"name": "HITEC City",                    "lat": 17.4435, "lng": 78.3772},
    "H": {"name": "Gachibowli",                    "lat": 17.4401, "lng": 78.3489},
    "I": {"name": "Osmania General Hospital",      "lat": 17.3713, "lng": 78.4731},
    "J": {"name": "Charminar",                     "lat": 17.3616, "lng": 78.4747},
}


class RoutingError(Exception):
    """Raised when the road routing service cannot be reached or finds no route."""


def _ssl_context():
    """Use certifi's certificate bundle if available (fixes 'certificate verify failed' on some PCs)."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


def _osrm_request(points, alternatives):
    """Ask OSRM for driving routes through [(lat, lng), ...]. Tries each routing server in turn."""
    coords = ";".join(f"{lng:.6f},{lat:.6f}" for lat, lng in points)
    errors = []
    for base in OSRM_SERVERS:
        for alt in ([alternatives, "true"] if alternatives not in ("true", "false") else [alternatives]):
            query = urllib.parse.urlencode({
                "alternatives": alt, "steps": "true",
                "overview": "full", "geometries": "geojson",
            })
            url = f"{base}/route/v1/driving/{coords}?{query}"
            req = urllib.request.Request(url, headers={"User-Agent": "UrbanFlowAI-MVP/1.0 (student project)"})
            host = urllib.parse.urlparse(base).netloc
            try:
                with urllib.request.urlopen(req, timeout=25, context=_ssl_context()) as resp:
                    data = json.load(resp)
            except urllib.error.HTTPError as exc:
                if exc.code == 400 and alt != "true" and alternatives not in ("true", "false"):
                    continue                              # server rejects numeric alternatives: retry with true
                errors.append(f"{host}: HTTP {exc.code}")
                break
            except Exception as exc:                      # DNS, SSL, timeout, proxy, firewall...
                reason = getattr(exc, "reason", exc)
                errors.append(f"{host}: {type(reason).__name__}: {reason}")
                print("Routing request failed:", errors[-1])
                break
            if data.get("code") == "Ok" and data.get("routes"):
                return data["routes"]
            errors.append(f"{host}: {data.get('code')} {data.get('message', '')}".strip())
            break
    raise RoutingError("Could not get a road route. " + " | ".join(errors))


# ---------- geometry helpers ----------
def _haversine(a, b):
    r = 6371000.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def _project(pt, lat0):
    return pt[1] * math.cos(math.radians(lat0)) * 111320.0, pt[0] * 110540.0


def _dist_point_to_route(pt, coords):
    """Shortest distance (metres) from a point to a polyline."""
    px, py = _project(pt, pt[0])
    best = float("inf")
    for a, b in zip(coords, coords[1:]):
        ax, ay = _project(a, pt[0])
        bx, by = _project(b, pt[0])
        dx, dy = bx - ax, by - ay
        seg = dx * dx + dy * dy
        t = 0.0 if seg == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg))
        best = min(best, math.hypot(px - (ax + t * dx), py - (ay + t * dy)))
    return best


def _point_at_fraction(coords, fraction):
    """Point + vertex index that lies `fraction` of the way along a polyline."""
    legs = [_haversine(a, b) for a, b in zip(coords, coords[1:])]
    target, run = sum(legs) * fraction, 0.0
    for i, leg in enumerate(legs):
        if run + leg >= target:
            return coords[i], i
        run += leg
    return coords[-1], len(coords) - 1


def _bearing(a, b):
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dl = math.radians(b[1] - a[1])
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def _offset(pt, bearing_deg, metres):
    az = math.radians(bearing_deg)
    return (pt[0] + metres * math.cos(az) / 110540.0,
            pt[1] + metres * math.sin(az) / (111320.0 * math.cos(math.radians(pt[0]))))


# ---------- turn-by-turn text ----------
def fmt_distance(m):
    if m < 1000:
        return f"{int(round(m / 10.0) * 10) if m >= 100 else int(round(m))} m"
    return f"{m / 1000.0:.1f} km"


def fmt_minutes(seconds):
    mins = int(round(seconds / 60.0))
    if mins < 60:
        return f"{max(mins, 1)} min"
    return f"{mins // 60} h {mins % 60:02d} min"


def _turns_text(n):
    return f"{n} turn" + ("" if n == 1 else "s")


def fmt_hour(h):
    return f"{int(h):02d}:{int(round((h % 1) * 60)):02d}"


def _compass(bearing):
    names = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]
    return names[int((bearing + 22.5) // 45) % 8]


_ICONS = {"left": "↰", "sharp left": "↰", "slight left": "↖", "right": "↱", "sharp right": "↱",
          "slight right": "↗", "straight": "↑", "uturn": "↶"}


def _step_icon(maneuver):
    t = maneuver.get("type", "")
    if t == "depart":
        return "●"
    if t == "arrive":
        return "⚑"
    if t in ("roundabout", "rotary", "roundabout turn", "exit roundabout", "exit rotary"):
        return "⟳"
    return _ICONS.get(maneuver.get("modifier", "straight"), "↑")


def _describe_step(step):
    m = step["maneuver"]
    t, mod = m.get("type", ""), m.get("modifier", "")
    name = step.get("name") or ""
    onto = f" onto {name}" if name else ""
    if t == "depart":
        return f"Head {_compass(m.get('bearing_after', 0))}" + (f" on {name}" if name else "")
    if t == "arrive":
        return "Arrive at your destination" + (f" on {name}" if name else "")
    if t in ("roundabout", "rotary", "roundabout turn"):
        ex = m.get("exit")
        return f"At the roundabout take exit {ex}{onto}" if ex else f"Enter the roundabout{onto}"
    if t in ("exit roundabout", "exit rotary"):
        return f"Exit the roundabout{onto}"
    if t == "merge":
        return f"Merge {mod}{onto}".replace("  ", " ")
    if t in ("on ramp", "off ramp"):
        return f"Take the ramp {mod}{onto}".replace("  ", " ")
    if t == "fork":
        return f"Keep {mod} at the fork{onto}".replace("  ", " ")
    if t == "end of road":
        return f"At the end of the road turn {mod}{onto}"
    if t in ("continue", "new name"):
        return f"Continue {mod if mod and mod != 'straight' else 'straight'}{onto}"
    if mod == "uturn":
        return f"Make a U-turn{onto}"
    if mod == "straight" or not mod:
        return f"Continue straight{onto}"
    return f"Turn {mod}{onto}"


def _parse_route(raw):
    coords = [[lat, lng] for lng, lat in raw["geometry"]["coordinates"]]
    steps, legs = [], raw.get("legs", [])
    for li, leg in enumerate(legs):
        for st in leg.get("steps", []):
            t = st["maneuver"].get("type")
            if (t == "arrive" and li < len(legs) - 1) or (t == "depart" and li > 0):
                continue                                   # internal via-point, not a real turn
            loc = st["maneuver"]["location"]
            steps.append({
                "instruction": _describe_step(st),
                "icon": _step_icon(st["maneuver"]),
                "road": st.get("name", ""),
                "distance_m": round(st["distance"]),
                "distance_text": fmt_distance(st["distance"]) if t != "arrive" else "",
                "duration_s": round(st["duration"]),
                "location": [loc[1], loc[0]],
                "is_turn": t not in ("depart", "arrive") and st["maneuver"].get("modifier") not in (None, "straight"),
            })
    return {"distance_m": round(raw["distance"]), "duration_s": round(raw["duration"]),
            "coordinates": coords, "steps": steps}


def _top_roads(route, n=3):
    totals = {}
    for st in route["steps"]:
        if st["road"]:
            totals[st["road"]] = totals.get(st["road"], 0) + st["distance_m"]
    return [name for name, _ in sorted(totals.items(), key=lambda kv: -kv[1])[:n]]


def _road_at_fraction(route, fraction):
    target, run = route["distance_m"] * fraction, 0
    for st in route["steps"]:
        run += st["distance_m"]
        if run >= target and st["road"]:
            return st["road"]
    return ""


def plan_routes(start, end, incident):
    """Real road routes from start to end. If `incident`, put a crash on the fastest
    route and pick a route that stays clear of it (alternatives first, then detours)."""
    routes = [_parse_route(r) for r in _osrm_request([start, end], "3")]
    routes.sort(key=lambda r: r["duration_s"])
    primary = routes[0]
    plan = {"incident_point": None, "incident_road": "", "avoided": None, "routes": routes, "selected": 0}
    if not incident:
        return plan

    ip, idx = _point_at_fraction(primary["coordinates"], INCIDENT_POSITION)
    plan["incident_point"] = list(ip)
    plan["incident_road"] = _road_at_fraction(primary, INCIDENT_POSITION)
    for r in routes:
        r["clearance_m"] = round(_dist_point_to_route(ip, r["coordinates"]))
        r["blocked"] = r["clearance_m"] < INCIDENT_CLEARANCE_M

    if all(r["blocked"] for r in routes):
        # No alternative avoids the crash: force a detour through a waypoint beside it.
        coords = primary["coordinates"]
        heading = _bearing(coords[max(idx - 3, 0)], coords[min(idx + 3, len(coords) - 1)])
        for metres in (1200, 2500, 4000):
            found = False
            for side in (90, -90):
                via = _offset(ip, heading + side, metres)
                try:
                    cand = _parse_route(_osrm_request([start, via, end], "false")[0])
                except RoutingError:
                    continue
                cand["clearance_m"] = round(_dist_point_to_route(ip, cand["coordinates"]))
                cand["blocked"] = cand["clearance_m"] < INCIDENT_CLEARANCE_M
                if not cand["blocked"]:
                    routes.append(cand)
                    found = True
            if found:
                break

    safe = [i for i, r in enumerate(routes) if not r["blocked"]]
    if safe:
        plan["selected"] = min(safe, key=lambda i: routes[i]["duration_s"])
        plan["avoided"] = True
    else:
        plan["avoided"] = False       # nothing avoids it: caller shows the fastest route + a warning
    return plan


# =====================================================================
# 1. AGENT ADAPTERS: Bridging the ML models to the Agent JSON system
# =====================================================================

def svm_detector_adapter(sensor_data):
    """Monitor Agent uses this to check for accidents."""
    detector = get_detector()

    speed = sensor_data.get("average_speed_kmph", 40)
    density = sensor_data.get("vehicle_count", 20)
    if speed < 10:
        density = max(density, 80)  # a stopped road is a jam, whatever the slider says
    road_id = sensor_data.get("road_id", 0)

    # The detector needs 6 ticks of past speeds and 2 flagged ticks before it can
    # alert, so one lone reading can never trigger it. Replay a short normal
    # history, then feed the live reading for a few ticks (reset per simulation,
    # because the detector is shared and remembers earlier clicks).
    detector.reset()
    for _ in range(6):
        detector.update(road_id=road_id, speed=45, density=40)
    result = {}
    for _ in range(3):
        result = detector.update(road_id=road_id, speed=speed, density=density)

    detected = result["status"] == "Alert"
    return {
        "incident_detected": detected,
        "incident_type": "road_blockage" if detected else "none",
        "confidence": result.get("score", 0.0),
        "details": result.get("reason", "Normal flow")
    }


def decision_tree_adapter(event_payload):
    """Analyzer Agent uses this to forecast congestion."""
    sensor = event_payload.get("sensor_data", {})
    vehicles = sensor.get("vehicle_count", 50)
    hour = sensor.get("time_of_day", 8.5)
    weather = sensor.get("weather_severity", 0)

    try:
        # Part 1 Decision Tree returns a 0-100 congestion score
        congestion_score = predict_congestion(time_of_day=hour, weather_severity=weather, current_vehicle_count=vehicles)
    except FileNotFoundError:
        # Fallback just in case traffic_model.joblib is missing during the live demo
        congestion_score = 85 if vehicles > 60 else 25

    level = "low"
    if congestion_score > 70: level = "high"
    elif congestion_score > 40: level = "medium"

    return {
        "congestion_level": level,
        "congestion_score": round(congestion_score, 1),
        "forecast_horizon_minutes": 30,
        "predicted_travel_time_minutes": 15 if level == "high" else 5
    }


def astar_route_adapter(event_payload):
    """Dispatcher Agent uses this to plan the real road route (and avoid a crash)."""
    loc = event_payload.get("location", {})
    start = LANDMARKS[loc.get("current_node", "A")]
    end = LANDMARKS[loc.get("destination_node", "J")]
    incident = bool(event_payload.get("incident", {}).get("detected"))

    plan = plan_routes((start["lat"], start["lng"]), (end["lat"], end["lng"]), incident)
    chosen = plan["routes"][plan["selected"]]
    return {
        "route": [loc.get("current_node", "A"), loc.get("destination_node", "J")],
        "total_cost": round(chosen["duration_s"] / 60.0, 1),
        "algorithm": "OpenStreetMap road routing (OSRM)",
        "routes_are_different": plan["selected"] != 0,
        "plan": plan,
    }


# Initialize the Multi-Agent Coordinator System
coordinator = UrbanFlowCoordinator(
    monitor=Monitor(svm_detector_adapter),
    analyzer=Analyzer(decision_tree_adapter),
    dispatcher=Dispatcher(astar_route_adapter)
)

# =====================================================================
# 2. RESPONSE BUILDERS (shape the data main.js renders)
# =====================================================================

def _route_view(plan, congestion_score):
    """Routes, directions and ETA for the dashboard. ETA = road time x (1 + congestion/100)."""
    factor = 1.0 + max(congestion_score, 0) / 100.0
    chosen = plan["routes"][plan["selected"]]
    others = [r for i, r in enumerate(plan["routes"]) if i != plan["selected"]]

    rows = []
    for i, r in enumerate(plan["routes"]):
        selected = i == plan["selected"]
        if selected:
            name = "Best"
        elif r.get("blocked"):
            name = "Blocked"
        else:
            name = f"Alternate {len(rows)}"
        via = ", ".join(_top_roads(r, 2)) or "local roads"
        rows.append({
            "name": name,
            "estimated_time": fmt_minutes(r["duration_s"] * factor),
            "distance": f"{fmt_distance(r['distance_m'])} · {via}",
            "selected": selected,
        })
    rows.sort(key=lambda row: not row["selected"])

    return {
        "routes": rows,
        "route_coordinates": chosen["coordinates"],
        "alternate_routes_coordinates": [r["coordinates"] for r in others],
        "directions": chosen["steps"],
        "route_summary": {
            "distance_m": chosen["distance_m"],
            "distance_text": fmt_distance(chosen["distance_m"]),
            "free_flow_text": fmt_minutes(chosen["duration_s"]),
            "eta_text": fmt_minutes(chosen["duration_s"] * factor),
            "delay_text": fmt_minutes(max(chosen["duration_s"] * (factor - 1.0), 0)) if factor > 1.005 else "0 min",
            "via_roads": _top_roads(chosen, 3),
            "turns": sum(1 for s in chosen["steps"] if s["is_turn"]),
        },
    }


def build_dashboard_response(result, sensor_data):
    """Flatten the agent pipeline output into the fields static/main.js renders."""
    incident = result["incident"]
    forecast = result["forecast"]
    dispatch = result["dispatch"]
    now = datetime.now().strftime("%H:%M:%S")

    logs = [
        {"time": now, "agent": "Monitor", "type": "alert" if incident["detected"] else "info",
         "message": "Crash detected" if incident["detected"] else "Normal flow"},
        {"time": now, "agent": "Analyzer", "type": "info",
         "message": f"Congestion {forecast['congestion_level'].upper()} ({forecast.get('congestion_score', '—')})"},
    ]

    response = {
        "status": "success",
        "mode": "simulation",
        "congestion_score": forecast.get("congestion_score", 0),
        "active_incidents": 1 if incident["detected"] else 0,
        "vehicle_count": sensor_data["vehicle_count"],
        "decision_by": "Dispatcher",
        "confidence": incident["confidence"] if incident["detected"] else None,
        "pipeline_result": None,   # filled below without the (large) route geometry
    }

    if dispatch.get("dispatched"):
        plan = dispatch["route_result"]["plan"]
        view = _route_view(plan, forecast.get("congestion_score", 0))
        s = view["route_summary"]
        via = ", ".join(s["via_roads"]) or "local roads"
        road = plan["incident_road"] or "route"
        if plan["incident_point"]:
            if plan["avoided"]:
                text = f"Crash on {road}. Rerouted via {via}. {s['distance_text']}, {s['eta_text']}."
            else:
                text = f"Crash on {road}. No clear route. Use caution. {s['distance_text']}, {s['eta_text']}."
        else:
            text = f"Via {via}. {s['distance_text']}, {s['eta_text']}."
        logs.append({"time": now, "agent": "Dispatcher", "type": "success",
                     "message": f"{dispatch['vehicle_priority']}: {s['distance_text']}, {s['eta_text']}, {_turns_text(s['turns'])}"})
        response.update(view)
        response["decision"] = {"heading": "REROUTED" if plan["incident_point"] else "ROUTE", "text": text}
        response["eta_text"] = f"≈ {s['eta_text']} · {s['distance_text']}"
        if plan["incident_point"]:
            response["incident_coordinates"] = plan["incident_point"]
    else:
        logs.append({"time": now, "agent": "Dispatcher", "type": "info", "message": dispatch.get("reason", "No action.")})
        response["decision"] = {"heading": "NO ACTION", "text": dispatch.get("reason", "")}
        response["routes"], response["directions"] = [], []

    response["agent_log"] = logs
    slim = json.loads(json.dumps(result))
    if slim.get("dispatch", {}).get("route_result"):
        slim["dispatch"]["route_result"].pop("plan", None)
    response["pipeline_result"] = slim
    return response


def _parse_inputs(data):
    start, end = data.get("start_node", "A"), data.get("end_node", "J")
    if start not in LANDMARKS or end not in LANDMARKS:
        raise ValueError("Unknown start or end location.")
    if start == end:
        raise ValueError("Start and end must be different locations.")
    return start, end, {
        "time_of_day": float(data.get("time", 8.5)),
        "weather_severity": float(data.get("weather", 0)),
        "vehicle_count": int(data.get("vehicle_count", 85)),
    }


# =====================================================================
# 3. FLASK WEB SERVER ENDPOINTS
# =====================================================================

@app.route('/')
def home():
    """Serves the index.html dashboard from the templates folder."""
    return render_template('index.html')


@app.route('/api/nodes')
def api_nodes():
    """Real Hyderabad locations used by the dropdowns and the map."""
    return jsonify(LANDMARKS)


@app.route('/simulate_incident', methods=['POST'])
def simulate_incident():
    """Run the agents with a crash on the route (Monitor -> Analyzer -> Dispatcher)."""
    try:
        data = request.get_json(silent=True) or {}
        start, end, params = _parse_inputs(data)
        is_crash = data.get("trigger_crash", True)

        sensor_data = {
            "event_id": "UF-LIVE-01",
            "location": {"current_node": start, "destination_node": end,
                         "road": LANDMARKS[start]["name"]},
            "average_speed_kmph": 5 if is_crash else 45,
            "road_id": 0,
            **params,
        }
        result = coordinator.run(sensor_data)
        return jsonify(build_dashboard_response(result, sensor_data))

    except (ValueError, RoutingError) as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/predict_all', methods=['POST'])
def predict_all():
    """No incident: forecast congestion for the chosen conditions, over the whole day,
    and predict the road route, distance, turns and ETA."""
    try:
        data = request.get_json(silent=True) or {}
        start, end, params = _parse_inputs(data)
        now = datetime.now().strftime("%H:%M:%S")

        forecast = decision_tree_adapter({"sensor_data": params})
        score = forecast["congestion_score"]

        curve = []
        for i in range(48):
            h = i / 2.0
            s = predict_congestion(time_of_day=h, weather_severity=params["weather_severity"],
                                   current_vehicle_count=params["vehicle_count"])
            curve.append({"hour": h, "label": fmt_hour(h), "score": round(s, 1)})
        peak = max(curve, key=lambda c: c["score"])
        best = min(curve, key=lambda c: (c["score"], abs(c["hour"] - params["time_of_day"])))

        plan = plan_routes((LANDMARKS[start]["lat"], LANDMARKS[start]["lng"]),
                           (LANDMARKS[end]["lat"], LANDMARKS[end]["lng"]), incident=False)
        view = _route_view(plan, score)
        s = view["route_summary"]
        chosen = plan["routes"][0]
        best_eta = fmt_minutes(chosen["duration_s"] * (1 + best["score"] / 100.0))
        via = ", ".join(s["via_roads"]) or "local roads"

        response = {
            "status": "success",
            "mode": "prediction",
            "congestion_score": score,
            "active_incidents": 0,
            "vehicle_count": params["vehicle_count"],
            "decision_by": "Analyzer",
            "confidence": None,
            "decision": {
                "heading": "FORECAST",
                "text": (f"{forecast['congestion_level'].upper()} congestion at {fmt_hour(params['time_of_day'])}. "
                         f"{s['distance_text']}, {s['eta_text']}. Best time: {best['label']}."),
            },
            "eta_text": f"≈ {s['eta_text']} · {s['distance_text']}",
            "agent_log": [
                {"time": now, "agent": "Analyzer", "type": "info",
                 "message": f"Congestion {forecast['congestion_level'].upper()} ({score})"},
                {"time": now, "agent": "Router", "type": "success",
                 "message": f"{s['distance_text']}, {s['eta_text']}, {_turns_text(s['turns'])}"},
            ],
            "prediction": {
                "time_label": fmt_hour(params["time_of_day"]),
                "weather": "Rain" if params["weather_severity"] >= 1 else "Clear",
                "vehicle_count": params["vehicle_count"],
                "congestion_score": score,
                "congestion_level": forecast["congestion_level"],
                "curve": curve,
                "peak": peak,
                "best": best,
                "best_eta": best_eta,
                "route": s,
            },
            **view,
        }
        return jsonify(response)

    except (ValueError, RoutingError) as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    # Start the local server
    print("Starting UrbanFlow AI Command Center...")
    app.run(debug=True, port=5000)
