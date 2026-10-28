import csv
import io
import os
import re
from datetime import date, datetime
from functools import wraps
from pathlib import Path

import pandas as pd
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

from config import BASE_DIR, config, verify_login
from services import assets, common, optimized_orders
from services.logistics import service as logistics, store as trip_store

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

ALLOWED_EXTENSIONS = {".xls", ".xlsx"}


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route("/favicon.ico")
def favicon():
    return redirect(url_for("static", filename="img/login-hero.jpg"))


@app.route("/")
def index():
    if session.get("logged_in"):
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if verify_login(username, password):
            session.clear()
            session["logged_in"] = True
            session["username"] = username.strip()
            return redirect(url_for("dashboard"))
        error = "Invalid username or password."

    return render_template(
        "login.html",
        error=error,
        hero_exists=(BASE_DIR / "static" / "img" / "login-hero.jpg").exists(),
        logo_exists=(BASE_DIR / "static" / "img" / "CT-Logo.jpg").exists(),
        ct_logo_exists=(BASE_DIR / "static" / "img" / "CT-Logo.jpg").exists(),
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template(
        "dashboard.html",
        title="Menengai Orders Logistics",
        warehouses=list(common.WAREHOUSES.keys()),
        logo_exists=(BASE_DIR / "static" / "img" / "CT-Logo.jpg").exists(),
        ct_logo_exists=(BASE_DIR / "static" / "img" / "CT-Logo.jpg").exists(),
        bg_exists=(BASE_DIR / "static" / "img" / "bg.jpg").exists(),
        controltech_url=config.CONTROLTECH_URL,
    )


@app.route("/api/assets")
@login_required
def api_assets():
    query = request.args.get("q", "").strip()
    limit = request.args.get("limit", "100")
    try:
        limit_val = max(1, min(int(limit), 500))
    except ValueError:
        limit_val = 100
    try:
        return jsonify(
            {
                "error": 0,
                "assets": assets.assets_to_json(limit=limit_val, query=query),
                "total": len(assets.load_assets_catalog()),
            }
        )
    except Exception as exc:
        return jsonify({"error": 1, "message": str(exc)}), 500


@app.route("/api/match-asset", methods=["POST"])
@login_required
def api_match_asset():
    order_files = request.files.getlist("orders")
    if not order_files or all(not f.filename for f in order_files):
        return jsonify({"error": 1, "message": "Upload at least one orders Excel file."}), 400

    try:
        combined_orders, truck_number_norm = optimized_orders.process_multiple_excels(
            order_files
        )
        asset = assets.find_asset_by_truck(truck_number_norm)
        return jsonify(
            {
                "error": 0,
                "truck_number": truck_number_norm,
                "summary": {
                    "files": len(order_files),
                    "delivery_points": len(combined_orders),
                    "tonnage": round(float(combined_orders["TONNAGE"].sum()), 2),
                    "amount": round(float(combined_orders["AMOUNT"].sum()), 2),
                },
                "asset": (
                    {
                        "item_id": asset["item_id"],
                        "name": asset["name"],
                    }
                    if asset
                    else None
                ),
            }
        )
    except Exception as exc:
        return jsonify({"error": 1, "message": str(exc)}), 400


@app.route("/api/optimized/dispatch", methods=["POST"])
@login_required
def api_optimized_dispatch():
    order_files = request.files.getlist("orders")
    asset_item_id = request.form.get("asset_item_id", "").strip()
    if not order_files or any(not f.filename for f in order_files):
        return jsonify({"error": 1, "message": "Please upload orders Excel file(s)."}), 400
    if not asset_item_id:
        return jsonify({"error": 1, "message": "Please select an asset."}), 400

    for f in order_files:
        if not allowed_file(f.filename):
            return jsonify({"error": 1, "message": "Invalid orders file type."}), 400

    try:
        unit_id = int(asset_item_id)
    except ValueError:
        return jsonify({"error": 1, "message": "Invalid asset selection."}), 400

    asset = assets.get_asset_by_item_id(unit_id)
    if not asset:
        return jsonify({"error": 1, "message": "Selected asset was not found."}), 400

    warehouse_choice = request.form.get("warehouse", "MORL")
    if warehouse_choice not in common.WAREHOUSES:
        return jsonify({"error": 1, "message": f"Unknown warehouse: {warehouse_choice}"}), 400

    try:
        tf, tt = common.compute_route_timestamps_from_range(
            request.form.get("route_from"),
            request.form.get("route_end"),
        )
    except ValueError as exc:
        return jsonify({"error": 1, "message": str(exc)}), 400

    try:
        gdf_joined, truck_number_norm = optimized_orders.process_multiple_excels(
            order_files
        )
        if gdf_joined is None or gdf_joined.empty:
            return jsonify(
                {
                    "error": 1,
                    "message": "No delivery rows with valid coordinates were found.",
                }
            ), 400

        if truck_number_norm and asset["normalized_name"] != truck_number_norm:
            if truck_number_norm not in asset["normalized_name"]:
                return jsonify(
                    {
                        "error": 1,
                        "message": (
                            f"Selected asset '{asset['name']}' does not match truck "
                            f"in orders ({truck_number_norm})."
                        ),
                    }
                ), 400

        result = optimized_orders.send_orders_and_create_route(
            config.WIALON_TOKEN,
            config.WIALON_RESOURCE_ID,
            asset["item_id"],
            asset["name"],
            gdf_joined,
            tf,
            tt,
            warehouse_choice,
        )
        route_uid = result.pop("route_uid", None)
        route_name = result.pop("route_name", None)
        ordered_stops = result.pop("ordered_stops", None)
        if result.get("error") == 0:
            result["summary"] = {
                "delivery_points": len(gdf_joined),
                "tonnage": round(float(gdf_joined["TONNAGE"].sum()), 2),
                "amount": round(float(gdf_joined["AMOUNT"].sum()), 2),
                "warehouse": warehouse_choice,
            }
            result["trip_id"], result["trip_warning"] = _record_trip(
                route_uid, route_name, asset, warehouse_choice, ordered_stops, tf, tt, order_files
            )
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": 1, "message": str(exc)}), 500


def _record_trip(route_uid, route_name, asset, warehouse, stops, tf, tt, order_files):
    """Keeps the dispatched route for the Trips tab; the route is already in Wialon either way."""
    if not trip_store.configured():
        return None, "The trip was not saved for the Trips tab: the app database is not configured."
    if not route_uid or not stops:
        return None, "The trip was not saved for the Trips tab: Wialon did not return the route."
    try:
        trip_id = logistics.record_upload(
            resource_id=config.WIALON_RESOURCE_ID,
            route_uid=route_uid,
            route_name=route_name or asset["name"],
            unit_id=asset["item_id"],
            unit_name=asset["name"],
            warehouse=warehouse,
            depot=common.WAREHOUSES[warehouse],
            stops=stops,
            start=tf,
            end=tt,
            files=", ".join(f.filename for f in order_files if f.filename),
            created_by=session.get("username", ""),
        )
        return trip_id, None
    except Exception as exc:
        print(f"[logistics] could not save the dispatched trip: {exc}")
        return None, f"The trip was not saved for the Trips tab: {exc}"


# ---- Trips (Logistics dashboard) ----

DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MAX_DAYS = 31


def _day_range():
    day_from = request.args.get("from", "")
    day_to = request.args.get("to", "") or day_from
    if not DAY_RE.match(day_from) or not DAY_RE.match(day_to):
        raise ValueError("Choose the days")
    span = (date.fromisoformat(day_to) - date.fromisoformat(day_from)).days + 1
    if span < 1:
        raise ValueError("The end day is before the start day")
    if span > MAX_DAYS:
        raise ValueError(f"Choose at most {MAX_DAYS} days")
    return day_from, day_to


def _trips_error(exc):
    if isinstance(exc, trip_store.NotFound):
        return jsonify({"error": 1, "message": str(exc)}), 404
    if isinstance(exc, ValueError):
        return jsonify({"error": 1, "message": str(exc)}), 400
    print(f"[logistics] {exc}")
    return jsonify({"error": 1, "message": str(exc)}), 500


@app.route("/api/logistics/trips")
@login_required
def api_trips():
    if not trip_store.configured():
        return jsonify({"error": 0, "storage": False, "rows": [], "total": 0, "page": 1, "pageSize": 15,
                        "counts": {k: 0 for k in ("in_progress", "scheduled", "completed", "ended")}})
    try:
        day_from, day_to = _day_range()
        page = max(1, int(request.args.get("page", "1")))
        page_size = max(5, min(100, int(request.args.get("pageSize", "15"))))
        state = request.args.get("state") or None
        data = logistics.list_page(
            config.WIALON_TOKEN,
            config.WIALON_RESOURCE_ID,
            day_from,
            day_to,
            q=request.args.get("q", ""),
            state=state,
            page=page,
            page_size=page_size,
            sort=request.args.get("sort") or None,
        )
        return jsonify({"error": 0, "storage": True, **data})
    except Exception as exc:
        return _trips_error(exc)


@app.route("/api/logistics/trips/<trip_id>", methods=["GET", "DELETE"])
@login_required
def api_trip(trip_id):
    try:
        if request.method == "DELETE":
            trip_store.remove_trip(config.WIALON_RESOURCE_ID, trip_id)
            return jsonify({"error": 0})
        return jsonify({"error": 0, **logistics.trip_detail(config.WIALON_TOKEN, config.WIALON_RESOURCE_ID, trip_id)})
    except Exception as exc:
        return _trips_error(exc)


STATE_LABEL = {
    "in_progress": "On the road",
    "scheduled": "Scheduled",
    "completed": "Completed",
    "ended": "Ended with missed stops",
}
VISIT_LABEL = {"on_time": "On time", "late": "Late", "early": "Early", "missed": "Missed", "pending": "Not yet"}


def _clock(t):
    return datetime.fromtimestamp(int(t), tz=logistics.EAT).strftime("%Y-%m-%d %H:%M") if t else ""


def _stop_rows(row, tracking, summary):
    visits = {v["seq"]: v for v in (tracking or {}).get("visits", [])}
    km = 0.0
    out = []
    for st in row["plan"]["stops"]:
        km += st.get("legKm", 0)
        v = visits.get(st["seq"], {})
        arrived = v.get("arrivedAt")
        out.append(
            {
                "Day": summary["day"],
                "Route": summary["routeName"],
                "Vehicle": summary["unitName"],
                "Stop #": st["seq"],
                "Customer": st["name"],
                "Location": st.get("location", ""),
                "Invoice": st.get("invoice", ""),
                "Weight (kg)": st.get("weightKg", 0),
                "Amount": st.get("amount", 0),
                "Planned km": round(km, 1),
                "ETA": _clock(st["eta"]),
                "Arrived": _clock(arrived),
                "Left": _clock(v.get("leftAt")),
                "Minutes vs ETA": round((arrived - st["eta"]) / 60) if arrived else "",
                "Status": VISIT_LABEL.get(v.get("status", "pending"), "Not yet"),
                "Closest (m)": v.get("closestM") if v.get("closestM") is not None else "",
            }
        )
    return out


@app.route("/api/logistics/trips/<trip_id>/export")
@login_required
def api_trip_export(trip_id):
    try:
        detail = logistics.trip_detail(config.WIALON_TOKEN, config.WIALON_RESOURCE_ID, trip_id)
    except Exception as exc:
        return _trips_error(exc)
    row = {"plan": detail["plan"]}
    rows = _stop_rows(row, detail["tracking"], detail)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()) if rows else ["Stop #"])
    writer.writeheader()
    writer.writerows(rows)
    name = re.sub(r"[^A-Za-z0-9 _.-]+", "", detail["routeName"]).strip() or "trip"
    return Response(
        "\ufeff" + buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}.csv"'},
    )


@app.route("/api/logistics/report")
@login_required
def api_trips_report():
    try:
        day_from, day_to = _day_range()
        trips = logistics.report_trips(config.WIALON_TOKEN, config.WIALON_RESOURCE_ID, day_from, day_to)
    except Exception as exc:
        return _trips_error(exc)
    trip_rows, stop_rows = [], []
    for row, tracking, s in trips:
        p = s["progress"] or {}
        trip_rows.append(
            {
                "Day": s["day"],
                "Route": s["routeName"],
                "Vehicle": s["unitName"],
                "Warehouse": s["warehouse"],
                "Source": "Uploaded" if s["source"] == "upload" else "Wialon Logistics",
                "Window from": _clock(s["windowFrom"]),
                "Window to": _clock(s["windowTo"]),
                "Planned return": _clock(s["returnAt"]),
                "Stops": s["stops"],
                "On time": p.get("onTime", 0),
                "Late": p.get("late", 0),
                "Missed": p.get("missed", 0),
                "Not yet": p.get("pending", s["stops"]),
                "Planned km": s["km"],
                "Driven km": (tracking or {}).get("km", ""),
                "Left warehouse": _clock((tracking or {}).get("startedAt")),
                "Back at warehouse": _clock((tracking or {}).get("returnedAt")),
                "Weight (kg)": s["weightKg"],
                "Amount": s["amount"],
                "State": STATE_LABEL[s["state"]],
                "Uploaded by": s["createdBy"],
                "Files": s["files"],
            }
        )
        stop_rows.extend(_stop_rows(row, tracking, s))
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        pd.DataFrame(trip_rows).to_excel(writer, sheet_name="Trips", index=False)
        pd.DataFrame(stop_rows).to_excel(writer, sheet_name="Stops", index=False)
    out.seek(0)
    period = day_from if day_from == day_to else f"{day_from} to {day_to}"
    return send_file(
        out,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"Menengai trips {period}.xlsx",
    )


if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        use_reloader=False,
    )
