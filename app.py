import os
from functools import wraps
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from config import BASE_DIR, config, verify_login
from services import assets, common, optimized_orders

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
        _, truck_number_norm = optimized_orders.process_multiple_excels(order_files)
        asset = assets.find_asset_by_truck(truck_number_norm)
        return jsonify(
            {
                "error": 0,
                "truck_number": truck_number_norm,
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
        if result.get("error") == 0:
            result["summary"] = {
                "delivery_points": len(gdf_joined),
                "tonnage": round(float(gdf_joined["TONNAGE"].sum()), 2),
                "amount": round(float(gdf_joined["AMOUNT"].sum()), 2),
                "warehouse": warehouse_choice,
            }
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": 1, "message": str(exc)}), 500


if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        use_reloader=False,
    )
