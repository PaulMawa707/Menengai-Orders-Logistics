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
from services import common, optimized_orders

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


@app.route("/api/optimized/dispatch", methods=["POST"])
@login_required
def api_optimized_dispatch():
    order_files = request.files.getlist("orders")
    assets_file = request.files.get("assets")
    if not order_files or not assets_file or assets_file.filename == "":
        return jsonify(
            {"error": 1, "message": "Please upload orders Excel and assets Excel."}
        ), 400

    for f in order_files:
        if not f.filename or not allowed_file(f.filename):
            return jsonify({"error": 1, "message": "Invalid orders file type."}), 400
    if not allowed_file(assets_file.filename):
        return jsonify({"error": 1, "message": "Invalid assets file type."}), 400

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

        unit_id, vehicle_name = optimized_orders.read_asset_id_from_excel(
            assets_file, truck_number_norm
        )
        if not unit_id:
            return jsonify(
                {
                    "error": 1,
                    "message": (
                        f"Could not find unit ID for truck (normalized): "
                        f"{truck_number_norm or 'UNKNOWN'}."
                    ),
                }
            ), 400

        result = optimized_orders.send_orders_and_create_route(
            config.WIALON_TOKEN,
            config.WIALON_RESOURCE_ID,
            unit_id,
            vehicle_name,
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
