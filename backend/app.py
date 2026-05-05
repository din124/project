"""Flask backend for the Smart Process Scheduling System (v2.1).

Exposes REST endpoints for process management, scheduling, analytics,
and real-time system monitoring.
"""

# app.py
# ─────────────────────────────────────────────────────────────────────────────
#  SMART PROCESS SCHEDULING SYSTEM — Flask Backend  (v2.1 + Real-time Monitor)
# ─────────────────────────────────────────────────────────────────────────────

from flask import Flask, request, jsonify
from flask_cors import CORS
from db import (
    setup_schema, insert_process, get_all_processes, clear_results,
    insert_result, log_run, get_run_history, update_algorithm_stats,
    get_algorithm_stats, clear_processes
)
from scheduler import smart_schedule, ALGORITHM_COMPLEXITY
from ml_model import get_model_info
from system_monitor import get_stats, get_cpu_history   # ← NEW

app = Flask(__name__)
CORS(app)

try:
    setup_schema()
    print("✓ Database schema ready.")
except Exception as e:
    print(f"⚠ DB setup skipped: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  REAL-TIME SYSTEM MONITOR  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/system/stats", methods=["GET"])
def system_stats():
    """Return a live snapshot of CPU, RAM, disk, and top OS processes."""
    return jsonify(get_stats())

@app.route("/system/history", methods=["GET"])
def system_history():
    """Return the last 60 CPU data-points for the live chart."""
    return jsonify(get_cpu_history())

@app.route("/system/processes", methods=["GET"])
def system_processes():
    """Return the top real OS processes sorted by CPU usage."""
    stats = get_stats()
    return jsonify(stats.get("processes", []))


# ─────────────────────────────────────────────────────────────────────────────
#  PROCESS MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/add_process", methods=["POST"])
def add_process():
    """Add a process to the scheduling queue.

    Request body (JSON):
        process_id (str): Unique process identifier.
        burst_time (int): CPU burst time.
        arrival_time (int): Arrival time.
        process_type (str, optional): ``'CPU'`` or ``'IO'``. Defaults to ``'CPU'``.
        priority (int, optional): 1 (highest) to 10 (lowest). Defaults to 5.

    Returns:
        JSON: ``{message, process_id}`` on success, ``{error}`` on failure.
    """
    data = request.json
    try:
        insert_process(
            data["process_id"],
            int(data["burst_time"]),
            int(data["arrival_time"]),
            data.get("process_type", "CPU"),
            int(data.get("priority", 5))
        )
        return jsonify({"message": "Process added successfully", "process_id": data["process_id"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/processes", methods=["GET"])
def get_processes():
    """Return all processes currently in the scheduling queue."""
    try:
        procs = get_all_processes()
        for p in procs:
            if "created_at" in p and p["created_at"]:
                p["created_at"] = str(p["created_at"])
        return jsonify(procs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/clear_processes", methods=["DELETE"])
def clear_all_processes():
    """Delete all processes from the scheduling queue."""
    try:
        clear_processes()
        return jsonify({"message": "All processes cleared"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
#  SCHEDULING
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/generate_schedule", methods=["GET"])
def generate_schedule():
    """Run all scheduling algorithms and return the ML-recommended result.

    Reads the current process queue, executes FCFS, SJF, Priority, Round Robin,
    and Smart (ML), persists the results, and returns a comparison.

    Returns:
        JSON: ``{primary, comparison, recommendation, run_id}`` on success,
        ``{error, trace}`` on failure.
    """
    try:
        processes = get_all_processes()
        if not processes:
            return jsonify({"error": "No processes found. Add processes first."}), 400

        for p in processes:
            p["burst_time"]   = int(p["burst_time"])
            p["arrival_time"] = int(p["arrival_time"])
            p["priority"]     = int(p.get("priority") or 5)
            if "created_at" in p:
                del p["created_at"]

        result     = smart_schedule(processes)
        primary    = result["primary"]
        comparison = result["comparison"]
        rec        = result["recommendation"]

        clear_results()
        run_id = log_run(
            algorithm_used = rec.get("algorithm", "Smart (ML)"),
            num_processes  = len(primary),
            avg_waiting    = round(sum(p["waiting_time"]    for p in primary) / len(primary), 2) if primary else 0,
            avg_turnaround = round(sum(p["turnaround_time"] for p in primary) / len(primary), 2) if primary else 0,
            ml_confidence  = rec.get("confidence", 0)
        )

        for p in primary:
            insert_result(
                pid         = p["process_id"],
                score       = p.get("score", 0),
                order_no    = p["order"],
                wt          = p["waiting_time"],
                tat         = p["turnaround_time"],
                algorithm   = rec.get("algorithm", "Smart (ML)"),
                start_time  = p.get("start_time", 0),
                finish_time = p.get("finish_time", 0),
                run_id      = run_id
            )

        for comp in comparison:
            if "error" not in comp:
                update_algorithm_stats(comp["algorithm"], comp["avg_waiting"], comp["avg_turnaround"])

        return jsonify({"primary": primary, "comparison": comparison, "recommendation": rec, "run_id": run_id})

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ─────────────────────────────────────────────────────────────────────────────
#  ANALYTICS & INFO
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/history", methods=["GET"])
def history():
    """Return the 20 most recent scheduling run records."""
    try:
        runs = get_run_history(limit=20)
        for r in runs:
            if "ran_at" in r and r["ran_at"]:
                r["ran_at"] = str(r["ran_at"])
        return jsonify(runs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/algorithm_stats", methods=["GET"])
def algorithm_stats():
    """Return aggregated performance statistics per scheduling algorithm."""
    try:
        stats = get_algorithm_stats()
        for s in stats:
            if "last_used" in s and s["last_used"]:
                s["last_used"] = str(s["last_used"])
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/complexity", methods=["GET"])
def complexity():
    """Return time/space complexity and data structure info for each algorithm."""
    return jsonify(ALGORITHM_COMPLEXITY)

@app.route("/model_info", methods=["GET"])
def model_info():
    """Return metadata about the ML model (type, features, training size)."""
    return jsonify(get_model_info())

@app.route("/")
def home():
    """Health check endpoint listing all available API routes."""
    return jsonify({
        "status":  "running",
        "project": "Smart Process Scheduling System",
        "version": "2.1",
        "endpoints": [
            "POST   /add_process",
            "GET    /processes",
            "DELETE /clear_processes",
            "GET    /generate_schedule",
            "GET    /history",
            "GET    /algorithm_stats",
            "GET    /complexity",
            "GET    /model_info",
            "GET    /system/stats      ← NEW",
            "GET    /system/history    ← NEW",
            "GET    /system/processes  ← NEW",
        ]
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)