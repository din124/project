"""Database module for the Smart Process Scheduling System.

Manages four MySQL tables:
    - processes:        input process queue
    - schedule_results: current scheduling output
    - run_history:      audit log of every scheduling run
    - algorithm_stats:  aggregated performance metrics per algorithm
"""

import mysql.connector
from config import DB_CONFIG


def get_connection():
    """Return a new MySQL connection using DB_CONFIG."""
    return mysql.connector.connect(**DB_CONFIG)


# ─────────────────────────────────────────────────────────────────────────────
#  SCHEMA SETUP — call once on startup
# ─────────────────────────────────────────────────────────────────────────────
def setup_schema():
    """Create all four database tables if they do not already exist.

    Tables created: ``processes``, ``schedule_results``, ``run_history``,
    and ``algorithm_stats``. Safe to call on every application startup
    because all statements use ``CREATE TABLE IF NOT EXISTS``.
    """
    conn = get_connection()
    cur  = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS processes (
            id           INT AUTO_INCREMENT PRIMARY KEY,
            process_id   VARCHAR(20)  NOT NULL,
            burst_time   INT          NOT NULL,
            arrival_time INT          NOT NULL DEFAULT 0,
            process_type ENUM('CPU','IO') NOT NULL DEFAULT 'CPU',
            priority     INT          NOT NULL DEFAULT 5,
            created_at   TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS schedule_results (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            process_id      VARCHAR(20)  NOT NULL,
            algorithm       VARCHAR(30)  NOT NULL DEFAULT 'Smart (ML)',
            rank_score      FLOAT,
            execution_order INT,
            waiting_time    FLOAT,
            turnaround_time FLOAT,
            start_time      INT,
            finish_time     INT,
            run_id          INT,
            created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS run_history (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            algorithm_used  VARCHAR(30)  NOT NULL,
            num_processes   INT,
            avg_waiting     FLOAT,
            avg_turnaround  FLOAT,
            ml_confidence   FLOAT,
            ran_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS algorithm_stats (
            id               INT AUTO_INCREMENT PRIMARY KEY,
            algorithm        VARCHAR(30) UNIQUE NOT NULL,
            total_runs       INT     NOT NULL DEFAULT 0,
            total_avg_wait   FLOAT   NOT NULL DEFAULT 0,
            total_avg_tat    FLOAT   NOT NULL DEFAULT 0,
            best_wait        FLOAT,
            best_tat         FLOAT,
            last_used        TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
#  PROCESSES TABLE
# ─────────────────────────────────────────────────────────────────────────────
def insert_process(pid, bt, at, ptype, priority=5):
    """Insert a new process into the processes table.

    Args:
        pid (str): Unique process identifier.
        bt (int): Burst time (CPU time required).
        at (int): Arrival time.
        ptype (str): Process type — ``'CPU'`` or ``'IO'``.
        priority (int): Scheduling priority (1 = highest, 10 = lowest). Defaults to 5.
    """
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(
        """INSERT INTO processes
           (process_id, burst_time, arrival_time, process_type, priority)
           VALUES (%s, %s, %s, %s, %s)""",
        (pid, bt, at, ptype, priority)
    )
    conn.commit()
    conn.close()


def get_all_processes():
    """Fetch all processes ordered by creation time.

    Returns:
        list[dict]: All rows from the processes table.
    """
    conn = get_connection()
    cur  = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM processes ORDER BY created_at ASC")
    data = cur.fetchall()
    conn.close()
    return data


def clear_processes():
    """Delete all rows from the processes table."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("DELETE FROM processes")
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
#  SCHEDULE RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────
def clear_results():
    """Delete all rows from the schedule_results table."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("DELETE FROM schedule_results")
    conn.commit()
    conn.close()


def insert_result(pid, score, order_no, wt, tat, algorithm="Smart (ML)",
                  start_time=0, finish_time=0, run_id=None):
    """Insert a single scheduled-process result.

    Args:
        pid (str): Process identifier.
        score (float): ML rank score assigned to the process.
        order_no (int): Execution order position.
        wt (float): Waiting time.
        tat (float): Turnaround time.
        algorithm (str): Algorithm that produced this result. Defaults to ``'Smart (ML)'``.
        start_time (int): Absolute start time on the CPU timeline. Defaults to 0.
        finish_time (int): Absolute finish time on the CPU timeline. Defaults to 0.
        run_id (int | None): Foreign key to run_history. Defaults to None.
    """
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(
        """INSERT INTO schedule_results
           (process_id, algorithm, rank_score, execution_order,
            waiting_time, turnaround_time, start_time, finish_time, run_id)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (pid, algorithm, score, order_no, wt, tat, start_time, finish_time, run_id)
    )
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
#  RUN HISTORY TABLE
# ─────────────────────────────────────────────────────────────────────────────
def log_run(algorithm_used, num_processes, avg_waiting, avg_turnaround, ml_confidence=0.0):
    """Insert a scheduling run record into run_history and return its ID.

    Args:
        algorithm_used (str): Name of the algorithm selected for this run.
        num_processes (int): Number of processes scheduled.
        avg_waiting (float): Average waiting time across all processes.
        avg_turnaround (float): Average turnaround time across all processes.
        ml_confidence (float): ML model confidence score (0–100). Defaults to 0.0.

    Returns:
        int: Auto-incremented ID of the newly inserted run record.
    """
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(
        """INSERT INTO run_history
           (algorithm_used, num_processes, avg_waiting, avg_turnaround, ml_confidence)
           VALUES (%s, %s, %s, %s, %s)""",
        (algorithm_used, num_processes, avg_waiting, avg_turnaround, ml_confidence)
    )
    run_id = cur.lastrowid
    conn.commit()
    conn.close()
    return run_id


def get_run_history(limit=10):
    """Return the most recent scheduling runs.

    Args:
        limit (int): Maximum number of rows to return. Defaults to 10.

    Returns:
        list[dict]: Run records ordered newest-first.
    """
    conn = get_connection()
    cur  = conn.cursor(dictionary=True)
    cur.execute(
        "SELECT * FROM run_history ORDER BY ran_at DESC LIMIT %s", (limit,)
    )
    data = cur.fetchall()
    conn.close()
    return data


# ─────────────────────────────────────────────────────────────────────────────
#  ALGORITHM STATS TABLE (DBMS Aggregation Demo)
# ─────────────────────────────────────────────────────────────────────────────
def update_algorithm_stats(algorithm, avg_wait, avg_tat):
    """Upsert aggregated performance stats for a scheduling algorithm.

    Uses ``INSERT ... ON DUPLICATE KEY UPDATE`` for an atomic upsert.
    Increments run count, accumulates totals, and tracks best (lowest)
    waiting and turnaround times seen so far.

    Args:
        algorithm (str): Algorithm name (must match the UNIQUE column).
        avg_wait (float): Average waiting time from the latest run.
        avg_tat (float): Average turnaround time from the latest run.
    """
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(
        """INSERT INTO algorithm_stats
               (algorithm, total_runs, total_avg_wait, total_avg_tat, best_wait, best_tat, last_used)
           VALUES (%s, 1, %s, %s, %s, %s, NOW())
           ON DUPLICATE KEY UPDATE
               total_runs     = total_runs + 1,
               total_avg_wait = total_avg_wait + VALUES(total_avg_wait),
               total_avg_tat  = total_avg_tat  + VALUES(total_avg_tat),
               best_wait      = LEAST(COALESCE(best_wait, 9999), VALUES(best_wait)),
               best_tat       = LEAST(COALESCE(best_tat,  9999), VALUES(best_tat)),
               last_used      = NOW()
        """,
        (algorithm, avg_wait, avg_tat, avg_wait, avg_tat)
    )
    conn.commit()
    conn.close()


def get_algorithm_stats():
    """Return aggregated performance statistics for every algorithm.

    Computes overall averages by dividing accumulated totals by run count
    directly in SQL for efficiency.

    Returns:
        list[dict]: Rows ordered by ``overall_avg_wait`` ascending. Each row
        contains ``algorithm``, ``total_runs``, ``overall_avg_wait``,
        ``overall_avg_tat``, ``best_wait``, ``best_tat``, and ``last_used``.
    """
    conn = get_connection()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT
            algorithm,
            total_runs,
            ROUND(total_avg_wait  / total_runs, 2) AS overall_avg_wait,
            ROUND(total_avg_tat   / total_runs, 2) AS overall_avg_tat,
            best_wait,
            best_tat,
            last_used
        FROM algorithm_stats
        ORDER BY overall_avg_wait ASC
    """)
    data = cur.fetchall()
    conn.close()
    return data