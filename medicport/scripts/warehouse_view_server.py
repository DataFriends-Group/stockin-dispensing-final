#!/usr/bin/env python3
"""
On-demand HTTP wrapper around the warehouse_view / warehouse_view_robot
static site generators.

Standalone server - does not touch main.py's in-memory state, since the
generators only read the inventory JSON from disk.

Endpoints:
  GET /                - links to both views
  GET /view/refresh     - regenerate the read-only pages from R3_DF.json, then
                           redirect to /view/index.html
  GET /view/...          - serves the last generated read-only pages (static,
                           no regeneration)
  GET /robot/refresh     - regenerate the robot-control pages, then redirect
                           to /robot/index.html
  GET /robot/...         - serves the last generated robot-control pages
                           (static, no regeneration)

Usage:
    python3 warehouse_view_server.py
    python3 warehouse_view_server.py --input ../data/R3_DF.json --port 8003
"""
import argparse
import os
import sys

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# warehouse_view / warehouse_view_robot live next to this file; add that dir
# to sys.path so the imports below work regardless of the importer's cwd
# (e.g. when run_servers.py imports this module from the parent directory).
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import warehouse_view
import warehouse_view_robot

ROOT_PAGE = """<!DOCTYPE html>
<html lang="cs">
<head><meta charset="utf-8"><title>MedicPort - Warehouse View</title>
<style>body{font-family:-apple-system,Helvetica,Arial,sans-serif;padding:24px;max-width:640px}
a{color:#2563eb}li{margin-bottom:8px}</style></head>
<body>
<h1>MedicPort - Warehouse View</h1>
<ul>
  <li><a href="/view/refresh">Prehled skladu (read-only) - obnovit a otevrit</a></li>
  <li><a href="/view/index.html">Prehled skladu - posledni vygenerovana verze</a></li>
  <li><a href="/robot/refresh">Prehled skladu s ovladanim robota - obnovit a otevrit</a></li>
  <li><a href="/robot/index.html">Prehled skladu s ovladanim robota - posledni vygenerovana verze</a></li>
</ul>
</body></html>"""


def build_app(input_path, view_dir, robot_dir):
    app = FastAPI(title="MedicPort Warehouse View", version="1.0.0")

    def refresh_view():
        warehouse_view.generate(input_path, view_dir)

    def refresh_robot():
        warehouse_view_robot.generate(input_path, robot_dir)

    # Make sure both directories exist and hold a fresh snapshot on startup.
    refresh_view()
    refresh_robot()

    # Registered before the StaticFiles mounts below: Starlette matches routes
    # in registration order, and a mount matches any "/view/*" path, so it
    # would otherwise shadow "/view/refresh" (and likewise for "/robot/refresh").
    @app.get("/view/refresh")
    def view_refresh():
        refresh_view()
        return RedirectResponse(url="/view/index.html")

    @app.get("/robot/refresh")
    def robot_refresh():
        refresh_robot()
        return RedirectResponse(url="/robot/index.html")

    @app.get("/", response_class=HTMLResponse)
    def root():
        return ROOT_PAGE

    app.mount("/view", StaticFiles(directory=view_dir, html=True), name="view")
    app.mount("/robot", StaticFiles(directory=robot_dir, html=True), name="robot")

    return app


DEFAULT_INPUT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "R3_DF.json"))
DEFAULT_VIEW_OUTPUT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "reports", "warehouse_view"))
DEFAULT_ROBOT_OUTPUT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "reports", "warehouse_view_robot"))

# Built with default paths at import time so other processes (e.g.
# run_servers.py) can just do `from warehouse_view_server import warehouse_app`
# and serve it alongside their other ASGI apps, the same way dispense_app /
# optimization_app are wired in.
warehouse_app = build_app(DEFAULT_INPUT, DEFAULT_VIEW_OUTPUT, DEFAULT_ROBOT_OUTPUT)


def main():
    parser = argparse.ArgumentParser(description="Serve on-demand warehouse view pages over HTTP.")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT, help="Path to inventory JSON (default: data/R3_DF.json)")
    parser.add_argument("--view-output", default=DEFAULT_VIEW_OUTPUT, help="Output dir for the read-only view")
    parser.add_argument("--robot-output", default=DEFAULT_ROBOT_OUTPUT, help="Output dir for the robot-control view")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    view_dir = os.path.abspath(args.view_output)
    robot_dir = os.path.abspath(args.robot_output)

    if not os.path.isfile(input_path):
        raise SystemExit(f"Input file not found: {input_path}")

    app = build_app(input_path, view_dir, robot_dir)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
