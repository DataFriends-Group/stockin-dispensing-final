#!/usr/bin/env python3
"""
OFFLINE editor for the warehouse layout (R3_DF.json): add, remove and resize
shelves within a rack.

Only run this while main.py is stopped. main.py owns the same inventory
file - it loads it into memory at startup and overwrites it whole on every
save_warehouse_state() call (see main.py:1506, main.py:1672, config.py). If
main.py is running at the same time as this editor, whichever one saves last
silently wins and the other side's changes are lost. At startup this script
checks whether something is already listening on port 8000 (main.py's
default port) and refuses to start unless --force is given.

Reads and writes the JSON directly on disk - there is no in-memory cache, so
every page load reflects the current file and every edit is saved
immediately (no separate "commit" step, no undo).

Rules enforced:
  - A shelf can only be deleted while it has zero items placed on it.
  - Changing a shelf's width/height does NOT move any other shelf's
    CoordinateY in the same rack - that stays purely manual.
  - A new shelf inherits CoordinateX/CoordinateZ from the rack's existing
    shelves (all shelves in a rack share those two); CoordinateY must be
    entered explicitly.

Usage:
    python3 warehouse_edit_server.py
    python3 warehouse_edit_server.py --input ../data/R3_DF.json --port 8004
    python3 warehouse_edit_server.py --force   # skip the main.py-running check
"""
import argparse
import html
import json
import os
import socket
import sys
from typing import Optional

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def port_is_open(host, port, timeout=0.3):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Data access - no in-memory cache: load fresh, mutate, save, every request.
# ---------------------------------------------------------------------------

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_data(path, data):
    # Write to a temp file and rename over the original so a crash mid-write
    # can never leave R3_DF.json half-written.
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def get_warehouse(data):
    return data["Warehouses"][0]


def find_rack(data, rack_id):
    for su in get_warehouse(data)["StorageUnits"]:
        if su["Id"] == rack_id:
            return su
    return None


def find_shelf(data, shelf_id):
    for su in get_warehouse(data)["StorageUnits"]:
        for shelf in (su.get("ChildUnitsType") or []):
            if shelf["Id"] == shelf_id:
                return su, shelf
    return None, None


def shelf_item_count(data, shelf):
    vsu_ids = {vsu["Id"] for vsu in (shelf.get("VirtualSuDimensions") or [])}
    if not vsu_ids:
        return 0
    return sum(
        1 for p in data.get("ItemPlacements", [])
        if p.get("VSURelation", {}).get("VSUnitId") in vsu_ids
    )


def next_free_id(data):
    ids = set()
    for su in get_warehouse(data)["StorageUnits"]:
        ids.add(su["Id"])
        for shelf in (su.get("ChildUnitsType") or []):
            ids.add(shelf["Id"])
            for vsu in (shelf.get("VirtualSuDimensions") or []):
                ids.add(vsu["Id"])
    return max(ids, default=0) + 1


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

CSS = """
body { font-family: -apple-system, Helvetica, Arial, sans-serif; background: #f4f5f7; color: #222; margin: 0; padding: 24px; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 24px 0 10px; }
a { color: #2563eb; text-decoration: none; }
.breadcrumb { margin-bottom: 16px; font-size: 13px; }
.warn { background: #fef3c7; border: 1px solid #f59e0b; border-radius: 6px; padding: 10px 14px; font-size: 13px; max-width: 700px; }
.notice { border-radius: 6px; padding: 8px 14px; font-size: 13px; margin-bottom: 14px; max-width: 700px; }
.notice.ok { background: #dcfce7; border: 1px solid #16a34a; }
.notice.err { background: #fee2e2; border: 1px solid #dc2626; }
ul.rack-list { list-style: none; padding: 0; }
ul.rack-list li { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; max-width: 400px; }

.shelf-edit-list { display: flex; flex-direction: column; gap: 8px; max-width: 800px; }
.shelf-edit-row { display: flex; align-items: center; gap: 14px; background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 10px 14px; flex-wrap: wrap; }
.shelf-edit-label { font-weight: 600; min-width: 60px; }
.shelf-edit-label .occ { font-weight: normal; color: #b45309; font-size: 12px; }
.inline-form, .add-form { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; font-size: 13px; }
.inline-form label, .add-form label { display: flex; flex-direction: column; gap: 2px; color: #555; }
.inline-form input, .add-form input { width: 90px; padding: 4px 6px; border: 1px solid #ccc; border-radius: 4px; }
.add-form input[name="text"] { width: 120px; }
button { cursor: pointer; border-radius: 6px; border: 1px solid #999; background: #fff; padding: 6px 12px; font-size: 13px; }
button:hover { background: #f0f0f0; }
button.danger { border-color: #dc2626; color: #dc2626; }
button:disabled { opacity: .4; cursor: not-allowed; }
.add-form { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 14px; max-width: 800px; }
.hint { font-size: 12px; color: #777; max-width: 700px; }
"""


def page(title, body):
    return f"""<!DOCTYPE html>
<html lang="cs">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""


def render_index(data, input_path):
    wh = get_warehouse(data)
    rows = []
    for su in wh["StorageUnits"]:
        if su.get("UnitType") != "Rack":
            continue
        n_shelves = len(su.get("ChildUnitsType") or [])
        rows.append(
            f'<li><a href="/rack/{su["Id"]}">{html.escape(su.get("Text", str(su["Id"])))}</a>'
            f' &middot; {n_shelves} polic</li>'
        )
    body = f"""
<h1>Editace layoutu skladu (OFFLINE)</h1>
<p class="warn">Tento nastroj pise primo do {html.escape(input_path)}.
Nepousteji soubezne s bezicim main.py - vitezi ten, kdo ulozi posledni.</p>
<ul class="rack-list">{"".join(rows)}</ul>
"""
    return page("Editace skladu", body)


def render_rack(data, rack, msg=None, error=None):
    shelves_html = []
    for shelf in (rack.get("ChildUnitsType") or []):
        dims = shelf.get("SuDimensions", {}) or {}
        count = shelf_item_count(data, shelf)
        occ_note = f' <span class="occ">obsazeno: {count} ks</span>' if count else ""
        disabled = "disabled" if count else ""
        title = 'title="Nejdrive vyskladnete obsah police"' if count else ""
        shelves_html.append(f"""
<div class="shelf-edit-row">
  <div class="shelf-edit-label">{html.escape(shelf.get("Text", str(shelf["Id"])))}{occ_note}</div>
  <form method="post" action="/shelf/{shelf['Id']}/update" class="inline-form">
    <label>Sirka<input type="number" step="0.01" name="width" value="{dims.get('Width', 0)}"></label>
    <label>Vyska<input type="number" step="0.01" name="height" value="{dims.get('Height', 0)}"></label>
    <label>Hloubka<input type="number" step="0.01" name="depth" value="{dims.get('Depth', 0)}"></label>
    <button type="submit">Ulozit</button>
  </form>
  <form method="post" action="/shelf/{shelf['Id']}/delete"
        onsubmit="return confirm('Opravdu smazat polici {html.escape(shelf.get('Text', ''))}?')">
    <button type="submit" class="danger" {disabled} {title}>Smazat</button>
  </form>
</div>""")

    existing = rack.get("ChildUnitsType") or []
    default_x = existing[0]["SuDimensions"].get("CoordinateX", 0.0) if existing else 0.0
    default_z = existing[0]["SuDimensions"].get("CoordinateZ", 0.0) if existing else 0.0

    notice = ""
    if msg:
        notice += f'<div class="notice ok">{html.escape(msg)}</div>'
    if error:
        notice += f'<div class="notice err">{html.escape(error)}</div>'

    body = f"""
<div class="breadcrumb"><a href="/">&laquo; Prehled racku</a></div>
<h1>{html.escape(rack.get("Text", str(rack["Id"])))}</h1>
{notice}
<div class="shelf-edit-list">{"".join(shelves_html) or "<p>Tento rack nema zadne police.</p>"}</div>

<h2>Pridat polici</h2>
<form method="post" action="/rack/{rack['Id']}/add-shelf" class="add-form">
  <label>Nazev<input type="text" name="text" required></label>
  <label>Sirka<input type="number" step="0.01" name="width" required></label>
  <label>Vyska<input type="number" step="0.01" name="height" required></label>
  <label>Hloubka<input type="number" step="0.01" name="depth" value="380"></label>
  <label>Y souradnice<input type="number" step="0.01" name="coordinate_y" required></label>
  <button type="submit">Pridat polici</button>
</form>
<p class="hint">X a Z souradnice se prevezmou automaticky ze stavajicich polic v tomto racku
(X={default_x}, Z={default_z}). Zmena sirky/vysky existujici police neposouva Y souradnice
ostatnich polic v racku - to je potreba doresit rucne.</p>
"""
    return page(f"Editace - {rack.get('Text', '')}", body)


def build_app(input_path):
    app = FastAPI(title="MedicPort Warehouse Editor (OFFLINE)", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    def index():
        data = load_data(input_path)
        return render_index(data, input_path)

    @app.get("/rack/{rack_id}", response_class=HTMLResponse)
    def rack_page(rack_id: int, msg: Optional[str] = None, error: Optional[str] = None):
        data = load_data(input_path)
        rack = find_rack(data, rack_id)
        if rack is None:
            return HTMLResponse("Rack nenalezen", status_code=404)
        return render_rack(data, rack, msg, error)

    @app.post("/shelf/{shelf_id}/update")
    def update_shelf(shelf_id: int, width: float = Form(...), height: float = Form(...), depth: float = Form(...)):
        data = load_data(input_path)
        su, shelf = find_shelf(data, shelf_id)
        if shelf is None:
            return RedirectResponse(url="/", status_code=303)
        dims = shelf.setdefault("SuDimensions", {})
        dims["Width"] = width
        dims["Height"] = height
        dims["Depth"] = depth
        save_data(input_path, data)
        return RedirectResponse(url=f"/rack/{su['Id']}?msg=Police+ulozena", status_code=303)

    @app.post("/shelf/{shelf_id}/delete")
    def delete_shelf(shelf_id: int):
        data = load_data(input_path)
        su, shelf = find_shelf(data, shelf_id)
        if shelf is None:
            return RedirectResponse(url="/", status_code=303)
        if shelf_item_count(data, shelf) > 0:
            return RedirectResponse(
                url=f"/rack/{su['Id']}?error=Police+je+obsazena+-+nejdrive+vyskladnete+obsah",
                status_code=303,
            )
        su["ChildUnitsType"] = [s for s in su["ChildUnitsType"] if s["Id"] != shelf_id]
        save_data(input_path, data)
        return RedirectResponse(url=f"/rack/{su['Id']}?msg=Police+smazana", status_code=303)

    @app.post("/rack/{rack_id}/add-shelf")
    def add_shelf(
        rack_id: int,
        text: str = Form(...),
        width: float = Form(...),
        height: float = Form(...),
        depth: float = Form(380.0),
        coordinate_y: float = Form(...),
    ):
        data = load_data(input_path)
        rack = find_rack(data, rack_id)
        if rack is None:
            return RedirectResponse(url="/", status_code=303)
        existing = rack.get("ChildUnitsType") or []
        coord_x = existing[0]["SuDimensions"].get("CoordinateX", 0.0) if existing else 0.0
        coord_z = existing[0]["SuDimensions"].get("CoordinateZ", 0.0) if existing else 0.0
        new_shelf = {
            "Id": next_free_id(data),
            "ChildUnitsType": None,
            "UnitType": "Shelf",
            "Text": text,
            "VirtualSuDimensions": [],
            "SuDimensions": {
                "Id": 0,
                "Width": width,
                "Height": height,
                "Depth": depth,
                "Weight": 40000,
                "CoordinateX": coord_x,
                "CoordinateY": coordinate_y,
                "CoordinateZ": coord_z,
            },
        }
        rack.setdefault("ChildUnitsType", []).append(new_shelf)
        save_data(input_path, data)
        return RedirectResponse(url=f"/rack/{rack_id}?msg=Police+pridana", status_code=303)

    return app


def main():
    default_input = os.path.join(SCRIPT_DIR, "..", "data", "R3_DF.json")

    parser = argparse.ArgumentParser(description="OFFLINE editor for shelves in the warehouse layout JSON.")
    parser.add_argument("--input", "-i", default=default_input, help="Path to inventory JSON (default: data/R3_DF.json)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8004)
    parser.add_argument("--force", action="store_true", help="Skip the check for a running main.py on port 8000")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        raise SystemExit(f"Input file not found: {input_path}")

    if not args.force and port_is_open("127.0.0.1", 8000):
        print("!" * 70)
        print("POZOR: na portu 8000 neco odpovida (pravdepodobne bezi main.py).")
        print(f"Tento editor pise primo do {input_path},")
        print("stejneho souboru, ktery main.py nacita pri startu a cely")
        print("prepisuje pri kazde skladove operaci. Soubezny beh muze vest")
        print("ke ztrate zmen (jednim, nebo druhym smerem).")
        print("Zastavte main.py, nebo spustte s --force pokud vite co delate.")
        print("!" * 70)
        sys.exit(1)

    app = build_app(input_path)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
