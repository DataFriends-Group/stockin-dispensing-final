#!/usr/bin/env python3
"""
On-demand HTML visualization of warehouse content.

Reads the MedicPort inventory JSON (Warehouses / ItemPlacements) and renders
a set of static, self-contained HTML pages:

  index.html          - all racks, with count of occupied shelves and total boxes
  rack_<id>.html       - one rack, elevation view of all its shelves
  shelf_<id>.html      - one shelf, top-down view of its slots (VSUs) and their
                         contents (barcode + dimensions)

Usage:
    python3 warehouse_view.py --input ../data/R3_DF.json --output ../reports/warehouse_view
    python3 warehouse_view.py --open      # also opens index.html in the browser
"""
import argparse
import html
import json
import os
import shutil
import sys
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime

SCALE = 0.5  # mm -> px for the shelf top-down drawing


@dataclass
class Item:
    barcode: str
    width: float
    height: float
    depth: float
    z_start: float
    z_end: float
    product_id: object
    batch: str
    expiration: str


@dataclass
class Vsu:
    id: int
    code: str
    width: float
    height: float
    depth: float
    coord_x: float
    items: list = field(default_factory=list)


@dataclass
class Shelf:
    id: int
    text: str
    width: float
    height: float
    depth: float
    coord_x: float
    coord_z: float
    rack_id: int
    rack_text: str
    vsus: list = field(default_factory=list)

    @property
    def item_count(self):
        return sum(len(v.items) for v in self.vsus)

    @property
    def has_slots(self):
        return len(self.vsus) > 0

    @property
    def occupied(self):
        return self.item_count > 0


@dataclass
class Rack:
    id: int
    text: str
    shelves: list = field(default_factory=list)

    @property
    def shelves_with_items(self):
        return sum(1 for s in self.shelves if s.occupied)

    @property
    def total_items(self):
        return sum(s.item_count for s in self.shelves)


def load_model(data):
    warehouse = data["Warehouses"][0]

    vsu_to_shelf_id = {}
    for su in warehouse["StorageUnits"]:
        for shelf_data in (su.get("ChildUnitsType") or []):
            for vsu_data in (shelf_data.get("VirtualSuDimensions") or []):
                vsu_to_shelf_id[vsu_data["Id"]] = shelf_data["Id"]

    items_by_vsu = {}
    for placement in data.get("ItemPlacements", []):
        vsu_id = placement["VSURelation"]["VSUnitId"]
        meta = placement.get("ItemMetadata", {})
        item = Item(
            barcode=meta.get("Barcode", ""),
            width=meta.get("Width", 0.0),
            height=meta.get("Height", 0.0),
            depth=meta.get("Depth", 0.0),
            z_start=placement.get("ZStart", 0.0),
            z_end=placement.get("ZEnd", 0.0),
            product_id=meta.get("ProductID"),
            batch=meta.get("Batch", ""),
            expiration=meta.get("Expiration", ""),
        )
        items_by_vsu.setdefault(vsu_id, []).append(item)

    racks = []
    for su in warehouse["StorageUnits"]:
        if su.get("UnitType") != "Rack":
            continue
        rack = Rack(id=su["Id"], text=su.get("Text", str(su["Id"])))
        for shelf_data in (su.get("ChildUnitsType") or []):
            dims = shelf_data.get("SuDimensions", {}) or {}
            shelf_coord_x = dims.get("CoordinateX", 0.0)
            shelf = Shelf(
                id=shelf_data["Id"],
                text=shelf_data.get("Text", str(shelf_data["Id"])),
                width=dims.get("Width", 0.0),
                height=dims.get("Height", 0.0),
                depth=dims.get("Depth", 0.0),
                coord_x=shelf_coord_x,
                coord_z=dims.get("CoordinateZ", 0.0),
                rack_id=rack.id,
                rack_text=rack.text,
            )
            for vsu_data in (shelf_data.get("VirtualSuDimensions") or []):
                vsu = Vsu(
                    id=vsu_data["Id"],
                    code=vsu_data.get("Code", str(vsu_data["Id"])),
                    width=vsu_data.get("Width", 0.0),
                    height=vsu_data.get("Height", 0.0),
                    depth=vsu_data.get("Depth", 0.0),
                    coord_x=vsu_data.get("CoordinateX", 0.0) - shelf_coord_x,
                    items=sorted(
                        items_by_vsu.get(vsu_data["Id"], []),
                        key=lambda it: min(it.z_start, it.z_end),
                    ),
                )
                shelf.vsus.append(vsu)
            rack.shelves.append(shelf)
        racks.append(rack)
    return warehouse, racks


CSS = """
body { font-family: -apple-system, Helvetica, Arial, sans-serif; background: #f4f5f7; color: #222; margin: 0; padding: 24px; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 16px; color: #555; font-weight: normal; }
a { color: inherit; text-decoration: none; }
.breadcrumb { margin-bottom: 16px; font-size: 13px; color: #666; }
.breadcrumb a { color: #2563eb; text-decoration: underline; }
.legend { font-size: 12px; color: #555; margin-bottom: 16px; display: flex; gap: 16px; flex-wrap: wrap; }
.legend span.swatch { display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }

.floor-plan { display: flex; gap: 20px; align-items: stretch; }
.floor-col { display: flex; flex-direction: column; gap: 14px; flex: 1; min-width: 220px; }
.floor-divider { width: 36px; flex: 0 0 36px; border-radius: 6px; background: repeating-linear-gradient(0deg, #d4d4d8, #d4d4d8 10px, #e9e9ec 10px, #e9e9ec 20px); display: flex; align-items: center; justify-content: center; }
.floor-divider span { writing-mode: vertical-rl; font-size: 12px; letter-spacing: 3px; color: #777; }
.rack-card { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 14px 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); transition: box-shadow .15s, transform .15s; }
.rack-card:hover { box-shadow: 0 4px 10px rgba(0,0,0,0.12); transform: translateY(-1px); }
.rack-card .name { font-size: 17px; font-weight: 600; margin-bottom: 6px; }
.rack-card .stat { font-size: 13px; color: #444; margin: 2px 0; }
.bar { background: #e5e7eb; border-radius: 4px; height: 8px; margin-top: 8px; overflow: hidden; }
.bar-fill { background: #16a34a; height: 100%; }

.shelf-stack { display: flex; flex-direction: column; gap: 6px; max-width: 720px; }
.shelf-bar { display: flex; align-items: center; justify-content: space-between; border-radius: 6px; padding: 10px 16px; font-size: 13px; border: 2px solid #ccc; }
.shelf-bar.occupied { background: #dcfce7; border-color: #16a34a; }
.shelf-bar.empty { background: #fff; border-style: dashed; border-color: #999; }
.shelf-bar.noslots { background: repeating-linear-gradient(45deg, #f1f1f1, #f1f1f1 8px, #e6e6e6 8px, #e6e6e6 16px); border-style: dotted; border-color: #bbb; color: #888; }
.shelf-bar .label { font-weight: 600; }
.shelf-bar .height { color: #666; font-variant-numeric: tabular-nums; }
.shelf-bar .count { font-variant-numeric: tabular-nums; }

.shelf-meta { font-size: 13px; color: #444; margin-bottom: 16px; }
.top-view { position: relative; background: #fafafa; border: 2px solid #333; box-sizing: content-box; }
.vsu-box { position: absolute; top: 0; bottom: 0; border-left: 1px solid #999; border-right: 1px solid #999; box-sizing: border-box; overflow: hidden; }
.vsu-box.empty { background: repeating-linear-gradient(45deg, #fff, #fff 6px, #f0f0f0 6px, #f0f0f0 12px); }
.vsu-code { position: absolute; top: 2px; left: 2px; font-size: 10px; color: #666; background: rgba(255,255,255,0.7); padding: 0 2px; z-index: 5; }
.item-box { position: absolute; left: 1px; right: 1px; background: #93c5fd; border: 1px solid #2563eb; border-radius: 2px; font-size: 9px; line-height: 1.15; overflow: hidden; padding: 1px 2px; box-sizing: border-box; color: #1e3a8a; }
.item-box:nth-child(odd) { background: #bfdbfe; }

table.contents { border-collapse: collapse; margin-top: 20px; font-size: 13px; }
table.contents th, table.contents td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; }
table.contents th { background: #f0f0f0; }
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


def rack_card(rack):
    total_shelves = len(rack.shelves)
    occ = rack.shelves_with_items
    pct = round(100 * occ / total_shelves) if total_shelves else 0
    return f"""
<a class="rack-card" href="rack_{rack.id}.html">
  <div class="name">{html.escape(rack.text)}</div>
  <div class="stat">Police s obsahem: {occ} / {total_shelves}</div>
  <div class="stat">Krabiček celkem: {rack.total_items}</div>
  <div class="bar"><div class="bar-fill" style="width:{pct}%"></div></div>
</a>"""


def render_index(warehouse, racks, generated_at):
    # Physical floor layout: a rack's first shelf CoordinateZ sign says which
    # side of the aisle it stands on; CoordinateX is its position along the
    # aisle. Racks are grouped into two facing columns with a divider (the
    # aisle) between them, each ordered by X so facing racks line up.
    with_coords = [r for r in racks if r.shelves]
    without_coords = [r for r in racks if not r.shelves]

    neg_side = sorted(
        (r for r in with_coords if r.shelves[0].coord_z < 0),
        key=lambda r: r.shelves[0].coord_x,
        reverse=True,
    )
    pos_side = sorted(
        (r for r in with_coords if r.shelves[0].coord_z >= 0),
        key=lambda r: r.shelves[0].coord_x,
        reverse=True,
    )

    neg_cards = "".join(rack_card(r) for r in neg_side)
    pos_cards = "".join(rack_card(r) for r in pos_side)
    extra_cards = "".join(rack_card(r) for r in without_coords)

    extra_html = (
        f'<div class="floor-col">{extra_cards}</div>' if extra_cards else ""
    )

    body = f"""
<h1>{html.escape(warehouse.get("Name", "Warehouse"))}</h1>
<h2>{html.escape(warehouse.get("Note", ""))} &middot; vygenerováno {generated_at}</h2>
<div class="floor-plan">
  <div class="floor-col">{neg_cards}</div>
  <div class="floor-divider"><span>ulička</span></div>
  <div class="floor-col">{pos_cards}</div>
  {extra_html}
</div>
"""
    return page("Sklad - přehled racků", body)


def render_rack(rack):
    bars = []
    for shelf in rack.shelves:
        if not shelf.has_slots:
            cls, label = "noslots", "bez definovaných pozic"
        elif shelf.occupied:
            cls, label = "occupied", f"{shelf.item_count} ks"
        else:
            cls, label = "empty", "prázdná"
        bars.append(f"""
<a class="shelf-bar {cls}" href="shelf_{shelf.id}.html">
  <span class="label">{html.escape(shelf.text)}</span>
  <span class="height">výška {shelf.height:.0f} mm</span>
  <span class="count">{label}</span>
</a>""")
    body = f"""
<div class="breadcrumb"><a href="index.html">&laquo; Přehled racků</a></div>
<h1>{html.escape(rack.text)}</h1>
<h2>{len(rack.shelves)} polic &middot; {rack.shelves_with_items} s obsahem &middot; {rack.total_items} krabiček celkem</h2>
<div class="legend">
  <span><span class="swatch" style="background:#dcfce7;border:1px solid #16a34a"></span>obsazená police</span>
  <span><span class="swatch" style="background:#fff;border:2px dashed #999"></span>prázdná police</span>
  <span><span class="swatch" style="background:#e6e6e6;border:1px dotted #bbb"></span>bez definovaných pozic</span>
</div>
<div class="shelf-stack">{"".join(bars)}</div>
"""
    return page(f"Rack {rack.text}", body)


def render_shelf(shelf):
    breadcrumb = (
        f'<div class="breadcrumb"><a href="index.html">Přehled racků</a> &raquo; '
        f'<a href="rack_{shelf.rack_id}.html">{html.escape(shelf.rack_text)}</a> &raquo; '
        f'{html.escape(shelf.text)}</div>'
    )
    meta = (
        f'<div class="shelf-meta">Rozměry police: {shelf.width:.0f} &times; '
        f'{shelf.height:.0f} &times; {shelf.depth:.0f} mm (Š&times;V&times;H) &middot; '
        f'{len(shelf.vsus)} pozic &middot; {shelf.item_count} krabiček</div>'
    )

    if not shelf.has_slots:
        body = f"""
{breadcrumb}
<h1>Police {html.escape(shelf.text)}</h1>
{meta}
<p>Tato police nemá definované žádné pozice (VSU) - nikdy do ní nebyla uložena krabička.</p>
"""
        return page(f"Police {shelf.text}", body)

    view_w = shelf.width * SCALE
    view_h = shelf.depth * SCALE

    vsu_divs = []
    rows = []
    for vsu in shelf.vsus:
        left = vsu.coord_x * SCALE
        width = vsu.width * SCALE
        empty_cls = " empty" if not vsu.items else ""
        inner = [f'<span class="vsu-code">{html.escape(vsu.code)}</span>']
        # ZStart/ZEnd are machine depth coordinates (sign/order vary by rack side),
        # so items are placed relative to the front-most item in this VSU rather
        # than taken as absolute offsets from the shelf front.
        vsu_min_z = min((min(it.z_start, it.z_end) for it in vsu.items), default=0.0)
        for item in vsu.items:
            lo = min(item.z_start, item.z_end)
            hi = max(item.z_start, item.z_end)
            top = (lo - vsu_min_z) * SCALE
            height = max(2, (hi - lo) * SCALE)
            inner.append(
                f'<div class="item-box" style="top:{top:.1f}px;height:{height:.1f}px" '
                f'title="Barcode {html.escape(item.barcode)}, {item.width:.0f}x{item.height:.0f}x{item.depth:.0f} mm">'
                f'{html.escape(item.barcode)}<br>{item.width:.0f}&times;{item.height:.0f}&times;{item.depth:.0f}</div>'
            )
            rows.append(
                f"<tr><td>{html.escape(vsu.code)}</td><td>{html.escape(item.barcode)}</td>"
                f"<td>{item.width:.0f} &times; {item.height:.0f} &times; {item.depth:.0f} mm</td>"
                f"<td>{html.escape(str(item.product_id))}</td>"
                f"<td>{html.escape(item.batch)}</td>"
                f"<td>{html.escape(item.expiration)}</td></tr>"
            )
        if not vsu.items:
            rows.append(
                f"<tr><td>{html.escape(vsu.code)}</td><td colspan='5'><em>prázdná pozice</em></td></tr>"
            )
        vsu_divs.append(
            f'<div class="vsu-box{empty_cls}" style="left:{left:.1f}px;width:{width:.1f}px">'
            f'{"".join(inner)}</div>'
        )

    table = f"""
<table class="contents">
<tr><th>Pozice</th><th>Barcode (BC)</th><th>Rozměry (Š&times;V&times;H)</th><th>ProductID</th><th>Batch</th><th>Expirace</th></tr>
{"".join(rows)}
</table>"""

    body = f"""
{breadcrumb}
<h1>Police {html.escape(shelf.text)}</h1>
{meta}
<div class="top-view" style="width:{view_w:.1f}px;height:{view_h:.1f}px">{"".join(vsu_divs)}</div>
{table}
"""
    return page(f"Police {shelf.text}", body)


def generate(input_path, output_dir):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    warehouse, racks = load_model(data)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(os.path.join(output_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(render_index(warehouse, racks, generated_at))

    for rack in racks:
        with open(os.path.join(output_dir, f"rack_{rack.id}.html"), "w", encoding="utf-8") as f:
            f.write(render_rack(rack))
        for shelf in rack.shelves:
            with open(os.path.join(output_dir, f"shelf_{shelf.id}.html"), "w", encoding="utf-8") as f:
                f.write(render_shelf(shelf))

    return os.path.join(output_dir, "index.html")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(script_dir, "..", "data", "R3_DF.json")
    default_output = os.path.join(script_dir, "..", "reports", "warehouse_view")

    parser = argparse.ArgumentParser(description="Render warehouse content as static HTML pages.")
    parser.add_argument("--input", "-i", default=default_input, help="Path to inventory JSON (default: data/R3_DF.json)")
    parser.add_argument("--output", "-o", default=default_output, help="Output directory (will be recreated)")
    parser.add_argument("--open", action="store_true", help="Open index.html in the default browser when done")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)

    if not os.path.isfile(input_path):
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    index_path = generate(input_path, output_dir)
    print(f"Generated: {index_path}")

    if args.open:
        webbrowser.open(f"file://{index_path}")


if __name__ == "__main__":
    main()
