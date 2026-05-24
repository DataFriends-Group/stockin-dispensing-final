"""
OPTIMIZATION SERVER - Port 8002 - MedicPort Warehouse System

Nightly warehouse reorganization:
- FEFO Compliance: earliest expiry at front (stock_index 0)
- Demand-based priority: high-dispense products optimized first
- Shelf height optimization: move products to smallest suitable shelf
- VSU consolidation: delete empty VSUs, close gaps, recalculate X-coordinates
- Zone-based robot coordination: Zone L (R1) / Buffer / Zone R (R2)

Port 8002 Endpoints:
- DELETE /vsu/empty                    -> Clear empty VSUs + consolidate gaps
- POST /ml/demandweight                -> Update weights from dispense logs
- GET /products/weights                -> List products by weight (high->low)
- POST /reorganise/suggest             -> Auto-suggest highest priority product
- GET /reorganise/suggest/{product_id} -> Suggest for specific product
- POST /reorganise/complete            -> Execute reorganization task
- POST /reorganise/fail                -> Report failure + rollback
- GET /inventory/expiry-report         -> Products approaching expiry
- GET /inventory/dead-stock            -> Slow-moving items
- GET /health                          -> Server health check
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import json
import os

# Import shared state from main server (same process = same memory)
from main import (
    items,
    robots,
    virtual_units,
    shelves,
    racks,
    OUTPUT_POSITIONS,
    save_warehouse_state,
    save_robots_to_file,
    INPUT_POSITION,
)


optimization_app = FastAPI(
    title="MedicPort Optimization Server",
    description="Nightly warehouse reorganization on port 8002",
    version="1.0.0"
)

optimization_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@optimization_app.get("/vsu/empty", tags=["VSU Management"])
async def get_empty_vsus():
    """List all empty VSUs currently in the warehouse."""
    empty = []
    for vsu_id, vsu in virtual_units.items():
        has_items = any(item.vsu_id == vsu_id for item in items.values())
        if (not vsu.items or len(vsu.items) == 0) and not has_items:
            shelf = shelves.get(vsu.shelf_id)
            empty.append({
                "vsu_id": vsu_id,
                "vsu_code": vsu.code,
                "shelf": shelf.name if shelf else None,
                "position": {"x": vsu.position.x, "y": vsu.position.y, "z": vsu.position.z},
                "dimensions": {"width": vsu.dimensions.width, "height": vsu.dimensions.height, "depth": vsu.dimensions.depth},
            })
    return {"count": len(empty), "empty_vsus": empty}


@optimization_app.get("/shelves/usable-gaps", tags=["VSU Management"])
async def get_usable_gaps(min_width: float = 150):
    """List all gaps wider than min_width (default 150mm), grouped by rack/shelf.

    Returns format:
      Rack 1 / Shelf 101 : x=200-400, x=500-800
    """
    report = []
    for shelf in shelves.values():
        gaps = find_shelf_gaps(shelf, virtual_units)
        big_gaps = [g for g in gaps if g["width"] >= min_width]
        if not big_gaps:
            continue
        rack = racks.get(shelf.rack_id)
        report.append({
            "rack": rack.name if rack else None,
            "shelf": shelf.name,
            "shelf_x_range": [shelf.position.x, shelf.position.x + shelf.dimensions.width],
            "gaps": [
                {
                    "x_start": g["start"],
                    "x_end": g["end"],
                    "width": g["width"],
                    "kind": "internal" if g["left_vsu_id"] and g["right_vsu_id"] else "edge",
                    "left_vsu_id": g["left_vsu_id"],
                    "right_vsu_id": g["right_vsu_id"],
                }
                for g in big_gaps
            ],
            "summary": ", ".join(f"x={int(g['start'])}-{int(g['end'])}" for g in big_gaps),
        })
    total_gaps = sum(len(s["gaps"]) for s in report)
    total_width = sum(g["width"] for s in report for g in s["gaps"])
    return {
        "min_width_threshold": min_width,
        "shelves_with_usable_gaps": len(report),
        "total_gaps": total_gaps,
        "total_usable_width_mm": total_width,
        "shelves": report,
    }


@optimization_app.get("/shelves/fragmentation", tags=["VSU Management"])
async def get_shelf_fragmentation():
    """List shelves with internal gaps (free space between VSUs). Read-only diagnostic."""
    report = []
    for shelf in shelves.values():
        gaps = find_shelf_gaps(shelf, virtual_units)
        internal = [g for g in gaps if g["left_vsu_id"] and g["right_vsu_id"]]
        edge = [g for g in gaps if not (g["left_vsu_id"] and g["right_vsu_id"])]
        if not internal and not edge:
            continue
        report.append({
            "shelf_id": shelf.id,
            "shelf_name": shelf.name,
            "shelf_x_range": [shelf.position.x, shelf.position.x + shelf.dimensions.width],
            "internal_gaps": internal,
            "edge_gaps": edge,
            "has_internal_fragmentation": len(internal) > 0,
        })
    return {"shelves_with_gaps": len(report), "shelves": report}


class ConsolidationStepCompleteRequest(BaseModel):
    step: Optional[int] = None  # echoed for verification; server uses its own counter


class ConsolidationFailRequest(BaseModel):
    reason: Optional[str] = "unspecified"
    failed_at_step: Optional[int] = None


@optimization_app.post("/vsu/consolidate", tags=["VSU Management"])
async def consolidate_vsus():
    """Plan consolidation per zone (left/right). Each zone's tasks can run in parallel.
    Tasks in left zone are assigned to R1, right zone to R2. Buffer/straddle shelves are skipped.
    Refuses to plan if empty VSUs exist — caller must run DELETE /vsu/empty first so the
    plan is computed on a clean layout.
    """
    empty_ids = [
        vid for vid, vsu in virtual_units.items()
        if not vsu.items and not any(it.vsu_id == vid for it in items.values())
    ]
    if empty_ids:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "empty_vsus_present",
                "message": f"{len(empty_ids)} empty VSU(s) found. Call DELETE /vsu/empty first to remove them, then try again.",
                "empty_vsu_ids": empty_ids,
            },
        )

    zb = _compute_zone_boundaries()

    zone_state = {
        "left":   {"shelves_inspected": 0, "shelves_skipped": 0, "tasks": [], "robot_id": ZONE_LEFT_ROBOT,
                   "boundary_x": [zb["left_zone"][0], zb["left_zone"][1]]},
        "right":  {"shelves_inspected": 0, "shelves_skipped": 0, "tasks": [], "robot_id": ZONE_RIGHT_ROBOT,
                   "boundary_x": [zb["right_zone"][0], zb["right_zone"][1]]},
        "buffer": {"shelves_inspected": 0, "boundary_x": [zb["buffer"][0], zb["buffer"][1]]},
        "straddle_warnings": [],
    }

    # Detect single-zone fallback: if no shelf fits cleanly in either zone, treat everything as left
    cleanly_zoned_count = sum(1 for sh in shelves.values() if _zone_for_shelf(sh, zb) in ("left", "right"))
    single_zone_mode = cleanly_zoned_count == 0 and shelves

    for shelf in shelves.values():
        zone = _zone_for_shelf(shelf, zb)
        if single_zone_mode:
            zone = "left"
        if zone == "buffer":
            zone_state["buffer"]["shelves_inspected"] += 1
            continue
        if zone == "straddle":
            zone_state["straddle_warnings"].append({"shelf_id": shelf.id, "shelf_name": shelf.name})
            continue
        if zone not in ("left", "right"):
            continue
        zs = zone_state[zone]
        zs["shelves_inspected"] += 1
        plan = plan_shelf_consolidation(shelf, virtual_units)
        if not plan:
            for g in find_shelf_gaps(shelf, virtual_units):
                if g["left_vsu_id"] and g["right_vsu_id"]:
                    zs["shelves_skipped"] += 1
                    break
            continue
        for shift in plan:
            # Skip empty-VSU shifts — those should be removed via DELETE /vsu/empty first
            if not shift.get("item_ids"):
                continue
            task = _create_consolidation_task(shift, zone=zone)
            if task is None:
                continue
            if single_zone_mode:
                task["zone"] = None  # disables temp-finder zone restriction
            consolidation_tasks[task["task_id"]] = task
            zs["tasks"].append({
                "task_id": task["task_id"],
                "source_vsu": task["source_vsu_code"],
                "source_old_x": task["source_old_x"],
                "target_x": task["target_x"],
                "items_count": len(task["items_in_order"]),
                "total_steps": task["total_steps"],
                "shelf": shelves[task["source_shelf_id"]].name if task["source_shelf_id"] in shelves else None,
                "robot_id": task["robot_id"],
                "next_step_endpoint": f"GET /consolidation/{task['task_id']}/next-step",
            })

    _persist_consolidation_tasks()

    total_tasks = len(zone_state["left"]["tasks"]) + len(zone_state["right"]["tasks"])
    return {
        "status": "consolidation_planned" if total_tasks else "nothing_to_consolidate",
        "warehouse_x_extent": [zb["min_x"], zb["max_x"]],
        "zones": {
            "left": {
                "boundary_x": zone_state["left"]["boundary_x"],
                "robot_id": ZONE_LEFT_ROBOT,
                "shelves_inspected": zone_state["left"]["shelves_inspected"],
                "shelves_skipped_under_150mm_rule": zone_state["left"]["shelves_skipped"],
                "tasks_created": len(zone_state["left"]["tasks"]),
                "tasks": zone_state["left"]["tasks"],
            },
            "buffer": {
                "boundary_x": zone_state["buffer"]["boundary_x"],
                "shelves_inspected": zone_state["buffer"]["shelves_inspected"],
                "note": "buffer zone is reserved for cross-zone handoffs; no consolidation runs here",
            },
            "right": {
                "boundary_x": zone_state["right"]["boundary_x"],
                "robot_id": ZONE_RIGHT_ROBOT,
                "shelves_inspected": zone_state["right"]["shelves_inspected"],
                "shelves_skipped_under_150mm_rule": zone_state["right"]["shelves_skipped"],
                "tasks_created": len(zone_state["right"]["tasks"]),
                "tasks": zone_state["right"]["tasks"],
            },
        },
        "straddle_warnings": zone_state["straddle_warnings"],
        "total_tasks": total_tasks,
        "parallelizable": True,
    }


def _busy_dispensing_robots() -> List[str]:
    """Return list of robot IDs currently DISPENSING (set by dispense_server)."""
    return [rid for rid, r in robots.items() if getattr(r, "status", "") == "DISPENSING"]


@optimization_app.get("/consolidation/{task_id}/next-step", tags=["VSU Management"])
async def consolidation_next_step(task_id: str):
    if task_id not in consolidation_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    task = consolidation_tasks[task_id]
    if task["status"] != "in_progress":
        return {"task_id": task_id, "status": task["status"], "message": "Task already finished"}
    if task["current_step"] is None or task["current_step"] > task["total_steps"]:
        return {"task_id": task_id, "status": "ready_to_complete",
                "message": "All steps executed. Call POST /consolidation/{task_id}/complete"}

    # Dispense priority: if the robot assigned to this task is currently DISPENSING,
    # pause consolidation and tell the caller to retry after dispense completes.
    busy = _busy_dispensing_robots()
    if task.get("robot_id") in busy:
        raise HTTPException(
            status_code=423,
            detail={
                "status": "paused_for_dispense",
                "robot_id": task.get("robot_id"),
                "message": "Robot is currently handling a dispense. Retry /next-step after dispense completes.",
                "current_step": task["current_step"],
                "total_steps": task["total_steps"],
            },
        )

    return _resolve_next_step_address(task)


@optimization_app.post("/consolidation/{task_id}/step-complete", tags=["VSU Management"])
async def consolidation_step_complete(task_id: str, request: ConsolidationStepCompleteRequest = ConsolidationStepCompleteRequest()):
    if task_id not in consolidation_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    task = consolidation_tasks[task_id]
    if task["status"] != "in_progress":
        raise HTTPException(status_code=409, detail=f"Task is {task['status']}, not in_progress")
    if task["current_step"] is None or task["current_step"] > task["total_steps"]:
        raise HTTPException(status_code=409, detail="No pending step")
    if request.step is not None and request.step != task["current_step"]:
        raise HTTPException(status_code=409, detail=f"Step {request.step} doesn't match server step {task['current_step']}")
    _commit_step(task)
    _persist_consolidation_tasks()
    return {
        "status": "ok",
        "completed_step": task["step_log"][-1]["step"],
        "next_step": task["current_step"],
        "remaining_steps": (task["total_steps"] - (task["current_step"] - 1)) if task["current_step"] else 0,
    }


def _release_robot_to_idle(robot_id: Optional[str]) -> Optional[Dict]:
    """Reset robot status to IDLE at INPUT_POSITION and persist robot_post.json."""
    if not robot_id or robot_id not in robots:
        return None
    r = robots[robot_id]
    # Don't override if robot is currently dispensing (dispense flow owns the status until it /completes)
    if getattr(r, "status", "") == "DISPENSING":
        return {"robot_id": robot_id, "skipped": True, "reason": "robot busy with dispense"}
    r.position = INPUT_POSITION
    r.status = "IDLE"
    r.current_task_id = None
    save_robots_to_file()
    return {
        "robot_id": robot_id,
        "status": "IDLE",
        "position": {"x": INPUT_POSITION.x, "y": INPUT_POSITION.y, "z": INPUT_POSITION.z},
    }


def _any_other_consolidation_in_progress_for(robot_id: str, exclude_task_id: str) -> bool:
    """True if another consolidation task is still using this robot."""
    return any(
        t for tid, t in consolidation_tasks.items()
        if tid != exclude_task_id and t.get("robot_id") == robot_id and t.get("status") == "in_progress"
    )


@optimization_app.post("/consolidation/{task_id}/complete", tags=["VSU Management"])
async def consolidation_complete(task_id: str):
    if task_id not in consolidation_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    task = consolidation_tasks[task_id]
    if task["status"] != "in_progress":
        return {"task_id": task_id, "status": task["status"]}
    if task["current_step"] is not None and task["current_step"] <= task["total_steps"]:
        raise HTTPException(status_code=409, detail=f"Steps remaining: {task['total_steps'] - task['current_step'] + 1}")
    task["status"] = "completed"
    task["completed_at"] = datetime.now().isoformat()
    save_warehouse_state()

    # Release the robot to IDLE at INPUT_POSITION on every task completion
    robot_release = _release_robot_to_idle(task.get("robot_id"))

    _persist_consolidation_tasks()
    return {
        "task_id": task_id, "status": "completed",
        "summary": {
            "vsu_relocated": task["source_vsu_code"],
            "from_x": task["source_old_x"], "to_x": task["target_x"],
            "items_moved": len(task["items_in_order"]),
            "steps_executed": task["total_steps"],
        },
        "robot_released": robot_release,
    }


@optimization_app.post("/consolidation/{task_id}/fail", tags=["VSU Management"])
async def consolidation_fail(task_id: str, request: ConsolidationFailRequest = ConsolidationFailRequest()):
    if task_id not in consolidation_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    task = consolidation_tasks[task_id]
    if task["status"] != "in_progress":
        return {"task_id": task_id, "status": task["status"]}
    task["status"] = "failed"
    task["failed_at"] = datetime.now().isoformat()
    task["fail_reason"] = request.reason
    task["failed_at_step"] = request.failed_at_step
    save_warehouse_state()

    robot_release = _release_robot_to_idle(task.get("robot_id"))

    _persist_consolidation_tasks()
    return {
        "task_id": task_id, "status": "failed", "reason": request.reason,
        "committed_steps": len(task["step_log"]),
        "current_state": {
            iid: loc for iid, loc in task["item_locations"].items()
        },
        "robot_released": robot_release,
    }


@optimization_app.get("/consolidation/tasks", tags=["VSU Management"])
async def list_consolidation_tasks(status: Optional[str] = None):
    items_out = list(consolidation_tasks.values())
    if status:
        items_out = [t for t in items_out if t["status"] == status]
    return {"count": len(items_out), "tasks": [
        {k: v for k, v in t.items() if not k.startswith("_")} for t in items_out
    ]}


@optimization_app.get("/consolidation/next-task", tags=["VSU Management"])
async def consolidation_next_task(robot_id: str):
    """Return the next in-progress task assigned to this robot.
    R1 polls with robot_id=R1 → gets next left-zone task.
    R2 polls with robot_id=R2 → gets next right-zone task.
    Both robots can poll simultaneously and work in parallel.
    """
    candidates = [t for t in consolidation_tasks.values()
                  if t["status"] == "in_progress" and t.get("robot_id") == robot_id]
    if not candidates:
        return {"robot_id": robot_id, "status": "no_pending_tasks",
                "message": f"No in-progress consolidation tasks for {robot_id}"}
    candidates.sort(key=lambda t: (t.get("source_shelf_id", 0), t.get("source_old_x", 0)))
    t = candidates[0]
    return {
        "robot_id": robot_id,
        "task_id": t["task_id"],
        "source_vsu": t["source_vsu_code"],
        "source_old_x": t["source_old_x"],
        "target_x": t["target_x"],
        "items_count": len(t["items_in_order"]),
        "current_step": t["current_step"],
        "total_steps": t["total_steps"],
        "zone": t.get("zone"),
        "next_step_endpoint": f"GET /consolidation/{t['task_id']}/next-step",
        "queued_after_this": max(0, len(candidates) - 1),
    }


@optimization_app.get("/consolidation/progress", tags=["VSU Management"])
async def consolidation_progress():
    """Aggregate view of all consolidation tasks for operator monitoring."""
    by_status = {"in_progress": 0, "completed": 0, "failed": 0}
    by_zone = {"left": {"total": 0, "in_progress": 0, "completed": 0, "failed": 0},
               "right": {"total": 0, "in_progress": 0, "completed": 0, "failed": 0}}
    by_robot = {}
    in_progress_tasks = []
    total_steps_planned = 0
    total_steps_executed = 0

    for t in consolidation_tasks.values():
        s = t["status"]
        by_status[s] = by_status.get(s, 0) + 1
        z = t.get("zone")
        if z in by_zone:
            by_zone[z]["total"] += 1
            by_zone[z][s] = by_zone[z].get(s, 0) + 1
        rid = t.get("robot_id")
        if rid:
            by_robot.setdefault(rid, {"total": 0, "in_progress": 0, "completed": 0, "failed": 0})
            by_robot[rid]["total"] += 1
            by_robot[rid][s] = by_robot[rid].get(s, 0) + 1
        total_steps_planned += t.get("total_steps", 0)
        total_steps_executed += len(t.get("step_log", []))
        if s == "in_progress":
            in_progress_tasks.append({
                "task_id": t["task_id"],
                "source_vsu": t.get("source_vsu_code"),
                "robot_id": t.get("robot_id"),
                "zone": t.get("zone"),
                "current_step": t.get("current_step"),
                "total_steps": t.get("total_steps"),
                "percent_complete": round(100 * len(t.get("step_log", [])) / max(1, t.get("total_steps", 1)), 1),
            })

    busy_dispensing = _busy_dispensing_robots()
    return {
        "totals": {
            "tasks": len(consolidation_tasks),
            "in_progress": by_status.get("in_progress", 0),
            "completed": by_status.get("completed", 0),
            "failed": by_status.get("failed", 0),
            "steps_planned": total_steps_planned,
            "steps_executed": total_steps_executed,
            "percent_complete": round(100 * total_steps_executed / max(1, total_steps_planned), 1),
        },
        "by_zone": by_zone,
        "by_robot": by_robot,
        "robots_busy_dispensing": busy_dispensing,
        "consolidation_paused": bool(busy_dispensing),
        "active_tasks": in_progress_tasks,
    }


@optimization_app.delete("/vsu/empty", tags=["VSU Management"])
async def delete_empty_vsus():
    """Clear all empty VSUs from the warehouse. First step of every nightly session."""
    result = clear_empty_vsus(virtual_units, shelves, items)
    save_warehouse_state()
    return result


class DemandWeightRequest(BaseModel):
    days: int = 7
    min_dispenses: int = 1


@optimization_app.post("/ml/demandweight", tags=["Demand Weights"])
async def update_demand_weights(request: DemandWeightRequest = DemandWeightRequest()):
    """
    Analyze dispense logs from the last N days and update product demand weights.
    Applies Min-Max normalization (0.0-1.0). Saves results to weights.json.
    """
    result = calculate_demand_weights(days=request.days, min_dispenses=request.min_dispenses)
    if result.get("status") not in ("success", "no_data", "no_qualifying_products"):
        raise HTTPException(status_code=500, detail=result)
    return result


@optimization_app.get("/products/weights", tags=["Demand Weights"])
async def get_weights():
    """
    Return all products sorted by demand weight (highest to lowest).
    Each entry includes weight, total_dispenses, and dispenses_per_output.
    """
    products = get_products_by_weight()
    weights_meta = load_json_file(WEIGHTS_FILE, {})
    return {
        "count": len(products),
        "last_updated": weights_meta.get("last_updated"),
        "period": weights_meta.get("period"),
        "products": products
    }


@optimization_app.get("/inventory/expiry-report", tags=["Inventory Monitoring"])
async def expiry_report(threshold_days: int = 30):
    """
    Products with items approaching expiry within the given threshold.
    Returns list sorted by days remaining (most urgent first).
    """
    now = datetime.now()
    threshold_date = now + timedelta(days=threshold_days)
    results = []

    for item_id, item in items.items():
        exp = item.metadata.expiration
        if not exp or exp.year >= 2099:
            continue
        days_remaining = (exp - now).days
        if days_remaining > threshold_days:
            continue

        vsu = virtual_units.get(item.vsu_id)
        shelf = shelves.get(vsu.shelf_id) if vsu else None
        rack = racks.get(shelf.rack_id) if shelf else None
        results.append({
            "item_id": item_id,
            "product_id": item.metadata.product_id,
            "barcode": item.metadata.barcode,
            "batch": item.metadata.batch,
            "expiration": exp.isoformat(),
            "days_remaining": days_remaining,
            "location": {
                "rack": rack.name if rack else None,
                "shelf": shelf.name if shelf else None,
                "vsu_code": vsu.code if vsu else None,
                "stock_index": item.stock_index,
            }
        })

    results.sort(key=lambda x: x["days_remaining"])
    return {"threshold_days": threshold_days, "count": len(results), "items": results}


@optimization_app.get("/inventory/dead-stock", tags=["Inventory Monitoring"])
async def dead_stock(min_days_since_dispense: int = 30):
    """
    Items in stock with zero or very low dispense frequency.
    Cross-references inventory against dispense logs.
    """
    logs = load_json_file(DISPENSE_LOG_FILE, {"products": {}})
    products_log = logs.get("products", {})
    now = datetime.now()

    product_inventory = {}
    for item_id, item in items.items():
        pid = item.metadata.product_id
        if pid not in product_inventory:
            product_inventory[pid] = {
                "product_id": pid,
                "barcode": item.metadata.barcode,
                "quantity_in_stock": 0,
                "locations": [],
            }
        vsu = virtual_units.get(item.vsu_id)
        shelf = shelves.get(vsu.shelf_id) if vsu else None
        rack = racks.get(shelf.rack_id) if shelf else None
        product_inventory[pid]["quantity_in_stock"] += 1
        product_inventory[pid]["locations"].append({
            "batch": item.metadata.batch,
            "rack": rack.name if rack else None,
            "shelf": shelf.name if shelf else None,
            "vsu_code": vsu.code if vsu else None,
            "stock_index": item.stock_index,
        })

    results = []
    for pid, inv in product_inventory.items():
        log_entry = products_log.get(str(pid), {})
        last_dispensed_str = log_entry.get("last_dispensed")
        total_dispensed = log_entry.get("total_dispensed", 0)

        if last_dispensed_str:
            last_dispensed = datetime.fromisoformat(last_dispensed_str)
            days_since = (now - last_dispensed).days
        else:
            last_dispensed = None
            days_since = 9999

        if days_since < min_days_since_dispense:
            continue

        results.append({
            "product_id": pid,
            "barcode": inv["barcode"],
            "quantity_in_stock": inv["quantity_in_stock"],
            "total_dispensed": total_dispensed,
            "last_dispensed": last_dispensed.isoformat() if last_dispensed else "never_dispensed",
            "days_since_last_dispense": days_since if days_since < 9999 else None,
            "locations": inv["locations"],
        })

    results.sort(key=lambda x: x["days_since_last_dispense"] or 99999, reverse=True)
    return {"min_days_since_dispense": min_days_since_dispense, "count": len(results), "products": results}


@optimization_app.get("/health", tags=["System"])
async def health_check():
    vsus_total = len(virtual_units)
    vsus_empty = len([v for v in virtual_units.values() if not v.items])
    robots_idle = len([r for r in robots.values() if r.status == "IDLE"])
    return {
        "status": "healthy",
        "server": "optimization",
        "port": 8002,
        "items_in_inventory": len(items),
        "vsus_total": vsus_total,
        "vsus_empty": vsus_empty,
        "robots_available": robots_idle,
        "shelves_loaded": len(shelves),
        "timestamp": datetime.now().isoformat(),
    }


# ==================== CONSTANTS ====================

WEIGHTS_FILE = "data/weights.json"
DISPENSE_LOG_FILE = "data/dispense_logs.json"
OPTIMIZED_VSUS_FILE = "data/optimized_vsus.json"
OPTIMIZATION_TASKS_FILE = "data/optimization_tasks.json"
CONSOLIDATION_TASKS_FILE = "data/consolidation_tasks.json"

VSU_GAP = 20  # 20mm gap between VSUs
ITEM_GAP = 3  # 3mm gap between items in VSU
MAX_CONSOLIDATION_SHIFT = 3  # Max VSUs to shift during gap consolidation
MIN_CONSOLIDATION_EDGE_GAP = 150  # Skip consolidation if resulting edge free space < 150mm
OUTPUT_CONCENTRATION_BOOST = 0.2  # Max additive boost for products consistently using one output

# Zone split (X-axis): 40% left / 20% buffer / 60% right (40+20+40 = 100)
ZONE_LEFT_FRACTION = 0.40
ZONE_BUFFER_FRACTION = 0.20
ZONE_LEFT_ROBOT = "R1"
ZONE_RIGHT_ROBOT = "R2"


# ==================== MODELS ====================

class NightlyOptimizationRequest(BaseModel):
    dry_run: bool = True
    min_improvement_threshold: float = 100  # min mm distance improvement
    max_relocations: int = 50
    weight_threshold: float = 0.3  # products >= this are "fast movers"


class ExpiryOptimizationRequest(BaseModel):
    product_id: Optional[int] = None  # None = auto-select highest weight
    dry_run: bool = True


class RelocationPlan(BaseModel):
    item_id: int
    product_id: int
    barcode: str
    current_vsu_id: int
    current_vsu_code: str
    current_shelf_name: str
    current_rack_name: str
    current_distance_to_output: float
    target_vsu_id: int
    target_vsu_code: str
    target_shelf_name: str
    target_rack_name: str
    target_distance_to_output: float
    distance_improvement: float
    pick_position: Dict[str, float]
    place_position: Dict[str, float]
    stock_index: int
    strategy: str  # "swap", "empty_vsu", "expiry_sort"
    swap_with_item_id: Optional[int] = None
    expiry: Optional[str] = None  # ISO format expiry date


class ReorganizationInstruction(BaseModel):
    step: int
    action: str  # "pick_and_place", "relocate"
    reason: str  # "expiry_optimization", "obstruction_removal"
    robot_id: str
    item_id: int
    product_id: int
    barcode: Optional[str] = None
    expiry: Optional[str] = None
    pick: Dict[str, Any]
    place: Dict[str, Any]


# ==================== FILE I/O HELPERS ====================

def load_json_file(filepath: str, default: Dict = None) -> Dict:
    """Load JSON file with error handling"""
    if default is None:
        default = {}
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return default
    except Exception as e:
        print(f"[OPTIMIZATION] Error loading {filepath}: {e}")
        return default


def save_json_file(filepath: str, data: Dict):
    """Save JSON file with error handling"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        print(f"[OPTIMIZATION] Error saving {filepath}: {e}")
        raise


# ==================== DISTANCE HELPERS ====================

def _manhattan_distance_dict(pos1: Dict[str, float], pos2) -> float:
    """Manhattan distance between a dict position and a Position object"""
    return abs(pos1["x"] - pos2.x) + abs(pos1["y"] - pos2.y)


def _find_nearest_output_distance(vsu_position, output_positions) -> float:
    """Get manhattan distance from VSU to nearest output"""
    if not output_positions:
        return 9999.0
    min_dist = None
    for out_pos in output_positions:
        d = abs(vsu_position.x - out_pos.x) + abs(vsu_position.y - out_pos.y)
        if min_dist is None or d < min_dist:
            min_dist = d
    return min_dist


def _find_most_used_output(product_id: int, weights_data: Dict) -> Optional[int]:
    """Find the output that this product is most frequently dispensed through"""
    product_info = weights_data.get("products", {}).get(str(product_id), {})
    dispenses_per_output = product_info.get("dispenses_per_output", {})

    if not dispenses_per_output:
        return None

    # Find output with max dispenses
    max_output = max(dispenses_per_output.items(), key=lambda x: x[1])
    return int(max_output[0])


# ==================== ITEM HELPERS ====================

def is_item_accessible(item_id: int, vsu, items: Dict) -> bool:
    """
    Check if item can be picked without obstruction handling.
    True if item is at stock_index 0, sole item, or all items in front are same product.
    """
    item = items[item_id]
    for other_id in vsu.items:
        if other_id == item_id:
            continue
        other = items.get(other_id)
        if other is None:
            continue
        if other.stock_index < item.stock_index:
            if other.metadata.product_id != item.metadata.product_id:
                return False
    return True


def _item_fits_vsu(item, vsu) -> bool:
    """Check if item dimensions fit within VSU dimensions"""
    return (item.metadata.dimensions.width <= vsu.dimensions.width and
            item.metadata.dimensions.height <= vsu.dimensions.height and
            item.metadata.dimensions.depth <= vsu.dimensions.depth)


def get_blocking_items(item_id: int, vsu, items: Dict) -> List[int]:
    """Get list of items blocking access to target item (different products in front)"""
    item = items[item_id]
    blocking = []

    for other_id in vsu.items:
        if other_id == item_id:
            continue
        other = items.get(other_id)
        if other is None:
            continue
        if other.stock_index < item.stock_index:
            if other.metadata.product_id != item.metadata.product_id:
                blocking.append(other_id)

    # Sort by stock_index (front items first)
    blocking.sort(key=lambda x: items[x].stock_index)
    return blocking


# ==================== ML DEMAND WEIGHTS ====================

def calculate_demand_weights(days: int = 7, min_dispenses: int = 1) -> Dict:
    """
    Analyze dispense logs from the last N days and calculate demand weights.
    Uses Min-Max normalization to scale dispense counts to 0.0 - 1.0 range.

    Returns dict with status, products analyzed, weights updated, top products.
    """
    print(f"\n{'='*60}")
    print("ML DEMAND WEIGHT CALCULATION")
    print(f"{'='*60}")
    print(f"  Analyzing last {days} days of dispense data...")

    logs = load_json_file(DISPENSE_LOG_FILE, {"products": {}, "daily_stats": {}})
    products_data = logs.get("products", {})

    if not products_data:
        print("  No dispense data found")
        return {"status": "no_data", "products_analyzed": 0, "weights_updated": 0}

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    period_str = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

    # Count dispenses per product
    product_dispenses = {}
    dispenses_per_output = defaultdict(lambda: defaultdict(int))

    for product_id, product_info in products_data.items():
        total_in_period = 0
        barcode = product_info.get("barcode", "UNKNOWN")

        daily_history = product_info.get("daily_history", {})
        for date_str, count in daily_history.items():
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d")
                if start_date <= date <= end_date:
                    total_in_period += count
            except ValueError:
                continue

        if total_in_period == 0:
            total_in_period = product_info.get("total_dispensed", 0)

        # Track dispenses per output
        output_history = product_info.get("output_history", {})
        for output_id, output_count in output_history.items():
            dispenses_per_output[product_id][output_id] = output_count

        if total_in_period >= min_dispenses:
            product_dispenses[product_id] = {
                "barcode": barcode,
                "dispenses": total_in_period,
                "last_dispense": product_info.get("last_dispense"),
                "dispenses_per_output": dict(dispenses_per_output[product_id])
            }

    if not product_dispenses:
        return {"status": "no_qualifying_products", "products_analyzed": len(products_data), "weights_updated": 0}

    # Min-Max Normalization
    dispense_counts = [p["dispenses"] for p in product_dispenses.values()]
    min_d, max_d = min(dispense_counts), max(dispense_counts)

    weights_data = {
        "last_updated": datetime.now().isoformat(),
        "period_days": days,
        "period": period_str,
        "model": "min_max_normalization",
        "products": {}
    }

    for product_id, info in product_dispenses.items():
        weight = (info["dispenses"] - min_d) / (max_d - min_d) if max_d > min_d else 1.0
        weight = max(0.1, weight)

        # Output concentration boost: products that consistently go to one output
        # get up to OUTPUT_CONCENTRATION_BOOST added to their weight.
        # concentration = max_single_output_count / total_dispenses (0.0 - 1.0)
        # A product always sent to the same output scores full boost.
        # A product split evenly across outputs scores half boost or less.
        output_counts = info.get("dispenses_per_output", {})
        concentration = 0.0
        if output_counts and info["dispenses"] > 0:
            max_output = max(output_counts.values())
            concentration = max_output / info["dispenses"]
        boost = round(OUTPUT_CONCENTRATION_BOOST * concentration, 3)
        weight = round(min(1.0, weight + boost), 3)

        weights_data["products"][product_id] = {
            "barcode": info["barcode"],
            "weight": weight,
            "dispenses_in_period": info["dispenses"],
            "last_dispense": info["last_dispense"],
            "dispenses_per_output": output_counts,
            "output_concentration": round(concentration, 3),
            "output_boost_applied": boost
        }

    save_json_file(WEIGHTS_FILE, weights_data)

    sorted_products = sorted(weights_data["products"].items(), key=lambda x: x[1]["weight"], reverse=True)
    top_products = [{"product_id": int(pid), "barcode": info["barcode"],
                     "dispenses": info["dispenses_in_period"], "weight": info["weight"]}
                    for pid, info in sorted_products[:5]]

    print(f"  Weights saved. Top product: {top_products[0] if top_products else 'N/A'}")
    print(f"{'='*60}\n")

    return {
        "status": "success",
        "products_analyzed": len(products_data),
        "weights_updated": len(weights_data["products"]),
        "top_products": top_products,
        "model_used": "min_max_normalization",
        "period": period_str
    }


def get_products_by_weight() -> List[Dict]:
    """Get all products sorted by weight (highest to lowest)."""
    weights_data = load_json_file(WEIGHTS_FILE, {"products": {}})
    products = weights_data.get("products", {})

    sorted_products = sorted(products.items(), key=lambda x: x[1].get("weight", 0), reverse=True)

    return [
        {
            "product_id": int(pid),
            "barcode": info.get("barcode", "UNKNOWN"),
            "weight": info.get("weight", 0),
            "total_dispenses": info.get("dispenses_in_period", 0),
            "dispenses_per_output": info.get("dispenses_per_output", {}),
            "output_concentration": info.get("output_concentration", 0.0),
            "output_boost_applied": info.get("output_boost_applied", 0.0)
        }
        for pid, info in sorted_products
    ]


# ==================== VSU MANAGEMENT ====================

def _delete_empty_vsus(virtual_units: Dict, shelves: Dict, items: Dict) -> Tuple[List[int], set]:
    """Remove empty VSUs. Returns (deleted_ids, set_of_affected_shelf_ids)."""
    empty_ids = []
    for vsu_id, vsu in virtual_units.items():
        if vsu.items and len(vsu.items) > 0:
            continue
        if any(it.vsu_id == vsu_id for it in items.values()):
            continue
        empty_ids.append(vsu_id)

    affected_shelves = set()
    for vsu_id in empty_ids:
        vsu = virtual_units.get(vsu_id)
        if not vsu:
            continue
        shelf = shelves.get(vsu.shelf_id)
        if shelf and vsu_id in shelf.virtual_units:
            shelf.virtual_units.remove(vsu_id)
            affected_shelves.add(shelf.id)
        del virtual_units[vsu_id]
    return empty_ids, affected_shelves


def find_shelf_gaps(shelf, virtual_units: Dict) -> List[Dict]:
    """Return gaps on a shelf as list of {start, end, width, left_vsu_id, right_vsu_id}.
    Only counts gaps large enough to have held a VSU (>= VSU_GAP)."""
    shelf_start = shelf.position.x
    shelf_end = shelf_start + shelf.dimensions.width
    shelf_vsus = sorted(
        [virtual_units[vid] for vid in shelf.virtual_units if vid in virtual_units],
        key=lambda v: v.position.x,
    )

    gaps = []
    cursor = shelf_start
    for v in shelf_vsus:
        v_left = v.position.x
        if v_left - cursor > VSU_GAP + 0.5:
            gaps.append({
                "start": cursor, "end": v_left, "width": v_left - cursor,
                "left_vsu_id": None if cursor == shelf_start else shelf_vsus[shelf_vsus.index(v)-1].id,
                "right_vsu_id": v.id,
            })
        cursor = v_left + v.dimensions.width
    if shelf_end - cursor > 0.5:
        gaps.append({
            "start": cursor, "end": shelf_end, "width": shelf_end - cursor,
            "left_vsu_id": shelf_vsus[-1].id if shelf_vsus else None,
            "right_vsu_id": None,
        })
    return gaps


def plan_shelf_consolidation(shelf, virtual_units: Dict) -> List[Dict]:
    """Plan VSU shifts on a shelf without modifying state.
    Returns list of {'vsu_id', 'vsu_code', 'old_x', 'new_x', 'shelf_id', 'item_ids'}.

    Rules enforced:
    - Detects left-edge gaps and internal gaps
    - Max MAX_CONSOLIDATION_SHIFT VSUs shifted per gap
    - Skip if resulting gap < MIN_CONSOLIDATION_EDGE_GAP
    - Cap-hit doesn't chase residual fragmentation
    """
    shelf_start = shelf.position.x
    shelf_end = shelf_start + shelf.dimensions.width
    shelf_vsus = sorted(
        [virtual_units[vid] for vid in shelf.virtual_units if vid in virtual_units],
        key=lambda v: v.position.x,
    )
    if not shelf_vsus:
        return []

    # Working copy of x positions for planning (don't touch real positions)
    positions = {v.id: v.position.x for v in shelf_vsus}
    plan = []

    start_idx = 0
    while start_idx < len(shelf_vsus):
        shelf_vsus.sort(key=lambda v: positions[v.id])

        cursor = shelf_start if start_idx == 0 else (
            positions[shelf_vsus[start_idx - 1].id]
            + shelf_vsus[start_idx - 1].dimensions.width
            + VSU_GAP
        )
        gap_idx = None
        for i in range(start_idx, len(shelf_vsus)):
            if positions[shelf_vsus[i].id] > cursor + 0.5:
                gap_idx = i
                break
            cursor = positions[shelf_vsus[i].id] + shelf_vsus[i].dimensions.width + VSU_GAP
        if gap_idx is None:
            break

        expected_x = cursor
        shift_candidates = shelf_vsus[gap_idx : gap_idx + MAX_CONSOLIDATION_SHIFT]
        unshifted = shelf_vsus[gap_idx + MAX_CONSOLIDATION_SHIFT:]

        shift_end = expected_x
        for c in shift_candidates:
            shift_end += c.dimensions.width + VSU_GAP
        shift_end -= VSU_GAP

        cap_hit = len(shift_candidates) == MAX_CONSOLIDATION_SHIFT and bool(unshifted)
        if unshifted:
            resulting_gap = positions[unshifted[0].id] - shift_end - VSU_GAP
        else:
            resulting_gap = shelf_end - shift_end

        if resulting_gap < MIN_CONSOLIDATION_EDGE_GAP:
            break

        new_x = expected_x
        for c in shift_candidates:
            old_x = positions[c.id]
            if abs(old_x - new_x) > 0.5:
                plan.append({
                    "vsu_id": c.id,
                    "vsu_code": c.code,
                    "old_x": old_x,
                    "new_x": new_x,
                    "shelf_id": shelf.id,
                    "item_ids": list(c.items),
                })
            positions[c.id] = new_x
            new_x += c.dimensions.width + VSU_GAP

        if cap_hit:
            break
        start_idx = gap_idx + len(shift_candidates)

    return plan


def clear_empty_vsus(virtual_units: Dict, shelves: Dict, items: Dict) -> Dict:
    """Delete empty VSUs only. Consolidation is a separate operation."""
    print(f"\n{'='*60}")
    print("CLEARING EMPTY VSUs")
    print(f"{'='*60}")

    deleted_ids, affected_shelves = _delete_empty_vsus(virtual_units, shelves, items)
    if not deleted_ids:
        print("  No empty VSUs found")
        return {"status": "success", "vsus_removed": 0, "vsu_ids": []}

    print(f"  Removed {len(deleted_ids)} empty VSUs on {len(affected_shelves)} shelves")
    print(f"{'='*60}\n")

    return {
        "status": "success",
        "vsus_removed": len(deleted_ids),
        "vsu_ids": deleted_ids,
        "shelves_affected": list(affected_shelves),
    }


# ==================== CONSOLIDATION ENGINE ====================

consolidation_tasks: Dict[str, dict] = {}


def _compute_zone_boundaries() -> Dict:
    """Compute warehouse X extent and zone boundaries (40/20/40 split)."""
    if not shelves:
        return {"min_x": 0, "max_x": 0, "left_end": 0, "right_start": 0,
                "extent": 0, "left_zone": (0, 0), "buffer": (0, 0), "right_zone": (0, 0)}
    xs = []
    for sh in shelves.values():
        xs.append(sh.position.x)
        xs.append(sh.position.x + sh.dimensions.width)
    min_x = min(xs)
    max_x = max(xs)
    extent = max_x - min_x
    left_end = min_x + extent * ZONE_LEFT_FRACTION
    right_start = min_x + extent * (ZONE_LEFT_FRACTION + ZONE_BUFFER_FRACTION)
    return {
        "min_x": min_x, "max_x": max_x, "extent": extent,
        "left_end": left_end, "right_start": right_start,
        "left_zone": (min_x, left_end),
        "buffer": (left_end, right_start),
        "right_zone": (right_start, max_x),
    }


def _zone_for_shelf(shelf, zb: Dict) -> str:
    """Return 'left', 'right', 'buffer', or 'straddle'."""
    s_left = shelf.position.x
    s_right = s_left + shelf.dimensions.width
    if s_right <= zb["left_end"] + 0.5:
        return "left"
    if s_left >= zb["right_start"] - 0.5:
        return "right"
    if s_left >= zb["left_end"] - 0.5 and s_right <= zb["right_start"] + 0.5:
        return "buffer"
    return "straddle"


def _zone_for_vsu(vsu, zb: Dict) -> str:
    sh = shelves.get(vsu.shelf_id)
    return _zone_for_shelf(sh, zb) if sh else "unknown"


def _persist_consolidation_tasks():
    save_json_file(CONSOLIDATION_TASKS_FILE, {"tasks": list(consolidation_tasks.values())})


def _load_consolidation_tasks():
    global consolidation_tasks
    data = load_json_file(CONSOLIDATION_TASKS_FILE, {"tasks": []})
    consolidation_tasks = {t["task_id"]: t for t in data.get("tasks", [])}


def _is_negative_z_shelf(vsu) -> bool:
    return vsu.position.z < 0


def _back_wall_z(vsu) -> float:
    """Z coordinate of the shelf back wall as seen from the VSU's side."""
    return vsu.position.z - vsu.dimensions.depth if _is_negative_z_shelf(vsu) else vsu.position.z + vsu.dimensions.depth


def _z_for_back_position(vsu, item_depth: float) -> Tuple[float, float]:
    """z_start, z_end for an item flush at the back wall (no items behind)."""
    bw = _back_wall_z(vsu)
    if _is_negative_z_shelf(vsu):
        return (bw + item_depth, bw)
    else:
        return (bw - item_depth, bw)


def _z_for_front_position(vsu, frontmost_z_start: float, item_depth: float) -> Tuple[float, float]:
    """z_start, z_end for an item placed in front of the current frontmost item with ITEM_GAP."""
    if _is_negative_z_shelf(vsu):
        z_end = frontmost_z_start + ITEM_GAP
        z_start = z_end + item_depth
    else:
        z_end = frontmost_z_start - ITEM_GAP
        z_start = z_end - item_depth
    return (z_start, z_end)


def _items_sorted_by_idx(vsu) -> List[int]:
    """Item ids in this VSU sorted by stock_index ASC (frontmost first)."""
    return sorted([iid for iid in vsu.items if iid in items],
                  key=lambda iid: items[iid].stock_index)


def _frontmost_item_in_vsu(vsu):
    sorted_ids = _items_sorted_by_idx(vsu)
    return items[sorted_ids[0]] if sorted_ids else None


def _vsu_remaining_depth(vsu) -> float:
    """Remaining depth in VSU accounting for existing items + ITEM_GAP per item."""
    n = len(vsu.items)
    if n == 0:
        return vsu.dimensions.depth
    used = sum(items[iid].metadata.dimensions.depth for iid in vsu.items if iid in items)
    used += n * ITEM_GAP
    return vsu.dimensions.depth - used


def _can_fit_in_front(vsu, item_id: int) -> bool:
    """Check if item can be placed at the front (mixed-product or empty stack)."""
    item = items.get(item_id)
    if not item:
        return False
    if item.metadata.dimensions.height > vsu.dimensions.height:
        return False
    if item.metadata.dimensions.width > vsu.dimensions.width:
        return False
    if item.metadata.dimensions.depth > _vsu_remaining_depth(vsu):
        return False
    front = _frontmost_item_in_vsu(vsu)
    if front and item.metadata.dimensions.width > front.metadata.dimensions.width:
        return False
    return True


def _find_temp_spot_for_item(item_id: int, exclude_vsu_ids: set, restrict_to_zone: Optional[str] = None) -> Optional[Dict]:
    """Pick a temp location for an item.
    Priority: same-shelf > same-zone > mixed-product front > empty VSU.
    If restrict_to_zone is set ('left' / 'right'), only VSUs in that zone are considered.
    Returns dict with vsu_id, idx (after re-indexing), z_start, z_end.
    """
    item = items.get(item_id)
    if not item:
        return None

    zb = _compute_zone_boundaries() if restrict_to_zone else None
    candidates = []
    for vsu in virtual_units.values():
        if vsu.id in exclude_vsu_ids:
            continue
        if restrict_to_zone is not None:
            vsu_zone = _zone_for_vsu(vsu, zb)
            if vsu_zone != restrict_to_zone:
                continue
        if not _can_fit_in_front(vsu, item_id):
            continue
        if vsu.items:
            front = _frontmost_item_in_vsu(vsu)
            front_z = front.z_start if front and front.z_start is not None else _back_wall_z(vsu)
            z_start, z_end = _z_for_front_position(vsu, front_z, item.metadata.dimensions.depth)
            kind = "front_stack"
        else:
            z_start, z_end = _z_for_back_position(vsu, item.metadata.dimensions.depth)
            kind = "empty_vsu_back"
        candidates.append({
            "vsu_id": vsu.id,
            "z_start": z_start,
            "z_end": z_end,
            "kind": kind,
            "shelf_id": vsu.shelf_id,
            "items_count": len(vsu.items),
        })

    if not candidates:
        return None

    # Prefer same-shelf, then mixed-product over empty
    item_vsu = virtual_units.get(item.vsu_id) if item.vsu_id else None
    item_shelf = item_vsu.shelf_id if item_vsu else None
    candidates.sort(key=lambda c: (
        0 if c["shelf_id"] == item_shelf else 1,
        0 if c["kind"] == "front_stack" else 1,
    ))
    return candidates[0]


def _vsu_to_address_dict(vsu, override_x: Optional[float] = None) -> Dict:
    shelf = shelves.get(vsu.shelf_id)
    rack = racks.get(shelf.rack_id) if shelf else None
    x = override_x if override_x is not None else vsu.position.x
    return {
        "rack": rack.name if rack else None,
        "shelf": shelf.name if shelf else None,
        "vsu_code": vsu.code,
        "vsu_id": vsu.id,
        "coordinates": {"x": x, "y": vsu.position.y, "z": vsu.position.z},
    }


def _make_step_plan(item_ids_front_to_back: List[int]) -> List[Dict]:
    """Generate the abstract step list for a VSU shift with N items.
    Each step is a single 'move' (pick + place atomic). Total steps = 2N-1.
    """
    plan = []
    n = len(item_ids_front_to_back)
    if n == 0:
        return []
    # Forward: move first N-1 items to temp, last item directly to target
    for i in range(n - 1):
        plan.append({"action": "move", "item_id": item_ids_front_to_back[i], "phase": "source_to_temp"})
    plan.append({"action": "move", "item_id": item_ids_front_to_back[n - 1], "phase": "source_to_target"})
    # Reverse: bring temp items back to target in reverse order so FEFO is preserved
    for i in range(n - 2, -1, -1):
        plan.append({"action": "move", "item_id": item_ids_front_to_back[i], "phase": "temp_to_target"})
    return plan


def _create_consolidation_task(shift: Dict, zone: str) -> Optional[Dict]:
    """Build a task for one VSU shift. zone in {'left','right'}."""
    vsu = virtual_units.get(shift["vsu_id"])
    if not vsu:
        return None
    item_ids_sorted = _items_sorted_by_idx(vsu)
    steps = _make_step_plan(item_ids_sorted) if item_ids_sorted else []

    task_id = f"CONSOL-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{shift['vsu_id']}"
    robot_id = ZONE_LEFT_ROBOT if zone == "left" else ZONE_RIGHT_ROBOT

    item_locations = {}
    for iid in item_ids_sorted:
        it = items[iid]
        item_locations[iid] = {
            "loc": "source",
            "vsu_id": vsu.id,
            "idx": it.stock_index,
            "z_start": it.z_start,
            "z_end": it.z_end,
        }

    task = {
        "task_id": task_id,
        "status": "in_progress",
        "created_at": datetime.now().isoformat(),
        "zone": zone,
        "source_vsu_id": vsu.id,
        "source_vsu_code": vsu.code,
        "source_shelf_id": vsu.shelf_id,
        "source_old_x": shift["old_x"],
        "target_x": shift["new_x"],
        "items_in_order": item_ids_sorted,
        "step_plan": steps,
        "current_step": 1 if steps else None,
        "total_steps": len(steps),
        "robot_id": robot_id,
        "step_log": [],
        "item_locations": item_locations,
        "target_vsu_created": False,
    }
    return task


def _resolve_next_step_address(task: Dict) -> Dict:
    """Compute concrete address for the current step using live state."""
    step_idx = task["current_step"] - 1
    step = task["step_plan"][step_idx]
    action = step["action"]
    item_id = step["item_id"]
    item = items.get(item_id)
    if not item:
        raise HTTPException(status_code=500, detail=f"Item {item_id} not found")

    # Mark the robot as actively engaged with this consolidation step
    rid = task.get("robot_id")
    if rid and rid in robots:
        r = robots[rid]
        if getattr(r, "status", "") != "DISPENSING":
            r.status = "CONSOLIDATING"
            r.current_task_id = task["task_id"]
            save_robots_to_file()

    response = {
        "task_id": task["task_id"],
        "step": task["current_step"],
        "of_total": task["total_steps"],
        "action": action,
        "robot_id": task["robot_id"],
        "phase": step.get("phase"),
        "item": {
            "item_id": item_id,
            "guid": getattr(item.metadata, "guid", None),
            "barcode": item.metadata.barcode,
            "batch": item.metadata.batch,
            "expiration": item.metadata.expiration.isoformat() if item.metadata.expiration else None,
            "depth": item.metadata.dimensions.depth,
            "width": item.metadata.dimensions.width,
            "height": item.metadata.dimensions.height,
        },
    }

    if action == "move":
        # FROM: live address of the item (handles dispense-induced relocations)
        cur_vsu = virtual_units.get(item.vsu_id) if item.vsu_id is not None else None
        if not cur_vsu:
            raise HTTPException(status_code=500, detail=f"Item {item_id} has no current VSU")
        from_addr = _vsu_to_address_dict(cur_vsu)
        from_addr["stock_index"] = item.stock_index
        from_addr["z_start"] = item.z_start
        from_addr["z_end"] = item.z_end
        response["from"] = from_addr

        # TO: depends on phase
        phase = step.get("phase")
        if phase == "source_to_temp":
            exclude = {task["source_vsu_id"]}
            spot = _find_temp_spot_for_item(item_id, exclude, restrict_to_zone=task.get("zone"))
            if not spot:
                raise HTTPException(status_code=409, detail=f"No temp spot available in zone={task.get('zone')}")
            target_vsu = virtual_units[spot["vsu_id"]]
            to_addr = _vsu_to_address_dict(target_vsu)
            to_addr["stock_index"] = 0
            to_addr["z_start"] = spot["z_start"]
            to_addr["z_end"] = spot["z_end"]
            response["to"] = to_addr
            response["temp_reason"] = spot["kind"]
            task["_pending_step_to"] = {
                "vsu_id": spot["vsu_id"],
                "z_start": spot["z_start"],
                "z_end": spot["z_end"],
                "kind": "temp",
            }
        else:
            # source_to_target OR temp_to_target → place into the source VSU at new x
            source_vsu = virtual_units.get(task["source_vsu_id"])
            if not source_vsu:
                raise HTTPException(status_code=500, detail="Source VSU missing")
            already_final = [iid for iid, loc in task["item_locations"].items() if loc["loc"] == "final"]
            if not already_final:
                z_start, z_end = _z_for_back_position(source_vsu, item.metadata.dimensions.depth)
            else:
                front_iid = min(already_final, key=lambda i: items[i].stock_index)
                front_z_start = task["item_locations"][front_iid]["z_start"]
                z_start, z_end = _z_for_front_position(source_vsu, front_z_start, item.metadata.dimensions.depth)
            target_x = task["target_x"]
            to_addr = _vsu_to_address_dict(source_vsu, override_x=target_x)
            to_addr["stock_index"] = 0
            to_addr["z_start"] = z_start
            to_addr["z_end"] = z_end
            to_addr["target_action"] = ("create_or_reposition_vsu"
                                         if not task["target_vsu_created"] else "stack_in_target")
            response["to"] = to_addr
            task["_pending_step_to"] = {
                "vsu_id": task["source_vsu_id"],
                "z_start": z_start,
                "z_end": z_end,
                "kind": "final",
                "is_first_final": not task["target_vsu_created"],
            }

    return response


def _commit_step(task: Dict):
    """Apply the just-completed step's effect to the warehouse state."""
    step_idx = task["current_step"] - 1
    step = task["step_plan"][step_idx]
    action = step["action"]
    item_id = step["item_id"]
    item = items.get(item_id)

    if action == "move":
        pending = task.pop("_pending_step_to", None)
        if not pending:
            raise HTTPException(status_code=409, detail="No pending move address; call /next-step first")

        # 1. Pick: remove from current VSU + re-index siblings
        cur_vsu = virtual_units.get(item.vsu_id) if item.vsu_id is not None else None
        if cur_vsu and item_id in cur_vsu.items:
            cur_vsu.items.remove(item_id)
        if cur_vsu:
            remaining_sorted = sorted(
                [(iid, items[iid].stock_index) for iid in cur_vsu.items if iid in items],
                key=lambda x: x[1],
            )
            for new_idx, (iid, _) in enumerate(remaining_sorted):
                items[iid].stock_index = new_idx

        # 2. Place: insert into target VSU at idx=0 + bump siblings
        target_vsu = virtual_units[pending["vsu_id"]]
        if pending.get("kind") == "final" and pending.get("is_first_final"):
            target_vsu.position.x = task["target_x"]
            task["target_vsu_created"] = True
        for iid in target_vsu.items:
            if iid in items:
                items[iid].stock_index += 1
        target_vsu.items.append(item_id)
        item.vsu_id = target_vsu.id
        item.stock_index = 0
        item.z_start = pending["z_start"]
        item.z_end = pending["z_end"]
        task["item_locations"][item_id] = {
            "loc": "temp" if pending.get("kind") == "temp" else "final",
            "vsu_id": target_vsu.id, "idx": 0,
            "z_start": pending["z_start"], "z_end": pending["z_end"],
        }

    task["step_log"].append({"step": task["current_step"], "action": action, "item_id": item_id})
    task["current_step"] += 1
    if task["current_step"] > task["total_steps"]:
        task["current_step"] = None

    # Persist warehouse state after every committed step so file always matches reality
    save_warehouse_state()


# ==================== PRODUCT CLASSIFICATION ====================

def classify_products(
    items: Dict,
    virtual_units: Dict,
    shelves: Dict,
    racks: Dict,
    product_weights: Dict,
    output_positions: List,
    weight_threshold: float,
    default_weight: float = 0.1
) -> Tuple[List[Dict], List[Dict]]:
    """
    Classify all items into fast movers and slow movers.
    Returns (fast_movers, slow_movers) sorted by weight/distance.
    """
    fast_movers = []
    slow_movers = []

    for item_id, item in items.items():
        if item.vsu_id is None:
            continue
        vsu = virtual_units.get(item.vsu_id)
        if vsu is None:
            continue

        shelf = shelves.get(vsu.shelf_id)
        rack = racks.get(vsu.rack_id) if vsu.rack_id else None

        distance = _find_nearest_output_distance(vsu.position, output_positions)
        weight = product_weights.get(item.metadata.product_id, default_weight)

        info = {
            "item_id": item_id,
            "product_id": item.metadata.product_id,
            "barcode": item.metadata.barcode,
            "vsu_id": item.vsu_id,
            "vsu_code": vsu.code,
            "shelf_id": vsu.shelf_id,
            "shelf_name": shelf.name if shelf else f"shelf_{vsu.shelf_id}",
            "rack_name": rack.name if rack else f"rack_{vsu.rack_id}",
            "distance": distance,
            "weight": weight,
            "stock_index": item.stock_index,
            "expiry": item.metadata.expiration.isoformat() if item.metadata.expiration else None,
            "position": {"x": vsu.position.x, "y": vsu.position.y, "z": vsu.position.z}
        }

        if weight >= weight_threshold:
            fast_movers.append(info)
        else:
            slow_movers.append(info)

    fast_movers.sort(key=lambda x: (-x["weight"], -x["distance"]))
    slow_movers.sort(key=lambda x: x["distance"])

    return fast_movers, slow_movers


# ==================== EXPIRY OPTIMIZATION ====================

def find_vsus_for_product(
    product_id: int,
    items: Dict,
    virtual_units: Dict,
    shelves: Dict,
    exclude_optimized: bool = True
) -> List[Dict]:
    """
    Find all VSUs containing a specific product, sorted by proximity.
    Returns list of VSU info dicts with items and accessibility status.
    """
    optimized_vsus = set()
    if exclude_optimized:
        data = load_json_file(OPTIMIZED_VSUS_FILE, {"optimized_vsus": []})
        optimized_vsus = set(data.get("optimized_vsus", []))

    vsu_map = {}

    for item_id, item in items.items():
        if item.metadata.product_id != product_id or item.vsu_id is None:
            continue

        vsu = virtual_units.get(item.vsu_id)
        if vsu is None or (exclude_optimized and item.vsu_id in optimized_vsus):
            continue

        if item.vsu_id not in vsu_map:
            shelf = shelves.get(vsu.shelf_id)
            vsu_map[item.vsu_id] = {
                "vsu_id": item.vsu_id,
                "vsu_code": vsu.code,
                "shelf_id": vsu.shelf_id,
                "shelf_name": shelf.name if shelf else f"shelf_{vsu.shelf_id}",
                "rack_id": vsu.rack_id,
                "position": {"x": vsu.position.x, "y": vsu.position.y, "z": vsu.position.z},
                "items_in_vsu": [],
                "product_at_front": True
            }

        vsu_map[item.vsu_id]["items_in_vsu"].append({
            "item_id": item_id,
            "product_id": item.metadata.product_id,
            "barcode": item.metadata.barcode,
            "expiry": item.metadata.expiration.isoformat() if item.metadata.expiration else None,
            "stock_index": item.stock_index
        })

    # Check accessibility and sort items
    for vsu_info in vsu_map.values():
        vsu_info["items_in_vsu"].sort(key=lambda x: x["stock_index"])

        # Check if product is blocked
        vsu = virtual_units.get(vsu_info["vsu_id"])
        for item_data in vsu_info["items_in_vsu"]:
            if not is_item_accessible(item_data["item_id"], vsu, items):
                vsu_info["product_at_front"] = False
                break

    # Sort by proximity (same shelf first, then by x position)
    vsu_list = list(vsu_map.values())
    vsu_list.sort(key=lambda v: (v["shelf_id"], v["position"]["x"]))

    return vsu_list


def compare_items_by_expiry(item_a: Dict, item_b: Dict) -> Tuple[Dict, Dict]:
    """
    Compare two items by expiry date.
    Returns (earlier_expiry_item, later_expiry_item).
    Earlier expiry -> FRONT (stock_index 0), Later expiry -> BACK.
    """
    expiry_a = item_a.get("expiry")
    expiry_b = item_b.get("expiry")

    if expiry_a is None and expiry_b is None:
        return (item_a, item_b)
    if expiry_a is None:
        return (item_b, item_a)
    if expiry_b is None:
        return (item_a, item_b)

    if isinstance(expiry_a, str):
        expiry_a = datetime.fromisoformat(expiry_a.replace('Z', '+00:00'))
    if isinstance(expiry_b, str):
        expiry_b = datetime.fromisoformat(expiry_b.replace('Z', '+00:00'))

    return (item_a, item_b) if expiry_a <= expiry_b else (item_b, item_a)


def find_nearby_vsu_pair(vsu_list: List[Dict]) -> Optional[Tuple[Dict, Dict]]:
    """Find two nearby VSUs for comparison. Returns (vsu_a, vsu_b) or None."""
    return (vsu_list[0], vsu_list[1]) if len(vsu_list) >= 2 else None


def suggest_expiry_reorganization(
    product_id: int,
    items: Dict,
    virtual_units: Dict,
    shelves: Dict,
    robots: Dict
) -> Dict:
    """
    Suggest expiry-based reorganization for a product.
    Compares 2 nearby VSUs and generates instructions to sort by expiry.

    Returns task with instructions or "already_optimised" status.
    """
    print(f"\n{'='*60}")
    print(f"EXPIRY REORGANIZATION - Product {product_id}")
    print(f"{'='*60}")

    # Find VSUs for this product
    vsu_list = find_vsus_for_product(product_id, items, virtual_units, shelves)

    if len(vsu_list) < 2:
        print(f"  Product {product_id} has < 2 VSUs - already optimised or nothing to compare")
        return {
            "status": "already_optimised",
            "product_id": product_id,
            "message": "Product already optimised - no further reorganization needed"
        }

    # Check if all pairs compared
    if is_product_fully_optimized(product_id, vsu_list):
        return {
            "status": "already_optimised",
            "product_id": product_id,
            "message": "All VSU pairs have been compared"
        }

    # Find nearby pair to compare
    vsu_a, vsu_b = find_nearby_vsu_pair(vsu_list)
    print(f"  Comparing {vsu_a['vsu_code']} and {vsu_b['vsu_code']}")

    # Get front items from each VSU
    item_a = vsu_a["items_in_vsu"][0] if vsu_a["items_in_vsu"] else None
    item_b = vsu_b["items_in_vsu"][0] if vsu_b["items_in_vsu"] else None

    if not item_a or not item_b:
        return {"status": "error", "message": "VSUs have no items"}

    # Compare expiry
    front_item, back_item = compare_items_by_expiry(item_a, item_b)

    print(f"  Front (earlier expiry): item {front_item['item_id']} - {front_item['expiry']}")
    print(f"  Back (later expiry): item {back_item['item_id']} - {back_item['expiry']}")

    # Generate instructions
    instructions = []
    step = 1

    # Assign robots (different shelves = different robots)
    available_robots = sorted([rid for rid, r in robots.items() if r.status in ["IDLE", "READY"]])
    if not available_robots:
        available_robots = sorted(robots.keys())

    # TODO: Generate detailed pick/place instructions
    # TODO: Handle obstructions (relocate blocking items first)
    # TODO: Create new VSU if needed

    # Generate task ID
    task_counters = load_json_file(OPTIMIZATION_TASKS_FILE, {"counters": {}})
    counter = task_counters.get("counters", {}).get(str(product_id), 0) + 1
    task_id = f"REORG_{product_id}_{counter:03d}"

    print(f"  Task ID: {task_id}")
    print(f"{'='*60}\n")

    return {
        "task_id": task_id,
        "product_id": product_id,
        "barcode": front_item.get("barcode"),
        "status": "pending",
        "instructions": instructions,
        "summary": {
            "vsus_compared": [vsu_a["vsu_code"], vsu_b["vsu_code"]],
            "items_to_move": 2,
            "new_vsu_created": False,
            "robots_assigned": available_robots[:2]
        }
    }


# ==================== OPTIMIZATION TRACKING ====================

def load_optimized_vsus() -> Dict:
    """Load optimized VSUs tracking data"""
    return load_json_file(OPTIMIZED_VSUS_FILE, {
        "optimized_vsus": [],
        "vsu_pairs_compared": [],
        "products_completed": [],
        "last_updated": None
    })


def save_optimized_vsus(data: Dict):
    """Save optimized VSUs tracking data"""
    data["last_updated"] = datetime.now().isoformat()
    save_json_file(OPTIMIZED_VSUS_FILE, data)


def reset_optimization_tracking():
    """Reset tracking for new nightly session"""
    data = {
        "optimized_vsus": [],
        "vsu_pairs_compared": [],
        "products_completed": [],
        "session_started": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat()
    }
    save_optimized_vsus(data)
    return data


def mark_vsu_pair_compared(vsu_a_id: int, vsu_b_id: int, product_id: int):
    """Mark a VSU pair as compared"""
    data = load_optimized_vsus()
    data["vsu_pairs_compared"].append({
        "vsu_a": vsu_a_id,
        "vsu_b": vsu_b_id,
        "product_id": product_id,
        "compared_at": datetime.now().isoformat()
    })
    save_optimized_vsus(data)


def mark_vsu_optimized(vsu_id: int):
    """Mark a VSU as optimized"""
    data = load_optimized_vsus()
    if vsu_id not in data["optimized_vsus"]:
        data["optimized_vsus"].append(vsu_id)
    save_optimized_vsus(data)


def is_product_fully_optimized(product_id: int, vsu_list: List[Dict]) -> bool:
    """Check if all VSU pairs for a product have been compared"""
    if len(vsu_list) < 2:
        return True

    data = load_optimized_vsus()
    compared_pairs = data.get("vsu_pairs_compared", [])
    vsu_ids = [v["vsu_id"] for v in vsu_list]

    for i, vsu_a in enumerate(vsu_ids):
        for vsu_b in vsu_ids[i+1:]:
            pair_key = tuple(sorted([vsu_a, vsu_b]))
            pair_found = any(
                tuple(sorted([p["vsu_a"], p["vsu_b"]])) == pair_key
                and p["product_id"] == product_id
                for p in compared_pairs
            )
            if not pair_found:
                return False

    return True


# ==================== DISTANCE OPTIMIZATION (EXISTING) ====================

def find_best_swap(
    fast_item: Dict,
    slow_movers: List[Dict],
    items: Dict,
    virtual_units: Dict,
    moved_items: set,
    occupied_targets: set,
    threshold: float
) -> Optional[Tuple[RelocationPlan, RelocationPlan]]:
    """Find a slow-mover in a closer VSU to swap with."""
    fast_item_obj = items[fast_item["item_id"]]
    fast_vsu = virtual_units[fast_item["vsu_id"]]

    for slow in slow_movers:
        if slow["item_id"] in moved_items or slow["vsu_id"] in occupied_targets:
            continue
        if fast_item["vsu_id"] in occupied_targets or slow["vsu_id"] == fast_item["vsu_id"]:
            continue

        improvement = fast_item["distance"] - slow["distance"]
        if improvement < threshold:
            continue

        slow_item_obj = items[slow["item_id"]]
        slow_vsu = virtual_units[slow["vsu_id"]]

        if not _item_fits_vsu(fast_item_obj, slow_vsu) or not _item_fits_vsu(slow_item_obj, fast_vsu):
            continue
        if not is_item_accessible(fast_item["item_id"], fast_vsu, items):
            continue
        if not is_item_accessible(slow["item_id"], slow_vsu, items):
            continue

        fast_vsu_count = len([i for i in fast_vsu.items if i in items])
        slow_vsu_count = len([i for i in slow_vsu.items if i in items])
        if fast_vsu_count > 1 or slow_vsu_count > 1:
            continue

        fast_plan = RelocationPlan(
            item_id=fast_item["item_id"], product_id=fast_item["product_id"],
            barcode=fast_item["barcode"], current_vsu_id=fast_item["vsu_id"],
            current_vsu_code=fast_item["vsu_code"], current_shelf_name=fast_item["shelf_name"],
            current_rack_name=fast_item["rack_name"], current_distance_to_output=fast_item["distance"],
            target_vsu_id=slow["vsu_id"], target_vsu_code=slow["vsu_code"],
            target_shelf_name=slow["shelf_name"], target_rack_name=slow["rack_name"],
            target_distance_to_output=slow["distance"], distance_improvement=improvement,
            pick_position=fast_item["position"], place_position=slow["position"],
            stock_index=0, strategy="swap", swap_with_item_id=slow["item_id"]
        )

        slow_plan = RelocationPlan(
            item_id=slow["item_id"], product_id=slow["product_id"],
            barcode=slow["barcode"], current_vsu_id=slow["vsu_id"],
            current_vsu_code=slow["vsu_code"], current_shelf_name=slow["shelf_name"],
            current_rack_name=slow["rack_name"], current_distance_to_output=slow["distance"],
            target_vsu_id=fast_item["vsu_id"], target_vsu_code=fast_item["vsu_code"],
            target_shelf_name=fast_item["shelf_name"], target_rack_name=fast_item["rack_name"],
            target_distance_to_output=fast_item["distance"], distance_improvement=0,
            pick_position=slow["position"], place_position=fast_item["position"],
            stock_index=0, strategy="swap", swap_with_item_id=fast_item["item_id"]
        )

        return (fast_plan, slow_plan)

    return None


def find_best_empty_vsu(
    fast_item: Dict,
    items: Dict,
    virtual_units: Dict,
    shelves: Dict,
    racks: Dict,
    output_positions: List,
    moved_items: set,
    occupied_targets: set,
    threshold: float
) -> Optional[RelocationPlan]:
    """Find an empty VSU closer to output for the fast-mover item."""
    fast_item_obj = items[fast_item["item_id"]]
    candidates = []

    for vsu_id, vsu in virtual_units.items():
        if vsu_id in occupied_targets or (vsu.items and len(vsu.items) > 0):
            continue
        if not _item_fits_vsu(fast_item_obj, vsu):
            continue

        target_distance = _find_nearest_output_distance(vsu.position, output_positions)
        improvement = fast_item["distance"] - target_distance
        if improvement < threshold:
            continue

        shelf = shelves.get(vsu.shelf_id)
        rack = racks.get(vsu.rack_id) if vsu.rack_id else None

        candidates.append({
            "vsu_id": vsu_id, "vsu_code": vsu.code,
            "shelf_name": shelf.name if shelf else f"shelf_{vsu.shelf_id}",
            "rack_name": rack.name if rack else f"rack_{vsu.rack_id}",
            "target_distance": target_distance, "improvement": improvement,
            "position": {"x": vsu.position.x, "y": vsu.position.y, "z": vsu.position.z}
        })

    if not candidates:
        return None

    candidates.sort(key=lambda x: x["target_distance"])
    best = candidates[0]

    item_depth = fast_item_obj.metadata.dimensions.depth
    depth_slot_size = item_depth + ITEM_GAP
    max_slots = int(virtual_units[best["vsu_id"]].dimensions.depth // depth_slot_size)
    stock_index = max(0, max_slots - 1)

    return RelocationPlan(
        item_id=fast_item["item_id"], product_id=fast_item["product_id"],
        barcode=fast_item["barcode"], current_vsu_id=fast_item["vsu_id"],
        current_vsu_code=fast_item["vsu_code"], current_shelf_name=fast_item["shelf_name"],
        current_rack_name=fast_item["rack_name"], current_distance_to_output=fast_item["distance"],
        target_vsu_id=best["vsu_id"], target_vsu_code=best["vsu_code"],
        target_shelf_name=best["shelf_name"], target_rack_name=best["rack_name"],
        target_distance_to_output=best["target_distance"], distance_improvement=best["improvement"],
        pick_position=fast_item["position"], place_position=best["position"],
        stock_index=stock_index, strategy="empty_vsu", swap_with_item_id=None
    )


# ==================== MAIN PLANNING ====================

def plan_nightly_optimization(
    request: NightlyOptimizationRequest,
    items: Dict,
    virtual_units: Dict,
    shelves: Dict,
    racks: Dict,
    product_weights: Dict,
    output_positions: List
) -> Dict[str, Any]:
    """
    Main optimization algorithm for distance-based fast-mover relocation.
    Returns dict with relocations, total_distance_saved, and analysis.
    """
    print("\n" + "=" * 60)
    print("NIGHTLY OPTIMIZATION - FAST MOVER RELOCATION")
    print("=" * 60)

    fast_movers, slow_movers = classify_products(
        items, virtual_units, shelves, racks,
        product_weights, output_positions, request.weight_threshold
    )

    if not fast_movers:
        return {"relocations": [], "total_distance_saved": 0, "analysis": {"fast_movers_identified": 0}}

    relocations: List[RelocationPlan] = []
    moved_items: set = set()
    occupied_targets: set = set()

    for fast_item in fast_movers:
        if fast_item["item_id"] in moved_items or len(relocations) >= request.max_relocations:
            break

        fast_vsu = virtual_units.get(fast_item["vsu_id"])
        if not fast_vsu or not is_item_accessible(fast_item["item_id"], fast_vsu, items):
            continue

        swap_result = find_best_swap(
            fast_item, slow_movers, items, virtual_units,
            moved_items, occupied_targets, request.min_improvement_threshold
        )

        if swap_result and len(relocations) + 2 <= request.max_relocations:
            fast_plan, slow_plan = swap_result
            relocations.extend([fast_plan, slow_plan])
            moved_items.update([fast_item["item_id"], slow_plan.item_id])
            occupied_targets.update([fast_plan.target_vsu_id, slow_plan.target_vsu_id])
            continue

        empty_result = find_best_empty_vsu(
            fast_item, items, virtual_units, shelves, racks,
            output_positions, moved_items, occupied_targets,
            request.min_improvement_threshold
        )

        if empty_result:
            relocations.append(empty_result)
            moved_items.add(fast_item["item_id"])
            occupied_targets.add(empty_result.target_vsu_id)

    improvements = [r.distance_improvement for r in relocations if r.distance_improvement > 0]

    return {
        "relocations": relocations,
        "total_distance_saved": sum(improvements),
        "analysis": {
            "fast_movers_identified": len(fast_movers),
            "relocations_planned": len(relocations),
            "avg_improvement": sum(improvements) / len(improvements) if improvements else 0
        }
    }


# ==================== INSTRUCTION GENERATION ====================

def generate_optimization_instructions(
    relocations: List[RelocationPlan],
    items: Dict,
    robots: Dict,
    virtual_units: Dict,
    shelves: Dict,
    racks: Dict,
) -> Tuple[List[Dict], str]:
    """Generate robot pick-and-place instructions for optimization relocations."""
    shelf_groups = defaultdict(list)
    for reloc in relocations:
        vsu = virtual_units.get(reloc.current_vsu_id)
        if vsu:
            shelf_groups[vsu.shelf_id].append(reloc)

    available_robots = sorted([
        rid for rid, r in robots.items() if r.status in ["IDLE", "READY"]
    ]) or sorted(robots.keys())

    robot_assignments = defaultdict(list)
    for idx, (shelf_id, relocs) in enumerate(sorted(shelf_groups.items(), key=lambda x: -len(x[1]))):
        robot_id = available_robots[idx % len(available_robots)]
        robot_assignments[robot_id].extend(relocs)

    instructions = []
    trip_number = 1

    for robot_id, assigned_relocs in robot_assignments.items():
        for reloc in assigned_relocs:
            instructions.append({
                "robot_id": robot_id,
                "trip_number": trip_number,
                "action": "optimize_relocate",
                "reason": "nightly_optimization",
                "items": [{
                    "item_id": reloc.item_id,
                    "product_id": reloc.product_id,
                    "barcode": reloc.barcode,
                    "pick_position": reloc.pick_position,
                    "place_position": reloc.place_position,
                    "current_vsu": reloc.current_vsu_code,
                    "target_vsu": reloc.target_vsu_code,
                    "stock_index": reloc.stock_index,
                    "strategy": reloc.strategy,
                }]
            })
            trip_number += 1

    task_id = f"NIGHTLY-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return instructions, task_id


# ==================== EXECUTION ====================

def execute_optimization_relocations(
    relocations: List[Dict],
    items: Dict,
    virtual_units: Dict,
    shelves: Dict,
) -> Dict:
    """Apply inventory updates for completed nightly optimization."""
    from relocation import swap_items_between_vsus, relocate_item_to_target_vsu

    completed = 0
    failed = 0
    errors = []
    processed_swaps = set()

    for reloc in relocations:
        item_id = reloc["item_id"] if isinstance(reloc, dict) else reloc.item_id
        strategy = reloc["strategy"] if isinstance(reloc, dict) else reloc.strategy
        swap_with = reloc.get("swap_with_item_id") if isinstance(reloc, dict) else reloc.swap_with_item_id
        target_vsu_id = reloc["target_vsu_id"] if isinstance(reloc, dict) else reloc.target_vsu_id

        swap_key = tuple(sorted([item_id, swap_with])) if swap_with else None
        if swap_key and swap_key in processed_swaps:
            continue

        try:
            if strategy == "swap" and swap_with:
                swap_items_between_vsus(item_id, swap_with, items, virtual_units, "nightly_optimization")
                processed_swaps.add(swap_key)
                completed += 2
            elif strategy in ["empty_vsu", "expiry_sort"]:
                stock_index = reloc["stock_index"] if isinstance(reloc, dict) else reloc.stock_index
                relocate_item_to_target_vsu(item_id, target_vsu_id, stock_index, items, virtual_units, "nightly_optimization")
                completed += 1
        except Exception as e:
            failed += 1
            errors.append({"item_id": item_id, "error": str(e)})

    return {"completed": completed, "failed": failed, "errors": errors}


# Load persisted consolidation tasks on module import
_load_consolidation_tasks()
