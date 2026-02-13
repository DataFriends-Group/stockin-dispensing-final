"""
DISPENSE SERVER - Port 8001
Handles all dispensing and relocation operations on a separate port.
Shares memory with main server (stock-in on port 8000).
"""

import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import shared data and functions from main
from main import (
    items,
    robots,
    virtual_units,
    shelves,
    racks,
    OUTPUT_POSITIONS,
    save_robots_to_file,
    save_warehouse_state,
    VirtualStorageUnit,
    Dimensions,
    Position,
)

# Import relocation function
from relocation import relocate_item

# Import dispense request/response models and endpoint functions
from dispensing import (
    DispenseRequest,
    CompleteDispenseRequest,
    FailDispenseRequest,
    create_dispense_task_endpoint,
    complete_dispense_endpoint,
    fail_dispense_endpoint,
    get_dispense_logs_endpoint,
    get_product_dispense_history_endpoint,
    get_task_status_endpoint,
)

# Create dispense app
dispense_app = FastAPI(
    title="MedicPort Dispense Server",
    description="Dispensing operations on port 8001",
    version="1.0.0"
)

dispense_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Task counter for dispense (shared via import)
task_counter = 0


@dispense_app.post("/dispense/create", tags=["Dispensing Operations"])
async def create_dispense_task(request: DispenseRequest):
    """
    Create a new dispensing task with multi-pick batching
    """
    global task_counter
    from main import task_counter as main_task_counter

    response, updated_task_counter, updated_relocate_tasks = create_dispense_task_endpoint(
        request,
        items,
        robots,
        virtual_units,
        shelves,
        racks,
        OUTPUT_POSITIONS,
        main_task_counter
    )

    # Update main's task_counter via import
    import main
    main.task_counter = updated_task_counter
    main.relocate_tasks_store.update(updated_relocate_tasks)

    return response


@dispense_app.post("/dispense/complete", tags=["Dispensing Operations"])
async def complete_dispense(request: CompleteDispenseRequest):
    """
    Mark a dispense task as complete - removes items from inventory
    """
    from main import relocate_tasks_store
    return complete_dispense_endpoint(
        request,
        items,
        robots,
        virtual_units,
        shelves,
        save_robots_to_file,
        save_warehouse_state,
        relocate_tasks_store
    )


@dispense_app.post("/dispense/fail", tags=["Dispensing Operations"])
async def fail_dispense(request: FailDispenseRequest):
    """
    Mark a dispense task as failed with optional partial success.

    If successful_trips is provided, those items will be removed from inventory.
    Items from trips NOT in successful_trips remain in inventory.

    Example:
        {
            "task_id": "DISP_001",
            "successful_trips": [1, 2, 3]  // Trips 1-3 dispensed successfully
        }
    """
    return fail_dispense_endpoint(
        request,
        items,
        robots,
        virtual_units,
        shelves,
        save_robots_to_file,
        save_warehouse_state
    )


@dispense_app.get("/dispense/logs", tags=["Dispensing Logs & History"])
async def get_dispense_logs():
    """
    Get comprehensive dispense statistics
    """
    return get_dispense_logs_endpoint()


@dispense_app.get("/dispense/product/{product_id}", tags=["Dispensing Logs & History"])
async def get_product_dispense_history(product_id: int):
    """
    Get detailed dispense history for a specific product
    """
    return get_product_dispense_history_endpoint(product_id)


@dispense_app.get("/dispense/task/{task_id}", tags=["Dispensing Logs & History"])
async def get_task_status(task_id: str):
    """
    Get status of a dispense task
    """
    return get_task_status_endpoint(task_id)


@dispense_app.get("/health", tags=["System"])
async def health_check():
    """Health check for dispense server"""
    return {
        "status": "healthy",
        "server": "dispense",
        "port": 8001,
        "items_in_inventory": len(items),
        "robots_available": len([r for r in robots.values() if r.status == "IDLE"])
    }


# Relocation Models and Endpoints
class TemporaryRelocateRequest(BaseModel):
    """Request to relocate item to temporary storage - ONLY accepts task_id from dispense instruction"""
    task_id: str = Field(..., description="Task ID from dispense instruction (e.g., RELOCATE-001) - REQUIRED")


class CompleteRelocateRequest(BaseModel):
    """Request to complete a pending relocation"""
    task_id: str = Field(..., description="Task ID from the relocate suggest response")


@dispense_app.post("/api/temporary/relocate", tags=["Relocation"])
async def suggest_relocation(request: TemporaryRelocateRequest):
    """
    Step 1: Suggest relocation destination (does NOT update inventory)

    This endpoint calculates where an obstructing item should be moved,
    but does NOT actually move it. Call /api/temporary/relocate/complete
    after the robot physically moves the item.

    Process:
    1. Lookup task_id from dispense instruction
    2. Find VSU using priority: same-product > mixed-product > create new > empty
    3. Return destination coordinates (inventory NOT updated)

    Request body:
    {
        "task_id": "RELOCATE-001"
    }

    Response:
    {
        "status": "pending",
        "task_id": "RELOCATE-001",
        "item_id": 123,
        "from": {"vsu_id": 29, "vsu_code": "vu29", ...},
        "to": {"vsu_id": 32, "vsu_code": "vu32", ...},
        "is_new_vsu": true/false,
        "committed": false
    }

    Next step: Call /api/temporary/relocate/complete with the task_id
    """
    try:
        import main

        print(f"\n{'='*60}")
        print(f"RELOCATION SUGGEST REQUEST")
        print(f"{'='*60}")
        print(f"Task ID: {request.task_id}")

        # Try to load from file if not in memory (cross-process sharing)
        if request.task_id not in main.relocate_tasks_store:
            relocate_file = Path("data/relocate_tasks.json")
            if relocate_file.exists():
                with open(relocate_file) as f:
                    file_tasks = json.load(f)
                    main.relocate_tasks_store.update(file_tasks)
                    print(f"[RELOCATE] Loaded {len(file_tasks)} tasks from file")

        if request.task_id not in main.relocate_tasks_store:
            raise HTTPException(
                status_code=404,
                detail=f"Task ID {request.task_id} not found. Make sure to call /dispense/create first to get valid task IDs."
            )

        task_info = main.relocate_tasks_store[request.task_id]
        item_id = task_info["item_id"]

        if task_info.get("status") == "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Task {request.task_id} already completed"
            )

        # If previously failed, allow retry
        if task_info.get("status") == "failed":
            print(f"[RELOCATE] Task {request.task_id} previously failed - retrying")

        print(f"Item ID: {item_id}")
        print(f"Barcode: {task_info['barcode']}")
        print(f"Product ID: {task_info['product_id']}")

        if item_id not in items:
            raise HTTPException(status_code=404, detail=f"Item {item_id} not found in warehouse")

        item = items[item_id]
        original_vsu_id = item.vsu_id
        original_vsu_code = virtual_units[original_vsu_id].code if original_vsu_id and original_vsu_id in virtual_units else None

        # Suggest relocation (commit=False - does NOT update inventory)
        relocation_info = relocate_item(
            item_id=item_id,
            items=items,
            virtual_units=virtual_units,
            shelves=shelves,
            reason="obstruction_removal",
            vsu_counter=main.vsu_counter,
            VirtualStorageUnit=VirtualStorageUnit,
            Dimensions=Dimensions,
            Position=Position,
            commit=False  # Only suggest, don't commit
        )

        # Update vsu_counter if a new VSU was created
        if relocation_info.get("is_new_vsu") and relocation_info.get("vsu_counter"):
            main.vsu_counter = relocation_info["vsu_counter"]
            print(f"[RELOCATE] Updated vsu_counter to {main.vsu_counter}")

        # Store pending relocation info for complete step
        main.relocate_tasks_store[request.task_id].update({
            "status": "pending",
            "original_vsu_id": relocation_info.get("from", {}).get("vsu_id"),
            "original_vsu_code": relocation_info.get("from", {}).get("vsu_code"),
            "new_vsu_id": relocation_info.get("to", {}).get("vsu_id"),
            "new_vsu_code": relocation_info.get("to", {}).get("vsu_code"),
            "new_stock_index": relocation_info.get("to", {}).get("stock_index"),
            "is_new_vsu": relocation_info.get("is_new_vsu"),
            "suggested_at": relocation_info.get("relocated_at")
        })

        print(f"Relocation suggested for item {item_id}")
        print(f"  From: {relocation_info.get('from', {}).get('vsu_code')}")
        print(f"  To: {relocation_info.get('to', {}).get('vsu_code')}")
        print(f"  Status: PENDING (call /complete to finalize)")
        print(f"{'='*60}\n")

        return {
            "status": "pending",
            "task_id": request.task_id,
            **relocation_info
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error suggesting relocation: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to suggest relocation: {str(e)}"
        )


@dispense_app.post("/api/temporary/relocate/complete", tags=["Relocation"])
async def complete_relocation(request: CompleteRelocateRequest):
    """
    Step 2: Complete the relocation (updates inventory)

    Call this AFTER the robot has physically moved the item to the
    destination returned by /api/temporary/relocate.

    Uses the STORED destination from suggest step (does NOT recalculate).

    Request body:
    {
        "task_id": "RELOCATE-001"
    }

    Response:
    {
        "status": "success",
        "task_id": "RELOCATE-001",
        "item_id": 123,
        "from": {"vsu_id": 29, ...},
        "to": {"vsu_id": 32, ...},
        "committed": true
    }
    """
    try:
        import main
        from datetime import datetime
        from relocation import calc_z_positions

        print(f"\n{'='*60}")
        print(f"RELOCATION COMPLETE REQUEST")
        print(f"{'='*60}")
        print(f"Task ID: {request.task_id}")

        # Lookup task details from relocate_tasks_store
        if request.task_id not in main.relocate_tasks_store:
            raise HTTPException(
                status_code=404,
                detail=f"Task ID {request.task_id} not found"
            )

        task_info = main.relocate_tasks_store[request.task_id]

        if task_info.get("status") == "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Task {request.task_id} already completed"
            )

        if task_info.get("status") != "pending":
            raise HTTPException(
                status_code=400,
                detail=f"Task {request.task_id} not in pending state. Call /api/temporary/relocate first."
            )

        item_id = task_info["item_id"]
        new_vsu_id = task_info["new_vsu_id"]
        new_vsu_code = task_info["new_vsu_code"]
        new_stock_index = task_info["new_stock_index"]
        original_vsu_id = task_info["original_vsu_id"]
        original_vsu_code = task_info["original_vsu_code"]

        if item_id not in items:
            raise HTTPException(status_code=404, detail=f"Item {item_id} not found in warehouse")

        item = items[item_id]
        original_vsu = virtual_units[original_vsu_id]
        new_vsu = virtual_units[new_vsu_id]

        print(f"Item ID: {item_id}")
        print(f"Barcode: {task_info['barcode']}")
        print(f"Using STORED destination: VSU {new_vsu_code} (stock_index={new_stock_index})")

        if item_id in original_vsu.items:
            original_vsu.items.remove(item_id)
        if not original_vsu.items:
            original_vsu.occupied = False

        has_existing_items = new_vsu.items and len(new_vsu.items) > 0

        if has_existing_items:
            # STACKING: Normalize existing items' stock indices
            print(f"[RELOCATE] Stacking in VSU with {len(new_vsu.items)} existing items")
            existing_items_sorted = sorted(
                [(eid, items[eid].stock_index) for eid in new_vsu.items if eid in items],
                key=lambda x: x[1]
            )
            for new_idx, (existing_item_id, old_idx) in enumerate(existing_items_sorted, start=1):
                items[existing_item_id].stock_index = new_idx
                print(f"  Item {existing_item_id}: stock_index {old_idx} -> {new_idx}")

        if not new_vsu.items:
            new_vsu.items = []
        new_vsu.items.append(item_id)
        new_vsu.occupied = True

        old_stock_index = item.stock_index
        item.vsu_id = new_vsu_id
        item.stock_index = new_stock_index

        # Calculate Z positions using proper function (accounts for 3mm gaps)
        new_z_start, new_z_end = calc_z_positions(new_vsu, item, items)

        save_warehouse_state()

        completed_at = datetime.now()

        main.relocate_tasks_store[request.task_id].update({
            "status": "completed",
            "completed_at": completed_at.isoformat()
        })

        print(f"Item {item_id} relocation COMPLETED (using stored destination)")
        print(f"  From: {original_vsu_code}")
        print(f"  To: {new_vsu_code} (stock_index={new_stock_index})")
        print(f"{'='*60}\n")

        return {
            "status": "success",
            "task_id": request.task_id,
            "item_id": item_id,
            "product_id": item.metadata.product_id,
            "barcode": item.metadata.barcode,
            "from": {
                "vsu_id": original_vsu_id,
                "vsu_code": original_vsu_code,
                "stock_index": old_stock_index,
                "x": original_vsu.position.x,
                "y": original_vsu.position.y
            },
            "to": {
                "vsu_id": new_vsu_id,
                "vsu_code": new_vsu_code,
                "stock_index": new_stock_index,
                "x": new_vsu.position.x,
                "y": new_vsu.position.y,
                "z_start": new_z_start,
                "z_end": new_z_end
            },
            "is_new_vsu": task_info.get("is_new_vsu", False),
            "committed": True,
            "relocated_at": completed_at.isoformat(),
            "reason": "obstruction_removal"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error completing relocation: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to complete relocation: {str(e)}"
        )
