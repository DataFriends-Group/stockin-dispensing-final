"""
DISPENSING MODULE - MedicPort Sequential Stock System

Features:
- Multi-product, multi-quantity dispensing
- MULTI-PICK BATCHING: Robots pick ALL items from same VSU in one trip
- Intelligent robot coordination (both robots can work together)
- Collision avoidance (robots never on same shelf)
- OUTPUT QUEUING: Robots wait if output is occupied (sequential access)
- Smart per-trip output selection (each trip goes to nearest output)
- Manual output override (force all robots to specific output)
- Complete/Fail APIs for task finalization
- Inventory updates on successful dispensing
- Comprehensive dispense logs with daily/weekly tracking
- ML weight tracking for future optimization

Robot Capacity & Multi-Pick:
- Robots pick ALL items from same VSU in one trip (no limit)
- Cannot pick from different VSUs in same trip
- Prioritize VSUs with more items (fewer trips)
- Pick front items first (stock_index 0, 1, 2...)

Output Selection & Queuing Logic:
- If output_id specified (manual mode): INTERLEAVED TRIPS
  * All robots go to same output with spatial offsets (R1: -100mm, R2: +100mm)
  * Trip sequence: R1 Trip 1, R2 Trip 1, R1 Trip 2, R2 Trip 2 (alternating)
  * Each trip waits for previous trip to prevent simultaneous arrival
  * Example: R1 arrives at (550, 50), completes, then R2 arrives at (750, 50), completes

- If output_id = null (auto mode): NEAREST OUTPUT + ROBOT QUEUING
  * Each trip goes to nearest output to batch's average position
  * If conflict detected (2+ robots, same output): Sequential execution (R1 finishes ALL, then R2 starts)
  * wait_for_output flag indicates robots that must queue

Trip Planning:
- Items grouped by VSU (all items from same VSU = 1 trip)
- VSUs sorted by item count (descending) - prioritize fuller VSUs
- Each VSU batch = 1 trip with optimized output selection
"""

from fastapi import HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Tuple, TYPE_CHECKING, Any
from datetime import datetime, timedelta
from collections import defaultdict
import json
import os

if TYPE_CHECKING:
    from main import Position, Item, Robot, VirtualStorageUnit, Shelf, Rack

DEPTH_GAP_BETWEEN_ITEMS = 3  # 3mm gap between items

def calc_z_positions(vsu, item, items_dict) -> Tuple[float, float]:
    """Calculate z_start and z_end for an item in VSU"""
    back_wall_z = vsu.position.z + vsu.dimensions.depth
    depth_from_back = 0
    for other_id in vsu.items:
        other = items_dict.get(other_id)
        if other and other.stock_index > item.stock_index:
            depth_from_back += other.metadata.dimensions.depth + DEPTH_GAP_BETWEEN_ITEMS
    z_start = back_wall_z - depth_from_back - item.metadata.dimensions.depth
    return z_start, z_start + item.metadata.dimensions.depth

class DispenseRequest(BaseModel):
    """
    Request to dispense multiple products

    Fields:
        products: List of products to dispense
            - Can use product_id: [{"product_id": 1, "quantity": 2}, ...]
            - Or use barcode: [{"barcode": "ABC123", "quantity": 2}, ...]
        output_id: Optional output selection (None = auto-select nearest output per trip)
    """
    products: List[Dict[str, Any]]
    output_id: Optional[int] = None

class DispenseItem(BaseModel):
    """Single item in a pick instruction"""
    item_id: int
    product_id: int
    barcode: str
    vsu_code: str
    shelf_name: str
    rack_name: str
    coordinates: Dict[str, float]
    stock_index: int
    batch: Optional[str] = None
    expiration: Optional[str] = None
    z_start: Optional[float] = None  # Front face of box (where robot grips)
    z_end: Optional[float] = None    # Back face of box (z_start + depth)
    action: str = "pick"  # "pick" or "relocate"
    reason: str = "fulfill_order"  # "fulfill_order" or "obstruction_removal"
    temp_endpoint: Optional[str] = None  # "/api/temporary/relocate" for relocations
    relocate_task_id: Optional[str] = None  # Task ID for relocation (e.g., "RELOCATE-001")

class DispenseInstruction(BaseModel):
    """Instruction for robot (relocation or pick)"""
    robot_id: str
    trip_number: Optional[int] = None  # None for relocations, numbered for picks
    items: List[DispenseItem] = []
    output_position: Optional[Dict[str, float]] = None  # None for relocations, set for picks

class DispenseTask(BaseModel):
    """Complete dispensing task"""
    task_id: str
    status: str = "pending"
    instructions: List[DispenseInstruction] = []
    output_position: Dict[str, float]
    robot_output_positions: Dict[str, Dict[str, float]] = {}
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    failure_reason: Optional[str] = None

class CompleteDispenseRequest(BaseModel):
    """Request to mark dispensing as complete"""
    task_id: str

class FailDispenseRequest(BaseModel):
    """Request to mark dispensing as failed (with optional partial success)"""
    task_id: str
    reason: str = "partial_failure"
    successful_trips: List[int] = []  # Trip numbers that were successfully dispensed

dispense_tasks: Dict[str, DispenseTask] = {}
dispense_task_counter = 0
output_usage: Dict[int, Dict[str, Any]] = {}

INVENTORY_FILE = "data/ml_robot_updated.json"
DISPENSE_LOG_FILE = "data/dispense_logs.json"
HISTORY_FOLDER = "data/history"

def load_inventory() -> dict:
    """Load current inventory from file"""
    try:
        if os.path.exists(INVENTORY_FILE):
            with open(INVENTORY_FILE, 'r') as f:
                return json.load(f)
        elif os.path.exists("data/ml_robot.json"):
            with open("data/ml_robot.json", 'r') as f:
                return json.load(f)
        else:
            return {"items": []}
    except Exception as e:
        print(f"Error loading inventory: {e}")
        return {"items": []}

def save_inventory(inventory_data: dict):
    """Save updated inventory to file"""
    try:
        with open(INVENTORY_FILE, 'w') as f:
            json.dump(inventory_data, f, indent=2)
        print(f"Inventory saved to {INVENTORY_FILE}")
    except Exception as e:
        print(f"Error saving inventory: {e}")
        raise

def load_dispense_logs() -> dict:
    """
    Load dispense logs

    Structure:
    {
        "products": {
            "1411": {
                "barcode": "XXX",
                "total_dispensed": 15,
                "last_dispensed": "2025-11-12T10:30:00",
                "daily_dispenses": {
                    "2025-11-12": 5,
                    "2025-11-11": 3
                },
                "weekly_dispenses": {
                    "2025-W46": 15,
                    "2025-W45": 22
                },
                "monthly_dispenses": {
                    "2025-11": 45,
                    "2025-10": 38
                }
            }
        },
        "summary": {
            "total_dispenses": 150,
            "last_updated": "2025-11-12T10:30:00",
            "current_week": "2025-W46",
            "current_month": "2025-11"
        },
        "archive": {
            "weeks": {
                "2025-W45": {
                    "total": 22,
                    "products": {"1411": 22}
                }
            },
            "months": {
                "2025-10": {
                    "total": 38,
                    "products": {"1411": 38}
                }
            }
        }
    }
    """
    try:
        if os.path.exists(DISPENSE_LOG_FILE):
            with open(DISPENSE_LOG_FILE, 'r') as f:
                logs = json.load(f)
                if "archive" not in logs:
                    logs["archive"] = {"weeks": {}, "months": {}}
                if "current_week" not in logs["summary"]:
                    logs["summary"]["current_week"] = get_week_number(datetime.now())
                if "current_month" not in logs["summary"]:
                    logs["summary"]["current_month"] = get_month_string(datetime.now())
                return logs
        else:
            return {
                "products": {},
                "summary": {
                    "total_dispenses": 0,
                    "last_updated": datetime.now().isoformat(),
                    "current_week": get_week_number(datetime.now()),
                    "current_month": get_month_string(datetime.now())
                },
                "archive": {
                    "weeks": {},
                    "months": {}
                }
            }
    except Exception as e:
        print(f"Error loading dispense logs: {e}")
        return {
            "products": {},
            "summary": {
                "total_dispenses": 0,
                "last_updated": datetime.now().isoformat(),
                "current_week": get_week_number(datetime.now()),
                "current_month": get_month_string(datetime.now())
            },
            "archive": {
                "weeks": {},
                "months": {}
            }
        }

def save_dispense_logs(logs: dict):
    """Save dispense logs to file"""
    try:
        with open(DISPENSE_LOG_FILE, 'w') as f:
            json.dump(logs, f, indent=2)
        print(f"Dispense logs saved to {DISPENSE_LOG_FILE}")
    except Exception as e:
        print(f"Error saving dispense logs: {e}")
        raise

def get_week_number(date: datetime) -> str:
    """Get ISO week number in format YYYY-Www"""
    return f"{date.year}-W{date.isocalendar()[1]:02d}"

def get_month_string(date: datetime) -> str:
    """Get month string in format YYYY-MM"""
    return f"{date.year}-{date.month:02d}"

def archive_old_products_to_history(logs: dict, days_threshold: int = 30):
    """
    Archive products with last_dispensed older than threshold to history_dispense.json

    - Removes products not dispensed in last 30 days from dispense_logs.json
    - Moves them to history_dispense.json (unlimited size)
    - Preserves all historical data
    """
    from datetime import timedelta

    HISTORY_FILE = "history_dispense.json"
    current_date = datetime.now()
    threshold_date = current_date - timedelta(days=days_threshold)

    products_to_archive = []

    # Find products older than threshold
    for product_id, product_data in logs["products"].items():
        if product_data.get("last_dispensed"):
            last_dispensed = datetime.fromisoformat(product_data["last_dispensed"])
            if last_dispensed < threshold_date:
                products_to_archive.append((product_id, product_data))

    if not products_to_archive:
        return  # Nothing to archive

    # Load or create history file
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        else:
            history = {
                "archived_products": {},
                "summary": {
                    "total_archived_products": 0,
                    "last_updated": current_date.isoformat()
                }
            }
    except Exception as e:
        print(f"Error loading history file: {e}")
        history = {
            "archived_products": {},
            "summary": {
                "total_archived_products": 0,
                "last_updated": current_date.isoformat()
            }
        }

    # Archive old products
    for product_id, product_data in products_to_archive:
        # Add archive timestamp
        product_data["archived_at"] = current_date.isoformat()
        product_data["archived_reason"] = f"No dispenses in last {days_threshold} days"

        # Move to history
        history["archived_products"][product_id] = product_data

        # Remove from active logs
        del logs["products"][product_id]

        print(f"  Archived product {product_id} (last dispensed: {product_data['last_dispensed']})")

    # Update history summary
    history["summary"]["total_archived_products"] = len(history["archived_products"])
    history["summary"]["last_updated"] = current_date.isoformat()

    # Save history file
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Archived {len(products_to_archive)} products to {HISTORY_FILE}")
    except Exception as e:
        print(f"Error saving history file: {e}")


def archive_old_periods(logs: dict, current_week: str, current_month: str):
    """
    Archive old week/month data when a new period starts

    This keeps the main product logs clean while preserving history
    """
    # Archive old weeks
    for product_id, product_data in logs["products"].items():
        if "weekly_dispenses" in product_data:
            for week, count in list(product_data["weekly_dispenses"].items()):
                if week != current_week:
                    # Archive this week
                    if week not in logs["archive"]["weeks"]:
                        logs["archive"]["weeks"][week] = {
                            "total": 0,
                            "products": {}
                        }
                    logs["archive"]["weeks"][week]["products"][product_id] = count
                    logs["archive"]["weeks"][week]["total"] += count

                    # Remove from current (keep only current week)
                    # del product_data["weekly_dispenses"][week]

    # Archive old months
    for product_id, product_data in logs["products"].items():
        if "monthly_dispenses" in product_data:
            for month, count in list(product_data["monthly_dispenses"].items()):
                if month != current_month:
                    # Archive this month
                    if month not in logs["archive"]["months"]:
                        logs["archive"]["months"][month] = {
                            "total": 0,
                            "products": {}
                        }
                    logs["archive"]["months"][month]["products"][product_id] = count
                    logs["archive"]["months"][month]["total"] += count

                    # Keep all months for now (don't delete)
                    # del product_data["monthly_dispenses"][month]

    print(f"Archived data - Weeks: {len(logs['archive']['weeks'])}, Months: {len(logs['archive']['months'])}")



def get_month_name(date: datetime) -> str:
    """Get month name in format 'january_2025'"""
    month_names = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ]
    return f"{month_names[date.month - 1]}_{date.year}"


def ensure_history_folder():
    """Create history folder structure if it doesn't exist"""
    if not os.path.exists(HISTORY_FOLDER):
        os.makedirs(HISTORY_FOLDER)
        print(f"Created history folder: {HISTORY_FOLDER}")


def get_monthly_history_path(year: int, month: int) -> str:
    """Get path to monthly history file"""
    ensure_history_folder()
    year_folder = os.path.join(HISTORY_FOLDER, str(year))
    if not os.path.exists(year_folder):
        os.makedirs(year_folder)

    date = datetime(year, month, 1)
    filename = f"{get_month_name(date)}.json"
    return os.path.join(year_folder, filename)


def load_monthly_history(year: int, month: int) -> dict:
    """Load monthly history file or create new one"""
    filepath = get_monthly_history_path(year, month)

    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading monthly history: {e}")

    # Create new monthly history structure
    date = datetime(year, month, 1)
    return {
        "month": f"{year}-{month:02d}",
        "month_name": get_month_name(date).replace('_', ' ').title(),
        "year": year,
        "products": {},
        "summary": {
            "total_dispenses": 0,
            "first_dispense": None,
            "last_dispense": None,
            "total_products": 0
        }
    }


def save_monthly_history(year: int, month: int, data: dict):
    """Save monthly history file"""
    filepath = get_monthly_history_path(year, month)
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Saved monthly history: {filepath}")
    except Exception as e:
        print(f"Error saving monthly history: {e}")
        raise


def archive_old_daily_dispenses(logs: dict, days_threshold: int = 30):
    """
    AUTOMATIC: Archive daily dispense records older than 30 days to monthly history files

    - Runs automatically every time a dispense is logged
    - Archives to history/YYYY/month_YYYY.json based on DISPENSE date (not current date)
    - Removes old records from active dispense_logs.json
    - Keeps active log small (rolling 30-day window)
    """
    current_date = datetime.now()
    threshold_date = current_date - timedelta(days=days_threshold)

    archived_count = 0
    monthly_archives = {}  # Group by (year, month) for batch saving

    # Process each product
    for product_id, product_data in logs["products"].items():
        if "daily_dispenses" not in product_data:
            continue

        dates_to_remove = []

        # Check each daily dispense record
        for date_str, quantity in list(product_data["daily_dispenses"].items()):
            try:
                dispense_date = datetime.fromisoformat(date_str)

                # If older than 30 days, archive it
                if dispense_date.date() <= threshold_date.date():
                    year = dispense_date.year
                    month = dispense_date.month

                    # Load monthly archive (cache in memory for this run)
                    if (year, month) not in monthly_archives:
                        monthly_archives[(year, month)] = load_monthly_history(year, month)

                    archive = monthly_archives[(year, month)]

                    # Initialize product in archive if needed
                    if product_id not in archive["products"]:
                        archive["products"][product_id] = {
                            "barcode": product_data.get("barcode", ""),
                            "total_dispensed_in_month": 0,
                            "daily_dispenses": {}
                        }

                    # Add to monthly archive
                    archive["products"][product_id]["daily_dispenses"][date_str] = quantity
                    archive["products"][product_id]["total_dispensed_in_month"] += quantity

                    # Update archive summary
                    archive["summary"]["total_dispenses"] += quantity
                    if archive["summary"]["first_dispense"] is None or date_str < archive["summary"]["first_dispense"]:
                        archive["summary"]["first_dispense"] = date_str
                    if archive["summary"]["last_dispense"] is None or date_str > archive["summary"]["last_dispense"]:
                        archive["summary"]["last_dispense"] = date_str

                    # Mark for removal from active log
                    dates_to_remove.append(date_str)
                    archived_count += 1

            except Exception as e:
                print(f"Warning: Could not process date {date_str}: {e}")
                continue

        # Remove archived dates from active log
        for date_str in dates_to_remove:
            del product_data["daily_dispenses"][date_str]

    # Save all monthly archives
    for (year, month), archive in monthly_archives.items():
        archive["summary"]["total_products"] = len(archive["products"])
        save_monthly_history(year, month, archive)

    if archived_count > 0:
        print(f"AUTO-ARCHIVED: {archived_count} daily records to {len(monthly_archives)} monthly file(s)")
        print(f"  Active log now contains only last {days_threshold} days")

    return archived_count


def get_product_history_from_archives(product_id: str, months_back: int = 12) -> dict:
    """
    Retrieve historical data for a product from monthly archive files

    Returns combined data from archived monthly files for the last N months
    """
    current_date = datetime.now()
    history_data = {
        "daily_dispenses": {},
        "monthly_totals": {},
        "total_from_archives": 0
    }

    # Check last N months
    for i in range(months_back):
        target_date = current_date - timedelta(days=30 * i)
        year = target_date.year
        month = target_date.month

        filepath = get_monthly_history_path(year, month)

        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    archive = json.load(f)

                    if product_id in archive.get("products", {}):
                        product_data = archive["products"][product_id]

                        # Merge daily dispenses
                        if "daily_dispenses" in product_data:
                            history_data["daily_dispenses"].update(product_data["daily_dispenses"])

                        # Add monthly total
                        month_key = f"{year}-{month:02d}"
                        history_data["monthly_totals"][month_key] = product_data.get("total_dispensed_in_month", 0)
                        history_data["total_from_archives"] += product_data.get("total_dispensed_in_month", 0)

            except Exception as e:
                print(f"Warning: Could not load archive for {year}-{month:02d}: {e}")
                continue

    return history_data


def list_all_monthly_archives() -> List[str]:
    """List all available monthly archive files"""
    archives = []

    if not os.path.exists(HISTORY_FOLDER):
        return archives

    # Walk through history folder
    for year_folder in os.listdir(HISTORY_FOLDER):
        year_path = os.path.join(HISTORY_FOLDER, year_folder)
        if os.path.isdir(year_path):
            for filename in os.listdir(year_path):
                if filename.endswith('.json'):
                    archives.append(os.path.join(year_path, filename))

    return sorted(archives)


def update_dispense_log(product_id: int, barcode: str, quantity: int):
    """
    Update dispense logs with new dispensing data

    Tracks:
    - Total dispensed count
    - Daily dispenses (by date)
    - Weekly dispenses (by ISO week)
    - Monthly dispenses (by year-month)

    AUTOMATIC ARCHIVING:
    - Daily records older than 30 days -> Moved to history/YYYY/month_YYYY.json
    - Keeps dispense_logs.json small (rolling 30-day window)
    - Archives organized by month when data was dispensed
    - Example: Nov 4th dispense -> Archived to november_2025.json on Dec 4th
    """
    logs = load_dispense_logs()

    product_key = str(product_id)
    current_date = datetime.now()
    date_str = current_date.date().isoformat()
    week_str = get_week_number(current_date)
    month_str = get_month_string(current_date)

    # Check for week/month rollover and archive if needed
    if logs["summary"]["current_week"] != week_str:
        print(f"\n{'='*60}")
        print(f"NEW WEEK DETECTED: {logs['summary']['current_week']} -> {week_str}")
        print(f"{'='*60}")
        archive_old_periods(logs, week_str, month_str)
        logs["summary"]["current_week"] = week_str

    if logs["summary"]["current_month"] != month_str:
        print(f"\n{'='*60}")
        print(f"NEW MONTH DETECTED: {logs['summary']['current_month']} -> {month_str}")
        print(f"{'='*60}")
        archive_old_periods(logs, week_str, month_str)
        logs["summary"]["current_month"] = month_str

    # Auto-archive products older than 30 days to history_dispense.json
    archive_old_products_to_history(logs, days_threshold=30)

    # AUTOMATIC: Archive daily dispenses older than 30 days to monthly history files
    archive_old_daily_dispenses(logs, days_threshold=30)

    # Initialize product entry if doesn't exist
    if product_key not in logs["products"]:
        logs["products"][product_key] = {
            "barcode": barcode,
            "total_dispensed": 0,
            "last_dispensed": None,
            "daily_dispenses": {},
            "weekly_dispenses": {},
            "monthly_dispenses": {}
        }

    product_log = logs["products"][product_key]

    # Ensure monthly_dispenses exists (for backward compatibility)
    if "monthly_dispenses" not in product_log:
        product_log["monthly_dispenses"] = {}

    # Update totals
    product_log["total_dispensed"] += quantity
    product_log["last_dispensed"] = current_date.isoformat()

    # Update daily dispenses
    if date_str not in product_log["daily_dispenses"]:
        product_log["daily_dispenses"][date_str] = 0
    product_log["daily_dispenses"][date_str] += quantity

    # Update weekly dispenses
    if week_str not in product_log["weekly_dispenses"]:
        product_log["weekly_dispenses"][week_str] = 0
    product_log["weekly_dispenses"][week_str] += quantity

    # Update monthly dispenses
    if month_str not in product_log["monthly_dispenses"]:
        product_log["monthly_dispenses"][month_str] = 0
    product_log["monthly_dispenses"][month_str] += quantity

    # Update summary
    logs["summary"]["total_dispenses"] += quantity
    logs["summary"]["last_updated"] = current_date.isoformat()

    # Save updated logs
    save_dispense_logs(logs)

    print(f"Dispense log updated: Product {product_id} (+{quantity})")
    print(f"  Daily total: {product_log['daily_dispenses'][date_str]}")
    print(f"  Weekly total ({week_str}): {product_log['weekly_dispenses'][week_str]}")
    print(f"  Monthly total ({month_str}): {product_log['monthly_dispenses'][month_str]}")
    print(f"  All-time total: {product_log['total_dispensed']}")

def get_shelf_id_from_vsu(vsu_id: int, virtual_units: Dict[int, 'VirtualStorageUnit']) -> int:
    """Get shelf ID from VSU"""
    if vsu_id in virtual_units:
        return virtual_units[vsu_id].shelf_id
    return -1

def get_rack_id_from_shelf(shelf_id: int, shelves: Dict[int, 'Shelf']) -> int:
    """Get rack ID from shelf"""
    if shelf_id in shelves:
        return shelves[shelf_id].rack_id
    return -1

def check_shelf_collision(shelf_id1: int, shelf_id2: int) -> bool:
    """
    Check if two shelves would cause robot collision
    
    Robots collide if they're on the same shelf
    Returns True if collision would occur
    """
    return shelf_id1 == shelf_id2

def detect_obstructions(
    item_id: int,
    items: Dict[int, 'Item'],
    virtual_units: Dict[int, 'VirtualStorageUnit']
) -> List[int]:
    """
    Detect if an item is blocked by DIFFERENT PRODUCT items in the same VSU

    Same product stacking is OK (pick front-to-back naturally)
    Only DIFFERENT products blocking = obstruction requiring relocation

    Returns: List of item_ids that must be removed first (only DIFFERENT products)
    """
    if item_id not in items:
        return []

    item = items[item_id]

    if item.vsu_id is None or item.vsu_id not in virtual_units:
        return []

    target_product_id = item.metadata.product_id
    vsu = virtual_units[item.vsu_id]
    obstructions = []

    # Find all DIFFERENT PRODUCT items in front of target item
    for blocking_item_id in vsu.items:
        if blocking_item_id not in items:
            continue

        blocking_item = items[blocking_item_id]

        # If blocking item has lower stock_index AND is a DIFFERENT product
        if (blocking_item.stock_index < item.stock_index and
            blocking_item.metadata.product_id != target_product_id):
            obstructions.append(blocking_item_id)

    # Sort by stock_index (front to back)
    obstructions.sort(key=lambda id: items[id].stock_index)

    return obstructions

def find_items_for_product(
    product_id: int,
    quantity: int,
    items: Dict[int, 'Item'],
    virtual_units: Dict[int, 'VirtualStorageUnit']
) -> List[Tuple[int, int]]:
    """
    Find items for a product with MULTI-PICK and FRONT-BOX priority

    1. Prioritize VSUs with more items (fewer trips)
    2. Pick all items from same VSU together
    3. Pick front items first (stock_index 0, 1, 2...)

    Returns: List of (item_id, stock_index) tuples
    """
    available_items = []

    for item_id, item in items.items():
        if (item.metadata.product_id == product_id and
            item.vsu_id is not None and
            item.vsu_id in virtual_units):

            vsu = virtual_units[item.vsu_id]
            available_items.append({
                "item_id": item_id,
                "stock_index": item.stock_index,
                "vsu_id": item.vsu_id,
                "position": vsu.position
            })

    if len(available_items) < quantity:
        return []

    vsu_item_counts = defaultdict(int)
    for item in available_items:
        vsu_item_counts[item["vsu_id"]] += 1

    available_items.sort(key=lambda x: (
        -vsu_item_counts[x["vsu_id"]],
        x["vsu_id"],
        x["stock_index"]
    ))

    return [(item["item_id"], item["stock_index"]) for item in available_items[:quantity]]

def allocate_robots_to_shelves(
    item_locations: List[Dict],
    robots: Dict[str, 'Robot'],
    virtual_units: Dict[int, 'VirtualStorageUnit'],
    shelves: Dict[int, 'Shelf']
) -> Dict[str, List[Dict]]:
    """
    Allocate robots to items, ensuring no shelf collisions

    Algorithm:
    1. Group items by shelf
    2. Assign different shelves to different robots (round-robin)
    3. Balance load between robots

    Returns: {robot_id: [item_assignments]}
    """
    shelf_groups = defaultdict(list)
    for item_info in item_locations:
        vsu_id = item_info["vsu_id"]
        if vsu_id in virtual_units:
            shelf_id = virtual_units[vsu_id].shelf_id
            shelf_groups[shelf_id].append(item_info)

    available_robots = sorted([
        robot_id for robot_id, robot in robots.items()
        if robot.status in ["IDLE", "READY"]
    ])

    if not available_robots:
        raise HTTPException(status_code=503, detail="No robots available")

    print(f"\n[ROBOT ALLOCATION] Available robots: {available_robots}")
    print(f"[ROBOT ALLOCATION] Shelves with items: {list(shelf_groups.keys())}")

    sorted_shelves = sorted(shelf_groups.items(), key=lambda x: len(x[1]), reverse=True)

    robot_assignments = {robot_id: [] for robot_id in available_robots}
    shelf_to_robot = {}

    for idx, (shelf_id, items_in_shelf) in enumerate(sorted_shelves):
        robot_idx = idx % len(available_robots)
        assigned_robot = available_robots[robot_idx]

        robot_assignments[assigned_robot].extend(items_in_shelf)
        shelf_to_robot[shelf_id] = assigned_robot

        print(f"[ROBOT ALLOCATION] Shelf {shelf_id} ({len(items_in_shelf)} items) -> {assigned_robot}")

    for robot_id, items in robot_assignments.items():
        if items:
            print(f"[ROBOT ALLOCATION] {robot_id}: {len(items)} items")

    return {r: items for r, items in robot_assignments.items() if items}

def generate_task_id() -> str:
    """Generate unique task ID"""
    global dispense_task_counter
    dispense_task_counter += 1
    return f"DISP-{datetime.now().strftime('%Y%m%d')}-{dispense_task_counter:04d}"

def sort_items_by_proximity(item_list: List[Dict], items: Dict[int, 'Item'], virtual_units: Dict[int, 'VirtualStorageUnit']) -> List[Dict]:
    """
    Sort items by proximity to minimize robot travel distance

    Strategy:
    1. Group by shelf (same shelf items together)
    2. Within shelf, sort by X position (left to right)
    """
    def get_sort_key(item_info):
        item = items[item_info["item_id"]]
        vsu = virtual_units[item.vsu_id]
        # Sort by: shelf_id (primary), then x position (secondary)
        return (vsu.shelf_id, vsu.position.x)

    return sorted(item_list, key=get_sort_key)

def create_batched_dispense_instructions(
    robot_assignments: Dict[str, List[Dict]],
    items: Dict[int, 'Item'],
    virtual_units: Dict[int, 'VirtualStorageUnit'],
    shelves: Dict[int, 'Shelf'],
    racks: Dict[int, 'Rack'],
    output_positions: List,
    task_counter: int = 0,
    relocate_tasks_store: Dict = None
) -> Tuple[List[DispenseInstruction], int, Dict]:
    """
    Create dispense instructions for robots with obstruction handling

    Rules:
    - Relocations: 1 item per trip, NO output position, NO trip number
    - Picks: Batched by VSU (all items from same VSU = 1 trip), WITH output position, WITH trip number

    Returns:
        Tuple of (List of DispenseInstruction, updated task_counter, updated relocate_tasks_store)
    """
    if relocate_tasks_store is None:
        relocate_tasks_store = {}

    instructions = []

    for robot_id, item_list in robot_assignments.items():
        sorted_items = sort_items_by_proximity(item_list, items, virtual_units)

        relocations = [item for item in sorted_items if item.get("action") == "relocate"]
        picks = [item for item in sorted_items if item.get("action") != "relocate"]

        print(f"\n   Robot {robot_id}: {len(relocations)} relocations + {len(picks)} picks")

        for item_info in relocations:
            item_id = item_info["item_id"]
            item = items[item_id]
            vsu_id = item.vsu_id
            vsu = virtual_units[vsu_id]
            shelf = shelves[vsu.shelf_id]
            rack = racks[shelf.rack_id]

            task_counter += 1
            relocate_task_id = f"RELOCATE-{task_counter:03d}"

            relocate_tasks_store[relocate_task_id] = {
                "item_id": item_id,
                "barcode": item.metadata.barcode,
                "product_id": item.metadata.product_id,
                "vsu_code": vsu.code,
                "vsu_id": vsu_id
            }

            z_start, z_end = calc_z_positions(vsu, item, items)
            dispense_item = DispenseItem(
                item_id=item_id,
                product_id=item.metadata.product_id,
                barcode=item.metadata.barcode,
                vsu_code=vsu.code,
                shelf_name=shelf.name,
                rack_name=rack.name,
                coordinates={
                    "x": vsu.position.x,
                    "y": vsu.position.y,
                    "z": vsu.position.z
                },
                stock_index=item.stock_index,
                batch=item.metadata.batch if hasattr(item.metadata, 'batch') else None,
                expiration=item.metadata.expiration.isoformat() if hasattr(item.metadata, 'expiration') and item.metadata.expiration else None,
                z_start=z_start,
                z_end=z_end,
                action="relocate",
                reason="obstruction_removal",
                temp_endpoint="/api/temporary/relocate",
                relocate_task_id=relocate_task_id
            )

            instruction = DispenseInstruction(
                robot_id=robot_id,
                trip_number=None,
                items=[dispense_item],
                output_position=None
            )
            instructions.append(instruction)

        vsu_groups = defaultdict(list)
        for item_info in picks:
            item_id = item_info["item_id"]
            item = items[item_id]
            vsu_groups[item.vsu_id].append(item_info)

        sorted_vsu_ids = sorted(vsu_groups.keys(), key=lambda v: -len(vsu_groups[v]))

        trip_number = 1
        for vsu_id in sorted_vsu_ids:
            batch = vsu_groups[vsu_id]

            dispense_items = []
            vsu = virtual_units[vsu_id]
            shelf = shelves[vsu.shelf_id]
            rack = racks[shelf.rack_id]

            for item_info in batch:
                item_id = item_info["item_id"]
                item = items[item_id]
                z_start, z_end = calc_z_positions(vsu, item, items)

                dispense_item = DispenseItem(
                    item_id=item_id,
                    product_id=item.metadata.product_id,
                    barcode=item.metadata.barcode,
                    vsu_code=vsu.code,
                    shelf_name=shelf.name,
                    rack_name=rack.name,
                    coordinates={
                        "x": vsu.position.x,
                        "y": vsu.position.y,
                        "z": vsu.position.z
                    },
                    stock_index=item.stock_index,
                    batch=item.metadata.batch if hasattr(item.metadata, 'batch') else None,
                    expiration=item.metadata.expiration.isoformat() if hasattr(item.metadata, 'expiration') and item.metadata.expiration else None,
                    z_start=z_start,
                    z_end=z_end,
                    action="pick",
                    reason="fulfill_order",
                    temp_endpoint=None
                )
                dispense_items.append(dispense_item)

            vsu_position = {
                "x": vsu.position.x,
                "y": vsu.position.y,
                "z": 0
            }

            nearest_output = find_nearest_output_for_batch(vsu_position, output_positions)

            instruction = DispenseInstruction(
                robot_id=robot_id,
                trip_number=trip_number,
                items=dispense_items,
                output_position={
                    "x": nearest_output.x,
                    "y": nearest_output.y,
                    "z": nearest_output.z
                }
            )
            instructions.append(instruction)

            print(f"      Trip {trip_number}: {len(batch)} items from {vsu.code} -> Output ({nearest_output.x}, {nearest_output.y})")
            trip_number += 1

    return instructions, task_counter, relocate_tasks_store

def find_nearest_output_for_batch(avg_position: Dict[str, float], output_positions: List) -> 'Position':
    """Find nearest output position to a batch's average pick location"""
    if not output_positions:
        raise ValueError("No output positions available")

    nearest = output_positions[0]
    min_distance = manhattan_distance(avg_position, nearest)

    for output_pos in output_positions[1:]:
        distance = manhattan_distance(avg_position, output_pos)
        if distance < min_distance:
            min_distance = distance
            nearest = output_pos

    return nearest

def assign_smart_trip_sequence(instructions: List[DispenseInstruction], robots: Dict[str, 'Robot']) -> List[DispenseInstruction]:
    """
    Assign global trip numbers with smart ordering (fastest robot first)

    Strategy:
    1. Calculate distance for each robot's first trip
    2. Order robots by distance (shortest = fastest)
    3. Interleave trips: Fastest -> Others -> Fastest -> Others
    4. Assign sequential global trip_number

    Args:
        instructions: List of all dispense instructions
        robots: Dictionary of robot objects with current positions

    Returns:
        Instructions with renumbered trip_number in optimized sequence
    """

    robot_groups = defaultdict(list)
    for inst in instructions:
        robot_groups[inst.robot_id].append(inst)

    robot_priorities = []

    for robot_id, trips in robot_groups.items():
        if not trips or robot_id not in robots:
            continue

        first_trip = trips[0]
        robot_current_pos = robots[robot_id].position

        total_distance = 0

        first_item_pos = first_trip.items[0].coordinates
        total_distance += manhattan_distance(
            {"x": robot_current_pos.x, "y": robot_current_pos.y},
            first_item_pos
        )

        for i in range(len(first_trip.items) - 1):
            pos1 = first_trip.items[i].coordinates
            pos2 = first_trip.items[i + 1].coordinates
            total_distance += manhattan_distance(pos1, pos2)

        last_item_pos = first_trip.items[-1].coordinates
        output_pos = first_trip.output_position
        if output_pos is not None:
            total_distance += manhattan_distance(last_item_pos, output_pos)

        robot_priorities.append({
            'robot_id': robot_id,
            'distance': total_distance
        })

    robot_priorities.sort(key=lambda x: (x['distance'], x['robot_id']))
    sorted_robot_ids = [r['robot_id'] for r in robot_priorities]

    trip_sequence = []
    max_trips = max(len(trips) for trips in robot_groups.values())

    for trip_round in range(max_trips):
        for robot_id in sorted_robot_ids:
            if trip_round < len(robot_groups[robot_id]):
                trip_sequence.append(robot_groups[robot_id][trip_round])

    trip_counter = 1
    for inst in trip_sequence:
        if inst.output_position is not None:
            inst.trip_number = trip_counter
            trip_counter += 1

    print(f"\n   Smart trip ordering (distance-based):")
    for priority in robot_priorities:
        print(f"      {priority['robot_id']}: {priority['distance']:.1f}mm total distance")
    print(f"      -> {sorted_robot_ids[0]} goes first (shortest distance)")

    trip_seq_str = ' -> '.join([f"{inst.robot_id} T{inst.trip_number}" for inst in trip_sequence if inst.trip_number is not None])
    if trip_seq_str:
        print(f"\n   Trip sequence: {trip_seq_str}")

    return instructions

def resolve_output_conflicts(instructions: List[DispenseInstruction], manual_output_id: Optional[int] = None, output_positions: List = None, robots: Dict[str, 'Robot'] = None) -> List[DispenseInstruction]:
    """
    Apply output position overrides and smart trip sequencing

    Strategy:
    1. If manual_output_id specified:
       - Override all outputs to the specified output
       - Apply spatial offsets (R1: -100mm, R2: +100mm)
       - Smart trip sequencing applied (fastest robot first, then interleave)

    2. If auto-select (manual_output_id=None):
       - Keep nearest outputs (already assigned in batching)
       - Smart trip sequencing applied

    Args:
        instructions: List of all dispense instructions
        manual_output_id: Specific output to use (None = auto-select nearest)
        output_positions: List of available outputs
        robots: Dictionary of robot objects (for smart sequencing)

    Returns:
        Updated instructions with optimized trip_number sequence
    """

    if manual_output_id is not None and output_positions:
        target_output = output_positions[manual_output_id]

        robot_groups = {}
        for inst in instructions:
            if inst.robot_id not in robot_groups:
                robot_groups[inst.robot_id] = []
            robot_groups[inst.robot_id].append(inst)

        robot_ids = sorted(robot_groups.keys())
        total_robots = len(robot_ids)

        print(f"\n   Manual output mode: Output #{manual_output_id}")
        print(f"      Base output: ({target_output.x}, {target_output.y}, {target_output.z})")

        for robot_index, robot_id in enumerate(robot_ids):
            offset_position = calculate_robot_output_offset(
                robot_id,
                {"x": target_output.x, "y": target_output.y, "z": target_output.z},
                robot_index,
                total_robots
            )

            for inst in robot_groups[robot_id]:
                inst.output_position = offset_position

            print(f"      {robot_id}: ({offset_position['x']}, {offset_position['y']}, {offset_position['z']})")

    if robots:
        assign_smart_trip_sequence(instructions, robots)

    return instructions

def manhattan_distance(pos1, pos2) -> float:
    """Calculate Manhattan distance between two positions"""
    x1 = pos1.x if hasattr(pos1, 'x') else pos1['x']
    y1 = pos1.y if hasattr(pos1, 'y') else pos1['y']
    x2 = pos2.x if hasattr(pos2, 'x') else pos2['x']
    y2 = pos2.y if hasattr(pos2, 'y') else pos2['y']
    return abs(x1 - x2) + abs(y1 - y2)

def calculate_robot_output_offset(robot_id: str, base_output_pos: Dict[str, float], robot_index: int, total_robots: int) -> Dict[str, float]:
    """Calculate offset position for robot at output to avoid collision"""
    OFFSET_DISTANCE = 50

    if total_robots == 1:
        return base_output_pos.copy()

    offset_x = 0
    if robot_index == 0:
        offset_x = -OFFSET_DISTANCE
    elif robot_index == 1:
        offset_x = OFFSET_DISTANCE
    else:
        offset_x = (robot_index - (total_robots - 1) / 2) * OFFSET_DISTANCE

    return {
        "x": base_output_pos["x"] + offset_x,
        "y": base_output_pos["y"],
        "z": base_output_pos["z"]
    }

def find_nearest_output_for_dispense(
    instructions: List[DispenseInstruction],
    output_positions: List
):
    """Find nearest output to average pick position"""
    if not instructions:
        return output_positions[0]

    avg_x = sum(inst.coordinates['x'] for inst in instructions) / len(instructions)
    avg_y = sum(inst.coordinates['y'] for inst in instructions) / len(instructions)
    avg_pos = {'x': avg_x, 'y': avg_y}

    nearest = output_positions[0]
    min_dist = manhattan_distance(avg_pos, nearest)

    for output_pos in output_positions[1:]:
        dist = manhattan_distance(avg_pos, output_pos)
        if dist < min_dist:
            min_dist = dist
            nearest = output_pos

    return nearest

def find_optimal_outputs_per_robot(
    robot_assignments: Dict[str, List[DispenseInstruction]],
    output_positions: List
) -> Dict[str, Dict[str, float]]:
    """Find optimal output for each robot based on their assigned items"""
    robot_outputs = {}

    for robot_id, instructions in robot_assignments.items():
        if not instructions:
            continue

        avg_x = sum(inst.coordinates['x'] for inst in instructions) / len(instructions)
        avg_y = sum(inst.coordinates['y'] for inst in instructions) / len(instructions)
        avg_pos = {'x': avg_x, 'y': avg_y}

        nearest_output = output_positions[0]
        min_dist = manhattan_distance(avg_pos, nearest_output)

        for output_pos in output_positions[1:]:
            dist = manhattan_distance(avg_pos, output_pos)
            if dist < min_dist:
                min_dist = dist
                nearest_output = output_pos

        if hasattr(nearest_output, 'x'):
            robot_outputs[robot_id] = {
                "x": nearest_output.x,
                "y": nearest_output.y,
                "z": nearest_output.z
            }
        else:
            robot_outputs[robot_id] = nearest_output.copy()

    return robot_outputs

def create_dispense_task_endpoint(
    request: DispenseRequest,
    items: Dict[int, 'Item'],
    robots: Dict[str, 'Robot'],
    virtual_units: Dict[int, 'VirtualStorageUnit'],
    shelves: Dict[int, 'Shelf'],
    racks: Dict[int, 'Rack'],
    output_positions: List['Position'],
    task_counter: int = 0
):
    """Create a dispensing task"""
    try:
        if not request.products:
            raise HTTPException(status_code=400, detail="No products requested")

        all_item_locations = []
        product_summary = []
        items_as_relocations = set()

        for product_request in request.products:
            product_id = product_request.get("product_id")
            barcode = product_request.get("barcode")
            quantity = product_request.get("quantity", 1)

            if not product_id and barcode:
                for item_id, item in items.items():
                    if item.metadata.barcode == barcode and item.vsu_id is not None:
                        product_id = item.metadata.product_id
                        break

                if not product_id:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Product with barcode {barcode} not found in inventory"
                    )

            if not product_id or quantity <= 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid product request: {product_request}. Must provide product_id or barcode"
                )

            item_picks = find_items_for_product(product_id, quantity, items, virtual_units)

            if not item_picks:
                raise HTTPException(
                    status_code=404,
                    detail=f"Insufficient stock for product {product_id} (barcode: {barcode}). Requested: {quantity}"
                )

            for item_id, stock_index in item_picks:
                item = items[item_id]

                obstructions = detect_obstructions(item_id, items, virtual_units)

                if obstructions:
                    print(f"   Item {item_id} blocked by {len(obstructions)} items: {obstructions}")
                    for obstruction_id in obstructions:
                        already_added = any(loc["item_id"] == obstruction_id for loc in all_item_locations)

                        if not already_added:
                            obstruction_item = items[obstruction_id]
                            all_item_locations.append({
                                "item_id": obstruction_id,
                                "product_id": obstruction_item.metadata.product_id,
                                "stock_index": obstruction_item.stock_index,
                                "vsu_id": obstruction_item.vsu_id,
                                "barcode": obstruction_item.metadata.barcode,
                                "action": "relocate",
                                "reason": "obstruction_removal"
                            })
                            items_as_relocations.add(obstruction_id)
                        else:
                            for loc in all_item_locations:
                                if loc["item_id"] == obstruction_id and loc.get("action") == "pick":
                                    loc["action"] = "relocate"
                                    loc["reason"] = "obstruction_removal"
                                    items_as_relocations.add(obstruction_id)
                                    print(f"      -> Item {obstruction_id} changed from pick to relocate")
                                    break

                if item_id not in items_as_relocations:
                    all_item_locations.append({
                        "item_id": item_id,
                        "product_id": product_id,
                        "stock_index": stock_index,
                        "vsu_id": item.vsu_id,
                        "barcode": item.metadata.barcode,
                        "action": "pick",
                        "reason": "fulfill_order"
                    })
            
            product_summary.append({
                "product_id": product_id,
                "quantity": quantity
            })
        
        print(f"\n{'='*60}")
        print(f"DISPENSE TASK CREATION")
        print(f"{'='*60}")
        print(f"Products requested: {len(request.products)}")
        print(f"Total items to pick: {len(all_item_locations)}")

        robot_assignments = allocate_robots_to_shelves(
            all_item_locations,
            robots,
            virtual_units,
            shelves
        )
        
        print(f"Robots allocated: {list(robot_assignments.keys())}")
        for robot_id, items_list in robot_assignments.items():
            print(f"  {robot_id}: {len(items_list)} items")

        instructions, task_counter, relocate_tasks_store = create_batched_dispense_instructions(
            robot_assignments,
            items,
            virtual_units,
            shelves,
            racks,
            output_positions,
            task_counter,
            {}
        )

        instructions = resolve_output_conflicts(
            instructions,
            manual_output_id=request.output_id,
            output_positions=output_positions,
            robots=robots
        )

        robot_output_positions = {}
        for inst in instructions:
            if inst.robot_id not in robot_output_positions and inst.output_position is not None:
                robot_output_positions[inst.robot_id] = inst.output_position

        output_dict = None
        for inst in instructions:
            if inst.output_position is not None:
                output_dict = inst.output_position
                break

        if output_dict is None:
            output_dict = {"x": 650, "y": 50, "z": 0}

        task_id = generate_task_id()
        task = DispenseTask(
            task_id=task_id,
            status="pending",
            instructions=instructions,
            output_position=output_dict,
            robot_output_positions=robot_output_positions
        )

        dispense_tasks[task_id] = task

        for robot_id in robot_assignments.keys():
            robots[robot_id].status = "DISPENSING"
            robots[robot_id].current_task_id = task_id
        
        print(f"Task created: {task_id}")
        print(f"{'='*60}\n")

        total_items = sum(len(inst.items) for inst in instructions)

        response_dict = {
            "status": "success",
            "task_id": task_id,
            "products": product_summary,
            "total_items": total_items,
            "total_trips": len(instructions),
            "robots_assigned": list(robot_assignments.keys()),
            "output_position": output_dict,
            "robot_output_positions": robot_output_positions,
            "instructions": [
                {
                    "robot_id": inst.robot_id,
                    "trip_number": inst.trip_number,
                    "output_position": inst.output_position,
                    "items": [
                        {
                            "item_id": item.item_id,
                            "product_id": item.product_id,
                            "barcode": item.barcode,
                            "batch": item.batch,
                            "expiration": item.expiration,
                            "location": {
                                "rack": item.rack_name,
                                "shelf": item.shelf_name,
                                "vsu": item.vsu_code,
                                "coordinates": item.coordinates
                            },
                            "stock_index": item.stock_index,
                            "z_start": item.z_start,
                            "z_end": item.z_end,
                            "action": next((loc.get("action", "pick") for loc in all_item_locations if loc["item_id"] == item.item_id), "pick"),
                            "reason": next((loc.get("reason", "fulfill_order") for loc in all_item_locations if loc["item_id"] == item.item_id), "fulfill_order"),
                            "temp_endpoint": "/api/temporary/relocate" if next((loc.get("action") for loc in all_item_locations if loc["item_id"] == item.item_id), None) == "relocate" else None,
                            "relocate_task_id": item.relocate_task_id if hasattr(item, 'relocate_task_id') else None
                        }
                        for item in inst.items
                    ]
                }
                for inst in instructions
            ]
        }

        return response_dict, task_counter, relocate_tasks_store
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error creating dispense task: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create dispense task: {str(e)}")

def complete_dispense_endpoint(
    request: CompleteDispenseRequest,
    items: Dict[int, 'Item'],
    robots: Dict[str, 'Robot'],
    virtual_units: Dict[int, 'VirtualStorageUnit'],
    shelves: Dict[int, 'StorageUnit'],
    save_robots_func,
    save_warehouse_func=None,
    relocate_tasks_store: Dict = None
):
    """Mark dispensing as successful and update inventory"""
    try:
        task_id = request.task_id

        if task_id not in dispense_tasks:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        task = dispense_tasks[task_id]
        
        if task.status == "completed":
            raise HTTPException(status_code=400, detail=f"Task {task_id} already completed")
        
        if task.status == "failed":
            raise HTTPException(status_code=400, detail=f"Task {task_id} has failed")
        
        print(f"\n{'='*60}")
        print(f"COMPLETING DISPENSE TASK: {task_id}")
        print(f"{'='*60}")

        if relocate_tasks_store is None:
            relocate_tasks_store = {}

        relocations_pending = []
        for instruction in task.instructions:
            for dispense_item in instruction.items:
                item_id = dispense_item.item_id
                action = getattr(dispense_item, 'action', 'pick')
                if action == 'relocate':
                    relocate_task_id = getattr(dispense_item, 'relocate_task_id', None)

                    original_vsu_id = None
                    if relocate_task_id and relocate_tasks_store and relocate_task_id in relocate_tasks_store:
                        original_vsu_id = relocate_tasks_store[relocate_task_id].get("vsu_id")

                    if item_id in items:
                        current_vsu_id = items[item_id].vsu_id

                        if original_vsu_id is not None and current_vsu_id == original_vsu_id:
                            relocations_pending.append({
                                "item_id": item_id,
                                "barcode": getattr(dispense_item, 'barcode', 'unknown'),
                                "status": "not_relocated",
                                "original_vsu_id": original_vsu_id,
                                "current_vsu_id": current_vsu_id,
                                "reason": "Item is still in original VSU - must call /api/temporary/relocate first"
                            })
                    else:
                        relocations_pending.append({
                            "item_id": item_id,
                            "barcode": getattr(dispense_item, 'barcode', 'unknown'),
                            "status": "missing",
                            "reason": "Item not found in inventory"
                        })

        if relocations_pending:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Cannot complete dispense: relocations not finished",
                    "message": f"{len(relocations_pending)} item(s) marked for relocation have not been properly relocated",
                    "pending_relocations": relocations_pending,
                    "instruction": "Call POST /api/temporary/relocate with task_id for each obstructing item before completing dispense"
                }
            )

        product_quantities = defaultdict(int)
        product_barcodes = {}

        items_removed = []

        from product_archive import archive_dispensed_item

        for instruction in task.instructions:
            for dispense_item in instruction.items:
                item_id = dispense_item.item_id
                action = getattr(dispense_item, 'action', 'pick')

                if action == 'relocate':
                    print(f"  Skipping item {item_id} (already relocated to temp storage)")
                    continue

                if item_id not in items:
                    print(f"WARNING: Item {item_id} not found, skipping")
                    continue

                item = items[item_id]
                product_id = item.metadata.product_id
                barcode = item.metadata.barcode

                product_quantities[product_id] += 1
                product_barcodes[product_id] = barcode

                vsu = None
                shelf_id = None
                shelf_name = "Unknown"

                if item.vsu_id and item.vsu_id in virtual_units:
                    vsu = virtual_units[item.vsu_id]
                    shelf_id = vsu.shelf_id

                    if shelf_id and shelf_id in shelves:
                        shelf = shelves[shelf_id]
                        shelf_name = shelf.name

                    archive_dispensed_item(
                        item_id=item_id,
                        item=item,
                        vsu=vsu,
                        shelf_id=shelf_id,
                        shelf_name=shelf_name,
                        task_id=task_id
                    )

                    if item_id in vsu.items:
                        vsu.items.remove(item_id)
                    if not vsu.items:
                        vsu.occupied = False

                items_removed.append({
                    "item_id": item_id,
                    "product_id": product_id,
                    "barcode": barcode
                })
                del items[item_id]
        
        print(f"Items removed from inventory: {len(items_removed)}")

        for product_id, quantity in product_quantities.items():
            barcode = product_barcodes[product_id]
            update_dispense_log(product_id, barcode, quantity)

        task.status = "completed"
        task.completed_at = datetime.now()

        robots_freed = []

        robot_ids_in_task = list(set(inst.robot_id for inst in task.instructions))
        total_robots = len(robot_ids_in_task)

        if task.robot_output_positions:
            output_to_robots = {}
            for robot_id in robot_ids_in_task:
                if robot_id in task.robot_output_positions:
                    output_pos = task.robot_output_positions[robot_id]
                    output_key = (output_pos['x'], output_pos['y'], output_pos['z'])
                    if output_key not in output_to_robots:
                        output_to_robots[output_key] = []
                    output_to_robots[output_key].append(robot_id)

            final_robot_positions = {}
            for output_key, robots_at_output in output_to_robots.items():
                base_output_dict = {"x": output_key[0], "y": output_key[1], "z": output_key[2]}

                if len(robots_at_output) == 1:
                    final_robot_positions[robots_at_output[0]] = base_output_dict
                else:
                    robot_index_map = {robot_id: idx for idx, robot_id in enumerate(sorted(robots_at_output))}
                    for robot_id in robots_at_output:
                        robot_index = robot_index_map[robot_id]
                        offset_pos = calculate_robot_output_offset(
                            robot_id,
                            base_output_dict,
                            robot_index,
                            len(robots_at_output)
                        )
                        final_robot_positions[robot_id] = offset_pos

            for robot_id, final_pos in final_robot_positions.items():
                if robot_id in robots:
                    robots[robot_id].status = "IDLE"
                    robots[robot_id].current_task_id = None
                    robots[robot_id].position.x = final_pos["x"]
                    robots[robot_id].position.y = final_pos["y"]
                    robots[robot_id].position.z = final_pos["z"]
                    robots_freed.append(robot_id)

            print(f"Robots freed at OUTPUTS (per-robot optimization):")
            for robot_id in sorted(final_robot_positions.keys()):
                final_pos = final_robot_positions[robot_id]
                base_output = task.robot_output_positions.get(robot_id, final_pos)
                if final_pos == base_output:
                    print(f"  {robot_id} -> ({final_pos['x']}, {final_pos['y']}, {final_pos['z']})")
                else:
                    print(f"  {robot_id} -> ({final_pos['x']}, {final_pos['y']}, {final_pos['z']}) [offset from ({base_output['x']}, {base_output['y']}, {base_output['z']})]")

        else:
            base_output_pos = task.output_position

            robot_index_map = {robot_id: idx for idx, robot_id in enumerate(sorted(robot_ids_in_task))}

            robot_output_positions = {}
            for robot_id in robot_ids_in_task:
                robot_index = robot_index_map[robot_id]
                offset_pos = calculate_robot_output_offset(
                    robot_id,
                    base_output_pos,
                    robot_index,
                    total_robots
                )
                robot_output_positions[robot_id] = offset_pos

            for robot_id, offset_pos in robot_output_positions.items():
                if robot_id in robots:
                    robots[robot_id].status = "IDLE"
                    robots[robot_id].current_task_id = None
                    robots[robot_id].position.x = offset_pos["x"]
                    robots[robot_id].position.y = offset_pos["y"]
                    robots[robot_id].position.z = offset_pos["z"]
                    robots_freed.append(robot_id)

                    if total_robots > 1:
                        offset_info = f"offset ({offset_pos['x']}, {offset_pos['y']}, {offset_pos['z']})"
                        print(f"  {robot_id} -> {offset_info}")

            if total_robots > 1:
                print(f"Robots freed at OUTPUT with collision avoidance (base: {base_output_pos['x']}, {base_output_pos['y']}, {base_output_pos['z']})")
                print(f"  Total robots: {total_robots}, Offset: 50mm on X-axis")
            else:
                print(f"Robot freed at output position ({base_output_pos['x']}, {base_output_pos['y']}, {base_output_pos['z']}): {set(robots_freed)}")

        if save_warehouse_func:
            save_warehouse_func()
            print(f"Warehouse state saved to file")
        else:
            print(f"Warning: save_warehouse_func not provided, inventory not saved to file")

        save_robots_func()

        print(f"Dispense task completed successfully")
        print(f"{'='*60}\n")
        
        return {
            "status": "success",
            "task_id": task_id,
            "items_dispensed": len(items_removed),
            "products_dispensed": [
                {
                    "product_id": pid,
                    "quantity": qty,
                    "barcode": product_barcodes[pid]
                }
                for pid, qty in product_quantities.items()
            ],
            "robots_freed": list(set(robots_freed)),
            "completed_at": task.completed_at.isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error completing dispense: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to complete dispense: {str(e)}")

def fail_dispense_endpoint(
    request: FailDispenseRequest,
    items: Dict[int, 'Item'],
    robots: Dict[str, 'Robot'],
    virtual_units: Dict[int, 'VirtualStorageUnit'],
    shelves: Dict[int, 'StorageUnit'],
    save_robots_func,
    save_warehouse_func=None
):
    """
    Mark dispensing as failed with optional partial success.

    If successful_trips is provided:
    - Items from those trips are removed from inventory (they were dispensed)
    - Items from trips NOT listed remain in inventory (they failed or weren't attempted)

    If successful_trips is empty:
    - No inventory changes (complete failure)
    """
    try:
        task_id = request.task_id

        if task_id not in dispense_tasks:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        task = dispense_tasks[task_id]

        if task.status == "completed":
            raise HTTPException(status_code=400, detail=f"Task {task_id} already completed")

        if task.status == "failed":
            raise HTTPException(status_code=400, detail=f"Task {task_id} already failed")

        successful_trips = request.successful_trips or []

        print(f"\n{'='*60}")
        print(f"FAILING DISPENSE TASK: {task_id}")
        print(f"Reason: {request.reason}")
        print(f"Successful trips: {successful_trips if successful_trips else 'None (complete failure)'}")
        print(f"{'='*60}")

        items_removed = []
        items_kept = []

        if successful_trips:
            from product_archive import archive_dispensed_item

            for instruction in task.instructions:
                trip_number = instruction.trip_number

                for dispense_item in instruction.items:
                    item_id = dispense_item.item_id
                    action = getattr(dispense_item, 'action', 'pick')

                    if action == 'relocate':
                        continue

                    if item_id not in items:
                        print(f"  WARNING: Item {item_id} not found, skipping")
                        continue

                    item = items[item_id]
                    barcode = item.metadata.barcode
                    product_id = item.metadata.product_id

                    if trip_number in successful_trips:
                        vsu = None
                        shelf_id = None
                        shelf_name = "Unknown"

                        if item.vsu_id and item.vsu_id in virtual_units:
                            vsu = virtual_units[item.vsu_id]
                            shelf_id = vsu.shelf_id

                            if shelf_id and shelf_id in shelves:
                                shelf = shelves[shelf_id]
                                shelf_name = shelf.name

                            archive_dispensed_item(
                                item_id=item_id,
                                item=item,
                                vsu=vsu,
                                shelf_id=shelf_id,
                                shelf_name=shelf_name,
                                task_id=task_id
                            )

                            if item_id in vsu.items:
                                vsu.items.remove(item_id)
                            if not vsu.items:
                                vsu.occupied = False

                        items_removed.append({
                            "item_id": item_id,
                            "product_id": product_id,
                            "barcode": barcode,
                            "trip_number": trip_number
                        })
                        del items[item_id]
                        print(f"  Trip {trip_number}: REMOVED item {item_id} ({barcode})")
                    else:
                        items_kept.append({
                            "item_id": item_id,
                            "product_id": product_id,
                            "barcode": barcode,
                            "trip_number": trip_number
                        })
                        print(f"  Trip {trip_number}: KEPT item {item_id} ({barcode}) - not in successful_trips")

            if save_warehouse_func:
                save_warehouse_func()

            task.status = "partial"
        else:
            task.status = "failed"

        task.failure_reason = request.reason
        task.completed_at = datetime.now()

        robots_freed = []
        robot_ids_in_task = list(set(inst.robot_id for inst in task.instructions))
        total_robots = len(robot_ids_in_task)

        if task.robot_output_positions:
            output_to_robots = {}
            for robot_id in robot_ids_in_task:
                if robot_id in task.robot_output_positions:
                    output_pos = task.robot_output_positions[robot_id]
                    output_key = (output_pos['x'], output_pos['y'], output_pos['z'])
                    if output_key not in output_to_robots:
                        output_to_robots[output_key] = []
                    output_to_robots[output_key].append(robot_id)

            final_robot_positions = {}
            for output_key, robots_at_output in output_to_robots.items():
                base_output_dict = {"x": output_key[0], "y": output_key[1], "z": output_key[2]}

                if len(robots_at_output) == 1:
                    final_robot_positions[robots_at_output[0]] = base_output_dict
                else:
                    robot_index_map = {robot_id: idx for idx, robot_id in enumerate(sorted(robots_at_output))}
                    for robot_id in robots_at_output:
                        robot_index = robot_index_map[robot_id]
                        offset_pos = calculate_robot_output_offset(
                            robot_id,
                            base_output_dict,
                            robot_index,
                            len(robots_at_output)
                        )
                        final_robot_positions[robot_id] = offset_pos

            for robot_id, final_pos in final_robot_positions.items():
                if robot_id in robots:
                    robots[robot_id].status = "IDLE"
                    robots[robot_id].current_task_id = None
                    robots[robot_id].position.x = final_pos["x"]
                    robots[robot_id].position.y = final_pos["y"]
                    robots[robot_id].position.z = final_pos["z"]
                    robots_freed.append(robot_id)

            print(f"Robots freed at OUTPUTS (per-robot positions):")
            for robot_id in sorted(final_robot_positions.keys()):
                final_pos = final_robot_positions[robot_id]
                print(f"  {robot_id} -> ({final_pos['x']}, {final_pos['y']}, {final_pos['z']})")

        else:
            base_output_pos = task.output_position

            robot_index_map = {robot_id: idx for idx, robot_id in enumerate(sorted(robot_ids_in_task))}

            robot_output_positions = {}
            for robot_id in robot_ids_in_task:
                robot_index = robot_index_map[robot_id]
                offset_pos = calculate_robot_output_offset(
                    robot_id,
                    base_output_pos,
                    robot_index,
                    total_robots
                )
                robot_output_positions[robot_id] = offset_pos

            for robot_id, offset_pos in robot_output_positions.items():
                if robot_id in robots:
                    robots[robot_id].status = "IDLE"
                    robots[robot_id].current_task_id = None
                    robots[robot_id].position.x = offset_pos["x"]
                    robots[robot_id].position.y = offset_pos["y"]
                    robots[robot_id].position.z = offset_pos["z"]
                    robots_freed.append(robot_id)

            print(f"Robots freed at OUTPUT positions:")
            for robot_id, offset_pos in robot_output_positions.items():
                print(f"  {robot_id} -> ({offset_pos['x']}, {offset_pos['y']}, {offset_pos['z']})")

        save_robots_func()

        print(f"\nSummary:")
        print(f"  Items removed (successful trips): {len(items_removed)}")
        print(f"  Items kept (failed/not attempted): {len(items_kept)}")
        print(f"  Task status: {task.status}")
        print(f"{'='*60}\n")

        return {
            "status": "success",
            "task_id": task_id,
            "task_status": task.status,
            "message": f"Task marked as {task.status}",
            "successful_trips": successful_trips,
            "items_removed": items_removed,
            "items_kept": items_kept,
            "robots_freed": list(set(robots_freed)),
            "failure_reason": request.reason,
            "failed_at": task.completed_at.isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error failing dispense: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to mark dispense as failed: {str(e)}")

def get_dispense_logs_endpoint():
    """
    GET DISPENSE LOGS
    
    Returns comprehensive dispense statistics
    """
    try:
        logs = load_dispense_logs()
        
        # Calculate statistics
        total_products = len(logs["products"])
        total_dispenses = logs["summary"]["total_dispenses"]
        
        # Get today's dispenses
        today = datetime.now().date().isoformat()
        today_total = sum(
            product_data["daily_dispenses"].get(today, 0)
            for product_data in logs["products"].values()
        )
        
        # Get this week's dispenses
        current_week = get_week_number(datetime.now())
        week_total = sum(
            product_data["weekly_dispenses"].get(current_week, 0)
            for product_data in logs["products"].values()
        )
        
        # Top products (by total dispensed)
        top_products = sorted(
            [
                {
                    "product_id": int(pid),
                    "barcode": data["barcode"],
                    "total_dispensed": data["total_dispensed"],
                    "last_dispensed": data["last_dispensed"]
                }
                for pid, data in logs["products"].items()
            ],
            key=lambda x: x["total_dispensed"],
            reverse=True
        )[:10]
        
        return {
            "status": "success",
            "summary": {
                "total_products_tracked": total_products,
                "total_dispenses_all_time": total_dispenses,
                "dispenses_today": today_total,
                "dispenses_this_week": week_total,
                "last_updated": logs["summary"]["last_updated"]
            },
            "top_products": top_products
        }
    
    except Exception as e:
        print(f"Error getting dispense logs: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get dispense logs: {str(e)}")

def get_product_dispense_history_endpoint(product_id: int):
    """
    GET PRODUCT DISPENSE HISTORY

    Get detailed history for a specific product
    Combines data from active logs (last 30 days) + monthly archives
    """
    try:
        logs = load_dispense_logs()
        product_key = str(product_id)

        # Get archived history from monthly files
        archived_history = get_product_history_from_archives(product_key, months_back=12)

        # Combine active + archived data
        combined_daily = {}
        combined_monthly = {}

        # Add archived data first
        combined_daily.update(archived_history["daily_dispenses"])
        combined_monthly.update(archived_history["monthly_totals"])

        # Get active product data
        if product_key in logs["products"]:
            product_data = logs["products"][product_key]

            # Merge active daily dispenses (overwrites archive if duplicate - shouldn't happen)
            combined_daily.update(product_data.get("daily_dispenses", {}))

            # Merge monthly data
            combined_monthly.update(product_data.get("monthly_dispenses", {}))

            barcode = product_data["barcode"]
            total_dispensed = product_data["total_dispensed"]
            last_dispensed = product_data["last_dispensed"]
        else:
            # Product only exists in archives (not dispensed recently)
            if not archived_history["daily_dispenses"]:
                raise HTTPException(status_code=404, detail=f"No dispense history for product {product_id}")

            barcode = "ARCHIVED"
            total_dispensed = archived_history["total_from_archives"]
            last_dispensed = max(archived_history["daily_dispenses"].keys()) if archived_history["daily_dispenses"] else None

        # Sort combined data
        daily_sorted = sorted(
            combined_daily.items(),
            key=lambda x: x[0],
            reverse=True
        )[:60]  # Last 60 days (30 active + 30 archive)

        monthly_sorted = sorted(
            combined_monthly.items(),
            key=lambda x: x[0],
            reverse=True
        )[:12]  # Last 12 months

        return {
            "status": "success",
            "product_id": product_id,
            "barcode": barcode,
            "total_dispensed": total_dispensed,
            "last_dispensed": last_dispensed,
            "daily_dispenses": [
                {"date": date, "quantity": qty}
                for date, qty in daily_sorted
            ],
            "monthly_dispenses": [
                {"month": month, "quantity": qty}
                for month, qty in monthly_sorted
            ],
            "data_sources": {
                "active_log_days": len([d for d in combined_daily.keys() if d not in archived_history["daily_dispenses"]]),
                "archived_days": len(archived_history["daily_dispenses"]),
                "total_days": len(combined_daily)
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting product history: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get product history: {str(e)}")

def get_task_status_endpoint(task_id: str):
    """
    GET TASK STATUS
    
    Check status of a dispense task
    """
    try:
        if task_id not in dispense_tasks:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        task = dispense_tasks[task_id]
        
        return {
            "status": "success",
            "task_id": task_id,
            "task_status": task.status,
            "created_at": task.created_at.isoformat(),
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "failure_reason": task.failure_reason,
            "items_count": len(task.instructions),
            "robots": list(set(inst.robot_id for inst in task.instructions))
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")
