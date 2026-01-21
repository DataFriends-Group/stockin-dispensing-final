from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
from contextlib import asynccontextmanager
import hashlib
import json
import random
import re
import shutil
import traceback

from pathlib import Path

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup diagnostics and scheduler"""
    import os
    from scheduler import start_scheduler, stop_scheduler

    # Startup
    print("\n" + "="*60)
    print("STARTUP DIAGNOSTICS")
    print("="*60)
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in directory: {[f for f in os.listdir('.') if not f.startswith('.')]}")
    print(f"Racks loaded: {len(racks)}")
    print(f"Shelves loaded: {len(shelves)}")
    print(f"VSUs loaded: {len(virtual_units)}")
    print(f"Items loaded: {len(items)}")
    print(f"Robots loaded: {len(robots)}")
    print("="*60)

    # Start daily archiving scheduler
    print("\n" + "="*60)
    print("STARTING AUTOMATIC ARCHIVING SCHEDULER")
    print("="*60)
    start_scheduler()
    print("="*60 + "\n")

    yield

    # Shutdown
    print("\n" + "="*60)
    print("SHUTTING DOWN SCHEDULER")
    print("="*60)
    stop_scheduler()
    print("="*60 + "\n")
    

app = FastAPI(title="Sequential Stock-in System - MedicPort", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VSU_HORIZONTAL_GAP = 5  # 5mm gap between VSUs (X-axis)
VSU_TOP_CLEARANCE = 3  # 3mm gap from item top to shelf
WIDTH_TOLERANCE = 5  # ±5mm tolerance for width matching when stacking
HEIGHT_TOLERANCE = 5  # ±5mm tolerance for height matching when stacking
DEPTH_GAP_BETWEEN_ITEMS = 3  # 3mm gap between items in same VSU (depth)

def safe_parse_datetime(dt_string: str) -> datetime:
    try:
        return datetime.fromisoformat(dt_string)
    except ValueError:
        fixed_string = re.sub(r'\.(\d{6})\d+', r'.\1', dt_string)
        try:
            return datetime.fromisoformat(fixed_string)
        except:
            fixed_string = re.sub(r'\.\d+', '', dt_string)
            return datetime.fromisoformat(fixed_string)

class Position(BaseModel):
    x: float
    y: float
    z: float = 0

class Dimensions(BaseModel):
    width: float
    height: float
    depth: float
    
    @property
    def volume(self):
        return self.width * self.height * self.depth

class ScannerInput(BaseModel):
    Barcode: str
    Batch: str
    Exp: str
    Width: float
    Height: float
    Depth: float
    Weight: float = 0
    DeliveryId: Optional[str] = None

class VirtualStorageUnit(BaseModel):
    id: int
    code: str
    dimensions: Dimensions
    position: Position
    shelf_id: int
    rack_id: int
    items: List[int] = []
    occupied: bool = False

class Shelf(BaseModel):
    id: int
    rack_id: int
    name: str
    dimensions: Dimensions
    position: Position
    virtual_units: List[int] = []

class Rack(BaseModel):
    id: int
    name: str
    shelf_ids: List[int] = []

class Robot(BaseModel):
    id: str
    position: Position
    status: str = "IDLE"
    battery: int = 100
    current_task_id: Optional[str] = None

class ItemMetadata(BaseModel):
    product_id: int
    barcode: str
    dimensions: Dimensions
    weight: float
    expiration: datetime
    batch: str
    delivery_id: str
    warehouse_id: int = 1

class Item(BaseModel):
    id: int
    metadata: ItemMetadata
    vsu_id: Optional[int] = None
    stock_index: int = 0

class Task(BaseModel):
    id: str
    item_id: int
    destination_vsu_id: int
    robot_id: str
    status: str = "pending"
    score: float = 0
    created_at: datetime = Field(default_factory=datetime.now)
    is_new_vsu: bool = False
    z_position: float = 0

def load_robots_from_file():
    """Load robot data from robot_post.json"""
    try:
        with open('data/robot_post.json', 'r') as f:
            robot_data = json.load(f)
        
        loaded_robots = {}
        for robot_id, robot_info in robot_data.items():
            pos = robot_info.get('position', {})
            loaded_robots[robot_id] = Robot(
                id=robot_id,
                position=Position(
                    x=pos.get('x', 0),
                    y=pos.get('y', 0),
                    z=pos.get('z', 0)
                ),
                status=robot_info.get('status', 'IDLE'),
                battery=robot_info.get('battery', 100),
                current_task_id=robot_info.get('current_task_id')
            )
        print(f"Loaded {len(loaded_robots)} robots from robot_post.json")
        return loaded_robots
    except FileNotFoundError:
        print("robot_post.json not found, using default robots")
        return {
            "R1": Robot(id="R1", position=Position(x=400, y=100, z=0)),
            "R2": Robot(id="R2", position=Position(x=900, y=100, z=0))
        }
    except Exception as e:
        print(f"Error loading robot_post.json: {e}, using default robots")
        return {
            "R1": Robot(id="R1", position=Position(x=400, y=100, z=0)),
            "R2": Robot(id="R2", position=Position(x=900, y=100, z=0))
        }

def save_robots_to_file():
    """Save current robot states to robot_post.json"""
    try:
        robot_data = {}
        for robot_id, robot in robots.items():
            robot_data[robot_id] = {
                "position": {
                    "x": robot.position.x,
                    "y": robot.position.y,
                    "z": robot.position.z
                },
                "status": robot.status,
                "battery": robot.battery,
                "current_task_id": robot.current_task_id
            }
        
        with open('data/robot_post.json', 'w') as f:
            json.dump(robot_data, f, indent=2)
        
        print(f"Saved robot states to robot_post.json")
        return True
    except Exception as e:
        print(f"Error saving robots: {e}")
        return False

racks: Dict[int, Rack] = {}
shelves: Dict[int, Shelf] = {}
virtual_units: Dict[int, VirtualStorageUnit] = {}
items: Dict[int, Item] = {}
tasks: Dict[str, Task] = {}
pending_placements: Dict[str, dict] = {}  # {task_id: {vsu_id, stock_index, item_depth}}
task_counter = 0
item_counter = 0
vsu_counter = 28  # Start after existing VSUs

robots: Dict[str, Robot] = load_robots_from_file()

# Will be loaded from warehouse_layout.json
INPUT_POSITION = Position(x=0, y=0, z=0)
OUTPUT_POSITIONS = []

DEFAULT_WEIGHT = 0.1  # Default weight for new/unknown products
product_weights: Dict[int, float] = {}  # Product weights from ML model (0.1 to 1.0)
progress = {"total_boxes": 0, "completed": 0, "failed": 0, "current_box": 0}


def load_warehouse_layout():
    """Load input/output positions from warehouse_layout.json"""
    global INPUT_POSITION, OUTPUT_POSITIONS

    try:
        with open('data/warehouse_layout.json', 'r') as f:
            layout_data = json.load(f)

        # Load input positions
        input_positions = layout_data.get('input_positions', [])
        if input_positions:
            active_inputs = [p for p in input_positions if p.get('active', True)]
            if active_inputs:
                first_input = active_inputs[0]
                coords = first_input.get('coordinates', {})
                INPUT_POSITION = Position(
                    x=coords.get('x', 0),
                    y=coords.get('y', 0),
                    z=coords.get('z', 0)
                )
                print(f"Loaded input position: {first_input.get('name', 'Input')} at ({INPUT_POSITION.x}, {INPUT_POSITION.y}, {INPUT_POSITION.z})")

        # Load output positions
        output_positions = layout_data.get('output_positions', [])
        OUTPUT_POSITIONS = []
        for output in output_positions:
            if output.get('active', True):
                coords = output.get('coordinates', {})
                pos = Position(
                    x=coords.get('x', 650),
                    y=coords.get('y', 50),
                    z=coords.get('z', 0)
                )
                OUTPUT_POSITIONS.append(pos)
                print(f"Loaded output position: {output.get('name', 'Output')} at ({pos.x}, {pos.y}, {pos.z})")

        if not OUTPUT_POSITIONS:
            # Fallback to default
            OUTPUT_POSITIONS.append(Position(x=650, y=50, z=0))
            print("No active outputs found, using default output position")

        print(f"Warehouse layout loaded: {len(OUTPUT_POSITIONS)} output position(s)")

    except FileNotFoundError:
        print("warehouse_layout.json not found - using default positions")
        OUTPUT_POSITIONS.append(Position(x=650, y=50, z=0))
    except Exception as e:
        print(f"Error loading warehouse layout: {e}")
        OUTPUT_POSITIONS.append(Position(x=650, y=50, z=0))

def load_product_weights():
    """Load product weights from weights.json (updated by ML model during optimization)"""
    global product_weights
    
    try:
        with open('data/weights.json', 'r') as f:
            weights_data = json.load(f)
        
        # Expected format: {"product_weights": {"1": 0.8, "2": 0.5, ...}}
        if "product_weights" in weights_data:
            for product_id_str, weight in weights_data["product_weights"].items():
                # Skip non-numeric keys (like "comment", "updated_at", etc.)
                try:
                    product_id = int(product_id_str)
                    product_weights[product_id] = float(weight)
                except (ValueError, TypeError):
                    # Skip non-numeric entries
                    continue
            print(f"Loaded {len(product_weights)} product weights from weights.json")
        else:
            print("No 'product_weights' key in weights.json")
            
    except FileNotFoundError:
        print("weights.json not found - all products will use default weight (0.1)")
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in weights.json: {e}")
    except Exception as e:
        print(f"Error loading weights.json: {e}")

def get_product_weight(product_id: int) -> float:
    """Get product weight (0.1 to 1.0), returns default 0.1 for unknown products"""
    return product_weights.get(product_id, DEFAULT_WEIGHT)

def find_nearest_output(position: Position) -> Position:
    """Find the nearest output position to a given position"""
    if not OUTPUT_POSITIONS:
        return INPUT_POSITION  # Fallback
    
    nearest = OUTPUT_POSITIONS[0]
    min_distance = manhattan_distance(position, nearest)
    
    for output_pos in OUTPUT_POSITIONS[1:]:
        distance = manhattan_distance(position, output_pos)
        if distance < min_distance:
            min_distance = distance
            nearest = output_pos
    
    return nearest

def manhattan_distance(pos1: Position, pos2: Position) -> float:
    return abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)

def get_product_id_from_barcode(barcode: str) -> int:
    for item in items.values():
        if item.metadata.barcode == barcode:
            return item.metadata.product_id
    hash_value = hashlib.md5(barcode.encode()).hexdigest()
    return int(hash_value[:8], 16) % 10000

def assign_robot(task: Task, destination_position: Position) -> str:
    """Assign robot based on total distance (current -> input -> destination)"""
    available = [r for r in robots.values() if r.status == "IDLE"]
    if not available:
        raise HTTPException(status_code=503, detail="No available robots")
    
    # Calculate total travel distance for each robot
    best_robot = None
    best_distance = float('inf')
    
    for robot in available:
        # Distance: robot current position -> input -> destination
        to_input = manhattan_distance(robot.position, INPUT_POSITION)
        input_to_dest = manhattan_distance(INPUT_POSITION, destination_position)
        total_distance = to_input + input_to_dest
        
        if total_distance < best_distance:
            best_distance = total_distance
            best_robot = robot
    
    best_robot.status = "BUSY"
    best_robot.current_task_id = task.id
    task.robot_id = best_robot.id
    
    # Save robot state immediately after assignment
    save_robots_to_file()
    
    return best_robot.id

def can_stack_in_vsu(item: Item, vsu: VirtualStorageUnit) -> bool:
    """
    Check if item can stack in existing VSU (placed in FRONT of existing items).

    Stacking rules:
    - Same product ID (same barcode)
    - Width within ±5mm tolerance
    - Height within ±5mm tolerance
    - Depth must be EXACTLY same
    - New item width must be ≤ frontmost item width (can't put wider box in front)
    """
    if not vsu.items:
        return False

    # Must be same product
    first_item = items[vsu.items[0]]
    if first_item.metadata.product_id != item.metadata.product_id:
        return False

    # Check item fits in VSU dimensions
    if item.metadata.dimensions.width > vsu.dimensions.width:
        print(f"      VSU {vsu.code}: Item width ({item.metadata.dimensions.width}mm) > VSU width ({vsu.dimensions.width}mm)")
        return False
    if item.metadata.dimensions.height > vsu.dimensions.height:
        print(f"      VSU {vsu.code}: Item height ({item.metadata.dimensions.height}mm) > VSU height ({vsu.dimensions.height}mm)")
        return False

    # Check width matches (±5mm tolerance) for stacking compatibility
    width_diff = abs(item.metadata.dimensions.width - first_item.metadata.dimensions.width)
    if width_diff > WIDTH_TOLERANCE:
        print(f"      VSU {vsu.code}: Width mismatch ({width_diff}mm > {WIDTH_TOLERANCE}mm tolerance)")
        return False

    # Check height matches (±5mm tolerance) for stacking compatibility
    height_diff = abs(item.metadata.dimensions.height - first_item.metadata.dimensions.height)
    if height_diff > HEIGHT_TOLERANCE:
        print(f"      VSU {vsu.code}: Height mismatch ({height_diff}mm > {HEIGHT_TOLERANCE}mm tolerance)")
        return False

    # Depth must be EXACTLY same for stacking
    if item.metadata.dimensions.depth != first_item.metadata.dimensions.depth:
        print(f"      VSU {vsu.code}: Depth mismatch ({item.metadata.dimensions.depth}mm != {first_item.metadata.dimensions.depth}mm) - must be exact")
        return False

    # CRITICAL: New item width must be ≤ frontmost item width
    # (Can't put wider box in front of narrower one)
    # Frontmost item is the last one added (lowest stock_index, placed most recently in front)
    frontmost_item = items[vsu.items[-1]]  # Last item in list is frontmost
    if item.metadata.dimensions.width > frontmost_item.metadata.dimensions.width:
        print(f"      VSU {vsu.code}: New item width ({item.metadata.dimensions.width}mm) > frontmost item width ({frontmost_item.metadata.dimensions.width}mm)")
        return False

    # Check if depth fits in VSU (including 3mm gap between items)
    num_existing_items = len(vsu.items)
    total_depth_used = first_item.metadata.dimensions.depth * num_existing_items  # All same depth

    # Account for pending placements in this VSU
    pending_for_vsu = [p for p in pending_placements.values() if p["vsu_id"] == vsu.id]
    pending_depth = sum(p["item_depth"] + DEPTH_GAP_BETWEEN_ITEMS for p in pending_for_vsu)

    total_depth_with_gaps = total_depth_used + (num_existing_items * DEPTH_GAP_BETWEEN_ITEMS) + pending_depth

    remaining_depth = vsu.dimensions.depth - total_depth_with_gaps

    if item.metadata.dimensions.depth > remaining_depth:
        print(f"      VSU {vsu.code}: Not enough depth (need {item.metadata.dimensions.depth}mm, only {remaining_depth}mm available, pending: {len(pending_for_vsu)})")
        return False

    # Check if there's a valid stock index available
    max_items = calculate_max_items_in_vsu(vsu, item.metadata.dimensions.depth)
    pending_count = len(pending_for_vsu)
    calculated_index = max_items - num_existing_items - pending_count - 1
    if calculated_index < 0:
        print(f"      VSU {vsu.code}: VSU full (max {max_items}, existing {num_existing_items}, pending {pending_count})")
        return False

    print(f"      VSU {vsu.code}: CAN STACK! (depth {remaining_depth}mm available, placing in front)")
    return True


def calculate_max_items_in_vsu(vsu: VirtualStorageUnit, item_depth: float) -> int:
    """
    Calculate maximum number of items that can fit in VSU depth-wise.
    Considers 3mm gap between items.

    Formula: max_items = floor(vsu_depth / (item_depth + gap))
    But last item doesn't need trailing gap, so:
    max_items = floor((vsu_depth + gap) / (item_depth + gap))
    """
    effective_item_depth = item_depth + DEPTH_GAP_BETWEEN_ITEMS
    max_items = int((vsu.dimensions.depth + DEPTH_GAP_BETWEEN_ITEMS) / effective_item_depth)
    return max(1, max_items)


def calculate_stock_index(vsu: VirtualStorageUnit, item_depth: float, is_new_vsu: bool) -> int:
    """
    Calculate stock index for back-to-front placement.

    First item placed at back gets highest index (max_capacity - 1).
    Next item placed in front gets (max_capacity - 2), and so on.
    Last item at front gets index 0.

    For stacking (existing VSU): index = max_capacity - current_count - pending_count - 1
    For new VSU: index = max_capacity - 1 - pending_count (account for pending)

    Returns -1 if VSU is full (no valid index available)
    """
    max_items = calculate_max_items_in_vsu(vsu, item_depth)
    current_count = len(vsu.items)

    # Count pending placements for this VSU
    pending_count = sum(1 for p in pending_placements.values() if p["vsu_id"] == vsu.id)

    if is_new_vsu or current_count == 0:
        # First item in VSU - place at back with highest index (minus any pending)
        index = max_items - 1 - pending_count
    else:
        # Stacking - place in front of existing items and pending items
        index = max_items - current_count - pending_count - 1

    # Return -1 if VSU is full
    return index if index >= 0 else -1


def calculate_item_z_position(vsu: VirtualStorageUnit, item_depth: float, stock_index: int) -> float:
    """
    Calculate Z coordinate for item based on stock index.

    Boxes are placed from the back wall forward:
    - First box (highest stock_index) touches the back wall
    - Each subsequent box is placed 3mm in front of the previous
    - Empty space is at the front of the VSU

    For SAME product stacking (all items have same depth):
    - Uses uniform depth calculation (efficient)

    For MIXED product stacking (items may have different depths):
    - Sums actual depths of items behind (higher stock_index)
    - Ensures no collision between boxes of different depths
    """
    back_wall_z = vsu.position.z + vsu.dimensions.depth

    # Check if VSU has existing items with different depths (mixed product scenario)
    if vsu.items:
        existing_depths = [items[item_id].metadata.dimensions.depth for item_id in vsu.items]

        # Check if all existing items have same depth (same product stacking)
        all_same_depth = len(set(existing_depths)) == 1

        if all_same_depth and existing_depths[0] == item_depth:
            # SAME PRODUCT: All items have same depth - use efficient uniform calculation
            max_items = calculate_max_items_in_vsu(vsu, item_depth)
            effective_item_depth = item_depth + DEPTH_GAP_BETWEEN_ITEMS
            positions_from_back = max_items - 1 - stock_index
            z_offset = positions_from_back * effective_item_depth
            return back_wall_z - item_depth - z_offset
        else:
            # MIXED PRODUCT: Items have different depths - calculate position based on
            # items that are "in front" (lower stock_index) - we need to leave room for them
            #
            # For items placed from back to front:
            # - Highest stock_index item touches back wall
            # - Each item in front needs: previous_item_z - gap - this_item_depth
            #
            # Since we're calculating for a specific stock_index, we sum depths of items
            # that will be IN FRONT of this position (lower stock_index values)

            # Get depths of existing items
            items_with_index = []
            for item_id in vsu.items:
                item_obj = items[item_id]
                items_with_index.append({
                    'stock_index': item_obj.stock_index,
                    'depth': item_obj.metadata.dimensions.depth
                })

            # Also include pending placements
            pending_for_vsu = [p for p in pending_placements.values() if p["vsu_id"] == vsu.id]
            for p in pending_for_vsu:
                items_with_index.append({
                    'stock_index': p['stock_index'],
                    'depth': p['item_depth']
                })

            # Find the maximum stock_index to determine the back-most position
            all_indices = [item_info['stock_index'] for item_info in items_with_index] + [stock_index]
            max_stock_index = max(all_indices)

            # Sum depths of items from the back wall up to (but not including) this item
            # Items with higher stock_index are placed first (closer to back)
            depth_from_back = 0
            gaps_from_back = 0
            for item_info in items_with_index:
                if item_info['stock_index'] > stock_index:
                    depth_from_back += item_info['depth']
                    gaps_from_back += 1

            # If this is the back-most item (highest stock_index), it touches the back wall
            if stock_index == max_stock_index:
                return back_wall_z - item_depth
            else:
                # Position = back_wall - depths_of_items_behind - gaps_between_them - gap_before_this - this_item_depth
                # gaps_from_back already counts the number of items behind, which equals the number of gaps
                # (including the gap between this item and the item immediately behind it)
                total_offset = depth_from_back + (gaps_from_back * DEPTH_GAP_BETWEEN_ITEMS)
                return back_wall_z - total_offset - item_depth
    else:
        # Empty VSU - first item goes at back wall
        return back_wall_z - item_depth


def find_vsu_for_stacking(item: Item) -> Optional[VirtualStorageUnit]:
    """Find existing VSU where item can stack (including pending placements)"""
    product_id = item.metadata.product_id

    # Debug: Find all VSUs with same product (actual items)
    matching_product_vsus = []
    for vsu in virtual_units.values():
        if vsu.items:
            first_item = items[vsu.items[0]]
            if first_item.metadata.product_id == product_id:
                matching_product_vsus.append(vsu)

    # Also check pending placements for same product
    pending_same_product = [p for p in pending_placements.values() if p["product_id"] == product_id]

    print(f"Looking for stacking VSU for Product {product_id}")
    print(f"   Found {len(matching_product_vsus)} VSUs with same product (actual)")
    print(f"   Found {len(pending_same_product)} pending placements with same product")

    # Check each matching VSU (actual items)
    for vsu in virtual_units.values():
        if can_stack_in_vsu(item, vsu):
            print(f"   STACKING in VSU {vsu.code} (has {len(vsu.items)} items)")
            return vsu

    # Check pending placements - find VSU where same product is pending
    for pending in pending_same_product:
        vsu = virtual_units.get(pending["vsu_id"])
        if vsu and can_stack_with_pending(item, vsu, pending):
            print(f"   STACKING in VSU {vsu.code} (has pending placement for same product)")
            return vsu

    print(f"   No suitable VSU found - will create new")
    return None


def can_stack_with_pending(item: Item, vsu: VirtualStorageUnit, pending: dict) -> bool:
    """Check if item can stack in VSU that has pending placement (may or may not have actual items)"""
    # Check dimensions match (same rules as can_stack_in_vsu)
    width_diff = abs(item.metadata.dimensions.width - pending["item_width"])
    if width_diff > WIDTH_TOLERANCE:
        return False

    height_diff = abs(item.metadata.dimensions.height - pending["item_height"])
    if height_diff > HEIGHT_TOLERANCE:
        return False

    # Depth must match exactly
    if item.metadata.dimensions.depth != pending["item_depth"]:
        return False

    # Check remaining depth (accounting for actual items AND all pending items)
    num_existing_items = len(vsu.items)
    existing_depth = 0
    if num_existing_items > 0:
        first_item = items[vsu.items[0]]
        existing_depth = first_item.metadata.dimensions.depth * num_existing_items + (num_existing_items * DEPTH_GAP_BETWEEN_ITEMS)

    pending_for_vsu = [p for p in pending_placements.values() if p["vsu_id"] == vsu.id]
    pending_depth = sum(p["item_depth"] + DEPTH_GAP_BETWEEN_ITEMS for p in pending_for_vsu)

    total_used = existing_depth + pending_depth
    remaining_depth = vsu.dimensions.depth - total_used

    if item.metadata.dimensions.depth > remaining_depth:
        return False

    # Check if there's a valid stock index available
    max_items = calculate_max_items_in_vsu(vsu, item.metadata.dimensions.depth)
    pending_count = len(pending_for_vsu)
    calculated_index = max_items - num_existing_items - pending_count - 1
    if calculated_index < 0:
        return False

    return True

def find_empty_vsu_for_item(item: Item) -> Optional[VirtualStorageUnit]:
    """
    Find an empty VSU that can accommodate this item.
    Prefers smallest suitable VSU (best fit).

    Returns empty VSU if found, None otherwise
    """
    print(f"   Checking for empty VSUs that can fit item ({item.metadata.dimensions.width}×{item.metadata.dimensions.height}×{item.metadata.dimensions.depth}mm)...")

    suitable_empty_vsus = []

    for vsu in virtual_units.values():
        # Must be empty
        if vsu.items:
            continue

        # Skip if there are pending placements for this VSU
        if any(p["vsu_id"] == vsu.id for p in pending_placements.values()):
            continue

        # Check if item fits in VSU dimensions
        if (item.metadata.dimensions.width <= vsu.dimensions.width and
            item.metadata.dimensions.height <= vsu.dimensions.height and
            item.metadata.dimensions.depth <= vsu.dimensions.depth):

            # Calculate efficiency score (lower = better fit = less wasted space)
            width_waste = vsu.dimensions.width - item.metadata.dimensions.width
            height_waste = vsu.dimensions.height - item.metadata.dimensions.height
            volume_waste = (vsu.dimensions.width * vsu.dimensions.height * vsu.dimensions.depth) - item.metadata.dimensions.volume

            # Score: prefer smallest VSU (least wasted space)
            # Primary: smallest height (space efficiency priority)
            # Secondary: least volume waste
            efficiency_score = (vsu.dimensions.height * 1000) + volume_waste

            shelf = shelves[vsu.shelf_id]
            suitable_empty_vsus.append({
                'vsu': vsu,
                'score': efficiency_score,
                'width_waste': width_waste,
                'height_waste': height_waste,
                'volume_waste': volume_waste,
                'shelf_name': shelf.name
            })

            print(f"      Empty VSU {vsu.code} ({shelf.name}): {vsu.dimensions.width}×{vsu.dimensions.height}×{vsu.dimensions.depth}mm")
            print(f"         Waste: W={width_waste:.1f}mm, H={height_waste:.1f}mm, Vol={volume_waste:.0f}mm³, Score={efficiency_score:.0f}")

    if not suitable_empty_vsus:
        print(f"   No suitable empty VSUs found")
        return None

    # Sort by efficiency score (smallest height first, then least volume waste)
    suitable_empty_vsus.sort(key=lambda x: x['score'])

    best = suitable_empty_vsus[0]
    print(f"   BEST FIT: VSU {best['vsu'].code} on {best['shelf_name']} (Score={best['score']:.0f}, least wasted space)")

    return best['vsu']


def find_mixed_product_vsu(item: Item) -> Optional[VirtualStorageUnit]:
    """
    FALLBACK: Find VSU where item physically fits in FRONT, regardless of product.

    Used when:
    - No stacking possible (no same-product VSU)
    - No empty VSU available
    - Cannot create new VSU (no shelf space)

    Criteria:
    - VSU has items (different product)
    - Item width <= frontmost item width (can't place wider box in front)
    - Item height <= VSU height (must physically fit)
    - Remaining depth >= item depth + 3mm gap

    IMPORTANT: Items can ONLY be placed in FRONT of existing items, never behind.
    This ensures boxes don't block each other.

    Prefers VSUs with TIGHTEST fit (minimum wasted space).
    """
    print(f"   FALLBACK: Checking for ANY VSU where item fits in FRONT...")

    item_width = item.metadata.dimensions.width
    item_height = item.metadata.dimensions.height
    item_depth = item.metadata.dimensions.depth

    suitable_vsus = []

    for vsu in virtual_units.values():
        # Must have items (non-empty)
        if not vsu.items:
            continue

        # Get first item to check product
        first_item = items[vsu.items[0]]

        # Skip if same product (that's stacking, handled earlier)
        if first_item.metadata.product_id == item.metadata.product_id:
            continue

        # Height: item must fit within VSU height
        if item_height > vsu.dimensions.height:
            print(f"      VSU {vsu.code}: Height {item_height}mm > VSU height {vsu.dimensions.height}mm")
            continue

        # CRITICAL: New item width must be <= frontmost item width
        # Can't place wider box in front of narrower one (would block it)
        frontmost_item = items[vsu.items[-1]]  # Last item in list is frontmost
        if item_width > frontmost_item.metadata.dimensions.width:
            print(f"      VSU {vsu.code}: Width {item_width}mm > frontmost item {frontmost_item.metadata.dimensions.width}mm")
            continue

        # Check remaining depth (including 3mm gap for new item)
        num_existing_items = len(vsu.items)
        total_depth_used = sum(items[i].metadata.dimensions.depth for i in vsu.items)

        # Account for pending placements in this VSU
        pending_for_vsu = [p for p in pending_placements.values() if p["vsu_id"] == vsu.id]
        pending_depth = sum(p["item_depth"] + DEPTH_GAP_BETWEEN_ITEMS for p in pending_for_vsu)

        # Include gaps between existing items + gap for new item + pending items
        total_depth_with_gaps = total_depth_used + (num_existing_items * DEPTH_GAP_BETWEEN_ITEMS) + pending_depth
        remaining_depth = vsu.dimensions.depth - total_depth_with_gaps

        if item_depth > remaining_depth:
            print(f"      VSU {vsu.code}: Need {item_depth}mm, only {remaining_depth}mm available (pending: {len(pending_for_vsu)})")
            continue

        # Calculate wasted space (lower = tighter fit = better)
        width_waste = frontmost_item.metadata.dimensions.width - item_width
        height_waste = vsu.dimensions.height - item_height
        depth_waste = remaining_depth - item_depth
        total_waste = width_waste + height_waste + depth_waste

        shelf = shelves[vsu.shelf_id]
        existing_product_id = first_item.metadata.product_id

        suitable_vsus.append({
            'vsu': vsu,
            'width_waste': width_waste,
            'height_waste': height_waste,
            'depth_waste': depth_waste,
            'total_waste': total_waste,
            'remaining_depth': remaining_depth,
            'existing_product_id': existing_product_id,
            'shelf_name': shelf.name,
            'items_count': len(vsu.items),
            'frontmost_width': frontmost_item.metadata.dimensions.width
        })

        print(f"      VSU {vsu.code} ({shelf.name}): CAN FIT in front")
        print(f"         Frontmost item width: {frontmost_item.metadata.dimensions.width}mm, New item: {item_width}mm")
        print(f"         Waste - W:{width_waste:.1f}mm H:{height_waste:.1f}mm D:{depth_waste:.1f}mm = Total:{total_waste:.1f}mm")

    if not suitable_vsus:
        print(f"   No VSU found where item can be placed in front")
        return None

    # Sort by total waste (tightest fit first)
    suitable_vsus.sort(key=lambda x: x['total_waste'])

    best = suitable_vsus[0]
    print(f"   MIXED-PRODUCT FIT: VSU {best['vsu'].code} on {best['shelf_name']}")
    print(f"      Existing Product: {best['existing_product_id']}, Items: {best['items_count']}")
    print(f"      Frontmost width: {best['frontmost_width']}mm, New item: {item_width}mm")
    print(f"      Total waste: {best['total_waste']:.1f}mm")

    return best['vsu']

def calculate_next_vsu_position(shelf: Shelf, item_width: float) -> Optional[float]:
    """
    Calculate X position for new VSU on shelf
    
    FIXED: Now properly validates that new VSU + gap fits within shelf bounds
    Includes tolerance for floating point precision and exact-fit scenarios
    Returns None if insufficient space
    """
    # Add small tolerance for floating point precision and exact fits (0.5mm)
    WIDTH_FIT_TOLERANCE = 0.5
    
    shelf_start = shelf.position.x
    shelf_end = shelf_start + shelf.dimensions.width
    
    # Get existing VSUs on this shelf
    shelf_vsus = [virtual_units[vid] for vid in shelf.virtual_units]
    
    if not shelf_vsus:
        # First VSU on shelf - check if it fits (with tolerance)
        if shelf_start + item_width <= shelf_end + WIDTH_FIT_TOLERANCE:
            return shelf_start
        else:
            print(f"      Item too wide ({item_width}mm) for shelf ({shelf.dimensions.width}mm)")
            return None
    
    # Find rightmost VSU
    rightmost = max(shelf_vsus, key=lambda v: v.position.x + v.dimensions.width)
    next_x = rightmost.position.x + rightmost.dimensions.width + VSU_HORIZONTAL_GAP
    
    # CRITICAL FIX: Check if new VSU fits completely within shelf bounds
    # Add tolerance for floating point precision and exact-fit scenarios
    if next_x + item_width <= shelf_end + WIDTH_FIT_TOLERANCE:
        available_space = shelf_end - next_x
        print(f"      Next position: {next_x}mm, Available: {available_space:.1f}mm, Need: {item_width}mm")
        return next_x
    else:
        available_space = shelf_end - next_x
        print(f"      Insufficient space: Available {available_space:.1f}mm < Need {item_width}mm")
        return None

def create_new_vsu(item: Item, shelf: Shelf) -> Optional[VirtualStorageUnit]:
    """Create new VSU for item on shelf with validation"""
    global vsu_counter

    # Validate box dimensions against shelf
    if item.metadata.dimensions.width > shelf.dimensions.width:
        print(f"      Item width ({item.metadata.dimensions.width}mm) > shelf width ({shelf.dimensions.width}mm)")
        return None
    if item.metadata.dimensions.height > shelf.dimensions.height - VSU_TOP_CLEARANCE:
        print(f"      Item height ({item.metadata.dimensions.height}mm) > shelf height ({shelf.dimensions.height - VSU_TOP_CLEARANCE}mm)")
        return None
    if item.metadata.dimensions.depth > shelf.dimensions.depth:
        print(f"      Item depth ({item.metadata.dimensions.depth}mm) > shelf depth ({shelf.dimensions.depth}mm)")
        return None

    # Calculate position
    x_pos = calculate_next_vsu_position(shelf, item.metadata.dimensions.width)
    if x_pos is None:
        return None

    # VSU dimensions
    vsu_width = item.metadata.dimensions.width
    vsu_height = shelf.dimensions.height - VSU_TOP_CLEARANCE  # Leave 0.3mm clearance
    vsu_depth = shelf.dimensions.depth

    vsu_counter += 1

    new_vsu = VirtualStorageUnit(
        id=vsu_counter,
        code=f"vu{vsu_counter}",
        dimensions=Dimensions(width=vsu_width, height=vsu_height, depth=vsu_depth),
        position=Position(x=x_pos, y=shelf.position.y, z=shelf.position.z),
        shelf_id=shelf.id,
        rack_id=shelf.rack_id,
        items=[],
        occupied=False
    )

    # Add to global virtual_units dict
    virtual_units[new_vsu.id] = new_vsu

    # Add to the shelf's virtual_units list (both passed shelf AND global shelves dict)
    if new_vsu.id not in shelf.virtual_units:
        shelf.virtual_units.append(new_vsu.id)

    # Also ensure the global shelves dict is updated (in case shelf is a different reference)
    if shelf.id in shelves:
        if new_vsu.id not in shelves[shelf.id].virtual_units:
            shelves[shelf.id].virtual_units.append(new_vsu.id)
            print(f"[VSU] Added VSU {new_vsu.id} to global shelves[{shelf.id}].virtual_units")
    else:
        print(f"[VSU] WARNING: shelf.id {shelf.id} not found in global shelves dict!")

    print(f"[VSU] Created new VSU {new_vsu.code} (ID: {new_vsu.id}) at ({x_pos}, {shelf.position.y}, {shelf.position.z}) on shelf {shelf.name}")
    print(f"[VSU] Shelf {shelf.id} now has virtual_units: {shelves[shelf.id].virtual_units if shelf.id in shelves else 'N/A'}")

    return new_vsu

def find_or_create_vsu_for_item(item: Item) -> tuple[Optional[VirtualStorageUnit], bool]:
    """
    Find existing VSU to stack, use empty VSU, create new VSU, or use mixed-product VSU

    Priority Order:
    1. Stacking (same product) - ALWAYS FIRST
    2. Empty VSU - REUSE before creating new
    3. Create new VSU on smallest suitable shelf
    4. FALLBACK: Mixed-product VSU (different product, similar dimensions)

    Returns:
        tuple: (vsu, is_new_vsu) where is_new_vsu indicates if VSU was newly created
    """

    # STEP 1: Try stacking first (same product)
    stacking_vsu = find_vsu_for_stacking(item)
    if stacking_vsu:
        print(f"   Using existing VSU for stacking")
        return stacking_vsu, False  # Reusing existing VSU

    print(f"   No stacking VSU found - checking for empty VSUs...")

    # STEP 2: Try to find empty VSU
    empty_vsu = find_empty_vsu_for_item(item)
    if empty_vsu:
        print(f"   Using empty VSU (no need to create new)")
        return empty_vsu, False  # Reusing existing empty VSU

    print(f"   No suitable empty VSUs - evaluating shelves for new VSU...")

    # STEP 3: Try to create new VSU - find all suitable shelves
    suitable_shelves = []

    for shelf in shelves.values():
        # Check if item fits in shelf height (with clearance)
        if item.metadata.dimensions.height > shelf.dimensions.height - VSU_TOP_CLEARANCE:
            continue

        # Check if shelf has horizontal space for new VSU
        next_x = calculate_next_vsu_position(shelf, item.metadata.dimensions.width)
        if next_x is not None:
            suitable_shelves.append(shelf)

    if suitable_shelves:
        print(f"   Found {len(suitable_shelves)} suitable shelves")

        # Sort by smallest height first (space efficiency)
        suitable_shelves.sort(key=lambda s: s.dimensions.height)
        print(f"   Sorted shelves by height: {[f'{s.name}(H={s.dimensions.height})' for s in suitable_shelves[:5]]}")

        # Prioritize smallest height, use scoring as tiebreaker
        best_shelf = None
        best_score = float('-inf')
        smallest_height = float('inf')

        for shelf in suitable_shelves:
            # Create a VIRTUAL VSU (preview) to calculate score
            x_pos = calculate_next_vsu_position(shelf, item.metadata.dimensions.width)
            if x_pos is None:
                continue

            virtual_vsu = VirtualStorageUnit(
                id=-1,  # Temporary ID for scoring
                code=f"preview_{shelf.id}",
                dimensions=Dimensions(
                    width=item.metadata.dimensions.width,
                    height=shelf.dimensions.height - VSU_TOP_CLEARANCE,
                    depth=shelf.dimensions.depth
                ),
                position=Position(
                    x=x_pos,
                    y=shelf.position.y,
                    z=shelf.position.z
                ),
                shelf_id=shelf.id,
                rack_id=shelf.rack_id,
                items=[],
                occupied=False
            )

            # Calculate placement score for this shelf
            score = calculate_placement_score(item, virtual_vsu)

            print(f"      Shelf {shelf.name}: H={shelf.dimensions.height}mm, Score={score:.2f}")

            # Always prefer smaller shelf height (space efficiency priority)
            # Only use score if heights are equal (tiebreaker for weight/demand)
            if shelf.dimensions.height < smallest_height:
                # Found a smaller shelf - ALWAYS take it!
                best_shelf = shelf
                best_score = score
                smallest_height = shelf.dimensions.height
                print(f"         -> NEW BEST (smaller shelf!)")
            elif shelf.dimensions.height == smallest_height and score > best_score:
                # Same height, but better score - take it (weight/demand matters here!)
                best_shelf = shelf
                best_score = score
                print(f"         -> NEW BEST (better score, same height)")

        if best_shelf:
            print(f"   WINNER: Shelf {best_shelf.name} (H={best_shelf.dimensions.height}mm) with score {best_score:.2f}")

            # Actually create VSU on the winning shelf
            new_vsu = create_new_vsu(item, best_shelf)
            if new_vsu:
                print(f"   Created VSU {new_vsu.code} on {best_shelf.name}")
                return new_vsu, True  # NEWLY CREATED VSU

    # STEP 4: FALLBACK - Try mixed-product VSU (different product, similar dimensions)
    print(f"   No space for new VSU - trying mixed-product fallback...")
    mixed_vsu = find_mixed_product_vsu(item)
    if mixed_vsu:
        print(f"   Using mixed-product VSU {mixed_vsu.code} (FALLBACK)")
        return mixed_vsu, False  # Reusing existing VSU with different product

    print(f"   ALL OPTIONS EXHAUSTED - No placement possible")
    return None, False

def calculate_placement_score(item: Item, vsu: VirtualStorageUnit) -> float:
    """Calculate placement score (higher = better)"""
    score = 0
    
    # Stacking bonus (same product) - HIGHEST PRIORITY
    if vsu.items:
        first_item = items[vsu.items[0]]
        if first_item.metadata.product_id == item.metadata.product_id:
            score += 100
    
    # Weight-based placement (higher weight = closer to output)
    # Weight range: 0.1 (low demand) to 1.0 (high demand)
    product_id = item.metadata.product_id
    weight = get_product_weight(product_id)
    
    # Find nearest output position
    nearest_output = find_nearest_output(vsu.position)
    
    # Calculate distance from VSU to nearest output
    distance_to_output = manhattan_distance(vsu.position, nearest_output)
    
    # High-weight products should be closer to output
    # Lower distance = higher score for high-weight products
    # Score formula: weight × (1 / distance) × scaling_factor
    # Add small constant to avoid division by zero
    if distance_to_output > 0:
        proximity_score = weight * (1000.0 / distance_to_output)
        score += proximity_score
    
    # Alternative: Height-based scoring for high-weight products
    # Lower shelves are easier for robots to access
    shelf = shelves[vsu.shelf_id]
    max_height = max(s.position.z for s in shelves.values()) if shelves else 1
    height_factor = (max_height - shelf.position.z) / max_height if max_height > 0 else 0
    score += weight * height_factor * 50
    
    # Space efficiency (tighter fit = better)
    width_diff = abs(vsu.dimensions.width - item.metadata.dimensions.width)
    score -= width_diff * 0.1
    
    return score

def save_warehouse_state():
    """Save warehouse state to ml_robot_updated.json in the original format"""
    try:
        # Build warehouse structure matching original format
        warehouse_data = {
            "Warehouses": [],
            "ItemPlacements": [],
            "Items": [],
            "Restrictions": []
        }
        
        # Build Warehouses array
        warehouse = {
            "Id": 1,
            "Name": "Virtual MedicPort Storage",
            "Note": "Storage PoC ML",
            "Type": "MedicPort Robot V1",
            "StorageUnits": []
        }
        
        # Build StorageUnits (Racks)
        for rack in racks.values():
            storage_unit = {
                "Id": rack.id,
                "ChildUnitsType": [],
                "UnitType": "Rack",
                "Text": rack.name,
                "VirtualSuDimensions": [],
                "SuDimensions": None
            }
            
            # Build ChildUnitsType (Shelves)
            for shelf_id in rack.shelf_ids:
                if shelf_id not in shelves:
                    continue
                    
                shelf = shelves[shelf_id]
                child_unit = {
                    "Id": shelf.id,
                    "ChildUnitsType": None,
                    "UnitType": "Shelf",
                    "Text": shelf.name,
                    "VirtualSuDimensions": [],
                    "SuDimensions": {
                        "Id": 0,
                        "Width": shelf.dimensions.width,
                        "Height": shelf.dimensions.height,
                        "Depth": shelf.dimensions.depth,
                        "Weight": 40000,
                        "CoordinateX": shelf.position.x,
                        "CoordinateY": shelf.position.y,
                        "CoordinateZ": shelf.position.z
                    }
                }
                
                # Build VirtualSuDimensions (VSUs)
                print(f"[SAVE] Shelf {shelf.id} ({shelf.name}) has VSU IDs: {shelf.virtual_units}")
                for vsu_id in shelf.virtual_units:
                    if vsu_id not in virtual_units:
                        print(f"[SAVE] WARNING: VSU {vsu_id} in shelf.virtual_units but NOT in virtual_units dict!")
                        continue

                    vsu = virtual_units[vsu_id]
                    vsu_data = {
                        "Id": vsu.id,
                        "Width": vsu.dimensions.width,
                        "Height": vsu.dimensions.height,
                        "Depth": vsu.dimensions.depth,
                        "Code": vsu.code,
                        "CoordinateX": vsu.position.x,
                        "CoordinateY": vsu.position.y,
                        "CoordinateZ": vsu.position.z,
                        "Volume": vsu.dimensions.volume
                    }
                    child_unit["VirtualSuDimensions"].append(vsu_data)
                    print(f"[SAVE] Added VSU {vsu_id} ({vsu.code}) to shelf {shelf.id} VirtualSuDimensions")
                
                storage_unit["ChildUnitsType"].append(child_unit)
            
            warehouse["StorageUnits"].append(storage_unit)
        
        warehouse_data["Warehouses"].append(warehouse)
        
        # Build Items array
        for item in items.values():
            item_data = {
                "ProductID": item.metadata.product_id,
                "Barcode": item.metadata.barcode,
                "Width": item.metadata.dimensions.width,
                "Height": item.metadata.dimensions.height,
                "Depth": item.metadata.dimensions.depth,
                "Weight": item.metadata.weight,
                "Expiration": item.metadata.expiration.isoformat(),
                "Batch": item.metadata.batch,
                "DeliveryId": item.metadata.delivery_id,
                "WarehouseID": item.metadata.warehouse_id
            }
            warehouse_data["Items"].append(item_data)
        
        # Build ItemPlacements array
        for item in items.values():
            placement = {
                "VSURelation": {
                    "VSUnitId": item.vsu_id if item.vsu_id else None,
                    "WarehouseId": item.metadata.warehouse_id
                },
                "ProductId": item.metadata.product_id,
                "ItemMetadata": {
                    "ProductID": item.metadata.product_id,
                    "Barcode": item.metadata.barcode,
                    "Width": item.metadata.dimensions.width,
                    "Height": item.metadata.dimensions.height,
                    "Depth": item.metadata.dimensions.depth,
                    "Weight": item.metadata.weight,
                    "Expiration": item.metadata.expiration.isoformat(),
                    "Batch": item.metadata.batch,
                    "DeliveryId": item.metadata.delivery_id,
                    "WarehouseID": item.metadata.warehouse_id
                },
                "IND": True,
                "StockIndex": item.stock_index,
                "ReservationId": None
            }
            warehouse_data["ItemPlacements"].append(placement)
        
        with open('data/ml_robot_updated.json', 'w') as f:
            json.dump(warehouse_data, f, indent=2)
        
        print("Saved warehouse state to ml_robot_updated.json")
    except Exception as e:
        print(f"Error saving warehouse: {e}")
        traceback.print_exc()

def load_warehouse():
    """Load warehouse data from ml_robot_updated.json with its actual structure"""
    global item_counter, vsu_counter

    try:
        import os
        if not os.path.exists('data/ml_robot_updated.json'):
            print("ml_robot_updated.json not found! Please ensure the file exists in the same directory.")
            return False

        with open('data/ml_robot_updated.json', 'r') as f:
            data = json.load(f)
        
        print(f"Loading warehouse data...")
        print(f"JSON keys found: {list(data.keys())}")
        
        # Parse Warehouses structure
        warehouses = data.get("Warehouses", [])
        if not warehouses:
            print("No warehouses found in JSON!")
            return False
        
        warehouse = warehouses[0]  # Use first warehouse
        print(f"Loading warehouse: {warehouse.get('Name', 'Unknown')}")
        
        rack_id_counter = 1
        shelf_id_counter = 1
        
        # Parse StorageUnits (these are Racks)
        for storage_unit in warehouse.get("StorageUnits", []):
            rack_id = storage_unit["Id"]
            rack_name = storage_unit.get("Text", f"Rack {rack_id}")
            
            rack_shelf_ids = []
            
            # Parse ChildUnitsType (these are Shelves)
            for child_unit in storage_unit.get("ChildUnitsType", []):
                shelf_id = child_unit["Id"]
                shelf_name = child_unit.get("Text", f"shelf_{shelf_id}")
                shelf_dims = child_unit.get("SuDimensions", {})
                
                # Create shelf
                shelf = Shelf(
                    id=shelf_id,
                    rack_id=rack_id,
                    name=shelf_name,
                    dimensions=Dimensions(
                        width=shelf_dims.get("Width", 0),
                        height=shelf_dims.get("Height", 0),
                        depth=shelf_dims.get("Depth", 0)
                    ),
                    position=Position(
                        x=shelf_dims.get("CoordinateX", 0),
                        y=shelf_dims.get("CoordinateY", 0),
                        z=shelf_dims.get("CoordinateZ", 0)
                    ),
                    virtual_units=[]
                )
                
                # Parse VirtualSuDimensions (these are VSUs)
                for vsu_data in child_unit.get("VirtualSuDimensions", []):
                    vsu_id = vsu_data["Id"]
                    vsu = VirtualStorageUnit(
                        id=vsu_id,
                        code=vsu_data.get("Code", f"vu{vsu_id}"),
                        dimensions=Dimensions(
                            width=vsu_data.get("Width", 0),
                            height=vsu_data.get("Height", 0),
                            depth=vsu_data.get("Depth", 0)
                        ),
                        position=Position(
                            x=vsu_data.get("CoordinateX", 0),
                            y=vsu_data.get("CoordinateY", 0),
                            z=vsu_data.get("CoordinateZ", 0)
                        ),
                        shelf_id=shelf_id,
                        rack_id=rack_id,
                        items=[],
                        occupied=False
                    )
                    virtual_units[vsu_id] = vsu
                    shelf.virtual_units.append(vsu_id)
                    vsu_counter = max(vsu_counter, vsu_id)
                
                shelves[shelf_id] = shelf
                rack_shelf_ids.append(shelf_id)
            
            # Create rack
            rack = Rack(
                id=rack_id,
                name=rack_name,
                shelf_ids=rack_shelf_ids
            )
            racks[rack_id] = rack
        
        print(f"  Loaded {len(racks)} racks")
        print(f"  Loaded {len(shelves)} shelves")
        print(f"  Loaded {len(virtual_units)} VSUs")
        
        # Parse ItemPlacements (these contain the actual items with their locations)
        item_placements = data.get("ItemPlacements", [])
        
        for idx, placement in enumerate(item_placements, start=1):
            item_id = idx
            
            # Get item metadata from placement
            item_metadata = placement.get("ItemMetadata", {})
            if not item_metadata:
                print(f"    Skipping item {idx}: no metadata")
                continue
            
            # Get VSU location
            vsu_relation = placement.get("VSURelation", {})
            vsu_id = vsu_relation.get("VSUnitId")
            stock_index = placement.get("StockIndex", 0)
            
            # Parse expiration date
            exp_str = item_metadata.get("Expiration", "2025-12-31T00:00:00")
            
            try:
                item = Item(
                    id=item_id,
                    metadata=ItemMetadata(
                        product_id=item_metadata.get("ProductID", 0),
                        barcode=item_metadata.get("Barcode", ""),
                        dimensions=Dimensions(
                            width=item_metadata.get("Width", 0),
                            height=item_metadata.get("Height", 0),
                            depth=item_metadata.get("Depth", 0)
                        ),
                        weight=item_metadata.get("Weight", 0),
                        expiration=safe_parse_datetime(exp_str),
                        batch=item_metadata.get("Batch", ""),
                        delivery_id=item_metadata.get("DeliveryId", ""),
                        warehouse_id=item_metadata.get("WarehouseID", 1)
                    ),
                    vsu_id=vsu_id,
                    stock_index=stock_index
                )
                
                items[item_id] = item
                item_counter = max(item_counter, item_id)
                
                # Add item to VSU if placed
                if vsu_id and vsu_id in virtual_units:
                    vsu = virtual_units[vsu_id]
                    if item_id not in vsu.items:
                        vsu.items.append(item_id)
                        vsu.occupied = True
            except Exception as e:
                print(f"    Error loading item {idx}: {e}")
                continue
        
        print(f"  Loaded {len(items)} items")
        
        print(f"Warehouse loaded successfully!")
        print(f"   {len(racks)} racks | {len(shelves)} shelves | {len(virtual_units)} VSUs | {len(items)} items")
        return True
        
    except FileNotFoundError:
        print(f"ml_robot_updated.json not found in current directory!")
        print(f"   Current directory: {os.getcwd()}")
        return False
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in ml_robot_updated.json: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"Error loading warehouse: {e}")
        traceback.print_exc()
        return False

# Load warehouse on startup
print("Starting Sequential Stock-in System")
warehouse_loaded = load_warehouse()
if not warehouse_loaded:
    print("WARNING: Warehouse data not loaded! Server will start but functionality limited.")

# Load warehouse layout (input/output positions)
load_warehouse_layout()

# Load product weights from ML model
load_product_weights()

print(f"{len(robots)} robots ready")


@app.post("/stockin/suggest", tags=["Stock-In Operations"])
async def suggest_placement(scanner_input: ScannerInput):
    """
    Step 1: Suggest optimal placement for incoming item.

    - Finds best VSU (stacking same product, empty VSU, or creates new)
    - Calculates stock_index based on back-to-front placement
    - Returns task_id and placement details
    - Does NOT modify inventory - call /task/{task_id}/complete to confirm
    """
    global item_counter, task_counter

    try:
        # Create item (temporary - will be saved on complete)
        item_counter += 1
        product_id = get_product_id_from_barcode(scanner_input.Barcode)

        item = Item(
            id=item_counter,
            metadata=ItemMetadata(
                product_id=product_id,
                barcode=scanner_input.Barcode,
                dimensions=Dimensions(
                    width=scanner_input.Width,
                    height=scanner_input.Height,
                    depth=scanner_input.Depth
                ),
                weight=scanner_input.Weight,
                expiration=safe_parse_datetime(scanner_input.Exp),
                batch=scanner_input.Batch,
                delivery_id=scanner_input.DeliveryId or "UNKNOWN"
            )
        )

        # Find or create VSU (VSU is created but item not added yet)
        target_vsu, is_new_vsu = find_or_create_vsu_for_item(item)

        if not target_vsu:
            item_counter -= 1  # Rollback counter
            raise HTTPException(status_code=400, detail={
                "error": "No suitable storage location found",
                "reason": "Warehouse full or item too large",
                "item_dimensions": {
                    "width": scanner_input.Width,
                    "height": scanner_input.Height,
                    "depth": scanner_input.Depth
                },
                "action_needed": "Manual placement required or warehouse optimization needed"
            })

        # Calculate stock index (back-to-front placement)
        stock_index = calculate_stock_index(target_vsu, item.metadata.dimensions.depth, is_new_vsu)

        # Calculate exact Z position for this item
        z_position = calculate_item_z_position(target_vsu, item.metadata.dimensions.depth, stock_index)

        # Calculate max capacity for this VSU
        max_capacity = calculate_max_items_in_vsu(target_vsu, item.metadata.dimensions.depth)

        # Store item temporarily (not in VSU yet)
        items[item.id] = item
        item.stock_index = stock_index

        # Get location details
        shelf = shelves[target_vsu.shelf_id]
        rack = racks[shelf.rack_id]

        # Create task (pending confirmation)
        task_counter += 1
        task_id = f"TASK_{task_counter:03d}"

        task = Task(
            id=task_id,
            item_id=item.id,
            destination_vsu_id=target_vsu.id,
            robot_id="",
            score=calculate_placement_score(item, target_vsu),
            is_new_vsu=is_new_vsu,
            z_position=z_position
        )
        tasks[task_id] = task

        # Reserve this placement (prevents duplicate suggestions)
        pending_placements[task_id] = {
            "vsu_id": target_vsu.id,
            "stock_index": stock_index,
            "item_depth": item.metadata.dimensions.depth,
            "item_width": item.metadata.dimensions.width,
            "item_height": item.metadata.dimensions.height,
            "product_id": item.metadata.product_id
        }

        # Update progress (pending)
        progress["total_boxes"] += 1

        return {
            "task_id": task_id,
            "product_id": product_id,
            "placement": {
                "rack": rack.name,
                "shelf": shelf.name,
                "vsu_code": target_vsu.code,
                "vsu_id": target_vsu.id,
                "stock_index": stock_index,
                "max_capacity": max_capacity,
                "coordinates": {
                    "x": int(target_vsu.position.x),
                    "y": int(target_vsu.position.y),
                    "z": int(z_position)
                }
            },
            "is_new_vsu": is_new_vsu,
            "vsu_dimensions": {
                "width": target_vsu.dimensions.width,
                "height": target_vsu.dimensions.height,
                "depth": target_vsu.dimensions.depth
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.post("/task/{task_id}/complete", tags=["Stock-In Operations"])
async def complete_task(task_id: str):
    """
    Step 2: Complete the stock-in task and update inventory.

    - Adds item to VSU
    - Logs the stock-in operation
    - Saves warehouse state to file
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = tasks[task_id]

    if task.status == "completed":
        return {"message": "Task already completed", "task_id": task_id}

    try:
        # Update task status
        task.status = "completed"

        # Remove from pending placements (reservation complete)
        if task_id in pending_placements:
            del pending_placements[task_id]

        # Get item and VSU
        item = items[task.item_id]
        vsu = virtual_units[task.destination_vsu_id]

        # Add item to VSU (now actually register it)
        if item.id not in vsu.items:
            vsu.items.append(item.id)
            item.vsu_id = vsu.id
            # stock_index was already calculated in suggest

        vsu.occupied = True

        # Get Z position from task (stored during suggest)
        z_position = getattr(task, 'z_position', vsu.position.z)

        # Update progress
        progress["completed"] += 1

        # Log stock-in operation
        shelf = shelves.get(vsu.shelf_id)
        log_stockin(
            product_id=item.metadata.product_id,
            barcode=item.metadata.barcode,
            batch=item.metadata.batch,
            vsu_code=vsu.code,
            shelf_name=shelf.name if shelf else None,
            coordinates={"x": vsu.position.x, "y": vsu.position.y, "z": z_position}
        )

        # Save warehouse state
        save_warehouse_state()

        return {
            "status": "success",
            "task_id": task_id,
            "message": "Item placed successfully",
            "product_id": item.metadata.product_id,
            "barcode": item.metadata.barcode,
            "vsu_code": vsu.code,
            "stock_index": item.stock_index,
            "coordinates": {
                "x": int(vsu.position.x),
                "y": int(vsu.position.y),
                "z": int(z_position)
            },
            "items_in_vsu": len(vsu.items)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to complete task: {str(e)}")

@app.post("/task/{task_id}/fail", tags=["Stock-In Operations"])
async def fail_task(task_id: str, reason: Optional[str] = "Unknown error"):
    """Mark task as failed and free the robot"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    task = tasks[task_id]
    
    if task.status in ["completed", "failed"]:
        return {"message": f"Task already {task.status}", "task_id": task_id}
    
    try:
        # Update task status
        task.status = "failed"

        # Release pending placement reservation
        if task_id in pending_placements:
            del pending_placements[task_id]

        # FREE THE ROBOT - Return to input position and set IDLE
        robot = robots.get(task.robot_id)
        if robot:
            robot.position = INPUT_POSITION  # Robot returns to input
            robot.status = "IDLE"  # Robot is now available
            robot.current_task_id = None
            
            # Save robot state to file
            save_robots_to_file()
        
        # Update progress
        progress["failed"] += 1
        progress["current_box"] += 1
        
        return {
            "status": "failed",
            "task_id": task_id,
            "reason": reason,
            "robot_freed": robot.id if robot else None,
            "robot_status": robot.status if robot else None,
            "robot_position": {
                "x": INPUT_POSITION.x,
                "y": INPUT_POSITION.y,
                "z": INPUT_POSITION.z
            },
            "robot_returned_to": "INPUT_POSITION",
            "progress": progress
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to mark task as failed: {str(e)}")

@app.get("/health", tags=["System & Configuration"])
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "warehouse_loaded": len(virtual_units) > 0,
        "robots_available": sum(1 for r in robots.values() if r.status == "IDLE"),
        "total_vsus": len(virtual_units)
    }

@app.get("/robots/status", tags=["System & Configuration"])
async def get_robots_status():
    """Get current status and position of all robots"""
    return {
        "robots": [
            {
                "id": robot.id,
                "status": robot.status,
                "position": {
                    "x": robot.position.x,
                    "y": robot.position.y,
                    "z": robot.position.z
                },
                "battery": robot.battery,
                "current_task": robot.current_task_id,
                "available": robot.status == "IDLE"
            }
            for robot in robots.values()
        ],
        "summary": {
            "total_robots": len(robots),
            "idle": sum(1 for r in robots.values() if r.status == "IDLE"),
            "busy": sum(1 for r in robots.values() if r.status == "BUSY"),
            "error": sum(1 for r in robots.values() if r.status == "ERROR")
        }
    }

@app.post("/robots/reset", tags=["System & Configuration"])
async def reset_robots():
    """Reset all robots to IDLE state (for testing/recovery)"""
    global robots
    reset_count = 0
    for robot_id, robot in robots.items():
        if robot.status != "IDLE":
            robot.status = "IDLE"
            robot.current_task_id = None
            reset_count += 1

    save_robots_to_file()

    return {
        "status": "success",
        "message": f"Reset {reset_count} robots to IDLE",
        "robots": {
            robot_id: {"status": robot.status, "current_task_id": robot.current_task_id}
            for robot_id, robot in robots.items()
        }
    }

@app.get("/tasks/status", tags=["System & Configuration"])
async def get_tasks_status():
    """Get all tasks with their status"""
    task_list = []
    
    for task in tasks.values():
        item = items.get(task.item_id)
        vsu = virtual_units.get(task.destination_vsu_id)
        
        task_info = {
            "task_id": task.id,
            "status": task.status,
            "robot_id": task.robot_id,
            "created_at": task.created_at.isoformat(),
            "item": {
                "id": item.id,
                "barcode": item.metadata.barcode if item else None,
                "product_id": item.metadata.product_id if item else None
            } if item else None,
            "destination": {
                "vsu_id": vsu.id,
                "vsu_code": vsu.code,
                "position": {
                    "x": vsu.position.x,
                    "y": vsu.position.y,
                    "z": vsu.position.z
                }
            } if vsu else None
        }
        task_list.append(task_info)
    
    # Sort by creation time (newest first)
    task_list.sort(key=lambda t: t["created_at"], reverse=True)
    
    return {
        "tasks": task_list,
        "summary": {
            "total": len(tasks),
            "pending": sum(1 for t in tasks.values() if t.status == "pending"),
            "completed": sum(1 for t in tasks.values() if t.status == "completed"),
            "failed": sum(1 for t in tasks.values() if t.status == "failed")
        }
    }

@app.get("/tasks/{task_id}", tags=["System & Configuration"])
async def get_task_details(task_id: str):
    """Get detailed information about a specific task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    task = tasks[task_id]
    item = items.get(task.item_id)
    vsu = virtual_units.get(task.destination_vsu_id)
    robot = robots.get(task.robot_id)
    
    shelf = None
    rack = None
    if vsu:
        shelf = shelves.get(vsu.shelf_id)
        if shelf:
            rack = racks.get(shelf.rack_id)
    
    return {
        "task": {
            "id": task.id,
            "status": task.status,
            "created_at": task.created_at.isoformat(),
            "score": task.score
        },
        "item": {
            "id": item.id,
            "barcode": item.metadata.barcode,
            "product_id": item.metadata.product_id,
            "dimensions": {
                "width": item.metadata.dimensions.width,
                "height": item.metadata.dimensions.height,
                "depth": item.metadata.dimensions.depth
            },
            "batch": item.metadata.batch,
            "expiration": item.metadata.expiration.isoformat(),
            "stock_index": item.stock_index
        } if item else None,
        "destination": {
            "vsu_id": vsu.id,
            "vsu_code": vsu.code,
            "shelf": shelf.name if shelf else None,
            "rack": rack.name if rack else None,
            "position": {
                "x": vsu.position.x,
                "y": vsu.position.y,
                "z": vsu.position.z
            }
        } if vsu else None,
        "robot": {
            "id": robot.id,
            "status": robot.status,
            "position": {
                "x": robot.position.x,
                "y": robot.position.y,
                "z": robot.position.z
            },
            "battery": robot.battery
        } if robot else None
    }

@app.get("/warehouse/empty-vsus", tags=["Warehouse Management"])
async def get_empty_vsus():
    """
    Get all empty VSUs available for placement
    Sorted by smallest height first (most efficient)
    """
    empty_vsus = []
    
    for vsu in virtual_units.values():
        if not vsu.items:  # Empty VSU
            shelf = shelves.get(vsu.shelf_id)
            rack = racks.get(shelf.rack_id) if shelf else None
            
            empty_vsus.append({
                "vsu_code": vsu.code,
                "vsu_id": vsu.id,
                "dimensions": {
                    "width": vsu.dimensions.width,
                    "height": vsu.dimensions.height,
                    "depth": vsu.dimensions.depth,
                    "volume": vsu.dimensions.volume
                },
                "position": {
                    "x": vsu.position.x,
                    "y": vsu.position.y,
                    "z": vsu.position.z
                },
                "shelf": shelf.name if shelf else "unknown",
                "rack": rack.name if rack else "unknown"
            })
    
    # Sort by smallest height first, then smallest volume
    empty_vsus.sort(key=lambda v: (v["dimensions"]["height"], v["dimensions"]["volume"]))
    
    return {
        "total_empty_vsus": len(empty_vsus),
        "empty_vsus": empty_vsus,
        "grouped_by_size": group_vsus_by_size(empty_vsus)
    }

def group_vsus_by_size(vsus_list):
    """Group VSUs by similar dimensions"""
    from collections import defaultdict
    groups = defaultdict(list)
    
    for vsu in vsus_list:
        dims = vsu["dimensions"]
        size_key = f"{dims['width']}×{dims['height']}×{dims['depth']}"
        groups[size_key].append(vsu["vsu_code"])
    
    return {size: codes for size, codes in groups.items()}

@app.get("/warehouse/shelf-space", tags=["Warehouse Management"])
async def get_shelf_space():
    """
    Get available space on each shelf for creating new VSUs
    Shows which shelves have space and how much
    """
    shelf_spaces = []
    
    for shelf in shelves.values():
        shelf_start = shelf.position.x
        shelf_end = shelf_start + shelf.dimensions.width
        
        # Get VSUs on this shelf
        shelf_vsus = [virtual_units[vid] for vid in shelf.virtual_units if vid in virtual_units]
        
        if shelf_vsus:
            # Find rightmost VSU
            rightmost = max(shelf_vsus, key=lambda v: v.position.x + v.dimensions.width)
            next_x = rightmost.position.x + rightmost.dimensions.width + VSU_HORIZONTAL_GAP
            available_width = shelf_end - next_x
        else:
            # Empty shelf
            available_width = shelf.dimensions.width
        
        rack = racks.get(shelf.rack_id)
        
        shelf_spaces.append({
            "shelf": shelf.name,
            "rack": rack.name if rack else "unknown",
            "available_width": round(available_width, 1),
            "shelf_height": shelf.dimensions.height,
            "usable_height": round(shelf.dimensions.height - VSU_TOP_CLEARANCE, 1),
            "shelf_depth": shelf.dimensions.depth,
            "total_vsus": len(shelf.virtual_units),
            "empty_vsus": sum(1 for vid in shelf.virtual_units if vid in virtual_units and not virtual_units[vid].items),
            "can_create_new_vsu": available_width > 10,
            "position": {
                "x": shelf.position.x,
                "y": shelf.position.y,
                "z": shelf.position.z
            }
        })
    
    # Sort by available width (descending)
    shelf_spaces.sort(key=lambda s: s["available_width"], reverse=True)
    
    # Filter to only shelves with significant space
    shelves_with_space = [s for s in shelf_spaces if s["available_width"] > 10]
    
    return {
        "total_shelves": len(shelf_spaces),
        "shelves_with_space": len(shelves_with_space),
        "shelves": shelf_spaces,
        "summary": {
            "max_available_width": max((s["available_width"] for s in shelf_spaces), default=0),
            "shelves_with_50mm_plus": sum(1 for s in shelf_spaces if s["available_width"] >= 50),
            "shelves_with_100mm_plus": sum(1 for s in shelf_spaces if s["available_width"] >= 100),
            "completely_full": sum(1 for s in shelf_spaces if s["available_width"] <= 10)
        }
    }

@app.get("/warehouse/capacity", tags=["Warehouse Management"])
async def get_warehouse_capacity():
    """
    Get warehouse capacity analysis
    Shows available space per shelf and largest item that can fit
    """
    capacity_info = {
        "total_shelves": len(shelves),
        "total_vsus": len(virtual_units),
        "occupied_vsus": sum(1 for vsu in virtual_units.values() if vsu.items),
        "empty_vsus": sum(1 for vsu in virtual_units.values() if not vsu.items),
        "shelves_with_space": [],
        "largest_item_possible": {
            "width": 0,
            "height": 0,
            "depth": 0
        }
    }
    
    max_width = 0
    max_height = 0
    max_depth = 0
    
    for shelf in shelves.values():
        # Calculate available space
        shelf_start = shelf.position.x
        shelf_end = shelf_start + shelf.dimensions.width
        
        shelf_vsus = [virtual_units[vid] for vid in shelf.virtual_units if vid in virtual_units]
        
        if shelf_vsus:
            rightmost = max(shelf_vsus, key=lambda v: v.position.x + v.dimensions.width)
            next_x = rightmost.position.x + rightmost.dimensions.width + VSU_HORIZONTAL_GAP
            available_width = max(0, shelf_end - next_x)
        else:
            available_width = shelf.dimensions.width
        
        if available_width > 10:  # Only show if > 10mm available
            shelf_info = {
                "shelf_name": shelf.name,
                "rack_id": shelf.rack_id,
                "available_width": round(available_width, 1),
                "shelf_height": shelf.dimensions.height,
                "usable_height": round(shelf.dimensions.height - VSU_TOP_CLEARANCE, 1),
                "shelf_depth": shelf.dimensions.depth,
                "empty_vsus_count": sum(1 for vid in shelf.virtual_units if vid in virtual_units and not virtual_units[vid].items)
            }
            capacity_info["shelves_with_space"].append(shelf_info)
            
            # Track max dimensions possible
            if available_width > max_width:
                max_width = available_width
            if shelf.dimensions.height - VSU_TOP_CLEARANCE > max_height:
                max_height = shelf.dimensions.height - VSU_TOP_CLEARANCE
            if shelf.dimensions.depth > max_depth:
                max_depth = shelf.dimensions.depth
    
    # Sort by available width (descending)
    capacity_info["shelves_with_space"].sort(key=lambda x: x["available_width"], reverse=True)
    
    capacity_info["largest_item_possible"] = {
        "width": round(max_width, 1),
        "height": round(max_height, 1),
        "depth": round(max_depth, 1)
    }
    
    capacity_info["warehouse_status"] = "FULL" if max_width < 50 else "AVAILABLE"
    capacity_info["recommended_action"] = (
        "Warehouse optimization needed - most shelves full" 
        if max_width < 50 
        else "Normal operation - space available"
    )
    
    return capacity_info

@app.get("/warehouse/stats", tags=["Warehouse Management"])
async def get_warehouse_stats():
    """Get comprehensive warehouse statistics"""
    
    # VSU statistics
    total_vsus = len(virtual_units)
    occupied_vsus = sum(1 for vsu in virtual_units.values() if vsu.occupied)
    empty_vsus = total_vsus - occupied_vsus
    
    # Items per VSU
    vsu_utilization = {}
    for vsu in virtual_units.values():
        if vsu.items:
            vsu_utilization[vsu.code] = len(vsu.items)
    
    # Shelf statistics
    shelves_with_vsus = sum(1 for shelf in shelves.values() if shelf.virtual_units)
    empty_shelves = len(shelves) - shelves_with_vsus
    
    # Product diversity
    unique_products = len(set(item.metadata.product_id for item in items.values()))
    
    return {
        "warehouse": {
            "total_racks": len(racks),
            "total_shelves": len(shelves),
            "empty_shelves": empty_shelves,
            "shelves_in_use": shelves_with_vsus
        },
        "vsus": {
            "total": total_vsus,
            "occupied": occupied_vsus,
            "empty": empty_vsus,
            "utilization_percentage": round((occupied_vsus / total_vsus * 100), 1) if total_vsus > 0 else 0
        },
        "items": {
            "total_items": len(items),
            "unique_products": unique_products,
            "average_items_per_product": round(len(items) / unique_products, 1) if unique_products > 0 else 0
        },
        "top_vsus": sorted(
            [{"vsu": k, "items": v} for k, v in vsu_utilization.items()],
            key=lambda x: x["items"],
            reverse=True
        )[:10],
        "progress": progress
    }

@app.post("/warehouse/commit", tags=["Warehouse Management"])
async def commit_warehouse():
    """
    Make the updated warehouse permanent by replacing original file
    WARNING: This overwrites ml_robot.json with ml_robot_updated.json
    """
    import os
    
    if not os.path.exists('data/ml_robot_updated.json'):
        raise HTTPException(status_code=404, detail="No updated warehouse file found")
    
    try:
        # Backup original if not already backed up
        if not os.path.exists('ml_robot_backup.json'):
            shutil.copy('data/ml_robot.json', 'ml_robot_backup.json')
        
        # Replace original with updated
        shutil.copy('data/ml_robot_updated.json', 'data/ml_robot.json')
        
        return {
            "status": "success",
            "message": "Warehouse changes committed permanently",
            "backup": "ml_robot_backup.json",
            "active": "data/ml_robot.json",
            "note": "Changes will persist after server restart"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to commit: {str(e)}")

@app.post("/warehouse/rollback", tags=["Warehouse Management"])
async def rollback_warehouse():
    """
    Restore warehouse from backup (undo all changes)
    WARNING: This reverts to original ml_robot.json
    """
    import os
    
    if not os.path.exists('ml_robot_backup.json'):
        raise HTTPException(status_code=404, detail="No backup file found")
    
    try:
        # Restore from backup
        shutil.copy('ml_robot_backup.json', 'data/ml_robot.json')
        
        # Remove updated file
        if os.path.exists('data/ml_robot_updated.json'):
            os.remove('data/ml_robot_updated.json')
        
        return {
            "status": "success",
            "message": "Warehouse rolled back to original state",
            "note": "Restart server to load original data"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rollback: {str(e)}")

@app.post("/weights/reload", tags=["System & Configuration"])
async def reload_weights():
    """
    Reload product weights from weights.json
    Call this after nightly ML optimization updates the weights
    """
    try:
        load_product_weights()
        
        return {
            "status": "success",
            "message": "Product weights reloaded successfully",
            "total_products": len(product_weights),
            "default_weight": DEFAULT_WEIGHT,
            "sample_weights": dict(list(product_weights.items())[:5]) if product_weights else {}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload weights: {str(e)}")

@app.get("/weights/status", tags=["System & Configuration"])
async def get_weights_status():
    """
    Get current product weights status
    """
    return {
        "total_products_with_weights": len(product_weights),
        "default_weight": DEFAULT_WEIGHT,
        "weight_range": {
            "min": min(product_weights.values()) if product_weights else DEFAULT_WEIGHT,
            "max": max(product_weights.values()) if product_weights else DEFAULT_WEIGHT,
        },
        "sample_weights": dict(list(product_weights.items())[:10]) if product_weights else {},
        "output_positions": [
            {"x": pos.x, "y": pos.y, "z": pos.z} for pos in OUTPUT_POSITIONS
        ]
    }




# Track performance statistics
performance_stats = {
    "endpoints": {},
    "total_requests": 0,
    "start_time": datetime.now()
}

def update_performance_stats(endpoint: str, duration: float, status_code: int):
    """Update performance statistics for an endpoint"""
    if endpoint not in performance_stats["endpoints"]:
        performance_stats["endpoints"][endpoint] = {
            "count": 0,
            "total_time": 0,
            "min_time": float('inf'),
            "max_time": 0,
            "success_count": 0,
            "error_count": 0
        }
    
    stats = performance_stats["endpoints"][endpoint]
    stats["count"] += 1
    stats["total_time"] += duration
    stats["min_time"] = min(stats["min_time"], duration)
    stats["max_time"] = max(stats["max_time"], duration)
    stats["avg_time"] = stats["total_time"] / stats["count"]
    
    if 200 <= status_code < 300:
        stats["success_count"] += 1
    else:
        stats["error_count"] += 1
    
    performance_stats["total_requests"] += 1

@app.middleware("http")
async def performance_monitoring_middleware(request: Request, call_next):
    """
    PERFORMANCE MONITORING MIDDLEWARE
    
    Automatically tracks timing for ALL endpoints
    
    Features:
    - Measures execution time in milliseconds
    - Color-coded console output ()
    - Adds X-Process-Time-Ms header to response
    - Tracks statistics (min/max/avg)
    - Warns about slow endpoints (>500ms)
    """
    import time
    start_time = time.time()
    
    # Log incoming request with timestamp
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"\n{'='*80}")
    print(f"[{timestamp}] {request.method} {request.url.path}")
    
    # Process the request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    duration_ms = duration * 1000
    
    # Add timing headers (visible to client!)
    response.headers["X-Process-Time"] = f"{duration:.6f}"
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    
    # Color-coded logging based on speed
    if duration_ms < 100:
        icon = ""
        label = "FAST"
    elif duration_ms < 500:
        icon = ""
        label = "OK"
    elif duration_ms < 1000:
        icon = ""
        label = "SLOW"
    else:
        icon = ""
        label = "VERY SLOW"
    
    # Log the result
    print(f"{icon} {request.method} {request.url.path}")
    print(f"   Status: {response.status_code}")
    print(f"   Duration: {duration_ms:.2f}ms ({label})")
    
    # Warn about slow endpoints
    if duration_ms > 500:
        print(f"   WARNING: Slow endpoint detected!")
        print(f"   Consider optimization")
    
    # Update statistics
    endpoint = f"{request.method} {request.url.path}"
    update_performance_stats(endpoint, duration, response.status_code)
    
    print(f"{'='*80}\n")
    
    return response

@app.get("/performance/stats", tags=["Performance Monitoring"])
async def get_performance_stats():
    """
    Get comprehensive performance statistics
    
    Returns timing data for all endpoints:
    - Request counts
    - Min/Max/Average times
    - Success/Error rates
    - Sorted by average time (slowest first)
    """
    uptime = (datetime.now() - performance_stats["start_time"]).total_seconds()
    
    # Sort endpoints by average time (slowest first)
    sorted_endpoints = sorted(
        performance_stats["endpoints"].items(),
        key=lambda x: x[1]["avg_time"],
        reverse=True
    )
    
    endpoint_list = []
    for endpoint, stats in sorted_endpoints:
        endpoint_list.append({
            "endpoint": endpoint,
            "calls": stats["count"],
            "times_ms": {
                "min": round(stats["min_time"] * 1000, 2),
                "max": round(stats["max_time"] * 1000, 2),
                "avg": round(stats["avg_time"] * 1000, 2)
            },
            "success_rate": round(stats["success_count"] / stats["count"] * 100, 1) if stats["count"] > 0 else 0
        })
    
    return {
        "summary": {
            "total_requests": performance_stats["total_requests"],
            "unique_endpoints": len(performance_stats["endpoints"]),
            "uptime_seconds": round(uptime, 2),
            "requests_per_minute": round((performance_stats["total_requests"] / uptime * 60), 2) if uptime > 0 else 0
        },
        "endpoints": endpoint_list
    }

@app.get("/performance/slowest", tags=["Performance Monitoring"])
async def get_slowest_endpoints(limit: int = 10):
    """
    Get the slowest endpoints
    
    Useful for finding bottlenecks
    """
    sorted_endpoints = sorted(
        performance_stats["endpoints"].items(),
        key=lambda x: x[1]["avg_time"],
        reverse=True
    )[:limit]
    
    return {
        "slowest_endpoints": [
            {
                "endpoint": endpoint,
                "avg_time_ms": round(stats["avg_time"] * 1000, 2),
                "max_time_ms": round(stats["max_time"] * 1000, 2),
                "count": stats["count"]
            }
            for endpoint, stats in sorted_endpoints
        ]
    }

@app.post("/performance/reset", tags=["Performance Monitoring"])
async def reset_performance_stats():
    """
    Reset all performance statistics
    
    Useful for starting fresh after making optimizations
    """
    global performance_stats
    performance_stats = {
        "endpoints": {},
        "total_requests": 0,
        "start_time": datetime.now()
    }
    return {
        "status": "success",
        "message": "Performance statistics reset",
        "reset_at": datetime.now().isoformat()
    }
from dispensing import (
    DispenseRequest,
    CompleteDispenseRequest,
    FailDispenseRequest,
    create_dispense_task_endpoint,
    complete_dispense_endpoint,
    fail_dispense_endpoint,
    get_dispense_logs_endpoint,
    get_product_dispense_history_endpoint,
    get_task_status_endpoint
)

from stockin_logging import (
    log_stockin,
    get_stockin_logs,
    get_product_stockin_history
)

# ==================== DISPENSE ENDPOINTS MOVED TO PORT 8001 ====================
# Dispense operations are now on dispense_server.py (port 8001)
# Run both servers with: python run_servers.py
# ============================================================================

# ==================== STOCKIN LOGGING ENDPOINTS ====================

@app.get("/stockin/logs", tags=["Stock-In Logs & History"])
async def get_stockin_logs_endpoint():
    """
    GET STOCK-IN LOGS
    
    Returns summary of all stock-in operations
    """
    return get_stockin_logs()

@app.get("/stockin/product/{product_id}", tags=["Stock-In Logs & History"])
async def get_product_stockin(product_id: int):
    """
    GET PRODUCT STOCK-IN HISTORY

    Returns detailed stock-in history for a specific product
    """
    return get_product_stockin_history(product_id)

@app.post("/stockin/logs/reset", tags=["Stock-In Logs & History"])
async def reset_stockin_logs():
    """
    RESET STOCK-IN LOGS

    Archives current stockin_logs.json and starts fresh
    """
    from stockin_logging import reset_stockin_logs
    return reset_stockin_logs()

# ==================== INVENTORY MONITORING ENDPOINTS ====================

@app.get("/inventory/expiring", tags=["Inventory Monitoring"])
async def get_expiring_items(days: int = 30):
    """
    GET EXPIRING ITEMS

    Returns all items expiring within the specified number of days

    Query params:
    - days: Number of days to look ahead (default: 30)

    Returns product details with location information
    """
    from datetime import datetime, timedelta

    expiry_threshold = datetime.now() + timedelta(days=days)
    expiring_items = []

    for item in items.values():
        # Handle timezone-aware/naive comparison
        item_expiry = item.metadata.expiration.replace(tzinfo=None) if item.metadata.expiration.tzinfo else item.metadata.expiration
        if item_expiry <= expiry_threshold:
            # Find VSU and shelf for this item
            vsu = virtual_units.get(item.vsu_id)
            shelf = shelves.get(vsu.shelf_id) if vsu else None
            rack = racks.get(shelf.rack_id) if shelf else None

            days_until_expiry = (item_expiry - datetime.now()).days

            expiring_items.append({
                "item_id": item.id,
                "product_id": item.metadata.product_id,
                "barcode": item.metadata.barcode,
                "batch": item.metadata.batch,
                "expiration": item.metadata.expiration.isoformat(),
                "days_until_expiry": days_until_expiry,
                "status": "EXPIRED" if days_until_expiry < 0 else "EXPIRING_SOON",
                "location": {
                    "rack": rack.name if rack else "Unknown",
                    "shelf": shelf.name if shelf else "Unknown",
                    "vsu_code": vsu.code if vsu else "Unknown",
                    "coordinates": {
                        "x": vsu.position.x if vsu else 0,
                        "y": vsu.position.y if vsu else 0,
                        "z": vsu.position.z if vsu else 0
                    }
                }
            })

    # Sort by days until expiry (most urgent first)
    expiring_items.sort(key=lambda x: x["days_until_expiry"])

    return {
        "status": "success",
        "threshold_days": days,
        "total_items": len(expiring_items),
        "expired_count": sum(1 for x in expiring_items if x["days_until_expiry"] < 0),
        "expiring_soon_count": sum(1 for x in expiring_items if x["days_until_expiry"] >= 0),
        "items": expiring_items
    }

@app.get("/inventory/stock-levels", tags=["Inventory Monitoring"])
async def get_stock_levels(product_id: int = None, barcode: str = None):
    """
    GET STOCK LEVELS

    Returns stock levels and locations for products

    Query params:
    - product_id: Filter by product ID (optional)
    - barcode: Filter by barcode (optional)

    Returns quantity and location details
    """
    if not product_id and not barcode:
        return {"status": "error", "message": "Please provide product_id or barcode"}

    # Filter items
    filtered_items = []
    for item in items.values():
        if product_id and item.metadata.product_id != product_id:
            continue
        if barcode and item.metadata.barcode != barcode:
            continue
        filtered_items.append(item)

    if not filtered_items:
        return {
            "status": "success",
            "product_id": product_id,
            "barcode": barcode,
            "total_quantity": 0,
            "locations": [],
            "alert": "OUT_OF_STOCK"
        }

    # Group by location
    location_map = {}
    for item in filtered_items:
        vsu = virtual_units.get(item.vsu_id)
        shelf = shelves.get(vsu.shelf_id) if vsu else None
        rack = racks.get(shelf.rack_id) if shelf else None

        location_key = f"{rack.name if rack else 'Unknown'}|{shelf.name if shelf else 'Unknown'}|{vsu.code if vsu else 'Unknown'}"

        if location_key not in location_map:
            location_map[location_key] = {
                "rack": rack.name if rack else "Unknown",
                "shelf": shelf.name if shelf else "Unknown",
                "vsu_code": vsu.code if vsu else "Unknown",
                "coordinates": {
                    "x": vsu.position.x if vsu else 0,
                    "y": vsu.position.y if vsu else 0,
                    "z": vsu.position.z if vsu else 0
                },
                "quantity": 0,
                "items": []
            }

        location_map[location_key]["quantity"] += 1
        location_map[location_key]["items"].append({
            "item_id": item.id,
            "batch": item.metadata.batch,
            "expiration": item.metadata.expiration.isoformat()
        })

    locations = list(location_map.values())
    total_qty = sum(loc["quantity"] for loc in locations)

    # Determine alert level
    alert = "OK"
    if total_qty == 0:
        alert = "OUT_OF_STOCK"
    elif total_qty <= 5:
        alert = "LOW_STOCK"
    elif total_qty <= 10:
        alert = "MEDIUM_STOCK"

    return {
        "status": "success",
        "product_id": product_id or filtered_items[0].metadata.product_id,
        "barcode": barcode or filtered_items[0].metadata.barcode,
        "total_quantity": total_qty,
        "alert": alert,
        "locations": locations
    }

@app.get("/analytics/warehouse-utilization", tags=["Warehouse Analytics"])
async def get_warehouse_utilization():
    """
    GET WAREHOUSE UTILIZATION METRICS

    Returns VSU fill rates, shelf usage, and empty spaces
    """
    total_vsus = len(virtual_units)
    occupied_vsus = sum(1 for vsu in virtual_units.values() if len(vsu.items) > 0)
    empty_vsus = total_vsus - occupied_vsus

    # Calculate VSU fill rates (capacity is max items per VSU, assume 10 for now)
    vsu_details = []
    for vsu in virtual_units.values():
        shelf = shelves.get(vsu.shelf_id)
        rack = racks.get(shelf.rack_id) if shelf else None

        capacity = 10  # Default capacity per VSU
        fill_rate = len(vsu.items) / capacity if capacity > 0 else 0

        vsu_details.append({
            "vsu_code": vsu.code,
            "rack": rack.name if rack else "Unknown",
            "shelf": shelf.name if shelf else "Unknown",
            "capacity": capacity,
            "current_stock": len(vsu.items),
            "fill_rate": round(fill_rate * 100, 2),
            "status": "FULL" if len(vsu.items) >= capacity else ("EMPTY" if len(vsu.items) == 0 else "PARTIAL")
        })

    # Shelf utilization
    shelf_utilization = []
    for shelf in shelves.values():
        rack = racks.get(shelf.rack_id)
        total_shelf_vsus = len(shelf.virtual_units)
        occupied_shelf_vsus = sum(1 for vsu_id in shelf.virtual_units if vsu_id in virtual_units and len(virtual_units[vsu_id].items) > 0)

        shelf_utilization.append({
            "rack": rack.name if rack else "Unknown",
            "shelf": shelf.name,
            "total_vsus": total_shelf_vsus,
            "occupied_vsus": occupied_shelf_vsus,
            "empty_vsus": total_shelf_vsus - occupied_shelf_vsus,
            "utilization_rate": round((occupied_shelf_vsus / total_shelf_vsus * 100) if total_shelf_vsus > 0 else 0, 2)
        })

    return {
        "status": "success",
        "summary": {
            "total_vsus": total_vsus,
            "occupied_vsus": occupied_vsus,
            "empty_vsus": empty_vsus,
            "overall_utilization": round((occupied_vsus / total_vsus * 100) if total_vsus > 0 else 0, 2),
            "total_items": len(items)
        },
        "vsu_details": vsu_details,
        "shelf_utilization": shelf_utilization
    }


# ==================== SCHEDULER MANAGEMENT ENDPOINTS ====================

@app.get("/scheduler/status", tags=["Automatic Archiving Scheduler"])
async def get_scheduler_status():
    """
    GET SCHEDULER STATUS

    Check the status of the automatic daily archiving scheduler

    Returns:
    - Running status
    - Next scheduled run time
    - Job details
    """
    from scheduler import get_scheduler_status
    return get_scheduler_status()


@app.post("/scheduler/archive/manual", tags=["Automatic Archiving Scheduler"])
async def trigger_manual_archiving():
    """
    TRIGGER MANUAL ARCHIVING

    Manually trigger the archiving process immediately (doesn't wait for midnight)

    Useful for:
    - Testing the archiving system
    - Immediate cleanup of old data
    - Running archiving on-demand

    Returns:
    - Number of records archived
    - Success/error status
    """
    from scheduler import trigger_manual_archiving
    return trigger_manual_archiving()


@app.get("/scheduler/runs", tags=["Automatic Archiving Scheduler"])
async def get_scheduler_runs(limit: int = 10):
    """
    GET SCHEDULER RUN HISTORY

    View recent scheduler executions to see what happened at midnight

    Query params:
    - limit: Number of recent runs to show (default: 10, max: 30)

    Returns:
    - List of recent runs with timestamps
    - Records archived per run
    - Success/failure status
    - Duration of each run
    - Trigger type (scheduled vs manual)

    Example response:
    {
      "runs": [
        {
          "success": true,
          "timestamp": "2025-11-17T00:00:00",
          "records_archived": 15,
          "records_before": 26,
          "records_after": 11,
          "duration_seconds": 0.23,
          "trigger": "scheduled"
        }
      ],
      "total": 5,
      "showing": 5
    }
    """
    from scheduler import get_scheduler_runs
    limit = min(limit, 30)  # Max 30 runs
    return get_scheduler_runs(limit=limit)


# ==================== RELOCATION ENDPOINTS ====================

from relocation import (
    relocate_item,
    get_relocation_history,
    get_relocation_stats
)

class TemporaryRelocateRequest(BaseModel):
    """Request to relocate item to temporary storage - ONLY accepts task_id from dispense instruction"""
    task_id: str = Field(..., description="Task ID from dispense instruction (e.g., RELOCATE-001) - REQUIRED")

class TemporaryRestockRequest(BaseModel):
    """Request to restock items from temporary storage"""
    item_ids: Optional[List[int]] = None  # If None, restock all items

@app.post("/api/temporary/relocate", tags=["Temporary Storage"])
async def relocate_to_temporary_storage(request: TemporaryRelocateRequest):
    """
    Relocate an obstructing box to temporary storage using task_id

    This endpoint is called by the robot during dispense execution
    when it needs to move a blocking box out of the way.

    Process:
    1. Lookup task_id from dispense instruction (contains item details)
    2. Find available temp VSU on same shelf
    3. Move item to temp VSU (coordinates returned)
    4. Remove item from main inventory (ml_robot_updated.json)
    5. Add item to temporary storage (temporary_storage.json)

    Request body:
    {
        "task_id": "RELOCATE-001"
    }

    Response:
    {
        "status": "success",
        "task_id": "RELOCATE-001",
        "item_id": 123,
        "barcode": "TEST_BOX_123",
        "product_id": 20,
        "original_vsu_code": "vu15",
        "temp_vsu_id": 101,
        "temp_vsu_code": "temp_rack1_1_1",
        "temp_coordinates": {
            "x": 550.0,
            "y": 825.0,
            "z": 40.0
        },
        "relocated_at": "2025-12-06T10:30:00"
    }
    """
    try:
        global relocate_tasks_store

        print(f"\n{'='*60}")
        print(f"TEMPORARY RELOCATION REQUEST")
        print(f"{'='*60}")
        print(f"Task ID: {request.task_id}")

        # Lookup task details from relocate_tasks_store
        if request.task_id not in relocate_tasks_store:
            raise HTTPException(
                status_code=404,
                detail=f"Task ID {request.task_id} not found. Make sure to call /dispense/task first to get valid task IDs."
            )

        task_info = relocate_tasks_store[request.task_id]
        item_id = task_info["item_id"]

        print(f"Item ID: {item_id}")
        print(f"Barcode: {task_info['barcode']}")
        print(f"Product ID: {task_info['product_id']}")

        # Get item from warehouse
        if item_id not in items:
            raise HTTPException(status_code=404, detail=f"Item {item_id} not found in warehouse")

        item = items[item_id]
        barcode = item.metadata.barcode
        product_id = item.metadata.product_id
        original_vsu_id = item.vsu_id
        original_vsu_code = virtual_units[original_vsu_id].code if original_vsu_id and original_vsu_id in virtual_units else None

        # Relocate item (updates VSU in main inventory directly)
        # Pass VSU creation parameters so relocation can create new VSUs like stock-in
        global vsu_counter
        relocation_info = relocate_item(
            item_id=item_id,
            items=items,
            virtual_units=virtual_units,
            shelves=shelves,
            reason="obstruction_removal",
            vsu_counter=vsu_counter,
            VirtualStorageUnit=VirtualStorageUnit,
            Dimensions=Dimensions,
            Position=Position
        )

        # Update vsu_counter if a new VSU was created
        if relocation_info.get("is_new_vsu") and relocation_info.get("vsu_counter"):
            vsu_counter = relocation_info["vsu_counter"]
            print(f"[RELOCATE] Updated vsu_counter to {vsu_counter}")

        # Save updated warehouse state (item VSU was updated in-place)
        save_warehouse_state()

        # Update task status in store
        relocate_tasks_store[request.task_id].update({
            "original_vsu_id": relocation_info.get("original_vsu_id"),
            "original_vsu_code": relocation_info.get("original_vsu_code"),
            "new_vsu_id": relocation_info.get("new_vsu_id"),
            "new_vsu_code": relocation_info.get("new_vsu_code"),
            "status": "completed",
            "relocated_at": relocation_info.get("relocated_at")
        })

        print(f"Item {item_id} relocated (VSU updated in main inventory)")
        print(f"  From: {relocation_info.get('original_vsu_code')}")
        print(f"  To: {relocation_info.get('new_vsu_code')}")
        print(f"{'='*60}\n")

        return {
            "status": "success",
            "task_id": request.task_id,
            "item_id": item_id,
            "barcode": barcode,
            "product_id": product_id,
            "original_vsu_code": original_vsu_code,
            **relocation_info
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error relocating to temporary storage: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to relocate item: {str(e)}"
        )

# Dictionary to store relocate tasks (task_id -> task_data)
relocate_tasks_store = {}

@app.get("/api/relocation/history", tags=["Relocation"])
async def get_relocation_history_endpoint():
    """
    Get relocation history (audit trail)

    Response:
    {
        "status": "success",
        "total_relocations": 5,
        "relocations": [
            {
                "relocation_id": 1,
                "item_id": 123,
                "product_id": 1411,
                "barcode": "ABC123",
                "original_vsu_id": 15,
                "original_vsu_code": "vu15",
                "new_vsu_id": 20,
                "new_vsu_code": "vu20",
                "relocated_at": "2025-12-04T10:30:00",
                "reason": "obstruction_removal"
            }
        ]
    }
    """
    try:
        history = get_relocation_history()

        return {
            "status": "success",
            "total_relocations": len(history),
            "relocations": history
        }

    except Exception as e:
        print(f"Error getting relocation history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get relocation history: {str(e)}"
        )

@app.get("/api/relocation/stats", tags=["Relocation"])
async def get_relocation_stats_endpoint():
    """
    Get relocation statistics

    Response:
    {
        "status": "success",
        "total_relocations": 5,
        "last_updated": "2025-12-04T10:30:00"
    }
    """
    try:
        stats = get_relocation_stats()

        return {
            "status": "success",
            **stats
        }

    except Exception as e:
        print(f"Error getting relocation stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get relocation stats: {str(e)}"
        )

# ==================== PRODUCT ARCHIVE ====================

@app.get("/api/archive/items", tags=["Product Archive"])
async def get_archived_items():
    """
    Get all archived dispensed items

    Response:
    {
        "status": "success",
        "total_items": 10,
        "items": [
            {
                "item_id": 1,
                "product_id": 400,
                "barcode": "050991510066151",
                "vsu_code": "vu5",
                "shelf_name": "Shelf 213",
                "coordinates": {"x": 141.0, "y": 784.0, "z": -40.0},
                "dispensed_at": "2025-12-06T14:30:00",
                "task_id": "DISP-001"
            }
        ]
    }
    """
    try:
        from product_archive import load_product_archive

        archive = load_product_archive()

        return {
            "status": "success",
            "total_items": len(archive.items),
            "items": [
                {
                    "item_id": item.item_id,
                    "product_id": item.product_id,
                    "barcode": item.barcode,
                    "batch": item.batch,
                    "expiration": item.expiration,
                    "vsu_code": item.vsu_code,
                    "shelf_name": item.shelf_name,
                    "coordinates": item.coordinates,
                    "dispensed_at": item.dispensed_at.isoformat(),
                    "task_id": item.task_id
                }
                for item in archive.items.values()
            ]
        }

    except Exception as e:
        print(f"Error getting archived items: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get archived items: {str(e)}"
        )


@app.get("/api/archive/stats", tags=["Product Archive"])
async def get_archive_stats():
    """
    Get product archive statistics

    Response:
    {
        "status": "success",
        "total_items": 10,
        "total_products": 3,
        "products": [
            {
                "product_id": 400,
                "barcode": "050991510066151",
                "count": 5
            }
        ],
        "last_updated": "2025-12-06T14:30:00"
    }
    """
    try:
        from product_archive import get_archive_stats

        stats = get_archive_stats()

        return {
            "status": "success",
            **stats
        }

    except Exception as e:
        print(f"Error getting archive stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get archive stats: {str(e)}"
        )

# ==================== SERVER STARTUP ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
