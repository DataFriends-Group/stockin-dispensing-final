"""
Central configuration for MedicPort.

INVENTORY_FILE is the single working data file used across the app
(loaded at startup, saved after every state change, and used by the
commit/rollback endpoints). Change it here to switch datasets.
"""
import os

DATA_DIR = "data"

INVENTORY_FILENAME = "R3_DF.json"
INVENTORY_FILE = os.path.join(DATA_DIR, INVENTORY_FILENAME)
