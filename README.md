# MedicPort – Warehouse Robotics Backend

## Popis projektu

Backend systém pro automatizovaný farmaceutický sklad MedicPort. Řídí robotické zaskladnění a výdej léčiv s dodržováním FEFO (First Expiry First Out), optimalizací umístění na základě ML vah poptávky a koordinací více robotů s vyhýbáním kolizím.

Systém běží jako dvě FastAPI aplikace ve sdílené paměti:
- **Port 8000** — Zaskladnění (stock-in), správa skladu, inventář, monitoring
- **Port 8001** — Výdej (dispensing) s multi-pick batchingem a výstupním queueingem

## Struktura projektu

 Složka / soubor | Popis 

 `main.py` | Hlavní server (port 8000) – zaskladnění, inventář, warehouse management, performance monitoring, scheduler 
 `dispensing.py` | Výdejní logika – multi-pick batching, robot koordinace, collision avoidance, output queuing, dispense logy 
 `dispense_server.py` | FastAPI wrapper pro dispensing modul (port 8001), sdílí paměť s main 
 `relocation.py` | Přemísťování obstrukčních položek – hledání/vytváření VSU pro relokaci 
 `stockin_logging.py` | Logování zaskladnění – denní/týdenní/měsíční agregace, historie 
 `product_archive.py` | Archivace vydaných položek pro historické sledování 
 `scheduler.py` | APScheduler – denní archivace dispense logů starších 30 dní do měsíčních souborů 
 `run_servers.py` | Spouštěč obou serverů (hypercorn, sdílená paměť) 
 `data/` | Datové soubory (viz níže) 

### Datové soubory (`data/`)

 Soubor | Popis 
 
 `ml_robot.json` | Původní stav skladu (záloha) 
 `ml_robot_updated.json` | Aktuální stav skladu – hlavní inventář (items, VSU, shelves, racks) 
 `warehouse_layout.json` | Layout skladu – pozice vstupů (input) a výstupů (outputs) 
 `robot_post.json` | Stav robotů (pozice, status, baterie) 
 `weights.json` | ML váhy poptávky produktů (0.0–1.0), aktualizováno optimalizačním serverem 
 `stockin_logs.json` | Logy zaskladnění (denní/týdenní/měsíční agregace + historie) 
 `dispense_logs.json` | Logy výdejů (denní/týdenní/měsíční agregace + ML weight tracking) 
 `dispensed_archive.json` | Archiv vydaných položek 
 `relocation_history.json` | Historie přemístění obstrukčních položek 
 `scheduler_runs.json` | Historie běhů automatického archivačního scheduleru 

## Požadavky

- Python 3.10+
- Závislosti: `fastapi`, `pydantic`, `hypercorn`, `apscheduler`, `uvicorn`

## Potřebné úpravy před spuštěním

- `data/ml_robot.json` nebo `ml_robot_updated.json` — musí obsahovat layout skladu (racks, shelves, VSUs, items)
- `data/warehouse_layout.json` — pozice vstupů a výstupů
- `data/robot_post.json` — konfigurace robotů (volitelné, výchozí R1 + R2)
- `data/weights.json` — ML váhy (volitelné, výchozí váha 0.1)

## Instalace

1. Naklonujte repository:
```bash
git clone <URL repozitáře>
cd medicport-warehouse
```

2. Vytvořte a aktivujte virtuální prostředí:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Nainstalujte závislosti:
```bash
pip install -r requirements.txt
```

## Použití

### Spuštění obou serverů (doporučeno):
```bash
python run_servers.py
```
Spustí oba servery ve stejném procesu se sdílenou pamětí (port 8000 + 8001).

### Spuštění pouze stock-in serveru:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### API dokumentace:
Po spuštění dostupná na:
- Stock-in: `http://localhost:8000/docs`
- Dispensing: `http://localhost:8001/docs`

## API Endpointy

### Port 8000 – Zaskladnění & Správa skladu

**Stock-In Operations:**
- `POST /stockin/suggest` — Naskenování položky, návrh umístění (VSU + souřadnice)
- `POST /task/{task_id}/complete` — Potvrzení úspěšného zaskladnění
- `POST /task/{task_id}/fail` — Oznámení selhání zaskladnění
- `PATCH /item/{guid}/metadata` — Aktualizace metadat položky (batch, expirace)
- `GET /item/{guid}` — Detail položky podle GUID
- `GET /items/search` — Vyhledávání položek (barcode, product_id, batch)
- `PATCH /items/bulk-update` — Hromadná aktualizace metadat
- `GET /items/audit-log` — Audit log změn metadat

**Warehouse Management:**
- `GET /warehouse/empty-vsus` — Seznam prázdných VSU
- `GET /warehouse/shelf-space` — Volné místo na policích
- `GET /warehouse/capacity` — Celková kapacita skladu
- `GET /warehouse/stats` — Statistiky skladu
- `POST /warehouse/commit` — Uložení stavu (ml_robot_updated → ml_robot)
- `POST /warehouse/rollback` — Rollback na poslední uložený stav

**Inventory Monitoring:**
- `GET /inventory/expiring` — Položky s blížící se expirací
- `GET /inventory/stock-levels` — Stav zásob podle produktu

**System & Configuration:**
- `GET /health` — Healthcheck
- `GET /robots/status` — Stav robotů
- `POST /robots/reset` — Reset robotů do IDLE
- `GET /tasks/status` — Přehled všech tasků
- `POST /weights/reload` — Přenačtení ML vah z weights.json

**Performance Monitoring:**
- `GET /performance/stats` — Statistiky výkonu endpointů
- `GET /performance/slowest` — Nejpomalejší endpointy

**Stock-In Logs:**
- `GET /stockin/logs` — Souhrn zaskladnění
- `GET /stockin/product/{product_id}` — Historie zaskladnění produktu
- `POST /stockin/logs/reset` — Archivace a reset logů

**Relocation & Archive:**
- `POST /api/temporary/relocate` — Přemístění obstrukční položky
- `GET /api/relocation/history` — Historie relokací
- `GET /api/archive/items` — Archiv vydaných položek

**Scheduler:**
- `GET /scheduler/status` — Stav archivačního scheduleru
- `POST /scheduler/archive/manual` — Manuální spuštění archivace
- `GET /scheduler/runs` — Historie běhů scheduleru

### Port 8001 – Výdej (Dispensing)

- `POST /dispense/create` — Vytvoření výdejního tasku (multi-produkt, multi-množství, multi-pick)
- `POST /dispense/complete` — Potvrzení dokončení výdeje
- `POST /dispense/fail` — Selhání výdeje (s možností partial success per trip)
- `GET /dispense/logs` — Statistiky výdejů
- `GET /dispense/product/{product_id}` — Historie výdejů produktu
- `GET /dispense/task/{task_id}` — Stav výdejního tasku
- `GET /health` — Healthcheck

## Klíčové koncepty

### Zaskladnění (Stock-in)
Systém přijme scan položky (barcode, rozměry), najde optimální VSU podle scorovacího algoritmu (FEFO expirace, ML váhy poptávky, výška police, vzdálenost k výstupu) a vrátí přesné souřadnice pro robota. Položky se plní od zadní stěny (nejvyšší stock_index) směrem dopředu.

### Výdej (Dispensing)
Multi-pick batching — robot vyzvedne všechny položky ze stejného VSU v jednom tripu. Koordinace dvou robotů s vyhýbáním kolizím (nikdy na stejné polici současně). Output queuing při konfliktu na výstupním místě.

### Relokace
Při výdeji i zaskladnění může být cílová položka blokována jinou (obstrukce). Systém automaticky přemístí blokující položku do vhodného VSU (existujícího nebo nově vytvořeného).

### Scheduler
APScheduler spouští denně v 11:00 archivaci dispense logů starších 30 dní do měsíčních souborů. Lze spustit i manuálně.

## Poznámky

- Inventář se drží v paměti (načteno z `ml_robot_updated.json` při startu), změny se zapisují po každém complete/fail
- `ml_robot.json` slouží jako záloha — `POST /warehouse/commit` přepíše zálohu aktuálním stavem
- Při sdíleném běhu (run_servers.py) oba servery operují nad stejnými daty v paměti
- Performance monitoring middleware měří latenci všech endpointů

## Kontakt

Code owner: SURG Solutions – info@surg-solutions.com
Adam Bulka bulka@surg-solutions.com
