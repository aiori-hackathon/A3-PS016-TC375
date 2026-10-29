# PS-016 Handover Simulation Prototype

## What it does
- Models a user moving along a road covered by 3 abstract cell sites.
- Simulates two speeds: 15 m/s and 35 m/s.
- Compares a signal-based baseline against a supervised Random Forest decision policy.
- Repeats runs over five random seeds (11, 22, 33, 44, 55).
- Reports handover failures, ping-pong events, interruption duration, and a throughput proxy.
- Creates CSV tables and PNG graphs in `outputs/`.

## Important limitation
This is a reproducible **abstract Python prototype**, not an LTE/5G standards-capable simulator. The signal model and throughput are simplified assumptions. Do not report the throughput proxy as real LTE/5G throughput or claim standards conformance. PS-016 specifically asks for one working LTE/5G-capable simulator and a clearly identified radio model. Treat this code as an early demonstration and use a validated LTE/5G simulator (for example, ns-3 with a suitable LTE/5G module) for the final evaluation.

## Run on a computer
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Install dependencies:
   `pip install -r requirements.txt`
4. Run:
   `python handover_sim.py`
5. Check the `outputs` folder.

## Android
You can try it in an online Python environment that supports file downloads and package installation, but a desktop/laptop is much easier for final simulation work. This project is not an Android APK.

## Files
- `handover_sim.py`: simulation and graph generation
- `requirements.txt`: Python packages
- `outputs/`: generated after running
