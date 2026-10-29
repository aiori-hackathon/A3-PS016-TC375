"""
PS-016: Simulation-Based Handover Decision Evaluation for High-Mobility Users
Educational prototype (Python). This is a simplified abstract cellular mobility model,
NOT a standards-accurate LTE/5G PHY/MAC simulator.

Requirements:
  pip install numpy pandas matplotlib scikit-learn

Run:
  python handover_sim.py

Outputs:
  outputs/results.csv
  outputs/summary.csv
  outputs/*.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# Abstract model assumptions: 3 cells on a straight road, RSSI-like signal in dBm.
CELL_POSITIONS = np.array([0.0, 500.0, 1000.0])  # metres
CELL_TX_DBM = 30.0
PATHLOSS_EXP = 3.2
NOISE_STD_DB = 2.0
DT = 0.2  # seconds
ROAD_END = 1500.0
SEEDS = [11, 22, 33, 44, 55]
SPEEDS = {"low": 15.0, "high": 35.0}  # m/s (~54 and 126 km/h)
POLICIES = ["signal_baseline", "supervised_ml"]

def signal_dbm(user_x, rng):
    """Simplified log-distance signal model with log-normal shadowing."""
    d = np.maximum(np.abs(user_x - CELL_POSITIONS), 10.0)
    mean = CELL_TX_DBM - 10 * PATHLOSS_EXP * np.log10(d / 10.0)
    return mean + rng.normal(0, NOISE_STD_DB, size=3)

def build_training_data(n=5000, seed=2026):
    """
    Create labelled snapshots from the same abstract model.
    Label = strongest cell, except avoid switching for tiny gains:
    use best cell only if its signal is at least 3 dB above serving cell.
    """
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for _ in range(n):
        x = rng.uniform(0, ROAD_END)
        sig = signal_dbm(x, rng)
        serving = int(rng.integers(0, 3))
        best = int(np.argmax(sig))
        target = serving if best == serving or sig[best] - sig[serving] < 3.0 else best
        rows.append([x, *sig, serving, sig[serving], np.max(sig) - sig[serving],
                     np.max(sig), np.partition(sig, -2)[-2]])
        labels.append(target)
    columns = ["x_m", "sig0", "sig1", "sig2", "serving", "serving_sig",
               "best_gain", "best_sig", "second_sig"]
    return pd.DataFrame(rows, columns=columns), np.array(labels)

def fit_model():
    X, y = build_training_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=60, max_depth=10,
                                   class_weight="balanced", random_state=42)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    print("Supervised model validation accuracy:", round(accuracy_score(yte, pred), 3))
    print(classification_report(yte, pred, zero_division=0))
    return model

def features(x, sig, serving):
    return pd.DataFrame([[x, *sig, serving, sig[serving],
                          np.max(sig) - sig[serving], np.max(sig),
                          np.partition(sig, -2)[-2]]],
                        columns=["x_m", "sig0", "sig1", "sig2", "serving",
                                 "serving_sig", "best_gain", "best_sig", "second_sig"])

def simulate_one(speed, policy, seed, model):
    rng = np.random.default_rng(seed)
    x, serving = 0.0, 0
    elapsed = 0.0
    handovers = failures = pingpong = 0
    interruption = 0.0
    last_switch_time = -999.0
    previous_cell = None
    switch_times = []
    samples = []
    throughput_values = []
    # Approximate radio/link and handover parameters
    failure_prob = 0.08 if speed < 20 else 0.16
    handover_interrupt = 0.08  # seconds per successful handover
    while x <= ROAD_END:
        sig = signal_dbm(x, rng)
        best = int(np.argmax(sig))
        if policy == "signal_baseline":
            # Conventional strongest-signal policy with 3 dB hysteresis
            target = serving
            if best != serving and sig[best] >= sig[serving] + 3.0:
                target = best
        else:
            target = int(model.predict(features(x, sig, serving))[0])
            # Avoid obviously weak transitions; retain a modest hysteresis guard
            if target != serving and sig[target] < sig[serving] + 1.0:
                target = serving

        handover_now = target != serving
        if handover_now:
            handovers += 1
            # Rapid reverse handover within 2 s counted as ping-pong
            if previous_cell is not None and target == previous_cell and elapsed - last_switch_time <= 2.0:
                pingpong += 1
            old = serving
            previous_cell = old
            serving = target
            last_switch_time = elapsed
            switch_times.append(elapsed)
            if rng.random() < failure_prob:
                failures += 1
                interruption += 0.35
            else:
                interruption += handover_interrupt

        # Simple throughput proxy from serving signal, with loss during interruption.
        # Not a standards-based PHY throughput calculation.
        sinr_proxy = sig[serving] + 100.0  # fixed noise offset for a monotonic proxy
        rate = float(np.clip(2.0 * np.log2(1.0 + 10 ** (sinr_proxy / 10.0)), 0, 100))
        if handover_now:
            rate *= 0.5
        throughput_values.append(rate)
        samples.append((elapsed, x, serving, *sig, rate))
        x += speed * DT
        elapsed += DT

    avg_rate = float(np.mean(throughput_values)) if throughput_values else 0.0
    duration = elapsed
    return {
        "speed": speed, "policy": policy, "seed": seed,
        "handover_events": handovers,
        "handover_failures": failures,
        "ping_pong_events": pingpong,
        "interruption_duration_s": interruption,
        "avg_throughput_mbps_proxy": avg_rate,
        "simulation_duration_s": duration
    }, pd.DataFrame(samples, columns=["time_s", "x_m", "serving_cell",
                                      "sig_cell0_dbm", "sig_cell1_dbm",
                                      "sig_cell2_dbm", "throughput_mbps_proxy"])

def make_plots(results):
    metrics = [
        ("handover_failures", "Handover failures (lower is better)"),
        ("ping_pong_events", "Ping-pong events (lower is better)"),
        ("interruption_duration_s", "Total interruption duration (s; lower is better)"),
        ("avg_throughput_mbps_proxy", "Mean throughput proxy (higher is better)")
    ]
    for col, title in metrics:
        pivot = results.groupby(["speed_label", "policy"])[col].mean().unstack()
        ax = pivot.plot(kind="bar", figsize=(8, 5))
        ax.set_title(title)
        ax.set_xlabel("User mobility speed")
        ax.set_ylabel(col.replace("_", " "))
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / f"{col}.png", dpi=160)
        plt.close()

    # Show one reproducible trajectory example
    seed = int(results["seed"].iloc[0])
    fig, ax = plt.subplots(figsize=(9, 5))
    for cell in range(3):
        ax.plot(CELL_POSITIONS, [0, 1, 2], alpha=0)  # keep axes independent of radio samples
    ax.scatter(CELL_POSITIONS, [0, 0, 0], marker="^", s=100, label="Cell sites")
    ax.set_title("Three-cell layout (abstract road model)")
    ax.set_xlabel("Position along road (m)")
    ax.set_yticks([])
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "three_cell_layout.png", dpi=160)
    plt.close()

def main():
    model = fit_model()
    rows = []
    for speed_label, speed in SPEEDS.items():
        for seed in SEEDS:
            for policy in POLICIES:
                result, _trace = simulate_one(speed, policy, seed, model)
                result["speed_label"] = speed_label
                rows.append(result)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results.csv", index=False)
    summary = df.groupby(["speed_label", "policy"]).agg(
        handover_failures_mean=("handover_failures", "mean"),
        ping_pong_mean=("ping_pong_events", "mean"),
        interruption_mean_s=("interruption_duration_s", "mean"),
        throughput_proxy_mean=("avg_throughput_mbps_proxy", "mean"),
        handover_events_mean=("handover_events", "mean")
    ).reset_index()
    summary.to_csv(OUT / "summary.csv", index=False)
    print("\nSummary results:")
    print(summary.to_string(index=False))
    make_plots(df)
    print(f"\nFinished. Results and graphs saved in: {OUT.resolve()}")
    print("IMPORTANT: Throughput is a proxy, not measured LTE/5G PHY throughput.")
    print("For the final PS-016 deliverable, validate with a real LTE/5G simulator (e.g., ns-3 with LTE/5G modules) and document its radio model.")

if __name__ == "__main__":
    main()
