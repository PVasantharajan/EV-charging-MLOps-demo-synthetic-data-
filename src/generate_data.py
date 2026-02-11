import pandas as pd
import numpy as np
import uuid


def generate_synthetic_ev_data(
    n_vehicles=8,
    hours=48,
    seed=42,
):
    np.random.seed(seed)

    end_time = pd.Timestamp.utcnow().floor("min")
    start_time = end_time - pd.Timedelta(hours=hours)

    all_records = []

    for i in range(n_vehicles):

        vehicle_id = f"EV_{str(uuid.uuid4())[:8]}"

        soc = np.random.uniform(20, 80)
        charging_speed_per_hour = np.random.uniform(8, 15)  # % per hour
        discharge_rate_per_hour = np.random.uniform(1, 3)   # % per hour

        ts = start_time
        prev_ts = None
        prev_soc = None

        plugged_in = False
        charging_session_remaining_hours = 0

        while ts < end_time:

            # ---------------------------------------
            # Inject occasional long blackout gaps
            # ---------------------------------------
            if np.random.rand() < 0.02:  # small probability of outage
                gap_duration = np.random.uniform(4, 24)  # hours
                ts += pd.Timedelta(hours=gap_duration)
                plugged_in = False
                charging_session_remaining_hours = 0
                continue

            # ---------------------------------------
            # Regular telemetry interval (1–10 min)
            # ---------------------------------------
            step_minutes = np.random.randint(1, 11)
            next_ts = ts + pd.Timedelta(minutes=step_minutes)

            if next_ts > end_time:
                break

            # ---------------------------------------
            # Gap detection
            # ---------------------------------------
            if prev_ts is not None:
                gap_hours = (next_ts - prev_ts).total_seconds() / 3600
            else:
                gap_hours = 0

            gap_flag = gap_hours >= 4

            hour = next_ts.hour

            # ---------------------------------------
            # SOC behavior
            # ---------------------------------------

            elapsed_hours = step_minutes / 60

            if gap_flag:
                # Passive idle drift
                soc -= np.random.uniform(0.05, 0.3)
                plugged_in = False
                charging_session_remaining_hours = 0

            else:
                # Start charging session
                if not plugged_in:
                    if 17 <= hour <= 22 and np.random.rand() < 0.3:
                        plugged_in = True
                        charging_session_remaining_hours = np.random.uniform(1, 4)
                    elif 6 <= hour <= 9 and np.random.rand() < 0.1:
                        plugged_in = True
                        charging_session_remaining_hours = np.random.uniform(0.5, 2)

                if plugged_in:
                    soc += charging_speed_per_hour * elapsed_hours
                    charging_session_remaining_hours -= elapsed_hours
                    if charging_session_remaining_hours <= 0:
                        plugged_in = False
                else:
                    soc -= discharge_rate_per_hour * elapsed_hours

            soc = np.clip(soc, 5, 100)

            # ---------------------------------------
            # SOC delta masking
            # ---------------------------------------
            if prev_soc is not None and not gap_flag:
                soc_delta = soc - prev_soc
            else:
                soc_delta = np.nan

            all_records.append(
                {
                    "vehicle_id": vehicle_id,
                    "timestamp": next_ts,
                    "plugged_in": plugged_in,
                    "soc_percent": round(float(soc), 2),
                    "gap_flag": gap_flag,
                    "soc_delta": None
                    if np.isnan(soc_delta)
                    else round(float(soc_delta), 3),
                }
            )

            prev_ts = next_ts
            prev_soc = soc
            ts = next_ts

    df = pd.DataFrame(all_records)
    df = df.sort_values(["vehicle_id", "timestamp"]).reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = generate_synthetic_ev_data()
    df.to_csv("synthetic_ev_data.csv", index=False)
    print("Synthetic dataset generated: synthetic_ev_data.csv")
