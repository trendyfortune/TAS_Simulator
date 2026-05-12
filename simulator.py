# simulator.py
import os
import pandas as pd
from datacenter import Host, WorkloadStreamer, TelemetryProvider
from schedulers import round_robin_scheduler, greedy_consolidation_scheduler, tas_scheduler, ThermalPredictor

def run_simulation(scheduler_name, data_dir="data", steps=144):
    """Runs a 24-hour simulation (144 steps of 10 minutes)."""
    print(f"\n{'='*40}")
    print(f"STARTING SIMULATION: {scheduler_name.upper()}")
    print(f"{'='*40}")

    # 1. Boot up the Data Center
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        print("ERROR: No CSV files found in 'data/' folder!")
        return

    datacenter = []
    telemetry_streams = {}
    
    for idx, file in enumerate(csv_files):
        host_id = f"Host_{idx}"
        host = Host(host_id)
        # Turn on the first 2 hosts, leave the rest asleep
        if idx < 2:
            host.is_active = True
            
        datacenter.append(host)
        
        # Attach the real background telemetry to this specific host
        file_path = os.path.join(data_dir, file)
        telemetry_streams[host_id] = TelemetryProvider(file_path)

    # 2. Setup the Workload Generator (Using the first CSV as the traffic source)
    workload_generator = WorkloadStreamer(os.path.join(data_dir, csv_files[0]))
    
    # 3. Load ML Predictor (Used for TAS, and for logging true temps for RR/Greedy)
    predictor = ThermalPredictor("artifacts")

    # 4. Storage for our results (To plot later)
    simulation_log = []

    # ==========================================
    # THE MASTER CLOCK LOOP
    # ==========================================
    rr_index = 0  # To keep track of the Round Robin circle

    for step in range(steps):
        # 1. Grab current background telemetry for all hosts
        current_telemetry = {h.host_id: telemetry_streams[h.host_id].get_background_state() for h in datacenter}
        
        # 2. Incoming Traffic Arrives (Batch of 2 to 5 VMs every 10 minutes)
        import random
        incoming_vms = workload_generator.get_next_batch(batch_size=random.randint(2, 5))
        
        # 3. Ask the Scheduler to place the VMs
        if scheduler_name == "tas":
            tas_scheduler(incoming_vms, datacenter, current_telemetry, predictor)
        elif scheduler_name == "round_robin":
            _, rr_index = round_robin_scheduler(incoming_vms, datacenter, start_index=rr_index)
        elif scheduler_name == "greedy":
            greedy_consolidation_scheduler(incoming_vms, datacenter)

        # 4. Log the state of the Data Center at this minute
        active_hosts = [h for h in datacenter if h.is_active]
        total_power = sum(h.current_power for h in active_hosts)
        
        # Calculate maximum temperature in the data center to track hotspots
        max_temp = 0.0
        for h in active_hosts:
            t_data = current_telemetry[h.host_id]
            # Use the ML model to tell us how hot this server physically is right now
            temp = predictor.predict(h, h.current_cpu_utilization, h.current_power, t_data)
            if temp > max_temp:
                max_temp = temp

        simulation_log.append({
            "Step": step,
            "Active_Hosts": len(active_hosts),
            "Total_Power_W": total_power,
            "Max_Temperature_C": max_temp
        })
        
        if step % 20 == 0:
            print(f"Step {step:3}/{steps} | Active Hosts: {len(active_hosts):2} | Total Power: {total_power:7.1f}W | Max Temp: {max_temp:5.1f}C")

    # Save Results
    df_results = pd.DataFrame(simulation_log)
    os.makedirs("results", exist_ok=True)
    save_path = f"results/{scheduler_name}_log.csv"
    df_results.to_csv(save_path, index=False)
    print(f"\nSimulation complete! Saved logs to {save_path}")

# ==========================================
# RUN ALL 3 SIMULATIONS
# ==========================================
if __name__ == "__main__":
    run_simulation("round_robin", steps=144)
    run_simulation("greedy", steps=144)
    run_simulation("tas", steps=144)