# simulator.py

import os
import random
import pandas as pd
from datacenter import Host, TelemetryProvider
from schedulers import round_robin_scheduler, greedy_consolidation_scheduler, tas_scheduler, ThermalPredictor

def load_bitbrains_vms(data_dir="bitbrains_data", num_vms=750):
    """
    Reads the real-world Bitbrains dataset and converts them into VM objects.
    """
    print(f"Loading {num_vms} Virtual Machines from the Bitbrains dataset...")
    vms = []
    
    try:
        files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])[:num_vms]
    except FileNotFoundError:
        print(f"ERROR: Could not find the '{data_dir}' folder!")
        return []
    
    for i, file in enumerate(files):
        file_path = os.path.join(data_dir, file)
        
        try:
            df = pd.read_csv(file_path, sep=';\t', engine='python')
            row = df.iloc[0] 
            
            provisioned_cores = int(row.iloc[1])
            cpu_usage_percent = float(row.iloc[4]) / 100.0 
            
            from datacenter import VM
            vm = VM(f"BB_VM_{i}", f"Bitbrains_{provisioned_cores}C")
            
            vm.cores = max(1, provisioned_cores) 
            vm.cpu_utilization = cpu_usage_percent
            
            vms.append(vm)
            
        except Exception as e:
            print(f"Skipping {file} due to data anomaly: {e}")
            
    print(f"Successfully loaded {len(vms)} Bitbrains VMs!")
    return vms

def run_simulation(scheduler_name, bitbrains_vms, data_dir="data", steps=144):
    """Runs a 24-hour simulation (144 steps of 10 minutes)."""
    print(f"\n{'='*40}")
    print(f"STARTING SIMULATION: {scheduler_name.upper()}")
    print(f"{'='*40}")

    # 1. Boot up the physical Data Center (Your qh2 Dell Servers)
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

    # 2. Load ML Predictor (The Physics Engine)
    predictor = ThermalPredictor("artifacts")

    # 3. Storage for our results
    simulation_log = []

    # ==========================================
    # THE MASTER CLOCK LOOP
    # ==========================================
    rr_index = 0  # To keep track of the Round Robin circle
    
    # Make a fresh copy of the Bitbrains VMs for this specific simulation
    pending_vms = bitbrains_vms.copy()

    for step in range(steps):
        # 1. Grab current background telemetry for all hosts
        current_telemetry = {h.host_id: telemetry_streams[h.host_id].get_background_state() for h in datacenter}
        
        # 2. Incoming Traffic Arrives (Pop 5 to 10 Bitbrains VMs to simulate heavy enterprise traffic)
        incoming_vms = []
        if pending_vms:
            batch_size = random.randint(5, 10)
            for _ in range(min(batch_size, len(pending_vms))):
                incoming_vms.append(pending_vms.pop(0))
        
        # 3. Ask the Scheduler to place the VMs
        if incoming_vms:
            if scheduler_name == "tas":
                tas_scheduler(incoming_vms, datacenter, current_telemetry, predictor)
            elif scheduler_name == "round_robin":
                _, rr_index = round_robin_scheduler(incoming_vms, datacenter, start_index=rr_index)
            elif scheduler_name == "greedy":
                greedy_consolidation_scheduler(incoming_vms, datacenter)

        # 4. Log the physical state of the Data Center at this minute
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
    # Load the 750 real-world VMs ONCE, and pass a copy to each simulation!
    global_bitbrains_vms = load_bitbrains_vms("bitbrains_data", num_vms=750)
    
    if not global_bitbrains_vms:
        print("Aborting simulation. Please make sure the 750 Bitbrains CSV files are in the 'bitbrains_data' folder.")
    else:
        run_simulation("round_robin", global_bitbrains_vms, steps=144)
        run_simulation("greedy", global_bitbrains_vms, steps=144)
        run_simulation("tas", global_bitbrains_vms, steps=144)