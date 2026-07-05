import os
import random
import copy
import pandas as pd
import time  

from datacenter import Host, TelemetryProvider
from schedulers import round_robin_scheduler, greedy_consolidation_scheduler, tas_scheduler, ce_tas_scheduler, lotas_scheduler, ThermalPredictor

def load_bitbrains_vms(data_dir="bitbrains_data", num_vms=750):
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
            
            # REAL PHYSICS FIX: Give every VM a lifespan between 2 and 12 hours (12 to 72 ticks)
            vm.lifespan = random.randint(12, 72) 
            
            vms.append(vm)
        except Exception as e:
            pass
            
    print(f"Successfully loaded {len(vms)} Bitbrains VMs!")
    return vms

def run_simulation(scheduler_name, bitbrains_vms, data_dir="data", steps=144):
    print(f"\n{'='*40}")
    print(f"STARTING SIMULATION: {scheduler_name.upper()}")
    print(f"{'='*40}")

    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        print("ERROR: No CSV files found in 'data/' folder!")
        return 0.0

    datacenter = []
    telemetry_streams = {}
    
    for idx, file in enumerate(csv_files):
        host_id = f"Host_{idx}"
        host = Host(host_id)
        if idx < 2:
            host.is_active = True
        datacenter.append(host)
        file_path = os.path.join(data_dir, file)
        telemetry_streams[host_id] = TelemetryProvider(file_path)

    predictor = ThermalPredictor("artifacts")
    simulation_log = []
    rr_index = 0  
    
    total_scheduling_time = 0.0 
    pending_vms = copy.deepcopy(bitbrains_vms)

    for step in range(steps):
        current_telemetry = {h.host_id: telemetry_streams[h.host_id].get_background_state() for h in datacenter}
        
        # 1. Incoming Traffic Arrives
        incoming_vms = []
        if pending_vms:
            batch_size = random.randint(5, 10)
            for _ in range(min(batch_size, len(pending_vms))):
                incoming_vms.append(pending_vms.pop(0))
        
        # 2. Place VMs (WITH TIMERS)
        start_scheduling = time.perf_counter() 
        
        if incoming_vms:
            if scheduler_name == "tas":
                tas_scheduler(incoming_vms, datacenter, current_telemetry, predictor)
            elif scheduler_name == "ce_tas":
                ce_tas_scheduler(incoming_vms, datacenter, current_telemetry, predictor)
            elif scheduler_name == "lotas":
                lotas_scheduler(incoming_vms, datacenter, current_telemetry, predictor)
            elif scheduler_name == "round_robin":
                _, rr_index = round_robin_scheduler(incoming_vms, datacenter, start_index=rr_index)
            elif scheduler_name == "greedy":
                greedy_consolidation_scheduler(incoming_vms, datacenter)

        end_scheduling = time.perf_counter() 
        total_scheduling_time += (end_scheduling - start_scheduling)

        # 3. Log State
        active_hosts = [h for h in datacenter if h.is_active]
        total_power = sum(h.current_power for h in active_hosts)
        
        max_temp = 0.0
        for h in active_hosts:
            t_data = current_telemetry[h.host_id]
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

        # 4. VM Departures
        for host in datacenter:
            if host.is_active:
                vms_to_remove = []
                for vm in host.hosted_vms:
                    if hasattr(vm, 'lifespan'):
                        vm.lifespan -= 1
                        if vm.lifespan <= 0:
                            vms_to_remove.append(vm)
                
                for vm in vms_to_remove:
                    host.hosted_vms.remove(vm)
                
                if not host.hosted_vms:
                    host.is_active = False

    df_results = pd.DataFrame(simulation_log)
    os.makedirs("results", exist_ok=True)
    save_path = f"results/{scheduler_name}_log.csv"
    df_results.to_csv(save_path, index=False)
    
    print(f"\nSimulation complete! Saved logs to {save_path}")
    print(f"\n--- PERFORMANCE METRICS FOR {scheduler_name.upper()} ---")
    print(f"Total Time Spent in Scheduler: {total_scheduling_time:.4f} seconds")
    if scheduler_name in ["tas", "ce_tas", "lotas"]:
        predictor.print_performance_stats()
        
    return total_scheduling_time # <--- WE RETURN THE TIME TO THE MAIN BLOCK

if __name__ == "__main__":
    global_bitbrains_vms = load_bitbrains_vms("bitbrains_data", num_vms=750)
    
    if not global_bitbrains_vms:
        print("Aborting simulation. Please make sure the 750 Bitbrains CSV files are in the 'bitbrains_data' folder.")
    else:
        # Capture the times returned by each simulation run
        time_rr = run_simulation("round_robin", global_bitbrains_vms, steps=144)
        time_greedy = run_simulation("greedy", global_bitbrains_vms, steps=144)
        time_tas = run_simulation("tas", global_bitbrains_vms, steps=144)
        time_lotas = run_simulation("lotas", global_bitbrains_vms, steps=144) 
        time_ce_tas = run_simulation("ce_tas", global_bitbrains_vms, steps=144)
        
        # Link the times to the names
        exec_times = {
            'Round Robin': time_rr,
            'Greedy': time_greedy,
            'TAS (2021 Paper)': time_tas,
            'LOTAS (2023 Paper)': time_lotas,
            'CE-TAS (Novel)': time_ce_tas
        }
        
        # --- GENERATE AND PRINT THE FINAL SUMMARY TABLE ---
        print("\n" + "="*85)
        print(" FINAL SIMULATION RESULTS SUMMARY".center(85))
        print("="*85)
        
        files = {
            'Round Robin': 'results/round_robin_log.csv',
            'Greedy': 'results/greedy_log.csv',
            'TAS (2021 Paper)': 'results/tas_log.csv',
            'LOTAS (2023 Paper)': 'results/lotas_log.csv',
            'CE-TAS (Novel)': 'results/ce_tas_log.csv'
        }
        
        results = []
        for name, filepath in files.items():
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                avg_hosts = df['Active_Hosts'].mean()
                peak_temp = df['Max_Temperature_C'].max()
                sched_time = exec_times.get(name, 0.0)
                
                results.append({
                    'Algorithm': name,
                    'Average Active Hosts': f"{avg_hosts:.2f}",
                    'Peak Temp (°C)': f"{peak_temp:.2f}",
                    'Scheduling Time (s)': f"{sched_time:.4f}"
                })
        
        if results:
            print(f"{'Algorithm':<22} | {'Average Active Hosts':<22} | {'Peak Temp (°C)':<15} | {'Scheduling Time (s)'}")
            print("-" * 85)
            for res in results:
                print(f"{res['Algorithm']:<22} | {res['Average Active Hosts']:<22} | {res['Peak Temp (°C)']:<15} | {res['Scheduling Time (s)']}")
        else:
            print("Could not generate summary table. Log files not found.")
        print("="*85 + "\n")