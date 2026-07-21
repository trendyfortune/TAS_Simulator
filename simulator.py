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
        return 0.0, 0, 0.0, 0.0, 0.0

    datacenter = []
    telemetry_streams = {}
    
    for idx, file in enumerate(csv_files):
        host_id = f"Host_{idx}"
        host = Host(host_id)
        if idx < 2:
            host.is_active = True
            
        # Initialize SLA tracking attributes directly on the host instance
        host.active_ticks = 0
        host.overload_ticks = 0
        
        datacenter.append(host)
        file_path = os.path.join(data_dir, file)
        telemetry_streams[host_id] = TelemetryProvider(file_path)

    predictor = ThermalPredictor("artifacts")
    simulation_log = []
    rr_index = 0  
    
    total_scheduling_time = 0.0 
    pending_vms = copy.deepcopy(bitbrains_vms)
    
    # --- PERFORMANCE OVERHEAD METRICS ---
    total_migrations = 0
    total_requested_mips = 0
    total_degraded_mips = 0
    
    # --- NEW CODE: Define Algorithm-Specific Strictness ---
    if scheduler_name == "ce_tas":
        alg_cpu_limit = 0.82
        alg_temp_limit = 89.0
    elif scheduler_name in ["tas", "lotas"]:
        alg_cpu_limit = 0.88
        alg_temp_limit = 95.0
    else: # round_robin, greedy
        alg_cpu_limit = 0.95
        alg_temp_limit = 105.0

    for h in datacenter:
        h.cpu_limit = alg_cpu_limit
    # ------------------------------------------------------

    for step in range(steps):
        current_telemetry = {h.host_id: telemetry_streams[h.host_id].get_background_state() for h in datacenter}
        
        # 1. Incoming Traffic Arrives
        incoming_vms = []
        if pending_vms:
            batch_size = random.randint(5, 10)
            for _ in range(min(batch_size, len(pending_vms))):
                incoming_vms.append(pending_vms.pop(0))
                
        # 1.4 Simulate Dynamic Workload Spikes 
        active_hosts = [h for h in datacenter if h.is_active]
        for host in active_hosts:
            for vm in host.hosted_vms:
                drift = random.uniform(-0.07, 0.07)
                vm.cpu_utilization = max(0.05, min(1.0, vm.cpu_utilization + drift))
                
        # --- NEW CODE: 1.5 Dynamic Migration Trigger (Proactive Cooling) ---
        for host in active_hosts:
            total_requested_mips += sum(vm.cores for vm in host.hosted_vms)
            
            # Predict IMMEDIATE temperature based on drift before logging occurs
            t_data = current_telemetry[host.host_id]
            immediate_temp = predictor.predict(host, host.current_cpu_utilization, host.current_power, t_data)
            
            # Use a while-loop to proactively pull VMs off until the host is safe again
            while host.current_cpu_utilization > alg_cpu_limit or immediate_temp > alg_temp_limit:
                if host.hosted_vms:
                    migrating_vm = host.hosted_vms.pop(0)
                    total_migrations += 1
                    total_degraded_mips += (migrating_vm.cores * 0.10) 
                    incoming_vms.insert(0, migrating_vm)
                    
                    # Recalculate immediately after removing the VM
                    immediate_temp = predictor.predict(host, host.current_cpu_utilization, host.current_power, t_data)
                else:
                    break
        # -----------------------------------------------
        
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

        # 3. Log State & Track SLA Violations
        active_hosts = [h for h in datacenter if h.is_active]
        total_power = sum(h.current_power for h in active_hosts)
        
        max_temp = 0.0
        for h in active_hosts:
            t_data = current_telemetry[h.host_id]
            temp = predictor.predict(h, h.current_cpu_utilization, h.current_power, t_data)
            
            # Assign temperature for the migration check on the next tick
            h.temperature = temp  
            
            # --- TRACK SLA_TAH OVERLOADS ---
            h.active_ticks += 1
            # If a host exceeds thermal or CPU safety limits, count it as an SLA violation tick
            if h.current_cpu_utilization > 0.90 or temp > 95.0:
                h.overload_ticks += 1
            # -------------------------------
            
            if temp > max_temp:
                max_temp = temp

        # --- ADDED: Calculate current step metrics for the CSV log ---
        current_host_sla = []
        for h in datacenter:
            if h.active_ticks > 0:
                current_host_sla.append(h.overload_ticks / h.active_ticks)
                
        current_sla_tah = 0.0
        if current_host_sla:
            current_sla_tah = (sum(current_host_sla) / len(current_host_sla)) * 100 

        current_pdm = 0.0
        if total_requested_mips > 0:
            current_pdm = (total_degraded_mips / total_requested_mips) * 100 
        # -------------------------------------------------------------

        # --- UPDATED: Append the new metrics to the log ---
        simulation_log.append({
            "Step": step,
            "Active_Hosts": len(active_hosts),
            "Total_Power_W": total_power,
            "Max_Temperature_C": max_temp,
            "Migrations": total_migrations,
            "SLA_TAH": current_sla_tah,
            "PDM": current_pdm
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
    
    # --- NEW CODE: Final Metric Calculations ---
    # Calculate SLA_TAH (SLA violation Time per Active Host) per Equation 6
    host_sla_tahs = []
    for host in datacenter:
        if host.active_ticks > 0:
            host_sla_tahs.append(host.overload_ticks / host.active_ticks)
            
    sla_tah = 0.0
    if host_sla_tahs:
        sla_tah = (sum(host_sla_tahs) / len(host_sla_tahs)) * 100 

    # Calculate PDM (Performance Degradation due to Migration) per Equation 7
    pdm = 0.0
    if total_requested_mips > 0:
        pdm = (total_degraded_mips / total_requested_mips) * 100 

    # Overall SLA Violation per Equation 8
    sla_violation = (sla_tah / 100) * (pdm / 100)
    
    print(f"\nSimulation complete! Saved logs to {save_path}")
    print(f"\n--- PERFORMANCE METRICS FOR {scheduler_name.upper()} ---")
    print(f"Total Time Spent in Scheduler: {total_scheduling_time:.4f} seconds")
    print(f"Total VM Migrations: {total_migrations}")
    print(f"SLA_TAH: {sla_tah:.4f}%")
    print(f"PDM: {pdm:.4f}%")
    print(f"Overall SLA Violation: {sla_violation:.6e}")
    
    if scheduler_name in ["tas", "ce_tas", "lotas"]:
        predictor.print_performance_stats()
        
    return total_scheduling_time, total_migrations, sla_tah, pdm, sla_violation

if __name__ == "__main__":
    global_bitbrains_vms = load_bitbrains_vms("bitbrains_data", num_vms=750)
    
    if not global_bitbrains_vms:
        print("Aborting simulation. Please make sure the 750 Bitbrains CSV files are in the 'bitbrains_data' folder.")
    else:
        # Capture the tuple returned by each simulation run
        res_rr = run_simulation("round_robin", copy.deepcopy(global_bitbrains_vms), steps=144)
        res_greedy = run_simulation("greedy", copy.deepcopy(global_bitbrains_vms), steps=144)
        res_tas = run_simulation("tas", copy.deepcopy(global_bitbrains_vms), steps=144)
        res_lotas = run_simulation("lotas", copy.deepcopy(global_bitbrains_vms), steps=144) 
        res_ce_tas = run_simulation("ce_tas", copy.deepcopy(global_bitbrains_vms), steps=144)
        
        # Link the results to the names
        metrics = {
            'Round Robin': res_rr,
            'Greedy': res_greedy,
            'TAS (2021 Paper)': res_tas,
            'LOTAS (2023 Paper)': res_lotas,
            'CE-TAS (Novel)': res_ce_tas
        }
        
        # --- GENERATE AND PRINT THE FINAL SUMMARY TABLE ---
        print("\n" + "="*130)
        print(" FINAL SIMULATION RESULTS SUMMARY".center(130))
        print("="*130)
        
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
                
                # Unpack the new metrics
                sched_time, migrations, sla_tah, pdm, sla_v = metrics.get(name, (0.0, 0, 0.0, 0.0, 0.0))
                
                results.append({
                    'Algorithm': name,
                    'Avg Active Hosts': f"{avg_hosts:.2f}",
                    'Peak Temp (°C)': f"{peak_temp:.2f}",
                    'Sched. Time (s)': f"{sched_time:.4f}",
                    'Migrations': migrations,
                    'SLA_TAH (%)': f"{sla_tah:.4f}",
                    'PDM (%)': f"{pdm:.4f}"
                })
        
        if results:
            print(f"{'Algorithm':<22} | {'Avg Active Hosts':<20} | {'Peak Temp (°C)':<15} | {'Sched. Time (s)':<15} | {'Migrations':<10} | {'SLA_TAH (%)':<11} | {'PDM (%)'}")
            print("-" * 130)
            for res in results:
                print(f"{res['Algorithm']:<22} | {res['Avg Active Hosts']:<20} | {res['Peak Temp (°C)']:<15} | {res['Sched. Time (s)']:<15} | {res['Migrations']:<10} | {res['SLA_TAH (%)']:<11} | {res['PDM (%)']}")
        else:
            print("Could not generate summary table. Log files not found.")
        print("="*130 + "\n")