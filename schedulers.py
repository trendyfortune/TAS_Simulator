# schedulers.py

import joblib
import pandas as pd
import numpy as np
import os

def round_robin_scheduler(unplaced_vms, datacenter, start_index=0):
    """
    BASELINE 1: Spreads VMs equally across all hosts in a circle.
    Thermal Agnostic. Energy Inefficient (keeps too many hosts awake).
    """
    placed_count = 0
    host_count = len(datacenter)
    host_index = start_index

    for vm in unplaced_vms:
        loop_start = host_index
        
        while True:
            target_host = datacenter[host_index]
            
            if target_host.can_accept(vm):
                target_host.add_vm(vm)
                target_host.is_active = True  
                placed_count += 1
                
                host_index = (host_index + 1) % host_count
                break
            
            host_index = (host_index + 1) % host_count
            
            if host_index == loop_start:
                print(f"WARNING: Datacenter FULL. Could not place {vm.vm_id}")
                break
                
    return placed_count, host_index


def greedy_consolidation_scheduler(unplaced_vms, datacenter):
    """
    BASELINE 2: Packs VMs onto the fewest number of hosts possible.
    Thermal Agnostic. Energy Efficient, but creates massive Hotspots.
    """
    unplaced_vms.sort(key=lambda v: v.used_cores, reverse=True)
    placed_count = 0

    for vm in unplaced_vms:
        for target_host in datacenter:
            if target_host.can_accept(vm):
                target_host.add_vm(vm)
                target_host.is_active = True 
                placed_count += 1
                break
                
    return placed_count


class ThermalPredictor:
    """Loads your Kaggle-tier Stacking Ensemble and applies physical calibrations."""
    def __init__(self, artifacts_dir="artifacts"):
        print("Loading ML Models from artifacts folder...")
        self.rf = joblib.load(os.path.join(artifacts_dir, "rf_final.pkl"))
        self.xgb = joblib.load(os.path.join(artifacts_dir, "xgb_final.pkl"))
        self.en = joblib.load(os.path.join(artifacts_dir, "en_final.pkl"))
        self.en_scaler = joblib.load(os.path.join(artifacts_dir, "en_scaler.pkl"))
        self.meta_model = joblib.load(os.path.join(artifacts_dir, "meta_model_final.pkl"))
        self.meta_scaler = joblib.load(os.path.join(artifacts_dir, "global_meta_scaler.pkl"))
        self.features = joblib.load(os.path.join(artifacts_dir, "feature_order.pkl"))
        print("ML Models loaded successfully!")

    def predict(self, host, projected_cpu_util, projected_power, telemetry_row):
        """Calculates the hypothetical physics if we place a VM on this host."""
        data = telemetry_row.copy()
        prev_power = data.get("Power", 0.0)
        prev_cpu_load = data.get("CPU_Load", 0.0)
        
        projected_cpu_load = projected_cpu_util * 100.0
        data["CPU_Load"] = projected_cpu_load
        data["Power"] = projected_power
        
        data["CPU_util"] = projected_cpu_util
        data["CPU_cores_used"] = projected_cpu_util * host.max_cores
        data["Core_util"] = projected_cpu_util
        
        future_vm_count = len(host.hosted_vms) + 1
        data["VM_per_core"] = future_vm_count / (host.max_cores + 1e-6)
        data["Power_per_VM"] = projected_power / future_vm_count
        
        safe_cpu = max(projected_cpu_load, 10.0)
        data["Power_per_CPU"] = projected_power / safe_cpu
        
        data["Power_per_core"] = projected_power / (host.max_cores + 1e-6)
        data["Power_CPU_interaction"] = projected_power * projected_cpu_load
        data["Power_diff"] = projected_power - prev_power
        data["CPU_Load_diff"] = projected_cpu_load - prev_cpu_load
        data["Cooling_efficiency"] = data.get("Cooling_Power", 0.0) / (projected_power + 1e-6)
        
        df = pd.DataFrame([data])
        for f in self.features:
            if f not in df.columns:
                df[f] = 0.0
                
        X = df[self.features]
        
        # Base Predictions
        rf_pred = self.rf.predict(X)
        xgb_pred = self.xgb.predict(X)
        X_scaled = self.en_scaler.transform(X)
        en_pred = self.en.predict(X_scaled)
        
        meta_X = np.column_stack([rf_pred, xgb_pred, en_pred])
        meta_X_scaled = self.meta_scaler.transform(meta_X)
        base_temp = self.meta_model.predict(meta_X_scaled)[0]
        
       # ==========================================
        # PHYSICAL CALIBRATION (The Density Extrapolator)
        # ==========================================
        # Because the CSV dataset contains mostly light-workload VMs, the real physical 
        # CPU utilization stays low even when a server is 100% full of allocated cores.
        # However, packing too many VMs on a single server causes I/O and Network hotspots.
        # We penalize 'Zombie VM Packing' by triggering a meltdown prediction if 
        # the server has more than 8 VMs packed onto it.
        
        thermal_penalty = 0.0
        if future_vm_count > 8:
            # Add an exponential +12°C for every VM jammed onto the server beyond 8
            thermal_penalty = (future_vm_count - 8) * 12.0 
            
        final_temp = base_temp + thermal_penalty
        return final_temp


def tas_scheduler(unplaced_vms, datacenter, telemetry_data, predictor):
    """
    BASELINE 3: Thermal-Aware Scheduler (Your Algorithm).
    Places VMs on the host that will result in the lowest exhaust temperature.
    Consolidates power gracefully without melting servers.
    """
    placed_count = 0
    unplaced_vms.sort(key=lambda v: v.cores, reverse=True)
    
    for vm in unplaced_vms:
        best_host = None
        lowest_temp = float('inf')
        
        active_hosts = [h for h in datacenter if h.is_active and h.can_accept(vm)]
        
        for host in active_hosts:
            projected_cpu = host.current_cpu_utilization + (vm.cores / host.max_cores)
            projected_power = host.idle_power + (projected_cpu * (host.max_power - host.idle_power))
            
            t_data = telemetry_data.get(host.host_id, {})
            predicted_temp = predictor.predict(host, projected_cpu, projected_power, t_data)
            
            # Thermal Redline from the IEEE paper is 105°C
            if predicted_temp < 105.0 and predicted_temp < lowest_temp:
                lowest_temp = predicted_temp
                best_host = host
                
        if best_host is None:
            inactive_hosts = [h for h in datacenter if not h.is_active and h.can_accept(vm)]
            if inactive_hosts:
                best_host = inactive_hosts[0]
                best_host.is_active = True 
        
        if best_host:
            best_host.add_vm(vm)
            placed_count += 1
            
    return placed_count


# ==========================================
# QUICK TEST
# ==========================================
if __name__ == "__main__":
    from datacenter import Host, VM

    print("--- Testing TAS ML Scheduler ---")
    dc_tas = [Host(f"Host_{i}") for i in range(3)]
    dc_tas[0].is_active = True 
    
    vms_tas = [VM(f"VM_{i}", "VM4") for i in range(6)]
    for v in vms_tas: v.cpu_utilization = 0.25 
    
    dummy_telemetry = {"Host_0": {"Fan_speed1": 9000, "Network_RX": 1000}}
    ml_brain = ThermalPredictor("artifacts")
    
    tas_scheduler(vms_tas, dc_tas, dummy_telemetry, ml_brain)
    
    for h in dc_tas:
        print(h)