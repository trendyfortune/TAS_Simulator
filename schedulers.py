# schedulers.py

import joblib
import pandas as pd
import numpy as np
import os

def round_robin_scheduler(unplaced_vms, datacenter, start_index=0):
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
                break
    return placed_count, host_index


def greedy_consolidation_scheduler(unplaced_vms, datacenter):
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
        data = telemetry_row.copy()
        prev_power = data.get("Power", 0.0)
        prev_cpu_load = data.get("CPU_Load", 0.0)
        
        projected_cpu_load = projected_cpu_util * 100.0
        data["CPU_Load"] = projected_cpu_load
        data["Power"] = projected_power
        
        data["CPU_util"] = projected_cpu_util
        data["CPU_cores_used"] = projected_cpu_util * host.max_cores
        data["Core_util"] = projected_cpu_util
        
        # Float-point accuracy fix: ensures logging exactly matches reality
        is_hypothetical = projected_cpu_util > (host.current_cpu_utilization + 1e-6)
        
        if is_hypothetical:
            vm_count = len(host.hosted_vms) + 1
        else:
            vm_count = max(1, len(host.hosted_vms))
            
        data["VM_per_core"] = vm_count / (host.max_cores + 1e-6)
        data["Power_per_VM"] = projected_power / vm_count
        
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
        
        rf_pred = self.rf.predict(X)
        xgb_pred = self.xgb.predict(X)
        X_scaled = self.en_scaler.transform(X)
        en_pred = self.en.predict(X_scaled)
        
        meta_X = np.column_stack([rf_pred, xgb_pred, en_pred])
        meta_X_scaled = self.meta_scaler.transform(meta_X)
        base_temp = self.meta_model.predict(meta_X_scaled)[0]
        
        thermal_penalty = 0.0
        if vm_count > 8:
            thermal_penalty = (vm_count - 8) * 12.0 
            
        final_temp = base_temp + thermal_penalty
        return final_temp


def tas_scheduler(unplaced_vms, datacenter, telemetry_data, predictor):
    placed_count = 0
    unplaced_vms.sort(key=lambda v: v.cores, reverse=True)
    
    for vm in unplaced_vms:
        best_host = None
        lowest_temp = float('inf')
        
        active_hosts = [h for h in datacenter if h.is_active and h.can_accept(vm)]
        
        for host in active_hosts:
            projected_cpu = host.current_cpu_utilization + (vm.used_cores / host.max_cores)
            projected_power = host.idle_power + (projected_cpu * (host.max_power - host.idle_power))
            
            t_data = telemetry_data.get(host.host_id, {})
            predicted_temp = predictor.predict(host, projected_cpu, projected_power, t_data)
            
            # THE FIX: We use a 90°C placement threshold to leave a 15°C Safety Buffer!
            # This allows the servers to survive ambient weather changes in the afternoon.
            if predicted_temp < 90.0 and predicted_temp < lowest_temp:
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