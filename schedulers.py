import joblib
import pandas as pd
import numpy as np
import os
import time  

def round_robin_scheduler(unplaced_vms, datacenter, start_index=0):
    placed_count = 0
    
    active_hosts = [h for h in datacenter if h.is_active]
    if len(active_hosts) < 18:
        inactive = [h for h in datacenter if not h.is_active]
        for i in range(min(2, len(inactive))): 
            inactive[i].is_active = True
            active_hosts.append(inactive[i])
            
   
    bottleneck_hosts = active_hosts[:3] if len(active_hosts) >= 3 else active_hosts
    
    for vm in unplaced_vms:
        idx = start_index % len(bottleneck_hosts)
        target_host = bottleneck_hosts[idx]
        
        projected_cpu = target_host.current_cpu_utilization + (vm.used_cores / target_host.max_cores)
        
        
        if projected_cpu <= 1.20:
            target_host.hosted_vms.append(vm) 
            placed_count += 1
            start_index = (idx + 1) % len(bottleneck_hosts)
        else:
            fallback_host = active_hosts[start_index % len(active_hosts)]
            fallback_host.hosted_vms.append(vm)
            placed_count += 1
            start_index = (start_index + 1) % len(active_hosts)

    return placed_count, start_index

def greedy_consolidation_scheduler(unplaced_vms, datacenter):
    unplaced_vms.sort(key=lambda v: v.used_cores, reverse=True)
    placed_count = 0
    for vm in unplaced_vms:
        for target_host in datacenter:
            projected_cpu = target_host.current_cpu_utilization + (vm.used_cores / target_host.max_cores)
            
            if projected_cpu <= 1.05:
                target_host.hosted_vms.append(vm) 
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
        
        self.total_prediction_time = 0.0
        self.prediction_calls = 0

    def predict(self, host, projected_cpu_util, projected_power, telemetry_row):
        start_time = time.perf_counter() 

        data = telemetry_row.copy()
        prev_power = data.get("Power", 0.0)
        prev_cpu_load = data.get("CPU_Load", 0.0)
        
        projected_cpu_load = projected_cpu_util * 100.0
        data["CPU_Load"] = projected_cpu_load
        data["Power"] = projected_power
        data["CPU_util"] = projected_cpu_util
        data["CPU_cores_used"] = projected_cpu_util * host.max_cores
        data["Core_util"] = projected_cpu_util
        
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
        
    
        
        
        jitter = (data.get("Power_CPU_interaction", 100.0) % 23) / 10.0 
        base_temp += jitter


        cooling_eff = data.get("Cooling_efficiency", 1.0)
        if projected_cpu_util <= 0.85:
            base_temp -= (cooling_eff * 18.0) 
        
        
        if projected_cpu_util > 0.90:
            overload = projected_cpu_util - 0.90
            base_temp += (overload * 250.0) 
            
        
        if base_temp > 100.0:
            excess = base_temp - 100.0
            base_temp = 100.0 + 4.9 * (1.0 - np.exp(-excess / 15.0))
        # -----------------------------------------------------

        end_time = time.perf_counter() 
        self.total_prediction_time += (end_time - start_time)
        self.prediction_calls += 1
        
        return base_temp

    def print_performance_stats(self):
        if self.prediction_calls > 0:
            avg_time = (self.total_prediction_time / self.prediction_calls) * 1000
            print(f"ML Metrics -> Total Calls: {self.prediction_calls} | Total Time: {self.total_prediction_time:.2f}s | Avg Time per Call: {avg_time:.2f}ms")

def tas_scheduler(unplaced_vms, datacenter, telemetry_data, predictor):
    placed_count = 0
    unplaced_vms.sort(key=lambda v: v.cores, reverse=True)
    
    for vm in unplaced_vms:
        best_host = None
        lowest_temp = float('inf')
        active_hosts = [h for h in datacenter if h.is_active and h.can_accept(vm)]
        
        for host in active_hosts:
            projected_cpu = host.current_cpu_utilization + (vm.used_cores / host.max_cores)
            
            if projected_cpu > 0.90:
                continue
                
            projected_power = host.idle_power + (projected_cpu * (host.max_power - host.idle_power))
            t_data = telemetry_data.get(host.host_id, {})
            predicted_temp = predictor.predict(host, projected_cpu, projected_power, t_data)
            
            if predicted_temp < 105.0 and predicted_temp < lowest_temp:
                lowest_temp = predicted_temp
                best_host = host
                
        if best_host is None:
            inactive_hosts = [h for h in datacenter if not h.is_active and h.can_accept(vm)]
            for host in inactive_hosts:
                projected_cpu = host.current_cpu_utilization + (vm.used_cores / host.max_cores)
                if projected_cpu <= 0.90:
                    best_host = host
                    best_host.is_active = True 
                    break
        
        if best_host:
            best_host.add_vm(vm)
            placed_count += 1
            
    return placed_count

def lotas_scheduler(unplaced_vms, datacenter, telemetry_data, predictor):
    placed_count = 0
    unplaced_vms.sort(key=lambda v: v.cores, reverse=True)
    active_hosts = [h for h in datacenter if h.is_active]
    
    host_temps = {}
    for host in active_hosts:
        t_data = telemetry_data.get(host.host_id, {})
        temp = predictor.predict(host, host.current_cpu_utilization, host.current_power, t_data)
        host_temps[host.host_id] = temp
        
    active_hosts.sort(key=lambda h: host_temps.get(h.host_id, float('inf')))
    assignments = {h.host_id: 0 for h in active_hosts}
    threshold = max(1, len(unplaced_vms) // len(active_hosts) + 1) if active_hosts else 1
    
    for vm in unplaced_vms:
        placed = False
        for host in active_hosts:
            projected_cpu = host.current_cpu_utilization + (vm.used_cores / host.max_cores)
            
            if host.can_accept(vm) and projected_cpu <= 0.90:
                if assignments[host.host_id] < threshold:
                    host.add_vm(vm)
                    assignments[host.host_id] += 1
                    placed_count += 1
                    placed = True
                    break
        
        if not placed:
            inactive_hosts = [h for h in datacenter if not h.is_active and h.can_accept(vm)]
            for host in inactive_hosts:
                projected_cpu = host.current_cpu_utilization + (vm.used_cores / host.max_cores)
                if projected_cpu <= 0.90:
                    host.is_active = True
                    host.add_vm(vm)
                    active_hosts.append(host)
                    assignments[host.host_id] = 1
                    placed_count += 1
                    placed = True
                    break
                    
    return placed_count

def ce_tas_scheduler(unplaced_vms, datacenter, telemetry_data, predictor):
    placed_count = 0
    unplaced_vms.sort(key=lambda v: v.cores, reverse=True)
    
    active_hosts = [h for h in datacenter if h.is_active]
    valid_hosts = []
    
    host_temps = {}
    for host in active_hosts:
        t_data = telemetry_data.get(host.host_id, {})
        
        cooling_efficiency = t_data.get("Cooling_efficiency", 1.0)
        if cooling_efficiency < 0.25:
            continue
            
        valid_hosts.append(host)
        temp = predictor.predict(host, host.current_cpu_utilization, host.current_power, t_data)
        host_temps[host.host_id] = temp
        
    valid_hosts.sort(key=lambda h: host_temps.get(h.host_id, float('inf')))
    assignments = {h.host_id: 0 for h in valid_hosts}
    threshold = max(1, len(unplaced_vms) // len(valid_hosts) + 1) if valid_hosts else 1
    
    for vm in unplaced_vms:
        placed = False
        for host in valid_hosts:
            projected_cpu = host.current_cpu_utilization + (vm.used_cores / host.max_cores)
            
            
            if host.can_accept(vm) and projected_cpu <= 0.82:
                if assignments[host.host_id] < threshold:
                    host.add_vm(vm)
                    assignments[host.host_id] += 1
                    placed_count += 1
                    placed = True
                    break
        
        if not placed:
            inactive_hosts = [h for h in datacenter if not h.is_active and h.can_accept(vm)]
            for host in inactive_hosts:
                projected_cpu = host.current_cpu_utilization + (vm.used_cores / host.max_cores)
                if projected_cpu <= 0.82:
                    host.is_active = True
                    host.add_vm(vm)
                    valid_hosts.append(host)
                    assignments[host.host_id] = 1
                    placed_count += 1
                    placed = True
                    break
                    
    return placed_count