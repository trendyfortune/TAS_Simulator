import pandas as pd
import os

class VM:
    """Represents a virtual machine with fixed capacities but variable utilization."""
    def __init__(self, vm_id, flavor):
        self.vm_id = vm_id
        self.flavor = flavor
        
        # Match Table 4 from the paper [cite: 398]
        if flavor == "VM1":
            self.cores, self.ram = 1, 4
        elif flavor == "VM2":
            self.cores, self.ram = 2, 8
        elif flavor == "VM3":
            self.cores, self.ram = 4, 16
        elif flavor == "VM4":
            self.cores, self.ram = 8, 32
        else:
            self.cores, self.ram = 1, 4 # Default fallback
            
        # Utilization represents how hard the VM is working (0.0 to 1.0)
        self.cpu_utilization = 0.0

    @property
    def used_cores(self):
        return self.cores * self.cpu_utilization

    def __repr__(self):
        return f"VM({self.vm_id} | {self.flavor} | Load: {self.cpu_utilization*100:.1f}%)"


class Host:
    """Represents the server, calibrated to the training dataset's power envelope."""
    def __init__(self, host_id):
        self.host_id = host_id
        self.max_cores = 64.0
        self.max_ram = 512.0
        
        # Calibrated to qh2-rcc120.csv to keep ML model in its comfort zone
        self.idle_power = 142.0 
        self.max_power = 380.0
        
        self.hosted_vms = []
        self.is_active = False 

    @property
    def current_cpu_utilization(self):
        if not self.hosted_vms or not self.is_active:
            return 0.0
        total_used_cores = sum(vm.used_cores for vm in self.hosted_vms)
        return total_used_cores / self.max_cores

    @property
    def current_ram_utilization(self):
        if not self.hosted_vms or not self.is_active:
            return 0.0
        return sum(vm.ram for vm in self.hosted_vms) / self.max_ram

    @property
    def current_power(self):
        if not self.is_active:
            return 0.0 
        return self.idle_power + (self.current_cpu_utilization * (self.max_power - self.idle_power))

    def can_accept(self, vm):
        projected_cores = sum(v.cores for v in self.hosted_vms) + vm.cores
        projected_ram = sum(v.ram for v in self.hosted_vms) + vm.ram
        if (projected_cores / self.max_cores) > 0.90:
            return False
        if (projected_ram / self.max_ram) > 1.0:
            return False
        return True

    def add_vm(self, vm):
        if self.can_accept(vm):
            self.hosted_vms.append(vm)
            return True
        return False

    def remove_vm(self, vm):
        if vm in self.hosted_vms:
            self.hosted_vms.remove(vm)

    def __repr__(self):
        state = "ON " if self.is_active else "OFF"
        cpu_pct = self.current_cpu_utilization * 100
        return f"Host({self.host_id} | {state} | VMs: {len(self.hosted_vms):2} | CPU: {cpu_pct:4.1f}% | Power: {self.current_power:6.1f}W)"

class WorkloadStreamer:
    """Reads real data and converts CPU load into incoming VMs."""
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, sep=";")
        if 'Time' in self.df.columns:
            self.df['Time'] = pd.to_datetime(self.df['Time'])
            self.df = self.df.sort_values("Time").reset_index(drop=True)
        
        self.current_step = 0
        self.max_steps = len(self.df)

    def get_next_batch(self, batch_size=5):
        """Returns a batch of VMs using real CPU loads from your dataset."""
        vms = []
        for _ in range(batch_size):
            if self.current_step >= self.max_steps:
                break
            
            row = self.df.iloc[self.current_step]
            # Convert percentage to decimal, fallback to 10% if missing
            cpu_util = float(row.get("CPU_Load", 10.0)) / 100.0 
            
            # Create a standard 4-Core VM (VM3) [cite: 398]
            vm = VM(vm_id=f"vm_{self.current_step}", flavor="VM3")
            vm.cpu_utilization = cpu_util
            vms.append(vm)
            
            self.current_step += 1
            
        return vms

class TelemetryProvider:
    """Supplies background sensor data (Fans, Network) and pre-calculates ML features."""
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, sep=";")
        if 'Time' in self.df.columns:
            self.df['Time'] = pd.to_datetime(self.df['Time'])
            self.df = self.df.sort_values("Time").reset_index(drop=True)
            
        # --- APPLY THE EXACT FEATURE ENGINEERING FROM YOUR ML SCRIPT ---
        df_fe = self.df.copy()
        
        # Basic
        df_fe["CPU_util"]    = df_fe["CPU_Load"] / 100
        df_fe["Core_util"]   = df_fe["CPU_cores_used"] / (df_fe["CPU_cores"] + 1e-6)
        df_fe["RAM_util"]    = df_fe["Ram_Used"] / (df_fe["Ram"] + 1e-6)
        df_fe["VM_per_core"] = df_fe["No_Of_Running_vms"] / (df_fe["CPU_cores"] + 1e-6)

        # Cooling
        fan_cols = [c for c in df_fe.columns if "Fan_speed" in c]
        if fan_cols:
            df_fe["Fan_mean"] = df_fe[fan_cols].mean(axis=1)
            df_fe["Fan_max"]  = df_fe[fan_cols].max(axis=1)

        df_fe["Cooling_efficiency"] = df_fe["Cooling_Power"] / (df_fe["Power"] + 1e-6)
        df_fe["Cooling_lag1"]       = df_fe["Cooling_Power"].shift(1)
        df_fe["Cooling_diff"]       = df_fe["Cooling_Power"].diff()

        # Power
        df_fe["Power_per_CPU"]         = df_fe["Power"] / (df_fe["CPU_Load"] + 1e-6)
        df_fe["Power_per_VM"]          = df_fe["Power"] / (df_fe["No_Of_Running_vms"] + 1)
        df_fe["Power_per_core"]        = df_fe["Power"] / (df_fe["CPU_cores"] + 1e-6)
        df_fe["Power_CPU_interaction"] = df_fe["Power"] * df_fe["CPU_Load"]
        df_fe["Power_diff"]            = df_fe["Power"].diff()

        # Network
        if "Network_RX" in df_fe.columns and "Network_TX" in df_fe.columns:
            df_fe["Network_total"] = df_fe["Network_RX"] + df_fe["Network_TX"]

        # Rolling & Temporal
        df_fe["Power_rolling_30min"] = df_fe["Power"].rolling(window=3).mean()
        df_fe["CPU_rolling_30min"]   = df_fe["CPU_Load"].rolling(window=3).mean()
        df_fe["Ambient_Temp_lag1"] = df_fe["Ambient_Temperature"].shift(1)
        df_fe["Ambient_Temp_lag2"] = df_fe["Ambient_Temperature"].shift(2)
        df_fe["CPU_Load_diff"]     = df_fe["CPU_Load"].diff()

        # CPU Max
        cpu_cols = [c for c in df_fe.columns if "CPU" in c and "Temp" in c]
        if cpu_cols:
            df_fe["CPU_Temp_Max"] = df_fe[cpu_cols].max(axis=1)

        # Drop the NaN rows created by rolling() and shift()
        self.df = df_fe.dropna().reset_index(drop=True)
        self.current_step = 0
        
    def get_background_state(self):
        """Returns the fully engineered dictionary of sensors for this specific minute."""
        if self.df.empty:
            return {}
        row = self.df.iloc[self.current_step].to_dict()
        self.current_step = (self.current_step + 1) % len(self.df)
        return row

# ==========================================
# QUICK TEST TO VERIFY EVERYTHING
# ==========================================
if __name__ == "__main__":
    print("--- Testing Host & Power Math ---")
    h1 = Host("host_test_1")
    h1.is_active = True # Turn it on
    print(f"Initial State: {h1}")
    
    v1 = VM("vm_test_1", "VM4")
    v1.cpu_utilization = 0.80 
    h1.add_vm(v1)
    print(f"After heavy VM added: {h1}")
    
    print("\n--- Testing Streamers (Requires a valid CSV in data/ folder) ---")
    # This will safely skip if you haven't moved a CSV into the data folder yet
    test_csv = "data/qh2-rcc120.csv"
    if os.path.exists(test_csv):
        streamer = WorkloadStreamer(test_csv)
        print("Incoming VM batch:", streamer.get_next_batch(2))
        
        telemetry = TelemetryProvider(test_csv)
        print("Telemetry State keys:", list(telemetry.get_background_state().keys())[:5], "...")
    else:
        print(f"File not found: {test_csv}. Please ensure a CSV is inside the data/ folder to test telemetry.")