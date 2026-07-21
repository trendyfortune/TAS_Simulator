import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. IEEE PAPER AESTHETICS (Matches strict academic styling)
# ---------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 2.0,
    'lines.markersize': 7
})

# ---------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------
print("Loading CSV logs...")
tas = pd.read_csv('results/tas_log.csv')
greedy = pd.read_csv('results/greedy_log.csv')
rr = pd.read_csv('results/round_robin_log.csv')
cetas = pd.read_csv('results/ce_tas_log.csv')
lotas = pd.read_csv('results/lotas_log.csv')

logs = {
    'Round Robin': rr,
    'Greedy': greedy,
    'TAS (2021 Paper)': tas,
    'LOTAS (2023 Paper)': lotas,
    'CE-TAS (Novel)': cetas
}

# ---------------------------------------------------------
# 3. EXTRACT OVERHEAD METRICS
# ---------------------------------------------------------
results_data = {}
for name, df in logs.items():
    results_data[name] = {
        'Migrations': df['Migrations'].iloc[-1] if 'Migrations' in df.columns else 0,
        'SLA_TAH': df['SLA_TAH'].iloc[-1] if 'SLA_TAH' in df.columns else 0,
        'PDM': df['PDM'].iloc[-1] if 'PDM' in df.columns else 0
    }

# ---------------------------------------------------------
# 4. PLOTTING CONFIGURATION
# ---------------------------------------------------------
steps = tas['Step']
markevery = 15  # Downsample markers so lines don't get cluttered

# Global Labels and Colors to ensure consistency across all bar charts
labels = ['Round Robin', 'Greedy', 'TAS (2021 Paper)', 'LOTAS (2023 Paper)', 'CE-TAS (Novel)']
short_labels = ['Round Robin', 'Greedy', 'TAS', 'LOTAS', 'CE-TAS'] # For narrower bar charts
colors = ['green', 'red', 'blue', 'orange', 'purple']

# =========================================================
# FIGURE 6: Maximum Host Temperature
# =========================================================
plt.figure(figsize=(10, 6))
plt.plot(steps, rr['Max_Temperature_C'], label='Round Robin', color='green', linestyle='-', marker='^', markevery=markevery)
plt.plot(steps, greedy['Max_Temperature_C'], label='Greedy', color='red', linestyle='--', marker='s', markevery=markevery)
plt.plot(steps, tas['Max_Temperature_C'], label='TAS (2021 Paper)', color='blue', linestyle='-', marker='o', markevery=markevery)
plt.plot(steps, lotas['Max_Temperature_C'], label='LOTAS (2023 Paper)', color='orange', linestyle=':', marker='*', markevery=markevery, markersize=10)
plt.plot(steps, cetas['Max_Temperature_C'], label='CE-TAS (Novel)', color='purple', linestyle='-.', marker='D', markevery=markevery)

plt.axhline(y=105, color='black', linestyle='-', linewidth=2.5, label='Safe Redline ($105^\circ$C)')

plt.xlabel('Simulation Time')
plt.ylabel('Max. Temperature ($^\circ$C)')
plt.legend(loc='lower right', ncol=2)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Fig_6_Max_Temperature.png', dpi=300, bbox_inches='tight')
plt.close()

# =========================================================
# FIGURE 7: Total Power Consumption
# =========================================================
plt.figure(figsize=(10, 6))
plt.plot(steps, rr['Total_Power_W'], label='Round Robin', color='green', linestyle='-', marker='^', markevery=markevery)
plt.plot(steps, greedy['Total_Power_W'], label='Greedy', color='red', linestyle='--', marker='s', markevery=markevery)
plt.plot(steps, tas['Total_Power_W'], label='TAS (2021 Paper)', color='blue', linestyle='-', marker='o', markevery=markevery)
plt.plot(steps, lotas['Total_Power_W'], label='LOTAS (2023 Paper)', color='orange', linestyle=':', marker='*', markevery=markevery, markersize=10)
plt.plot(steps, cetas['Total_Power_W'], label='CE-TAS (Novel)', color='purple', linestyle='-.', marker='D', markevery=markevery)

plt.xlabel('Simulation Time')
plt.ylabel('Total Power (Watts)')
plt.legend(loc='lower right', ncol=2)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Fig_7_Total_Power.png', dpi=300, bbox_inches='tight')
plt.close()

# =========================================================
# FIGURE 8: Number of Active Hosts
# =========================================================
plt.figure(figsize=(10, 6))
plt.plot(steps, rr['Active_Hosts'], label='Round Robin', color='green', linestyle='-', marker='^', markevery=markevery)
plt.plot(steps, greedy['Active_Hosts'], label='Greedy', color='red', linestyle='--', marker='s', markevery=markevery)
plt.plot(steps, tas['Active_Hosts'], label='TAS (2021 Paper)', color='blue', linestyle='-', marker='o', markevery=markevery)
plt.plot(steps, lotas['Active_Hosts'], label='LOTAS (2023 Paper)', color='orange', linestyle=':', marker='*', markevery=markevery, markersize=10)
plt.plot(steps, cetas['Active_Hosts'], label='CE-TAS (Novel)', color='purple', linestyle='-.', marker='D', markevery=markevery)

plt.xlabel('Simulation Time')
plt.ylabel('Number of Active Hosts')
plt.legend(loc='lower right', ncol=2)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Fig_8_Active_Hosts.png', dpi=300, bbox_inches='tight')
plt.close()

# =========================================================
# FIGURE 9: Average Power (Bar Chart)
# =========================================================
plt.figure(figsize=(10, 5))
avg_power = [rr['Total_Power_W'].mean(), greedy['Total_Power_W'].mean(), tas['Total_Power_W'].mean(), lotas['Total_Power_W'].mean(), cetas['Total_Power_W'].mean()]
bars1 = plt.bar(short_labels, avg_power, color=colors, alpha=0.8, edgecolor='black', width=0.6)
plt.ylabel('Average Power (Watts)')
plt.title('Average Power Consumption Comparison')

# Dynamic text offset
power_offset = max(avg_power) * 0.02
for i, v in enumerate(avg_power):
    plt.text(i, v + power_offset, f"{v:.1f}", ha='center', fontweight='bold', fontsize=10)
    
plt.ylim(0, max(avg_power) * 1.15)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Fig_9_Average_Power.png', dpi=300, bbox_inches='tight')
plt.close()

# =========================================================
# FIGURE 10: Average Hosts (Bar Chart)
# =========================================================
plt.figure(figsize=(10, 5))
avg_hosts = [rr['Active_Hosts'].mean(), greedy['Active_Hosts'].mean(), tas['Active_Hosts'].mean(), lotas['Active_Hosts'].mean(), cetas['Active_Hosts'].mean()]
bars2 = plt.bar(short_labels, avg_hosts, color=colors, alpha=0.8, edgecolor='black', width=0.6)
plt.ylabel('Average Active Hosts')
plt.title('Average Server Utilization Comparison')

# Dynamic text offset
host_offset = max(avg_hosts) * 0.02
for i, v in enumerate(avg_hosts):
    plt.text(i, v + host_offset, f"{v:.2f}", ha='center', fontweight='bold', fontsize=10)
    
plt.ylim(0, max(avg_hosts) * 1.15)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Fig_10_Average_Hosts.png', dpi=300, bbox_inches='tight')
plt.close()

# =========================================================
# FIGURE 11: Overhead Metrics (Subplots)
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Migrations
axes[0].bar(short_labels, [results_data[l]['Migrations'] for l in labels], color=colors, alpha=0.8, edgecolor='black')
axes[0].set_title('Number of VM Migrations')
axes[0].set_ylabel('Count')
axes[0].grid(axis='y', linestyle='--', alpha=0.6)

# Plot 2: SLA_TAH
axes[1].bar(short_labels, [results_data[l]['SLA_TAH'] for l in labels], color=colors, alpha=0.8, edgecolor='black')
axes[1].set_title('SLA_TAH (%)')
axes[1].set_ylabel('Percentage')
axes[1].grid(axis='y', linestyle='--', alpha=0.6)

# Plot 3: PDM
axes[2].bar(short_labels, [results_data[l]['PDM'] for l in labels], color=colors, alpha=0.8, edgecolor='black')
axes[2].set_title('PDM (%)')
axes[2].set_ylabel('Percentage')
axes[2].grid(axis='y', linestyle='--', alpha=0.6)

# Add text labels on top of the bars for Fig 11
for ax_idx, metric in enumerate(['Migrations', 'SLA_TAH', 'PDM']):
    data_values = [results_data[l][metric] for l in labels]
    offset = max(data_values) * 0.02 if max(data_values) > 0 else 0.1
    for i, v in enumerate(data_values):
        axes[ax_idx].text(i, v + offset, f"{v:.2f}", ha='center', fontweight='bold', fontsize=10)
    axes[ax_idx].set_ylim(0, max(data_values) * 1.15 if max(data_values) > 0 else 1.0)

plt.tight_layout()
plt.savefig('Fig_11_Overhead_Metrics.png', dpi=300, bbox_inches='tight')
plt.close()

print("All final IEEE graphs successfully generated!")