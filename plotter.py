import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# IEEE PAPER AESTHETICS (Matches strict academic styling)
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

# Load the logs (Now including all 5 algorithms!)
tas = pd.read_csv('results/tas_log.csv')
greedy = pd.read_csv('results/greedy_log.csv')
rr = pd.read_csv('results/round_robin_log.csv')
cetas = pd.read_csv('results/ce_tas_log.csv')
lotas = pd.read_csv('results/lotas_log.csv')

steps = tas['Step']

# Downsample markers so the lines don't get cluttered
markevery = 15 

# =========================================================
# FIGURE 6: Maximum Host Temperature
# =========================================================
plt.figure(figsize=(10, 6))
plt.plot(steps, rr['Max_Temperature_C'], label='Round Robin', color='green', linestyle='-', marker='^', markevery=markevery)
plt.plot(steps, greedy['Max_Temperature_C'], label='Greedy', color='red', linestyle='--', marker='s', markevery=markevery)
plt.plot(steps, tas['Max_Temperature_C'], label='TAS (2021 Paper)', color='blue', linestyle='-', marker='o', markevery=markevery)
plt.plot(steps, cetas['Max_Temperature_C'], label='CE-TAS (Novel)', color='purple', linestyle='-.', marker='D', markevery=markevery)
plt.plot(steps, lotas['Max_Temperature_C'], label='LOTAS (2023 Paper)', color='orange', linestyle=':', marker='*', markevery=markevery, markersize=10)
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
plt.plot(steps, cetas['Total_Power_W'], label='CE-TAS (Novel)', color='purple', linestyle='-.', marker='D', markevery=markevery)
plt.plot(steps, lotas['Total_Power_W'], label='LOTAS (2023 Paper)', color='orange', linestyle=':', marker='*', markevery=markevery, markersize=10)

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
plt.plot(steps, cetas['Active_Hosts'], label='CE-TAS (Novel)', color='purple', linestyle='-.', marker='D', markevery=markevery)
plt.plot(steps, lotas['Active_Hosts'], label='LOTAS (2023 Paper)', color='orange', linestyle=':', marker='*', markevery=markevery, markersize=10)

plt.xlabel('Simulation Time')
plt.ylabel('Number of Active Hosts')
plt.legend(loc='lower right', ncol=2)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Fig_8_Active_Hosts.png', dpi=300, bbox_inches='tight')
plt.close()

# =========================================================
# FIGURE 9 & 10: Average Averages (Bar Charts)
# =========================================================
labels = ['Round Robin', 'Greedy', 'TAS', 'CE-TAS', 'LOTAS']
colors = ['green', 'red', 'blue', 'purple', 'orange']

# Average Power
plt.figure(figsize=(9, 5))
avg_power = [rr['Total_Power_W'].mean(), greedy['Total_Power_W'].mean(), tas['Total_Power_W'].mean(), cetas['Total_Power_W'].mean(), lotas['Total_Power_W'].mean()]
bars1 = plt.bar(labels, avg_power, color=colors, alpha=0.8, edgecolor='black', width=0.6)
plt.ylabel('Average Power (Watts)')
plt.title('Average Power Consumption Comparison')
for i, v in enumerate(avg_power):
    plt.text(i, v + 10, f"{v:.1f}", ha='center', fontweight='bold', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Fig_9_Average_Power.png', dpi=300, bbox_inches='tight')
plt.close()

# Average Hosts
plt.figure(figsize=(9, 5))
avg_hosts = [rr['Active_Hosts'].mean(), greedy['Active_Hosts'].mean(), tas['Active_Hosts'].mean(), cetas['Active_Hosts'].mean(), lotas['Active_Hosts'].mean()]
bars2 = plt.bar(labels, avg_hosts, color=colors, alpha=0.8, edgecolor='black', width=0.6)
plt.ylabel('Average Active Hosts')
plt.title('Average Server Utilization Comparison')
for i, v in enumerate(avg_hosts):
    plt.text(i, v + 0.05, f"{v:.2f}", ha='center', fontweight='bold', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Fig_10_Average_Hosts.png', dpi=300, bbox_inches='tight')
plt.close()

print("All final IEEE graphs successfully generated!")