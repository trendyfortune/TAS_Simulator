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
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 2.0,
    'lines.markersize': 7
})

# THE FIX: Point to the 'results' folder where simulator.py saved the logs
tas = pd.read_csv('results/tas_log.csv')
greedy = pd.read_csv('results/greedy_log.csv')
rr = pd.read_csv('results/round_robin_log.csv')

steps = tas['Step']

# Downsample markers so the lines don't get cluttered (Paper style)
markevery = 15 

# =========================================================
# FIGURE 6: Maximum Host Temperature
# =========================================================
plt.figure(figsize=(8, 5))
plt.plot(steps, rr['Max_Temperature_C'], label='Round Robin', color='green', linestyle='-', marker='^', markevery=markevery)
plt.plot(steps, greedy['Max_Temperature_C'], label='Greedy', color='red', linestyle='--', marker='s', markevery=markevery)
plt.plot(steps, tas['Max_Temperature_C'], label='TAS (Proposed)', color='blue', linestyle='-', marker='o', markevery=markevery)
plt.axhline(y=105, color='black', linestyle=':', linewidth=2.5, label='Safe Redline ($105^\circ$C)')

plt.xlabel('Simulation Time')
plt.ylabel('Max. Temperature ($^\circ$C)')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Fig_6_Max_Temperature.png', dpi=300, bbox_inches='tight')
plt.close()

# =========================================================
# FIGURE 7: Total Power Consumption
# =========================================================
plt.figure(figsize=(8, 5))
plt.plot(steps, rr['Total_Power_W'], label='Round Robin', color='green', linestyle='-', marker='^', markevery=markevery)
plt.plot(steps, greedy['Total_Power_W'], label='Greedy', color='red', linestyle='--', marker='s', markevery=markevery)
plt.plot(steps, tas['Total_Power_W'], label='TAS (Proposed)', color='blue', linestyle='-', marker='o', markevery=markevery)

plt.xlabel('Simulation Time')
plt.ylabel('Total Power (Watts)')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Fig_7_Total_Power.png', dpi=300, bbox_inches='tight')
plt.close()

# =========================================================
# FIGURE 8: Number of Active Hosts
# =========================================================
plt.figure(figsize=(8, 5))
plt.plot(steps, rr['Active_Hosts'], label='Round Robin', color='green', linestyle='-', marker='^', markevery=markevery)
plt.plot(steps, greedy['Active_Hosts'], label='Greedy', color='red', linestyle='--', marker='s', markevery=markevery)
plt.plot(steps, tas['Active_Hosts'], label='TAS (Proposed)', color='blue', linestyle='-', marker='o', markevery=markevery)

plt.xlabel('Simulation Time')
plt.ylabel('Number of Active Hosts')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Fig_8_Active_Hosts.png', dpi=300, bbox_inches='tight')
plt.close()

# =========================================================
# FIGURE 9 & 10: Average Averages (Bar Charts)
# =========================================================
labels = ['Round Robin', 'Greedy', 'TAS (Proposed)']
colors = ['green', 'red', 'blue']

# Average Power
plt.figure(figsize=(7, 5))
avg_power = [rr['Total_Power_W'].mean(), greedy['Total_Power_W'].mean(), tas['Total_Power_W'].mean()]
bars1 = plt.bar(labels, avg_power, color=colors, alpha=0.8, edgecolor='black', width=0.5)
plt.ylabel('Average Power (Watts)')
for i, v in enumerate(avg_power):
    plt.text(i, v + 20, f"{v:.1f}", ha='center', fontweight='bold', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Fig_9_Average_Power.png', dpi=300, bbox_inches='tight')
plt.close()

# Average Hosts
plt.figure(figsize=(7, 5))
avg_hosts = [rr['Active_Hosts'].mean(), greedy['Active_Hosts'].mean(), tas['Active_Hosts'].mean()]
bars2 = plt.bar(labels, avg_hosts, color=colors, alpha=0.8, edgecolor='black', width=0.5)
plt.ylabel('Average Active Hosts')
for i, v in enumerate(avg_hosts):
    plt.text(i, v + 0.05, f"{v:.2f}", ha='center', fontweight='bold', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Fig_10_Average_Hosts.png', dpi=300, bbox_inches='tight')
plt.close()

print("All IEEE paper-style graphs successfully generated!")