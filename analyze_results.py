# analyze_results.py
import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("results/times.csv")
os.makedirs("results/plots", exist_ok=True)

# Базовое время
t_seq = df[df['procs'] == 1]['time'].iloc[0]

plt.figure(figsize=(12, 10))

# 1. Время выполнения
plt.subplot(2, 2, 1)
for prog in df['program'].unique():
    if prog == 'sequential': continue
    sub = df[df['program'] == prog]
    plt.plot(sub['procs'], sub['time'], 'o-', label=prog)
plt.yscale('log')
plt.xlabel('Число процессов')
plt.ylabel('Время (с, лог. шкала)')
plt.title('Время выполнения')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Ускорение
plt.subplot(2, 2, 2)
for prog in df['program'].unique():
    if prog == 'sequential': continue
    sub = df[df['program'] == prog]
    plt.plot(sub['procs'], sub['speedup'], 's-', label=prog)
plt.plot([1, 64], [1, 64], 'k--', label='Идеальное')
plt.xlabel('Число процессов')
plt.ylabel('Ускорение')
plt.title('Ускорение')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Эффективность
plt.subplot(2, 2, 3)
for prog in df['program'].unique():
    if prog == 'sequential': continue
    sub = df[df['program'] == prog]
    plt.plot(sub['procs'], sub['efficiency'], '^-', label=prog)
plt.axhline(1.0, color='k', linestyle='--')
plt.xlabel('Число процессов')
plt.ylabel('Эффективность')
plt.title('Эффективность')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/plots/performance.png", dpi=150)
plt.close()
print("Графики сохранены в results/plots/performance.png")