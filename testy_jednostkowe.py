from scipy.stats import ttest_rel
import numpy as np


# Ładowanie wynikw
results = np.loadtxt("results.csv", delimiter=",")


# Przygotwanie tablic do testów statystycznych
t_stat = np.zeros((4, 4))
p_val = np.zeros((4, 4))
better = np.zeros((4, 4), dtype=bool)
stat_significant = np.zeros((4, 4), dtype=bool)
stat_better = np.zeros((4, 4), dtype=bool)

aplha = 0.05

cls_names_short = ["our_LR", "our_MLP", "sk_LR", "sk_MLP"]


# Testy

for col in range(4):
    for row in range(4):
        if col == row:
            t_stat[row, col] = None
            p_val[row, col] = None
            better[row, col] = False
            stat_significant[row, col] = False
            stat_better[row, col] = None

        else:
            t_stat[row, col], p_val[row, col] = ttest_rel(results[:, row], results[:, col])

            if t_stat[row, col] > 0:
                better[row, col] = True
            else:
                better[row, col] = False

            if(aplha > p_val[row, col]):
                stat_significant[row, col] = True
            else:
                stat_significant[row, col] = False

            stat_better[row, col] = better[row, col] * stat_significant[row, col]
            if stat_better[row, col]:
                print(f"{cls_names_short[row], np.round(np.mean(results[:, row]), 3)} jest lepszy statystycznie od {cls_names_short[col], np.round(np.mean(results[:, col]), 3)}\n")


# Wypisywanie wyników testów

from tabulate import tabulate


print("T-statystyka")
print(tabulate(np.around(t_stat, 3), tablefmt="fancy_grid"), "\n")

print("Wartość p")
print(tabulate(p_val, tablefmt="fancy_grid"), "\n")

print("Lepsze klasyfikatory")
print(tabulate(better, tablefmt="fancy_grid"), "\n")

print("Przewaga statystyczna")
print(tabulate(stat_significant, tablefmt="fancy_grid"), "\n")

print("Klasyfikatory lepsze statystycznie")
print(tabulate(stat_better, tablefmt="fancy_grid"), "\n")


cls_results = [np.round(np.mean(results[:, row]), 3) for row in range(results.shape[1])]

cls_names = ["Nasza regresja logistyczna", "Nasze MLP", "Sklearn regresja logistyczna", "Sklearn MLP"]

cls_final = [(cls_names[i], cls_results[i]) for i in range(len(cls_results))]



print(tabulate(cls_final, headers=["Klasyfikator", "Znaczenie statystyczne"], tablefmt="fancy_grid"))
