import csv
import matplotlib.pyplot as plt
import time

from apriori_python import apriori
from efficient_apriori import apriori as e_apriori
from fpgrowth_py import fpgrowth

# Загрузка сета
products_set = [
    ['Перчатки', 'Шапка', 'Толстовка', 'Джинсы', 'Носки'],
    ['Худи', 'Джинсы', 'Носки', 'Перчатки', 'Футболка', 'Толстовка'],
    ['Носки', 'Перчатки', 'Худи'],
    ['Кроссовки', 'Шапка', 'Бейсболка', 'Носки', 'Очки', 'Перчатки'],
    ['Очки', 'Кроссовки', 'Бейсболка', 'Шапка'],
    ['Кросоовки', 'Шапка', 'Носки', 'Перчатки', 'Джинсы'],
    ['Носки', 'Перчатки', 'Толстовка'],
    ['Джинсы', 'Худи', 'Футболка', 'Перчатки', 'Носки'],
    ['Перчатки', 'Носки', 'Джинсы', 'Футболка', 'Очки', 'Шапка', 'Худи', 'Кроссовки'],
    ['Толстовка', 'Носки', 'Перчатки', 'Джинсы'],
    ['Футболка', 'Перчатки', 'Джинсы', 'Носки'],
    ['Худи', 'Футболка'],
    ['Худи', 'Шапка', 'Толстовка'],
    ['Шапка', 'Джинсы', 'Носки', 'Перчатки'],
    ['Носки', 'Перчатки', 'Худи', 'Джинсы', 'Толстовка'],
    ['Кроссовки', 'Бейсболка', 'Шапка', 'Перчатки', 'Носки', 'Джинсы'],
    ['Носки', 'Кроссовки', 'Шапка', 'Перчатки'],
    ['Футболка', 'Толстовка'],
    ['Носки', 'Худи', 'Джинсы', 'Перчатки'],
    ['Джинсы', 'Кроссовки', 'Перчатки', 'Носки']
]

# Обычный априори метод на сете
print("Rules for conf > 60%")
_, rules = apriori(products_set, minSup=0.3, minConf=0.6)
for rule in sorted(rules):
    if len(rule[1]) == 1:
        print(f"{rule[0]} --> {rule[1]} | conf = {rule[2]}")

print("Rules for conf > 80%")
_, rules = apriori(products_set, minSup=0.3, minConf=0.8)
for rule in sorted(rules):
    if len(rule[1]) == 1:
        print(f"{rule[0]} --> {rule[1]} | conf = {rule[2]}")

# Efficient apriori
list_of_transactions = []

with open('BreadBasket.csv') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        list_of_transactions.append(row)

print(list_of_transactions)

print("Rules for conf > 60%")
_, rules = e_apriori(list_of_transactions, min_support=0.3, min_confidence=0.6)
rules = filter(lambda rule: len(rule.lhs) <= 2 and len(rule.rhs) == 1, rules)
for rule in sorted(rules, key=lambda rule: rule.confidence):
    print(rule)

print("Rules for conf > 80%")
_, rules = e_apriori(list_of_transactions, min_support=0.3, min_confidence=0.8)
rules = filter(lambda rule: len(rule.lhs) <= 2 and len(rule.rhs) == 1, rules)
for rule in sorted(rules, key=lambda rule: rule.confidence):
    print(rule)

# FPGrowth

print("Rules for conf > 60%")
_, rules = fpgrowth(list_of_transactions, minSupRatio=0.29032, minConf=0.6)
for rule in sorted(rules):
    if len(rule[1]) == 1:
        print(f"{rule[0]} --> {rule[1]} | conf = {rule[2]}")

print("Rules for conf > 80%")
_, rules = fpgrowth(list_of_transactions, minSupRatio=0.2905, minConf=0.8)
for rule in sorted(rules):
    if len(rule[1]) == 1:
        print(f"{rule[0]} --> {rule[1]} | conf = {rule[2]}")

## Время
start_time = time.time()
_, _ = apriori(products_set * 100, minConf=0, minSup=0)
time_apriori = round(time.time() - start_time, 2)
print(time_apriori)

start_time = time.time()
_, _ = e_apriori(list_of_transactions * 100, min_confidence=0.0001, min_support=0.0001)
time_e_apriori = round(time.time() - start_time, 2)
print(time_e_apriori)

start_time = time.time()
_, _ = fpgrowth(list_of_transactions * 100, minSupRatio=0., minConf=0.)
time_fpgrowth = round(time.time() - start_time, 2)
print(time_fpgrowth)

colors = ['blue', 'green', 'red']
plt.figure(figsize=(5, 5))
plt.bar(["time_apriori", "time_e_apriori", "time_fpgrowth"], [time_apriori, time_e_apriori, time_fpgrowth],
        color=colors, width=.5)
plt.text(0, time_apriori, str(time_apriori) + 's', horizontalalignment='center', verticalalignment='bottom')
plt.text(1, time_e_apriori, str(time_e_apriori) + 's', horizontalalignment='center', verticalalignment='bottom')
plt.text(2, time_fpgrowth, str(time_fpgrowth) + 's', horizontalalignment='center', verticalalignment='bottom')
plt.ylim(0, 6)
plt.title("100 repeats of transactions")
plt.show()
