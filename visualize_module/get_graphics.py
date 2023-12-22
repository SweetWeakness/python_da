import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os


def get_boxplot(x, l, lf, idx):
    fig = plt.figure(figsize=(7, 6))
    plt.boxplot(x, labels=["Шкала"])
    plt.ylabel(l)
    plt.title("Boxplot (%s)" % l)
    plt.grid(True, which='major', color='dimgrey', linestyle='--')
    plt.tight_layout()
    plt.savefig("%s. Boxplot (%s).png" % (idx, lf))


def get_plot(x, y, xl, yl, lf, idx):
    fig = plt.figure(figsize=(6, 6))
    plt.plot(x, y, 'o', color="grey", alpha=0.2)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title("Зависимость цены от %s" % xl)
    plt.grid(True, which='major', color='dimgrey', linestyle='--')
    plt.savefig("%s. Зависимость цены от %s.png" % (idx, lf))


cur_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.abspath(os.path.join(cur_path, os.pardir))
csv_path = os.path.join(dir_path, "scrapy_module", "detmir.csv")

df = pd.read_csv(csv_path)

price_list = []
for a in list(np.array(df["Цена"])):
    price_list.append(re.sub("[^0-9]", "", a))
price_list = list(map(int, price_list))

for i, label in enumerate(df):
    if label != "Цена":
        l = " ".join(label.split()[:-1])
        get_boxplot(df[label], label, l[:-1], i+1)
    else:
        get_boxplot(price_list, "Цена", "Цена", i+1)

for i, label in enumerate(df):
    if label != "Цена":
        l = " ".join(label.split()[:-1])
        get_plot(df[label], price_list, label, "Цена", l[:-1], 6+i)
        
