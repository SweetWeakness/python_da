import os
import matplotlib.pyplot as plt


cur_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.abspath(os.path.join(cur_path, os.pardir))
ml_path = os.path.join(dir_path, "mldev_module")
cur = os.path.join(ml_path, "results")
curs = [x[0] for x in os.walk(cur)]
d1 = {}
d2 = {}
for cur in curs:
    folder = os.path.basename(os.path.normpath(cur))
    if folder == "results" or folder == "default":
        continue
    f = open(os.path.join(cur, "result.txt"), 'r')
    line1 = f.readline()
    line2 = f.readline()
    f.close()
    d1[int(folder)] = line1.split()[1]
    d2[int(folder)] = line2.split()[1]
    

def draw(d, title):
    d = sorted(list(d.items()))
    x = []
    y = []
    for _ in d:
        _ = list(_)
        x.append(_[0])
        y.append(float(_[1]))
    fig = plt.figure(figsize=(10, 8))
    plt.plot(x, y, color="red")
    plt.xlabel("Максимальная цена в выборке")
    plt.ylabel("Качество")
    plt.title(title + " score")
    plt.grid(True, which='major', color='dimgrey', linestyle='--')
    plt.savefig(title + " score.png")
    
draw(d1, "linear regression")
draw(d2, "lassolars")
