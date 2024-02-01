import numpy as np
import torch
import matplotlib.pyplot as plt

Directory = 'Data'
Files_num = [1]
results_path = []
random_results_path = []
for num in Files_num:
    file = f'results_{num}.pth'
    results_path.append(file)
    file = f'random_results_{num}.pth'
    random_results_path.append(file)

results = []
for path in results_path:
    results.append(torch.load(Directory+'/'+path))

random_results = []
for path in random_results_path:
    random_results.append(torch.load(Directory+'/'+path))

for i in range(len(results)):
    print(results_path[i], max(results[i]['results']), np.argmax(results[i]['results']), len(results[i]['results']))
    results[i]['avglosses'] = list(filter(lambda k:  0< k <100, results[i]['avglosses'] ))

with torch.no_grad():
    for i in range(len(results)):
        fig, ax_list = plt.subplots(3,1)
        fig.suptitle(results_path[i])
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        ax_list[0].plot(results[i]['results'])
        ax_list[1].plot(random_results[i])
        ax_list[2].plot(results[i]['avglosses']) 
        plt.tight_layout()

plt.show()