#%%
import matplotlib.pyplot as plt
import numpy as np

#%%
frob_18_res = np.array([0.108, 0.135, 0.143, 0.116, 0.150])
frob_17_res = np.array([0.142, 0.122, 0.106, 0.117, 0.121])
start_17_res = np.array([0.521,0.513,0.397,0.566, 0.406])

x_18 = np.arange(1, 6)
x_17 = np.arange(6, 11)

#%%
plt.figure(figsize=(6, 4))
plt.scatter(x_18, frob_18_res, marker='o', label='target vs trained')
plt.axhline(y=0.153, color='orange', linestyle='--', label='Fidelity @ 0.985')
plt.fill_between([0, 6], 0.157, 0.149, color='orange', alpha=0.1)
plt.axhline(y=0.122, color='green', linestyle='--', label='Fidelity @ 0.99')
plt.fill_between([0, 6], 0.125, 0.119, color='green', alpha=0.1)
plt.xlim(0.75, 5.25)


plt.title('18 active micro-heaters')
plt.grid(ls='--')
plt.xlabel('# target')
plt.ylabel('Frobenius norm')
plt.legend()
plt.xticks(x_18)
plt.savefig('./Immagini/frob_18.pdf')  # Salva la figura per Frob_18

#%%
plt.figure(figsize=(6, 4))
plt.scatter(x_17, frob_17_res, marker='o', label='target vs trained')
plt.scatter(x_17, start_17_res, marker='x', label='start vs trained')
plt.axhline(y=0.54, color='red', linestyle='--', label='Fidelity @ 0.82')
plt.fill_between([5, 11], 0.55, 0.53, color='red', alpha=0.1)
plt.axhline(y=0.153, color='orange', linestyle='--', label='Fidelity @ 0.985')
plt.fill_between([5, 11], 0.157, 0.149, color='orange', alpha=0.1)
plt.axhline(y=0.122, color='green', linestyle='--', label='Fidelity @ 0.99')
plt.fill_between([5, 11], 0.125, 0.119, color='green', alpha=0.1)
plt.title('17 active micro-heaters')
plt.grid(ls='--', alpha=0.3)
plt.xlabel('# target')
plt.ylabel('Frobenius norm')
plt.legend()
plt.xlim(5.75, 10.25)
plt.xticks(x_17)
plt.savefig('./Immagini/frob_17.pdf')  # Salva la figura per Frob_17

# %%
'''
FIGURE STABILITÀ
'''
#%%
with np.load("/media/dati_2/results/2026-06-15_02-28-13_128modi_stability_T1N_duration6_rep1-10_2.npz") as temp:
    stabilitySingles = temp["arr_1"]
    stabilityDoubles = temp["arr_2"]
    print(temp.keys())
#%%
colorsArray = ["darkviolet", "orange", "green", "red"]

labelSize = 28

# fig, axs = plt.subplots(2, 1, dpi=200, figsize=(6, 11))

# fig.tight_layout()

# fig.subplots_adjust( wspace=0.3, hspace=0.4)
x = np.linspace(0,14, len(stabilitySingles))
print(x*60*60)
print((6*4+60*6)*len(stabilitySingles)/60/60)
plt.plot(x,stabilitySingles, color = "darkviolet")
plt.title("Singles", size = labelSize, fontweight='bold', pad = 10)
plt.xlabel("Time [h]", size = 24)
plt.ylabel("Loss", size = 24)
plt.xticks(np.arange(0, 15, 2))
plt.yticks(np.arange(0.07, 0.105, 0.005))
plt.tick_params(axis='both', which='major', labelsize=20, pad = 10)
plt.tick_params(axis='both', which='minor', labelsize=20)
plt.grid(ls='--', alpha=0.3)
#plt.savefig('./Immagini/stability_singles.pdf')  # Salva la figura per Frob_17
plt.show()
#plt.xaxis.set_major_locator(plt.MaxNLocator(6))
#axs[0].yaxis.set_major_locator(plt.MaxNLocator(5))

x = np.linspace(0,14, len(stabilityDoubles))
plt.plot(x, stabilityDoubles, color = "darkviolet")
#plt].text( -0.25, 0.95,'b', transform=axs[1].transAxes, size = labelSize, fontweight='bold')
plt.title("Doubles", size = labelSize, fontweight='bold', pad = 10)
plt.xlabel("Time [h]", size = 24)
plt.ylabel("Loss", size = 24)
plt.tick_params(axis='both', which='major', labelsize=20, pad = 10)
plt.tick_params(axis='both', which='minor', labelsize=20)
plt.grid(ls='--', alpha=0.3)
plt.xticks(np.arange(0, 15, 2))
plt.yticks(np.arange(0.17, 0.24, 0.01))
#plt.savefig('./Immagini/stability_doubles.pdf')  # Salva la figura per Frob_17
plt.show()
# %%
target_1 = np.array([0.076,0.090,0.085])
days_1 = np.array([9, 16, 20])

all_0 = np.array([0.054, 0.055])
days_0 = np.array([7, 14])

plt.scatter(days_1, target_1, marker='o', label='target 1')
plt.scatter(days_0, all_0, marker='x', label='all 0')