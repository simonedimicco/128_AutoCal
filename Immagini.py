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
