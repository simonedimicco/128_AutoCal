#%%
import matplotlib.pyplot as plt
import numpy as np

#%%
frob_18_res = np.array([0.108, 0.135, 0.143, 0.116, 0.150])
frob_17_res = np.array([0.142, 0.122, 0.106, 0.117, 0.121])
start_17_res = np.array([0.473, 0.578, 0.374, 0.568, 0.379])

x_18 = np.arange(1, 6)
x_17 = np.arange(6, 11)

#%%
plt.figure(figsize=(6, 4))
plt.scatter(x_18, frob_18_res, marker='o', label='target vs trained')
plt.title('18 active micro-heaters')
plt.grid(ls='--')
plt.xlabel('# target')
plt.ylabel('Frobenius norm')
plt.legend()
plt.xticks(x_18)
plt.savefig('frob_18.pdf')  # Salva la figura per Frob_18

#%%
plt.figure(figsize=(6, 4))
plt.scatter(x_17, frob_17_res, marker='o', label='target vs trained')
plt.scatter(x_17, start_17_res, marker='x', label='start vs trained')
plt.title('17 active micro-heaters')
plt.grid(ls='--')
plt.xlabel('# target')
plt.ylabel('Frobenius norm')
plt.legend()
plt.xticks(x_17)
plt.savefig('frob_17.pdf')  # Salva la figura per Frob_17

# %%
