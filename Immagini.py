#%%
import os

from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np

#%%
# frob_18_res = np.array([0.108, 0.135, 0.143, 0.116, 0.150])
# frob_17_res = np.array([0.142, 0.122, 0.106, 0.117, 0.121])
# start_17_res = np.array([0.521,0.513,0.397,0.566, 0.406])

frob_18_res = np.array([10.496488, 13.123665, 16.307270, 10.689613, 13.727252])
start_18_res = np.array([17.911014,	17.309212,	18.995804	,18.661785,	16.857199])
frob_17_res = np.array([10.904798, 9.459026, 10.276665, 7.401263, 6.867322])
start_17_res = np.array([19.168938,	18.003755,	15.937966,	20.550905,	15.138947])
x_18 = np.arange(1, 6)
x_17 = np.arange(6, 11)
y_ticks = np.arange(4,21,2)
#%%
plt.figure(figsize=(5, 4))

#plt.axhline(y=0.153, color='orange', linestyle='--', label='Fidelity @ 0.985')
#plt.fill_between([0, 6], 0.157, 0.149, color='orange', alpha=0.1)
#plt.axhline(y=0.122, color='green', linestyle='--', label='Fidelity @ 0.99')
#plt.fill_between([0, 6], 0.125, 0.119, color='green', alpha=0.1)

# plt.axhline(y=18.9, color='orange', linestyle='--', label='Fidelity @ 0.32', alpha=0.3)
# plt.fill_between([0, 11], 18.9-0.4, 18.9+0.4, color='orange', alpha=0.1)

# plt.axhline(y=16.5, color='red', linestyle='--', label='Fidelity @ 0.48', alpha=0.3)
# plt.fill_between([0, 11], 16.5-0.4, 16.5+0.4, color='red', alpha=0.1)

# plt.axhline(y=13, color='green', linestyle='--', label='Fidelity @ 0.68', alpha=0.3)
# plt.fill_between([0, 11], 13-0.3, 13+0.3, color='green', alpha=0.1)

# plt.axhline(y=9.8, color='dodgerblue', linestyle='--', label='Fidelity @ 0.81', alpha=0.3)
# plt.fill_between([0, 11], 9.8-0.3, 9.8+0.3, color='dodgerblue', alpha=0.1)

# plt.axhline(y=7.2, color='purple', linestyle='--', label='Fidelity @ 0.90', alpha=0.3)
# plt.fill_between([0, 11], 7.2-0.2, 7.2+0.2, color='purple', alpha=0.1)
# plt.xlim(0.75,10.25)
plt.axhline(y=19, color='red', linestyle=':', label='random matrices')
plt.fill_between([0, 11], 19-1, 19+1, color='red', alpha=0.1)
plt.xlim(0.75,10.25)
plt.axhline(y=7.2, color='orange', linestyle=':', label='stability 18 micro-heaters')
plt.fill_between([0, 11], 7.2-1.6, 7.2+1.6, color='orange', alpha=0.1)
plt.xlim(0.75,10.25)
plt.scatter(x_18, frob_18_res, marker='o', label='target', c='darkviolet')
plt.scatter(x_18, start_18_res, marker='x', label='start', c='darkviolet')
plt.xlabel('# target', fontsize=16)
plt.ylabel(r'$L^1_{norm}$', fontsize=16)
plt.title('18 active micro-heaters')
plt.xticks(x_17, fontsize=16)
plt.yticks(y_ticks, fontsize=16)
plt.ylim(3.25, 21)
plt.xlim(0.75,5.25)
plt.xticks(x_18, fontsize=16)
plt.tight_layout()
plt.grid(ls='--', alpha=0.3)

plt.savefig('./Immagini/frob_18.pdf') 
plt.show()


#%%
plt.figure(figsize=(7.8, 4))
plt.axhline(y=19, color='red', linestyle=':', label='random matrices')
plt.fill_between([0, 11], 19-1, 19+1, color='red', alpha=0.1)
plt.xlim(0.75,10.25)
plt.axhline(y=7.2,xmin=0, xmax=0, color='orange', linestyle=':', label='stability 18')
plt.fill_between([0, 1], 7.2-1.6, 7.2+1.6, color='orange', alpha=0.1)
plt.xlim(0.75,10.25)
plt.axhline(y=4.2,xmin=0, xmax=1, color='green', linestyle=':', label='stability 17')
plt.fill_between([0, 11], 4.2-0.6, 4.2+0.6, color='green', alpha=0.1)
plt.scatter(x_17, frob_17_res, marker='o', label='target', c='darkviolet')
plt.scatter(x_17, start_17_res, marker='x', label='start', c='darkviolet')
plt.xlim(5.75,10.25)
plt.title('17 active micro-heaters')
plt.xlabel('# target', fontsize=16)
plt.ylabel(r'$L^1_{norm}$', fontsize=16)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
plt.xticks(x_17, fontsize=16)
plt.yticks(y_ticks, fontsize=16)
plt.ylim(3.25, 21)
plt.grid(ls='--', alpha=0.3)

plt.tight_layout()
plt.savefig('./Immagini/frob_17.pdf')  # Salva la figura per Frob_18
plt.show()



#%%
plt.figure(figsize=(10, 4))
plt.scatter(x_17, frob_17_res, marker='o', label='target vs trained')
plt.scatter(x_17, start_17_res, marker='x', label='start vs trained')
# plt.axhline(y=0.54, color='red', linestyle='--', label='Fidelity @ 0.82')
# plt.fill_between([5, 11], 0.55, 0.53, color='red', alpha=0.1)
# plt.axhline(y=0.153, color='orange', linestyle='--', label='Fidelity @ 0.985')
# plt.fill_between([5, 11], 0.157, 0.149, color='orange', alpha=0.1)
# plt.axhline(y=0.122, color='green', linestyle='--', label='Fidelity @ 0.99')
# plt.fill_between([5, 11], 0.125, 0.119, color='green', alpha=0.1)
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

labelSize = 20

# fig, axs = plt.subplots(2, 1, dpi=200, figsize=(6, 11))

# fig.tight_layout()

# fig.subplots_adjust( wspace=0.3, hspace=0.4)
x = np.linspace(0,14, len(stabilitySingles))
print(x*60*60)
print((6*4+60*6)*len(stabilitySingles)/60/60)
plt.plot(x,stabilitySingles, color = "darkviolet")
plt.title("Singles", size = labelSize, fontweight='bold', pad = 10)
plt.xlabel("Time [h]", size = 16)
plt.ylabel("Loss", size = 16)
plt.xticks(np.arange(0, 15, 2))
plt.yticks(np.arange(0.07, 0.105, 0.005))
plt.tick_params(axis='both', which='major', labelsize=16, pad = 10)
plt.tick_params(axis='both', which='minor', labelsize=16)
plt.grid(ls='--', alpha=0.3)
plt.tight_layout()
plt.savefig('./Immagini/stability_singles.pdf')  # Salva la figura per Frob_17
plt.show()
#plt.xaxis.set_major_locator(plt.MaxNLocator(6))
#axs[0].yaxis.set_major_locator(plt.MaxNLocator(5))

x = np.linspace(0,14, len(stabilityDoubles))
plt.plot(x, stabilityDoubles, color = "darkviolet")
#plt].text( -0.25, 0.95,'b', transform=axs[1].transAxes, size = labelSize, fontweight='bold')
plt.title("Doubles", size = labelSize, fontweight='bold', pad = 10)
plt.xlabel("Time [h]", size = 16)
plt.ylabel("Loss", size = 16)
plt.tick_params(axis='both', which='major', labelsize=16, pad = 10)
plt.tick_params(axis='both', which='minor', labelsize=16)
plt.grid(ls='--', alpha=0.3)
plt.xticks(np.arange(0, 15, 2))
plt.yticks(np.arange(0.17, 0.24, 0.01))
plt.tight_layout()
plt.savefig('./Immagini/stability_doubles.pdf')  # Salva la figura per Frob_17
plt.show()
# %%
target_1 = np.array([0.076,0.090,0.085])
days_1 = np.array([9, 16, 20])

all_0 = np.array([0.054, 0.055])
days_0 = np.array([7, 14])

plt.scatter(days_1, target_1, marker='o', label='target 1')
plt.scatter(days_0, all_0, marker='x', label='all 0')

#%%
path = '/media/dati_2/results/'
with np.load(os.path.join(path, "2026-06-15_16-51-08_128modi_LossLandscape_target1New_param0_duration6_1.npz")) as temp:
    lossLandscapeSingles_S1D6 = temp["arr_1"]
    lossLandscapeDoubles_S1D6 = temp["arr_2"]

with np.load(os.path.join(path, "2026-06-16_02-42-24_128modi_LossLandscape_target1New_param0_duration1_1.npz")) as temp:
    lossLandscapeSingles_S1D1 = temp["arr_1"]
    lossLandscapeDoubles_S1D1 = temp["arr_2"]


with np.load(os.path.join(path, "2026-06-16_05-34-24_128modi_LossLandscape_target1New_param0_duration6_32Start_1.npz")) as temp:
    lossLandscapeSingles_S32D6 = temp["arr_1"]
    lossLandscapeDoubles_S32D6 = temp["arr_2"]

x= np.linspace(2,62, len(lossLandscapeSingles_S1D6))
x_ticks = np.linspace(2,62, 7)
#%%
colorsArray = ["darkviolet", "orange", "green", "red"]


plt.plot(x,lossLandscapeSingles_S1D6, color = colorsArray[0], label='long duration')
plt.plot(x,lossLandscapeSingles_S1D1, color = colorsArray[1], label='short duration')
plt.plot(x,lossLandscapeSingles_S32D6, color = colorsArray[2], label='random start')
#plt.text( -0.25, 0.95,'a', transform=plt.gca().transAxes, size = 20, fontweight='bold')
plt.title("Singles", size = 16, fontweight='bold', pad = 10)
plt.xlabel("$V^2$", size = 16)
plt.ylabel("Loss", size = 16)
plt.tick_params(axis='both', which='major', labelsize=16, pad = 10)
plt.tick_params(axis='both', which='minor', labelsize=16)
plt.grid(ls='--', alpha=0.3)
plt.tight_layout()
plt.xticks(x_ticks)
plt.legend(fontsize=16, loc='lower left')
plt.savefig('./Immagini/loss_landscape_singles.pdf')  # Salva la figura per Frob_17
plt.show()
#%%
plt.plot(x,lossLandscapeDoubles_S1D6, color = colorsArray[0])
plt.plot(x,lossLandscapeDoubles_S1D1, color = colorsArray[1])
plt.plot(x,lossLandscapeDoubles_S32D6, color = colorsArray[2])
#plt.text( -0.25, 0.95,'b', transform=axs[0,1].transAxes, size = labelSize, fontweight='bold')
plt.title("Doubles", size = 16, fontweight='bold', pad = 10)
plt.xlabel("$V^2$", size = 16)
plt.ylabel("Loss", size = 16)
plt.tick_params(axis='both', which='major', labelsize=16, pad = 10)
plt.tick_params(axis='both', which='minor', labelsize=16)
plt.grid(ls='--', alpha=0.3)
plt.tight_layout()
plt.xticks(x_ticks)
plt.savefig('./Immagini/loss_landscape_doubles.pdf')  # Salva la figura per Frob_17
#axs[1].set_xticks([0.1, 0.01, 0.001, 0.0001])

plt.show()
#%%

plt.plot(x,lossLandscapeSingles_S32D6, color = colorsArray[2])
plt.xticks(x_ticks)
#plt.text( -0.25, 0.95,'c', transform=axs[1,0].transAxes, size = labelSize, fontweight='bold')
# plt.xlabel("$V^2$", size = 24)
# plt.ylabel("Loss", size = 24)
plt.tick_params(axis='both', which='major', labelsize=24, pad = 10)
plt.tick_params(axis='both', which='minor', labelsize=24)

plt.tight_layout()
plt.grid(ls='--', alpha=0.3)
plt.savefig('./Immagini/loss_landscape_singles_S32D6.pdf')  # Salva la figura per Frob_17
plt.show()
#%%
plt.plot(x,lossLandscapeDoubles_S32D6, color = colorsArray[2])
#plt.text( -0.25, 0.95,'d', transform=axs[1,1].transAxes, size = labelSize, fontweight='bold')
# plt.xlabel("$V^2$", size = 24)
# plt.ylabel("Loss", size = 24)
plt.tick_params(axis='both', which='major', labelsize=24, pad = 10)
plt.tick_params(axis='both', which='minor', labelsize=24)
plt.xticks(x_ticks)
plt.tight_layout()
plt.grid(ls='--', alpha=0.3)
plt.savefig('./Immagini/loss_landscape_doubles_S32D6.pdf')  # Salva la figura per Frob_17
plt.show()




# %%
path = '/media/dati_2/results/'
with np.load(os.path.join(path, "2026-04-24_18-40-26_128modi_training_target2_3PairsPre_32Start_intermediate_1.npz")) as temp:
    lossSingles = temp["arr_1"]
    lossDoubles = temp["arr_2"]
    for key in temp.keys():
        print(key)

plt.figure(figsize=(7, 4))
plt.plot(lossSingles, linewidth=4, color = "darkviolet", label='Singles')

plt.savefig('./Immagini/Loss.pdf')  # Salva la figura per Frob_17
#plt.plot(lossDoubles)
# %%

#%%

epsilon   = np.array([0.030, 0.040, 0.055, 0.075, 0.100, 0.120])
fid_avg   = np.array([0.9435, 0.9011, 0.8186, 0.6808, 0.4828, 0.3219])
fid_sigma = np.array([0.0006, 0.0011, 0.0019, 0.0031, 0.0045, 0.0053])
L1_avg    = np.array([5.48, 7.24, 9.82, 13.02, 16.56, 18.98])
L1_sigma  = np.array([0.14, 0.19, 0.25, 0.32, 0.40, 0.43])

common_style = dict(fmt='o', color='#1f77b4', ecolor='#1f77b4',
                     elinewidth=1.2, capsize=4, markersize=6,
                     markeredgecolor='black', markeredgewidth=0.5)

# 1) epsilon vs Fidelity, errore su y
fig1, ax1 = plt.subplots(figsize=(6, 5))
ax1.errorbar(epsilon, fid_avg, yerr=fid_sigma, **common_style)
ax1.set_xlabel(r'$\epsilon$', fontsize=13)
ax1.set_ylabel('Fidelity', fontsize=13)
ax1.set_title(r'Fidelity vs. $\epsilon$', fontsize=13)
ax1.grid(True, linestyle='--', alpha=0.5)
fig1.tight_layout()
fig1.savefig('./Immagini/fidelity_vs_epsilon.pdf')
# fig1.savefig('fidelity_vs_epsilon.png', dpi=200)

# 2) epsilon vs L1, errore su y
fig2, ax2 = plt.subplots(figsize=(6, 5))

ax2.errorbar(epsilon, L1_avg, yerr=L1_sigma, **common_style)
ax2.set_xlabel(r'$\epsilon$', fontsize=24)
ax2.set_ylabel(r'$L^1_{norm}$', fontsize=24)
ax2.set_title(r'$L^1_{norm}$ vs. $\epsilon$', fontsize=24)
ax2.grid(True, linestyle='--', alpha=0.5)
fig2.tight_layout()
fig2.savefig('./Immagini/L1_vs_epsilon.pdf')
# fig2.savefig('L1_vs_epsilon.png', dpi=200)

# 3) L1 vs Fidelity, errore su x e y
# 3) Fidelity vs L1, errore su x e y, un colore per ogni epsilon
fig3, ax3 = plt.subplots(figsize=(8, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(epsilon)))
for eps, x, y, xerr, yerr, c in zip(epsilon, fid_avg, L1_avg,
                                     fid_sigma, L1_sigma, colors):
    ax3.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', color=c,
                 ecolor=c, elinewidth=3, capsize=6, markersize=9,
                 markeredgecolor='black', markeredgewidth=1,
                 label=fr'$\epsilon = {eps:.3f}$')

ax3.set_xlabel('Fidelity', fontsize=24)
ax3.set_ylabel(r'$L^1_{norm}$', fontsize=24)
ax3.set_title(r'$L^1_{norm}$ vs. Fidelity', fontsize=24)
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.legend(fontsize=16)

ax3.tick_params(axis='x', labelsize=22)
ax3.tick_params(axis='y', labelsize=22)
fig3.tight_layout()
fig3.savefig('./Immagini/L1_vs_fidelity.pdf')
plt.show()
print("done")
    # %%
colors = plt.cm.tab10(np.linspace(0, 1, len(epsilon)))

plt.figure(figsize=(6, 5))

for eps, x, y, xerr, yerr, c in zip(epsilon, fid_avg, L1_avg,fid_sigma, L1_sigma, colors):
    plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', color=c,
                 ecolor=c, elinewidth=1.2, capsize=4, markersize=7,
                 markeredgecolor='black', markeredgewidth=1,
                 label=fr'$\epsilon = {eps:.3f}$')

plt.xlabel('Fidelity', fontsize=24)
plt.ylabel(r'$L^1_{norm}$', fontsize=24)
plt.title(r'$L^1_{norm}$ vs. Fidelity', fontsize=24)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=16)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.tight_layout()
#plt.savefig('L1_vs_fidelity.pdf')
plt.savefig('./Immagini/L1_vs_fidelity.pdf')
plt.show()
# %%
