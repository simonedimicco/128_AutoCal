#%%
import numpy as np
import os
from tqdm import tqdm
rng = np.random.default_rng()
#%%
path_target = '/media/dati_2/DATI_2026_05_27_misure_multiple/target_1_new/Ricostruzione_unitaria'
path_bad = '/media/dati_2/DATI_2026_06_16_misure_multiple/target_1_new/Ricostruzione_unitaria'
path_good = '/media/dati_2/DATI_2026_06_17_128modi_training_target1New_3Pairs_T1OldStart_1/Ricostruzione_unitaria'

singles= 'singles_distributions.npz'
doubles= 'couples_distributions.npz'

singles_unnorm= 'singles_distributions_unnorm.npz'
doubles_unnorm= 'couples_distributions_unnorm.npz'
#%%
with np.load(os.path.join(path_target, singles)) as data:
    S_target = data['distributions']
with np.load(os.path.join(path_target, doubles)) as data:
    C_target = data['distributions']
with np.load(os.path.join(path_target, singles_unnorm)) as data:
    S_target_unnorm = data['distributions']
with np.load(os.path.join(path_target, doubles_unnorm)) as data:
    C_target_unnorm = data['distributions']

with np.load(os.path.join(path_bad, singles)) as data:
    S_bad = data['distributions']
with np.load(os.path.join(path_bad, doubles)) as data:
    C_bad = data['distributions']
with np.load(os.path.join(path_bad, singles_unnorm)) as data:
    S_bad_unnorm = data['distributions']
with np.load(os.path.join(path_bad, doubles_unnorm)) as data:
    C_bad_unnorm = data['distributions']


with np.load(os.path.join(path_good, singles)) as data:
    S_good = data['distributions']
with np.load(os.path.join(path_good, doubles)) as data:
    C_good = data['distributions']
with np.load(os.path.join(path_good, singles_unnorm)) as data:
    S_good_unnorm = data['distributions']
with np.load(os.path.join(path_good, doubles_unnorm)) as data:
    C_good_unnorm = data['distributions']

#%%
def MyMaeExp(y_predicted, y_true):
    total_error_arr = 0
    for yp, yt in zip(y_predicted, y_true):
        #print("YP = ", yp)
        #print("YT = ", yt)
        yp = (yp/np.sum(yp))
        yt = (yt/np.sum(yt))
        #print("YP = ", yp)
        #print("YT = ", yt)
        #print("yp: ", yp, "yt: ", yt)
        total_error_arr += abs(yp - yt)
    total_error = np.sum(total_error_arr)
    #print("Total error is:",total_error)
    mae = total_error/len(y_predicted)
    #print("Mean absolute error is:",mae)
    return mae
#%%
loss_singles_bad = MyMaeExp(S_bad, S_target)
loss_singles_good = MyMaeExp(S_good, S_target)
loss_doubles_bad = MyMaeExp(C_bad, C_target)
loss_doubles_good = MyMaeExp(C_good, C_target)
loss_singles_relative = MyMaeExp(S_good, S_bad)
loss_doubles_relative = MyMaeExp(C_good, C_bad)

print("Loss bad singles:", loss_singles_bad)
print("Loss bad doubles:", loss_doubles_bad)
print("Loss good singles:", loss_singles_good)
print("Loss good doubles:", loss_doubles_good)
print("Loss relative singles:", loss_singles_relative)
print("Loss relative doubles:", loss_doubles_relative)
#%%
list_loss_good_sim=[]
list_loss_bad_sim=[]

for j in tqdm(range(1000)):

    S_target_sim = rng.poisson(S_target_unnorm).astype(np.float64)
    C_target_sim = rng.poisson(C_target_unnorm).astype(np.float64)
    S_bad_sim = rng.poisson(S_bad_unnorm).astype(np.float64)
    C_bad_sim = rng.poisson(C_bad_unnorm).astype(np.float64)
    S_good_sim = rng.poisson(S_good_unnorm).astype(np.float64)
    C_good_sim = rng.poisson(C_good_unnorm).astype(np.float64)

    for i in range(S_target_sim.shape[0]):
        S_target_sim[i] = S_target_sim[i]/np.sum(S_target_sim[i])
        S_bad_sim[i] = S_bad_sim[i]/np.sum(S_bad_sim[i])
        S_good_sim[i] = S_good_sim[i]/np.sum(S_good_sim[i])
    for i in range(C_target_sim.shape[0]):
        C_target_sim[i] = C_target_sim[i]/(np.sum(C_target_sim[i])/2)
        C_bad_sim[i] = C_bad_sim[i]/(np.sum(C_bad_sim[i])/2)
        C_good_sim[i] = C_good_sim[i]/(np.sum(C_good_sim[i])/2)

    loss_singles_bad_sim = MyMaeExp(S_bad_sim, S_target_sim)
    loss_singles_good_sim = MyMaeExp(S_good_sim, S_target_sim)
    loss_doubles_bad_sim = MyMaeExp(C_bad_sim, C_target_sim)
    loss_doubles_good_sim = MyMaeExp(C_good_sim, C_target_sim)

    list_loss_good_sim.append(loss_singles_good_sim+loss_doubles_good_sim)
    list_loss_bad_sim.append(loss_singles_bad_sim+loss_doubles_bad_sim)

error_loss_good=np.std(np.array(list_loss_good_sim))
error_loss_bad=np.std(np.array(list_loss_bad_sim))
#%%
print("Loss bad singles:", loss_singles_bad + loss_doubles_bad)
print("Loss good singles:", loss_singles_good + loss_doubles_good)
print('std sulle loss good:', error_loss_good)
print('std sulle loss bad:', error_loss_bad)
    


# %%
print(np.mean(list_loss_good_sim))

# %%
list_loss_good_sim
# %%
rng.poisson(S_target_unnorm).astype(np.float64)
# %%
