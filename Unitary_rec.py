# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 10:59:20 2026

@author: ControlCenter
"""

'''
IMPORT LIBRARIES SECTION:
'''
#%%
from qlab.devices.tdc import QuTag
import qlab.counting.counting as counting
import qlab.counting.cocount as cocount
from auto_classical import PowerSupplies
from qlab.devices.KeithleyPowerSupply import KeithleyPowerSupply
import logging
from dmx_controller import DMXController
#%%
import numpy as np
from numba import njit
import time 
import pyvisa as visa
from tqdm import tqdm
from WhiteLib import process_measurement
from datetime import datetime
strnow = lambda: datetime.now().strftime("%Y%m%d-%H%M%S")
strtoday = lambda: datetime.now().strftime("%Y_%m_%d")
strtimenow = lambda: datetime.now().strftime("%H:%M:%S")
import time
import os
#%%
'''
FUNCTIONS SECTION:
'''
#%%
def setloop(input):
    loop =[0,0,0,5]
    for channel in input:
        if channel != 0 and channel != 5:
            loop[channel-1]=channel 
        else:
            print('You are measuring the dark')

    return loop

def control_volts(pairs):
    """
    Controlla ogni coppia in `pairs`: se uno dei due valori 0 o > 8,
    stampa un errore e imposta la coppia a [0, 0].
    """
    for idx, (a, b) in enumerate(pairs):
        # verifica se a o b sono fuori dall'intervallo 0\Uffffffff8
        if not (0 <= a <= 8 and 0 <= b <= 8):
            print(f"Errore: elementi fuori range in pairs[{idx}] = {pairs[idx]}")
            return False
    
    return True
def flatten_list(pairs):
    # prende ogni coppia in pairs e ogni elemento x in quella coppia
    return [x for pair in pairs for x in pair]

def change_voltages(supply: PowerSupplies, volts) -> None:
    assert isinstance(supply, PowerSupplies)
    assert isinstance(volts, list)
    assert supply._num_supplies==len(volts)
    if control_volts(volts):  
        volts = flatten_list(volts)
        supply.voltages = volts
        print('Volts changed ->' + str(supply.voltages))
    else:
        volts = [[0, 0] for _ in range(supply._num_supplies)]
        supply.voltages = flatten_list(volts)
        print('Volts reset to 0 due to error ->' + str(supply.voltages))

#%%
'''
INIZIALIZATION KEYTHLEYS
'''
#%%
rm = visa.ResourceManager()
print(rm.list_resources())
list_conncted=[x for x in rm.list_resources()]
addresses = ['ASRL24::INSTR', 'ASRL25::INSTR', 'ASRL26::INSTR', 'ASRL27::INSTR', 'ASRL28::INSTR', 'ASRL29::INSTR','USB0::0x05E6::0x2230::9100018::INSTR', 'USB0::0x05E6::0x2230::9102515::INSTR', 'ASRL6::INSTR', 'ASRL8::INSTR']
print(f'You are connecting {len(addresses)} Keithleys')
supply = PowerSupplies(addresses)
#%%
volts = [[0,0] for _ in range(len(addresses))]
change_voltages(supply, volts)

#%%
raw = np.random.uniform(0, 54, size=len(addresses)*2)
# 2) trasformali con la radice quadrata → [0,8]
volt_vals = np.sqrt(raw)
# 3) rimodella in 6x2 e ottieni la list-of-lists
volts = volt_vals.reshape((len(addresses), 2)).tolist()
print(f'You are going to set the following voltages:')
print(volts)
#%%
change_voltages(supply, volts)

#%%
#pretraining voltages
volts[0]= [26.91770056, 20.25482259]
volts[1]= [25.13671606, 12.33392859] 
volts[2]= [14.39218764 ,38.11764358]
volts[3]= [15.06129933, 26.79019464] 
volts[4]= [32.93635505, 34.19253223]
volts[5]= [26.58369102, 22.61000908]
volts[6]= [24.09356394, 23.83949231]
volts[7]= [12.50908916, 17.35610151] 
volts[8]= [23.04587543, 28.30849853] 
volts[9]= [11.9300337 , 30.21736944]
volts_array = list( np.sqrt(volts))
volts = [[float(x), float(y)] for x, y in volts_array]
print(f'You are going to set the following voltages:')
print(volts)

#%%
#aftertraining voltages
volts[0]=[28.52133494, 19.29275721]
volts[1]=[26.45806454, 12.33392859]
volts[2]=[14.07755464, 39.44090354]
volts[3]=[15.34390228, 28.40254952]
volts[4]=[32.2095023 , 32.60211475]
volts[5]=[26.81896368, 21.92135263]
volts[6]=[23.63006376, 24.02677582]
volts[7]=[12.10942377, 15.74827279]
volts[8]=[22.66837316, 29.22583959]
volts[9]=[13.56544436, 30.54044808]
volts_array = list( np.sqrt(volts))
volts = [[float(x), float(y)] for x, y in volts_array]
print(f'You are going to set the following voltages:')
print(f'You are going to set the following voltages:')
print(volts)

#%%
#all_20
volts[0]=[20, 20]
volts[1]=[20, 20]
volts[2]=[20, 20]
volts[3]=[20, 20]
volts[4]=[20, 20]
volts[5]=[20, 20]
volts[6]=[20, 20]
volts[7]=[20, 20]
volts[8]=[20, 20]
volts[9]=[20, 20]
volts_array = list( np.sqrt(volts))
volts = [[float(x), float(y)] for x, y in volts_array]
print(f'You are going to set the following voltages:')
print(f'You are going to set the following voltages:')
print(volts)

#%%
#all_20
volts[0]=[20, 20]
volts[1]=[20, 20]
volts[2]=[20, 20]
volts[3]=[20, 20]
volts[4]=[20, 20]
volts[5]=[20, 20]
volts[6]=[20, 20]
volts[7]=[20, 20]
volts[8]=[20, 20]
volts[9]=[20, 20]
volts_array = list( np.sqrt(volts))
volts = [[float(x), float(y)] for x, y in volts_array]
print(f'You are going to set the following voltages:')
print(f'You are going to set the following voltages:')
print(volts)
#%%
# Target volts
volts[0]= [5.601,4.346]
volts[1]= [5.367,3.763]
volts[2]= [3.396,5.966]
volts[3]= [4.299,5.298]
volts[4]= [5.832,5.795]
volts[5]= [5.099,4.853]
volts[6]= [4.801,4.724]
volts[7]= [3.132,3.577]
#volts[8]= [4.594,5.756]
volts[8]= [4.594,0]
#volts[9]= [3.842,5.787]
volts[9]= [3.842,0]

print(f'You are going to set the following voltages:')
print(volts)
#%%
#Target_2
volts[0] = [4.982, 6.744]
volts[1] = [5.936, 4.612]
volts[2] = [6.481, 5.619]
volts[3] = [6.817, 4.076]
volts[4] = [4.217, 2.074]
volts[5] = [4.483, 5.363]
volts[6] = [5.077, 1.618]
volts[7] = [4.077, 7.307]
volts[8] = [5.451, 1.067]
volts[9] = [6.470, 6.944]

print(f'You are going to set the following voltages:')
print(volts)

#%%

#after training  21_04_2026

volts[0] = [31.74052433, 21.91479155]
volts[1] = [34.01837048,  3.37147212]
volts[2] = [22.64993622, 30.93264166]
volts[3] = [20.49874402, 32.44260608]
volts[4] = [23.5903211 , 29.66610636]
volts[5] = [18.21220227, 18.5196574 ]
volts[6] = [20.77825292, 15.89228753]
volts[7] = [9.76500426,  26.72253415]
volts[8] = [26.38833187,  0.        ]
volts[9] = [19.77356721,  0.        ]

volts_array = list( np.sqrt(volts))
volts = [[float(x), float(y)] for x, y in volts_array]
print(f'You are going to set the following voltages:')
print(volts)


#%%

#refine 2026-05-01

volts[0] = [30.72086116, 19.19779719]
volts[1] = [29.07245286, 13.4723269]
volts[2] = [11.24319881, 36.57669847]
volts[3] = [18.14860641, 29.0169418]
volts[4] = [34.31649245, 34.5729494]
volts[5] = [25.97631842, 23.90355533]
volts[6] = [23.69507688, 22.61472729]
volts[7] = [10.47858461, 12.14714416]
volts[8] = [20.72134531,  0.]
volts[9] = [14.09070056,  0.]

volts_array = list( np.sqrt(volts))
volts = [[float(x), float(y)] for x, y in volts_array]
print(f'You are going to set the following voltages:')
print(volts)

#%%

#train 2026-05-02

volts[0] = [42.07222399, 32.08724222]
volts[1] = [34.30225481, 10.68487193] 
volts[2] = [11.60007042, 32.47799552]
volts[3] = [14.9079791,  24.37225923] 
volts[4] = [12.58147829, 49.67670304] 
volts[5] = [33.62438231, 46.89339419]
volts[6] = [31.24192644, 25.09432715]
volts[7] = [11.2817808,  45.83195757]
volts[8] = [25.8980875,   0.]
volts[9] = [21.45948571,  0.]


volts_array = list( np.sqrt(volts))
volts = [[float(x), float(y)] for x, y in volts_array]
print(f'You are going to set the following voltages:')
print(volts)



#%%
change_voltages(supply, volts)
#%%
'''
INIZIALIZATION QU-TAGS
'''
#%%
boxes_ind = QuTag.discover()

#%%
ind = [b'T 02 0010', b'T 02 0021']
boxes = []
for i in ind:
    for j in boxes_ind:
        if j._sn == i:
           boxes.append(j)

if len(ind) != len(boxes):
    raise ValueError('scatole non trovate')

#%%
'''
INIZIALIZATION QDMX
'''
#%%
channel_a= 0
channel_b= 1
channel_c= 2
channel_d= 3
channel_e= 4
channel_f= 5
#%%
dmx = DMXController(log_level=logging.DEBUG)
#%%
dmx.stop_looping()
#%%
print(dmx.get_data())

#%%

dmx.set_dwell_time(channel=channel_a, dwell_time=26)
dmx.set_dwell_time(channel=channel_b, dwell_time=26)
dmx.set_dwell_time(channel=channel_c, dwell_time=26)
dmx.set_dwell_time(channel=channel_d, dwell_time=26)
dmx.set_dwell_time(channel=channel_e, dwell_time=16)
dmx.set_dwell_time(channel=channel_f, dwell_time=16)
#%%
'''
SET WORKING DIRECTORY
'''
#%%
path='C:/Users/ControlCenter/Desktop/128_AutoCal_dati/'
dir_name = path+'DATI_' + strtoday() + '_trainedMultiphase_T3_1'
#dir_name = path+'misure_cluce_classica'

if not os.path.exists(dir_name):
    os.mkdir(dir_name)
print(dir_name)
save_path = dir_name

#%%
with open(os.path.join(path,dir_name) + '/'+"readme.txt", "w") as file:
    file.write("target 1\t3 coppie\t start target2\n")
    file.write("trigger ch 17 box 2\n")
    file.write('sync channels: ch 3 box 1 and ch 27 box 2\n')
    file.write("voltages:\n")
    for voltage in supply.voltages_measure:
        file.write(f"{voltage:.3f}\n")
    file.write("currents:\n")
    for curr in supply.currents_measure:
        file.write(f'{curr:.3e}\n')
#%%
''''
EXPERIMENTAL SECTION
'''
#%%
esposizione = 0.1   #in secondi
durata= 60   #in secondi
ripetizioni= int(durata/esposizione)

#%%
names=['b','c', 'd', 'e']
#Voltages=[0 for _ in range(len(addresses))]
inputs = [(1,), (2,), (3,), (4,), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
save_path = os.path.join(dir_name,'Ricostruzione_unitaria')
os.makedirs(save_path, exist_ok=True)

for sequence in inputs:
    
    photons = len(sequence)
    
    if photons == 1:
        folder_dark= os.path.join(save_path, 'Buio' ,f'{names[sequence[0]-1]}')
        os.makedirs(folder_dark, exist_ok=True )
        folder_signal = os.path.join(save_path, "Singles", f'{names[sequence[0]-1]}')
        os.makedirs(folder_signal, exist_ok=True )
        loop = setloop((0,))
        dmx.set_active_outputs(loop)
        time.sleep(1)
        
        for i in range(5):
            print(f'Dark measurement channel {names[sequence[0]-1]} {i+1}/5 - started at {strtimenow()}')
            save_name_dark = os.path.join(folder_dark, f'misura_{i+1}')
            measure = counting.get_raw_timestamps_multiple(boxes,esposizione,num_acq=ripetizioni)
            times = [(t,c) for t,c in measure]
            times_box_1,channels_box_1 = times[0]
            times_box_2,channels_box_2 = times[1]
            t_tot, c_tot = process_measurement(times, photons=0)
            np.savez_compressed(save_name_dark, t_tot=t_tot, c_tot=c_tot)
        dmx.stop_looping()
        time.sleep(1)
        loop = setloop(sequence)
        dmx.set_active_outputs(loop)
        time.sleep(1)
        
        for i in range(5):
            print(f'Singles measurement channel {names[sequence[0]-1]} {i+1}/5 - started at {strtimenow()}')
            save_name_signal = os.path.join(folder_signal, f'misura_{i+1}')
            measure = counting.get_raw_timestamps_multiple(boxes,esposizione,num_acq=ripetizioni)
            times = [(t,c) for t,c in measure]
            times_box_1,channels_box_1 = times[0]
            
            times_box_2,channels_box_2 = times[1]
            t_tot, c_tot = process_measurement(times, photons=0)
            np.savez_compressed(save_name_signal, t_tot=t_tot, c_tot=c_tot)
        dmx.stop_looping()
        time.sleep(1)
        
    elif photons==2:
        folder_couples= os.path.join(save_path, f'measurement_2ph_{names[sequence[0]-1]}{names[sequence[1]-1]}')
        os.makedirs(folder_couples, exist_ok=True )
        loop = setloop(sequence)
        dmx.set_active_outputs(loop)
        time.sleep(1)
        for i in range(20):
            print(f'Couples measurement channels {names[sequence[0]-1]}{names[sequence[1]-1]} {i+1}/20 - started at {strtimenow()}')
            save_name_couples = os.path.join(folder_couples, f'misura_{i+1}')
            measure = counting.get_raw_timestamps_multiple(boxes,esposizione,num_acq=ripetizioni)
            times = [(t,c) for t,c in measure]
            times_box_1,channels_box_1 = times[0]
            times_box_2,channels_box_2 = times[1]
            t_tot, c_tot = process_measurement(times, photons=0)
            np.savez_compressed(save_name_couples, t_tot=t_tot, c_tot=c_tot)
        dmx.stop_looping()
        time.sleep(1)
        
    else:
        raise ValueError('Invalid number of chosen inputs')

#%%
#targets
volts_list=[[[6.89765693666106, 5.312488106125661], [3.630398718260098, 3.7826782884299113], [4.449507725181073, 6.881427817822481], [6.424057912116605, 4.640282380264631], [6.707728999008379, 0.9664659788888431], [5.310490186552734, 2.3262002204308554], [3.492294436440019, 0.8716489092342264], [5.943570828044322, 7.255526305612546], [5.480404021669932, 3.2401111416112496], [5.29111763567369, 6.624389530943558]],
[[7.2696327015303766, 7.193223510778225], [6.650965158058854, 5.045525701952319], [5.87551890280537, 3.3014711415699582], [7.1824080358395825, 6.158660528926107], [6.486887207018805, 6.703464323462679], [6.545786471918548, 5.691128672782624], [6.983712067857741, 5.076308935819567], [4.060321818799795, 6.65378324976755], [6.670522118691086, 4.915874300273161], [6.2937593835182515, 5.9584711324928445]],
[[5.443173028358117, 5.440706204921123], [1.4344639032021054, 3.8618614843766026], [6.07384607830751, 6.964698138129756], [5.1446762932354435, 4.787519149560442], [2.579647436726789, 4.717386358077844], [7.1763860856775725, 2.5643807391449633], [5.251987586672348, 3.7658306552808463], [6.962151013644113, 6.306505400381115], [6.641337970735474, 4.241713129951217], [4.473330532776752, 3.1109402951729805]],
[[5.72817711144528, 3.357328783759293], [1.9773489639773094, 3.79977634016576], [6.786361918944681, 5.124054598348149], [5.243776961462014, 3.31303179192386], [6.151689778711378, 7.138695930024983], [2.4198649626523046, 3.2841898182930365], [6.8773709543450705, 6.24139415334723], [6.994136039121712, 6.364383774685572], [2.6524622445743615, 1.7203578466796718], [7.075394074572337, 6.359872698759157]],
[[5.212557419524484, 4.436237950259489], [6.650744912324947, 4.80536012375207], [7.295695950255964, 5.485625596328706], [4.439576720627527, 5.222736953359593], [7.23540397799699, 1.9305835114303356], [5.56847866121124, 6.1020846036277065], [5.815419217014174, 6.227749960317922], [6.627733893424182, 7.338029125087826], [6.462192452898931, 3.9837847337524765], [5.979889708197363, 6.136454870060503]],
[[6.533929623371079, 1.5097419457966832], [2.0220404994938153, 5.514343042232568], [6.622073118683308, 4.551999290321775], [6.183767139069712, 6.758851784147259], [6.950939550445265, 3.9840152141238434], [3.5274974452528904, 4.936098003630582], [4.915145614919425, 0.9508092445986925], [6.70378496849807, 4.754653114810923], [3.7978488341479264, 5.771893095614393], [5.502514822794869, 5.940328218514365]],
[[5.1027659988602565, 4.4226341704803], [6.526074634514491, 3.557815312447542], [6.383977526788753, 7.195894894445404], [4.539531691308365, 5.602712230697879], [4.643116842438799, 4.964295931384257], [1.9065485763326127, 1.935806439062836], [3.2809544792388983, 4.061816287725975], [4.901827849854543, 6.703906621053668], [7.083603914385112, 5.060849982855844], [3.588320371150958, 6.161088659765744]]]
#%%
#misure intermedie
#2026-05-02_16-31-34_128modi_training_target3_3PairsPre_32Start_resultF2_1
#2026-05-02_16-31-34_128modi_training_target3_3PairsPre_32Start_resultF1_1
#2026-05-02_16-31-34_128modi_training_target3_3PairsPre_32Start_intermediate_1
volts_list=[[[6.484363489934788, 5.685906611667912], [5.854941573618467, 3.3918464744900674], [3.552527562644707, 5.771887719094912], [3.7288976501017577, 4.827180531407561], [3.6096627362526044, 7.076716113415171], [5.837358192324938, 6.813276701783147], [5.627177639503878, 5.029387034472074], [3.4750873492021035, 6.793868151000515], [5.1310601878711015, 0.0], [4.695200098436315, 0.0]],
            [[6.424368734543291, 5.772576561358753], [5.703219239360495, 3.648073169280633], [3.764355008566168, 5.705639833182349], [3.8382897087152132, 4.7462416520434125], [3.717778156163042, 7.01823099166367], [5.772129378937554, 6.698438853870754], [5.7014731424605625, 5.105573709341112], [3.583264460605313, 6.676635317553235], [5.1310601878711015, 0.0], [4.68166646197984, 0.0]]]

#%%
esposizione = 0.1   #in secondi
durata= 60   #in secondi
ripetizioni= int(durata/esposizione)    
path='C:/Users/ControlCenter/Desktop/128_AutoCal_dati/'
names=['b','c', 'd', 'e']
#Voltages=[0 for _ in range(len(addresses))]
inputs = [(1,), (2,), (3,), (4,), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
dir_name_sup = path+'DATI_' + strtoday() + '_target_ripetuti'
for kkk in range(len(volts_list)):
    volts= volts_list[kkk]
    change_voltages(supply, volts)

    dir_name = os.path.join(dir_name_sup, f'measurement_{kkk+1}')
    os.makedirs(dir_name, exist_ok=True)
    save_path = os.path.join(dir_name,'Ricostruzione_unitaria')
    os.makedirs(save_path, exist_ok=True)

    print(save_path)
    with open(os.path.join(path,dir_name) + '/'+"readme.txt", "w") as file:
        file.write("trigger ch 17 box 2\n")
        file.write('sync channels: ch 3 box 1 and ch 27 box 2\n')
        file.write("voltages:\n")
        for voltage in supply.voltages_measure:
            file.write(f"{voltage:.3f}\n")
        file.write("currents:\n")
        for curr in supply.currents_measure:
            file.write(f'{curr:.3e}\n')
            
    for sequence in inputs:
        
        photons = len(sequence)
        
        if photons == 1:
            folder_dark= os.path.join(save_path, 'Buio' ,f'{names[sequence[0]-1]}')
            os.makedirs(folder_dark, exist_ok=True )
            folder_signal = os.path.join(save_path, "Singles", f'{names[sequence[0]-1]}')
            os.makedirs(folder_signal, exist_ok=True )
            loop = setloop((0,))
            dmx.set_active_outputs(loop)
            time.sleep(1)
            
            for i in range(5):
                print(f'Dark measurement channel {names[sequence[0]-1]} {i+1}/5 - started at {strtimenow()}')
                save_name_dark = os.path.join(folder_dark, f'misura_{i+1}')
                measure = counting.get_raw_timestamps_multiple(boxes,esposizione,num_acq=ripetizioni)
                times = [(t,c) for t,c in measure]
                times_box_1,channels_box_1 = times[0]
                times_box_2,channels_box_2 = times[1]
                t_tot, c_tot = process_measurement(times, photons=0)
                np.savez_compressed(save_name_dark, t_tot=t_tot, c_tot=c_tot)
            dmx.stop_looping()
            time.sleep(1)
            loop = setloop(sequence)
            dmx.set_active_outputs(loop)
            time.sleep(1)
            
            for i in range(5):
                print(f'Singles measurement channel {names[sequence[0]-1]} {i+1}/5 - started at {strtimenow()}')
                save_name_signal = os.path.join(folder_signal, f'misura_{i+1}')
                measure = counting.get_raw_timestamps_multiple(boxes,esposizione,num_acq=ripetizioni)
                times = [(t,c) for t,c in measure]
                times_box_1,channels_box_1 = times[0]
                times_box_2,channels_box_2 = times[1]
                t_tot, c_tot = process_measurement(times, photons=0)
                np.savez_compressed(save_name_signal, t_tot=t_tot, c_tot=c_tot)
            dmx.stop_looping()
            time.sleep(1)
            
        elif photons==2:
            folder_couples= os.path.join(save_path, f'measurement_2ph_{names[sequence[0]-1]}{names[sequence[1]-1]}')
            os.makedirs(folder_couples, exist_ok=True )
            loop = setloop(sequence)
            dmx.set_active_outputs(loop)
            time.sleep(1)
            for i in range(20):
                print(f'Couples measurement channels {names[sequence[0]-1]}{names[sequence[1]-1]} {i+1}/20 - started at {strtimenow()}')
                save_name_couples = os.path.join(folder_couples, f'misura_{i+1}')
                measure = counting.get_raw_timestamps_multiple(boxes,esposizione,num_acq=ripetizioni)
                times = [(t,c) for t,c in measure]
                times_box_1,channels_box_1 = times[0]
                times_box_2,channels_box_2 = times[1]
                t_tot, c_tot = process_measurement(times, photons=0)
                np.savez_compressed(save_name_couples, t_tot=t_tot, c_tot=c_tot)
            dmx.stop_looping()
            time.sleep(1)
            
        else:
            raise ValueError('Invalid number of chosen inputs')
            
#%%

inputs = [(1,2,3)]
for sequence in inputs:
    photons = len(sequence)
    folder_txt = os.path.join(dir_name, 'threefold_coincidences')
    os.makedirs(folder_txt, exist_ok=True)
    loop = setloop(sequence)
    dmx.set_active_outputs(loop)
    time.sleep(1)
    for i in range(100):
        print(f'Boson sampling measurement {i+1}/100 - started at {strtimenow()}')
        file_txt = os.path.join(folder_txt, f'misura_{i+1}.txt')
        measure = counting.get_raw_timestamps_multiple(boxes,esposizione,num_acq=ripetizioni)
        times = [(t,c) for t,c in measure]
        times_box_1,channels_box_1 = times[0]
        times_box_2,channels_box_2 = times[1]
        coincidences = process_measurement(times, photons=photons)
        with open(file_txt, 'w') as f:
            for group in coincidences:
                f.write(f"{tuple(int(x) for x in group)}\n")
    dmx.stop_looping()
    time.sleep(1)
#%%
volts = [[0,0] for _ in range(len(addresses))]
change_voltages(supply, volts)

 #%%

del dmx