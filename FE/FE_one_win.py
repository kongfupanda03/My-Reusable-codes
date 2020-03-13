print("\n\n========================= FE on hist win =========================")

print("\n========================= FE (hist) on QMS =========================")

import sys
sys.path.append('./SourceCode')
from Config import *
Env_Config.fun_set_cwd(".")

#Show the window
print("win = [", Env_Config.hist_win_start.strftime('%Y-%m-%d'), ", ", Env_Config.hist_win_end.strftime('%Y-%m-%d'), "]")

file_exist = os.path.isfile(Env_Config.QMS_hist_all)

#If feature is not ready, create it
if file_exist == False:
    import FE.hist_FE_QMS
    del sys.modules["FE.hist_FE_QMS"]

#clear memory
from Global_fun import fun_del_all
fun_del_all(var_to_del = dir(), g_var = globals())


### generate top10 needs features
print('+++ FE(hist) on top10 needs ---')
import sys
sys.path.append('./SourceCode')
from Config import *
Env_Config.fun_set_cwd(".")

file_exist = os.path.isfile(Env_Config.hist_fea_top10_needs)

#If feature is not ready, create it
if file_exist == False:
    import FE.hist_FE_top10_needs
    del sys.modules["FE.hist_FE_top10_needs"]

#clear memory
from Global_fun import fun_del_all
fun_del_all(var_to_del = dir(), g_var = globals())


### generate all the other hist features
import sys
sys.path.append('./SourceCode')
from Config import *
Env_Config.fun_set_cwd(".")


file_exist = os.path.isfile(Env_Config.fea_hist_all)

#If feature is not ready, create it
if file_exist == False:
    import FE.FE_hist_main
    del sys.modules["FE.FE_hist_main"]

#clear memory
from Global_fun import fun_del_all
fun_del_all(var_to_del = dir(), g_var = globals())

print("\n\n========================= FE on event win =========================")

import sys
sys.path.append('./SourceCode')
from Config import *
Env_Config.fun_set_cwd(".")

#Show the window
print("win = [", Env_Config.event_win_start.strftime('%Y-%m-%d'), ", ", Env_Config.event_win_end.strftime('%Y-%m-%d'), "]")

file_exist = os.path.isfile(Env_Config.fea_event_all)

if file_exist == False:
    import FE.FE_event_main
    del sys.modules["FE.FE_event_main"]

#clear memory
from Global_fun import fun_del_all
fun_del_all(var_to_del = dir(), g_var = globals())


print("\n\n========================= Label generation =========================")

import sys
sys.path.append('./SourceCode')
from Config import *
Env_Config.fun_set_cwd(".")

from Data_cleaning.File_catalog import File_Catalog
File_Catalog.fun_set_file_catalog_b4_FE()

#Show the window
print("win = [", Env_Config.need_win_start.strftime('%Y-%m-%d'), ", ", Env_Config.need_win_end.strftime('%Y-%m-%d'), "]")

file_exist = os.path.isfile(Env_Config.Y_all)

if file_exist == False:
    data_qms = pd.read_feather(File_Catalog.fun_get_feather_filepath(file_name= 'vw_dlrg_sg_qms_closedsr'))

    #Generate labels only if data covers need win
    if max(data_qms.sr_date) >= Env_Config.need_win_end:        
        del data_qms 
        import FE.Y_QMS_topx
        del sys.modules["FE.Y_QMS_topx"]
    
#clear memory
from Global_fun import fun_del_all
fun_del_all(var_to_del = dir(), g_var = globals())


import gc
gc.collect()

print("\n\n========================= Overall feature integration -- L2 =========================")

import sys
sys.path.append('./SourceCode')
from Config import *
Env_Config.fun_set_cwd(".")

file_exist = os.path.isfile(Env_Config.fea_all)

if file_exist == False:
    import FE.FE_integrate
    del sys.modules["FE.FE_integrate"]

#clear memory
from Global_fun import fun_del_all
fun_del_all(var_to_del = dir(), g_var = globals())




