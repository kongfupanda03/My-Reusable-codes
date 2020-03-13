###############################################################################################################
# Project     : Call Reduction
#
# Coding      : Xiong Yuyu
#
# Date        : 11/02/2020
#
# Description : Main function to create/integrate all Training and Testing features across 4 weeks
#               
###############################################################################################################


print("\n\n=========================== FE on training data ===========================")
while True:
    import sys
    sys.path.append('./SourceCode')
    from Config import *
    Env_Config.fun_set_cwd(".")
    
    next_Y_start_date = Env_Config.fun_get_Y_start_time(file_name = fun_path_join(Env_Config.source_code_path, "Y_start_time_train.csv"))

    if next_Y_start_date != None:
        print("\n\n++++++ FE on training data for Y start date: {} ------".format(next_Y_start_date))
        import FE.FE_one_win    
        del sys.modules["FE.FE_one_win"] #COMMENT. Do remember to delete module, if it needs to be run multiple times by import
    else:
        break
        
    #Markt the Y_start_date has been handled
    from Config import Env_Config
    Env_Config.fun_mark_Y_start_time(file_name = fun_path_join(Env_Config.source_code_path, "Y_start_time_train.csv"))    
#End of while


# KELLY -- delete all variables and clear memory for above
#clear memory
from Global_fun import fun_del_all
fun_del_all(var_to_del = dir(), g_var = globals())

import gc
gc.collect()


print("\n\n=========================== Integrate features across multiple training windows ===========================")

import sys
sys.path.append('./SourceCode')
from Config import *
Env_Config.fun_set_cwd(".")
# Env_Config.print_variable_debugging_only()

from FE.sample_integrate import cls_sample_integrate

cls_fea_integrate = cls_sample_integrate()
fea_all = cls_fea_integrate.fun_sample_integrate()



# COMMENT - add y_start_date to FEA_multi_win for multiple sliding window training data
# UPDATED - added y_start_date to name
y_start_train_dt = pd.read_csv(fun_path_join(Env_Config.source_code_path, "Y_start_time_train.csv"))
fea_multi = 'FEA_multi_win_' + y_start_train_dt['Y_start_time'].max() + '.feather'
print("+++ Output features to: {} ---".format(fun_path_join(Env_Config.output_FE_train, fea_multi)))
print("+++ Number of features for FEA_multi_win {} ---".format(fea_all.shape))
fea_all.to_feather(fun_path_join(Env_Config.output_FE_train, fea_multi))

# KELLY -- delete all variables and clear memory for above
#clear memory
from Global_fun import fun_del_all
fun_del_all(var_to_del = dir(), g_var = globals())

import gc
gc.collect()


print("\n\n=========================== FE on testing data ===========================")

# Testing data on one window 

import sys
sys.path.append('./SourceCode')
from Config import *
Env_Config.fun_set_cwd(".")
from FE.sample_integrate import cls_sample_integrate

from Data_cleaning.File_catalog import File_Catalog
File_Catalog.fun_set_file_catalog_b4_FE()

next_Y_start_date = Env_Config.fun_get_Y_start_time(file_name = fun_path_join(Env_Config.source_code_path, "Y_start_time_test.csv"))

# UPDATE - do not update y_start_time_test to avoid sys.exit from Config file
if next_Y_start_date != None:
    print("\n\n++++++ FE on testing data for Y start date: {} ------".format(next_Y_start_date))
    import FE.FE_one_win   
    
    del sys.modules["FE.FE_one_win"] #COMMENT. Do remember to delete module, if it needs to be run multiple times by import
else:
    print('Testing Data created')

###Binarise features
## To check if lable exists
data_qms = pd.read_feather(File_Catalog.fun_get_feather_filepath(file_name= 'vw_dlrg_sg_qms_closedsr'))
label = max(data_qms.sr_date) >= Env_Config.need_win_end
del data_qms
print("+++ Does label exist? ",label)

print("+++ Start coverting Fea_all to binarises labels ---")
Y_start_date = Env_Config.need_win_start.strftime('%Y-%m-%d')
hist_win_end = Env_Config.hist_win_end.strftime('%Y-%m-%d')
#Read in data
DF_fea = pd.read_feather(fun_path_join(Env_Config.output_FE_test, "FEA_all_" + Y_start_date + ".feather"))

cls_fea_integrate = cls_sample_integrate()
index = np.array_split(range(DF_fea.shape[0]),10)

for i in range(10):
    output_path = fun_path_join(Env_Config.output_FE_test_par, "FEA_all_" + "part_" + str(i+1)+ "_"+Y_start_date + ".feather")
    if os.path.isfile(output_path):
        print("Output path {} exists".format(output_path))
        continue
    else:
        temp = DF_fea.iloc[index[i],:].reset_index(drop=True)#add verification to check if 10 needs exist
        #temp_binary = cls_fea_integrate.fun_model_binarisation(data=temp,hist_end_date = hist_win_end,label=label)
        ###fun_model_binarisation()add a flag to indicate whether to remove label.
        print("Output path is : {}".format(output_path))
        temp.to_feather(output_path)
    del temp

# KELLY -- delete all variables and clear memory for above
#clear memory
from Global_fun import fun_del_all
fun_del_all(var_to_del = dir(), g_var = globals())

import gc
gc.collect()

