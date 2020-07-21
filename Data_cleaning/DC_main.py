###############################################################################################################
# Project     : Customer Science for bizCare Call Reduction
#
# Coding      : Xiong Yuyu
#
# Date        : Since 2020-01-30
#
# Description : Main function to clean 
###############################################################################################################

print("\n========================= DC on XXX =========================")
import sys
sys.path.append('./SourceCode')
from Config import *
Env_Config.fun_set_cwd(".")

from Data_cleaning.File_catalog import File_Catalog

File_Catalog.fun_set_file_catalog_b4_FE()

file_exist=os.path.isfile(File_Catalog.fun_get_feather_filepath(file_name='table_name'))

if file_exist is False:
    import Data_cleaning.XXX
    del sys.modules['Data_cleaning.XXX']

#clear memory
from Global_fun import fun_del_all
fun_del_all(var_to_del = dir(), g_var = globals())
import gc
gc.collect()


