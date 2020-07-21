import sys
from datetime import datetime as date

sys.path.append('./SourceCode')

from Load_Package import *
from Config import Env_Config
Env_Config.fun_set_cwd(".")

from FE.fun_multi_FE import *

# With default settings 
multi_fe = multi_win_wrap()

#multi_fe = multi_win_wrap(training_win_prefix = "FEA_multi_win_y_2_n_10_")

while True:
    
    # initialise the class to access the variable 
    multi_win = Env_Config.fun_get_Y_start_time(file_name = fun_path_join(Env_Config.source_code_path,
                                                                          multi_fe.multi_win_source))
    
    if multi_win != None:
        
        train_fe, train_check = multi_fe.check_FE_train(dt=multi_win)
        test_fe, test_check = multi_fe.check_FE_test(dt=multi_win)
        
        print(f"\n=== Checking Training and Testing Features for Needs Window - {multi_win} ===")
        print(f"---Training data {train_fe} exist - {train_check} ---")
        print(f"---Testing data {test_fe} exist - {test_check} ---\n")
    
        # Check if FE was created
        if train_check is False or test_check is False:
            
            print(f'========== Generate Training and testing features for week {multi_win} ==========\n')
            # function to update Y_start_time_train.csv
            
            print(f'+++ Update {multi_fe.y_train_source} ---')
            multi_fe.fun_update_y_train(last_dt = multi_win)

            print(f'+++ Update {multi_fe.y_test_source} ---')
            # function to update Y_start_time_test.csv
            multi_fe.fun_update_y_test(last_dt = multi_win)
            
            print('+++ Training Features Windows ---')
            train = pd.read_csv(fun_path_join(Env_Config.source_code_path, multi_fe.y_train_source))
            print(train)
            print('\n+++ Testing Features Window ---')
            test = pd.read_csv(fun_path_join(Env_Config.source_code_path, multi_fe.y_test_source))
            print(test)
            
            print('\n+++ Run FE_main to generate features ---\n')
            
            import FE_main
            del sys.modules['FE_main'] # to restart the feature generation
            
            # update multi_start_time_test.csv
            multi_fe.fun_update_multi_win(dt = multi_win)
            print(f'\n========== Features for week {multi_win} completed ==========\n')
            
        else: multi_fe.fun_update_multi_win(dt = multi_win)  
        
    else:
        print(" --- FE for all windows generated --- ") 
        break
            
