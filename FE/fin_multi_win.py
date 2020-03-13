from Load_Package import *
from Config import Env_Config

class multi_win_wrap(object):
    
    def __init__(self, test = 10, training_win_prefix = "FEA_multi_win_" , testing_win_prefix = "FEA_all_", multi_win_source = 'Y_multi_start_time_test.csv', y_train_source = 'Y_start_time_train.csv', y_test_source = 'Y_start_time_test.csv'):
        
        self.test = 10
        self.training_win_prefix = training_win_prefix
        self.testing_win_prefix = testing_win_prefix
        self.y_train_source = y_train_source
        self.y_test_source = y_test_source
        self.multi_win_source =  multi_win_source
        
        
    def check_FE_train(self, dt):
        event_start_date = pd.to_datetime(dt) - pd.DateOffset(days = 7)
        event_start_str = event_start_date.strftime('%Y-%m-%d')
        training_file = self.training_win_prefix + event_start_str + '.feather'
        return training_file, os.path.isfile(fun_path_join(Env_Config.output_FE_train, training_file))
    
    def check_FE_test(self, dt):
        testing_file = self.testing_win_prefix + dt + '.feather'
        return testing_file, os.path.isfile(fun_path_join(Env_Config.output_FE_test, testing_file))

    
    def fun_update_y_train(self, last_dt):
        file = pd.read_csv(fun_path_join(Env_Config.source_code_path, self.y_train_source))
        size = Env_Config.need_win_size # 7 days
        last_dt = pd.to_datetime(last_dt) - pd.DateOffset(days = 7)
        dt_list = [0]*size
        dt_list[size-1] = last_dt

        while size-1 != 0:
            old_index = size - 1
            new_index = size - 2
            new_date = dt_list[old_index] - pd.DateOffset(days = 7)
            dt_list[new_index] = new_date
            size = size -1

        dt_list = [x.strftime('%Y-%m-%d') for x in dt_list]

        new_df = pd.DataFrame({
            'Y_start_time':dt_list
        })
        # update the value
        new_df = pd.merge(new_df, file, on='Y_start_time', how = 'left')
        new_df = new_df.fillna(0)
        new_df.handled = new_df.handled.astype('int')

        # update the values in Y_start_time_train before running FE_main
        new_df.to_csv(fun_path_join(Env_Config.source_code_path, self.y_train_source), index = False)
    
    
    def fun_update_y_test(self, last_dt):
        file = pd.read_csv(fun_path_join(Env_Config.source_code_path, self.y_test_source))
        # update the values in Y_start_time_test.csv before running FE_main
        file['Y_start_time'] = last_dt
        file.to_csv(fun_path_join(Env_Config.source_code_path, self.y_test_source), index = False)
    

    def fun_update_multi_win(self, dt):
        file = pd.read_csv(fun_path_join(Env_Config.source_code_path, self.multi_win_source))
        file.loc[(file.Y_start_time == dt),'handled'] = 1 # update value
        file.to_csv(fun_path_join(Env_Config.source_code_path, self.multi_win_source), index = False)
