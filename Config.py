###############################################################################################################
# Project     : Customer Behavior Prediction
#
# Coding      : Xiong Yuyu
#
# Date        : Since 2019-10-31
#
# Description : To configure project environment
#               1) folders used globally. Access folder by Env_Config.data_path instead of ".../.../Data"
#
###############################################################################################################

from Load_Package import *

class Env_Config:
    """Class to configure experimental environment:
       Use: call Env_Config.fun_set_cwd(cwd = "parent folder of SourceCode") to set current working directory, 
            then access static variables inside, e.g., Env_Config.data_path
       1) static variables to represent different folders
       2) fun_set_cwd: set working directory and initialize paths of all the folders
       3) __fun_init_paths: private function to initialize paths of all folders
    """
    cwd = "." #Current work directory
    
    #Sourcecode folder and its sub-folders
    #Build project folder structure
    source_code_path = None
    src_data_cleaning_path = None
    src_debugging_path = None
    src_EDA_path = None
    src_FE_path = None
    src_model_path = None
    src_eval_path = None
    src_notebook_path = None

    data_path = None    
    document_path = None
    archive_path = None
    
    #Output folder and its sub-folders
    output_path = None
    output_data_cleaning = None
    output_EDA = None
    output_FE = None  
    output_FE_train = None
    output_FE_test = None      
    output_model = None
    output_eval = None   

    ##########################################################################################
    # File path for each feature. One for history feature and the other for event feature
    ##########################################################################################
    #last layer of folders to store features
    hist_fea_folder = None
    event_fea_folder = None

    ### 1) Hist feature
    hist_fea_IPE_demo = None
    hist_fea_IPE_tran = None
    hist_fea_IDEAL_cash = None
    hist_fea_IPE_msg = None
    hist_fea_Giro=None
    hist_fea_Chq_In=None
    hist_fea_Chq_Out=None
    ### 2) Event feature
    event_fea_QMS = None
    event_fea_IPE_tran = None
    event_fea_trade_tran = None
    event_fea_IDEAL_cash = None
    event_fea_IPE_msg = None
    event_fea_Giro=None
    event_fea_Chq_In=None
    event_fea_Chq_Out=None

    ### 3) the big 5
    fea_hist_all = None
    QMS_hist_all = None
    fea_event_all = None
    Y_all = None
    fea_all = None

    
    ##########################################################################################
    # hist/event/need win
    ##########################################################################################
    hist_win_size = 6 #6 month
    event_win_size = 7 #7 day
    need_win_size = 7 #7 day

    #history window
    hist_win_start = None
    hist_win_end = None

    #event window
    event_win_start = None
    event_win_end = None

    #needs window
    need_win_start = None
    need_win_end = None

    ##########################################################################################
    # model training or testing
    ##########################################################################################
    is_FE_on_training_data = None # True 


    @classmethod
    def fun_set_cwd(cls, cwd):
        """Function: set working directory and initialize paths of all the folders
           Input:    1) cwd: current working directory. Set it to parent folder of SourceCode                     
           Output:   
        """
        cls.cwd = cwd
        cls.__fun_init_folder_paths() #Call private function to initialize paths of all folders

        cls.__fun_set_win_trainORtest() #set hist/event/need windows and whether it is FE on training or testing data

        cls.__fun_set_FE_output_path() #set feature output files
    #End of function fun_set_cwd

    @classmethod
    def __fun_init_folder_paths(cls):
        """Function: private function to initialize paths of all the folders
           Input:    
           Output:   exist if error happens
        """
        #Source code and its subfolders
        cls.source_code_path = fun_path_join(cls.cwd, "SourceCode")        
        cls.src_data_cleaning_path = fun_path_join(cls.source_code_path, "Data_cleaning")        
        cls.src_debugging_path = fun_path_join(cls.source_code_path, "Debugging")        
        cls.src_EDA_path = fun_path_join(cls.source_code_path, "EDA")
        cls.src_FE_path = fun_path_join(cls.source_code_path, "FE")
        cls.src_model_path = fun_path_join(cls.source_code_path, "Model")
        cls.src_eval_path = fun_path_join(cls.source_code_path, "Eval")
        cls.src_notebook_path = fun_path_join(cls.source_code_path, "Notebook")                

        #data, document, and output
        # cls.data_path = fun_path_join(cls.cwd, "Data")
        cls.data_path = "C:/Users/yuyuxiong/OneDrive - DBS Bank Ltd/Documents/Customer_Science/Data"
        cls.document_path = fun_path_join(cls.cwd, "Document")
        cls.output_path = fun_path_join(cls.cwd, "Output")    
        cls.archive_path =  fun_path_join(cls.cwd, "Archive")       

        #All the above folders should be created by user before running the program
        folder_set_exist = list({cls.source_code_path, cls.src_data_cleaning_path, cls.src_debugging_path, 
            cls.src_EDA_path, cls.src_FE_path, cls.src_model_path, cls.src_eval_path, cls.src_notebook_path, 
            cls.data_path, cls.document_path, cls.output_path, cls.archive_path})
        
        rslt_exist_check = [x for x in folder_set_exist if not fun_folder_existence_check(x)]
        
        #Print out all folders that should exist but not        
        if len(rslt_exist_check) > 0:            
            print("+++ Following folders NOT exist ---")
            for _, v in enumerate(rslt_exist_check):
                print(v)
            sys.exit("Solution -- Create folders and try again")

        #Create sub-folders in output folder, if any not exist        
        cls.output_data_cleaning = fun_path_join(cls.output_path, "Data_cleaning")
        cls.output_EDA = fun_path_join(cls.output_path, "EDA")
        cls.output_FE = fun_path_join(cls.output_path, "FE")
        cls.output_FE_train = fun_path_join(cls.output_FE, "Train")
        cls.output_FE_test = fun_path_join(cls.output_FE, "Test")
        cls.output_model = fun_path_join(cls.output_path, "Model")
        cls.output_eval = fun_path_join(cls.output_path, "Eval")

        folder_set_output = list({cls.output_data_cleaning, cls.output_EDA, cls.output_FE, cls.output_FE_train, 
            cls.output_FE_test, cls.output_model, cls.output_eval})  
        
        rslt_exist_check = [x for x in folder_set_output if not fun_folder_existence_check(x, create_if_not_exist= True)]
        
        #Print out folder that not exist and cannot be created
        if len(rslt_exist_check) > 0:
            print("+++ Following output folders NOT exist ---")
            for _, v in enumerate(rslt_exist_check):
                print(v)
            sys.exit("Solution -- Check why folders cannot be created and then try again")
    #End of Function '__fun_init_folder_paths'

    @classmethod
    def fun_get_Y_start_time(cls, file_name, update_file = False):
        """Function: Retrieve a date from file_name, and update that the date is handled if update_file == True
           Input:    1. file_name. 
                     2. update_file. 
           Output:   exist if error happens
        """        
        date_handled = pd.read_csv(filepath_or_buffer = file_name,
                                dtype = dict.fromkeys(range(2), str))
        
        idx = [x for x in list(range(date_handled.shape[0])) if date_handled.loc[x, "handled"] == "0"]
                        
        if len(idx) > 0:
            idx_unhandled = idx[0]
            
            #update the list of available dates
            if update_file is True:
                date_handled.loc[idx_unhandled, "handled"] = "1"      
                #update file
                date_handled.to_csv(path_or_buf = file_name, index=False)
            
            return date_handled.loc[idx_unhandled, "Y_start_time"] #return the 1st available date
        else:
            return None
    #End of Function 'fun_get_Y_start_time'

    @classmethod
    def __fun_set_win_trainORtest(cls):
        """Function: Read in the needs window start time from file, and generate hist/event/needs window 
           Input:    1. update_file. Update file_name on handed date if True
           Output:   exist if error happens
        """
        ava_date = cls.fun_get_Y_start_time(file_name = fun_path_join(Env_Config.source_code_path, "Y_start_time_train.csv"))

        #ava_date is for training
        if ava_date != None:
            cls.is_FE_on_training_data = True
        else:
            #ava_date is for testing
            ava_date = cls.fun_get_Y_start_time(file_name = fun_path_join(Env_Config.source_code_path, "Y_start_time_test.csv"))            
            if ava_date is None:
                sys.exit("Error: Config.__fun_set_win_trainORtest: no available date")
            else:
                cls.is_FE_on_training_data = False

        #Use ava_date to generate window
        
        #needs window
        cls.need_win_start = pd.to_datetime(ava_date)
        cls.need_win_end = cls.need_win_start + pd.DateOffset(days = cls.need_win_size - 1)

        #event window
        cls.event_win_end = cls.need_win_start - pd.DateOffset(days = 1)
        cls.event_win_start = cls.event_win_end - pd.DateOffset(days = 6)        

        #history window    
        need_month_1st_day = pd.to_datetime(cls.need_win_start.strftime('%Y-%m') + "-01")    
        cls.hist_win_end = need_month_1st_day - pd.DateOffset(days = 1)
        cls.hist_win_start = need_month_1st_day - pd.DateOffset(months = 6)        
    #End of Function '__fun_set_win_trainORtest'

    @classmethod
    def __fun_set_FE_output_path(cls):
        """Function: Create folders for FE on either training or testing. Set file names for output features
           Input:    
           Output:   exist if error happens
        """
        #Str format for the window boundaries
        str_hist_win_end = cls.hist_win_end.strftime('%Y-%m-%d')
        str_event_win_end = cls.event_win_end.strftime('%Y-%m-%d')
        str_need_win_end = cls.need_win_end.strftime('%Y-%m-%d')
        str_need_win_start = cls.need_win_start.strftime('%Y-%m-%d')
        
        #FE is on training or testing
        if cls.is_FE_on_training_data:
            folder_path = cls.output_FE_train   
        else:
            folder_path = cls.output_FE_test
        
        #create his/event folders if they not exist        
        cls.hist_fea_folder = fun_path_join(folder_path, "Hist_" + str_hist_win_end)                     
        cls.event_fea_folder = fun_path_join(folder_path, "Event_" + str_event_win_end)            

        folder_set_output = list({cls.hist_fea_folder, cls.event_fea_folder})  
        
        rslt_exist_check = [x for x in folder_set_output if not fun_folder_existence_check(x, create_if_not_exist= True)]
        
        #Print out folder that not exist and cannot be created
        if len(rslt_exist_check) > 0:
            print("+++ Following output folders NOT exist ---")
            for _, v in enumerate(rslt_exist_check):
                print(v)
            sys.exit("Solution -- Check why folders cannot be created and then try again")

        ### 1) Hist feature output files
        cls.hist_fea_IPE_demo = fun_path_join(cls.hist_fea_folder, "hist_fea_IPE_demo_" + str_hist_win_end + ".feather")
        cls.hist_fea_IPE_tran = fun_path_join(cls.hist_fea_folder, "hist_fea_IPE_tran_" + str_hist_win_end + ".feather")
        cls.hist_fea_IDEAL_cash = fun_path_join(cls.hist_fea_folder, "hist_fea_IDEAL_cash_" + str_hist_win_end + ".feather")
        cls.hist_fea_IPE_msg = fun_path_join(cls.hist_fea_folder, "hist_fea_IPE_msg_" + str_hist_win_end + ".feather")
        cls.hist_fea_Giro = fun_path_join(cls.hist_fea_folder, "hist_fea_Giro_" + str_hist_win_end + ".feather")
        cls.hist_fea_Chq_In = fun_path_join(cls.hist_fea_folder, "hist_fea_Chq_In_" + str_hist_win_end + ".feather")
        cls.hist_fea_Chq_Out = fun_path_join(cls.hist_fea_folder, "hist_fea_Chq_Out_" + str_hist_win_end + ".feather")

        ### 2) Event feature output files
        cls.event_fea_QMS = fun_path_join(cls.event_fea_folder, "event_fea_QMS_" + str_event_win_end + ".feather")
        cls.event_fea_IPE_tran = fun_path_join(cls.event_fea_folder, "event_fea_IPE_tran_" + str_event_win_end + ".feather")
        cls.event_fea_trade_tran = fun_path_join(cls.event_fea_folder, "event_fea_trade_tran_" + str_event_win_end + ".feather")
        cls.event_fea_IDEAL_cash = fun_path_join(cls.event_fea_folder, "event_fea_IDEAL_cash_" + str_event_win_end + ".feather")
        cls.event_fea_IPE_msg = fun_path_join(cls.event_fea_folder, "event_fea_IPE_msg_" + str_event_win_end + ".feather")
        cls.event_fea_Giro = fun_path_join(cls.event_fea_folder, "event_fea_Giro_" + str_event_win_end + ".feather")
        cls.event_fea_Chq_In = fun_path_join(cls.event_fea_folder, "event_fea_Chq_In_" + str_event_win_end + ".feather")
        cls.event_fea_Chq_Out = fun_path_join(cls.event_fea_folder, "event_fea_Chq_Out_" + str_event_win_end + ".feather")

        ### 3) the big 5 output files
        cls.fea_hist_all = fun_path_join(folder_path, "FEA_Hist_" + str_hist_win_end + ".feather")
        cls.QMS_hist_all = fun_path_join(folder_path, "QMS_Hist_" + str_hist_win_end + ".feather")
        cls.fea_event_all = fun_path_join(folder_path, "FEA_event_" + str_event_win_end + ".feather")
        cls.Y_all = fun_path_join(folder_path, "Y_" + str_need_win_end + ".feather")        
        cls.fea_all = fun_path_join(folder_path, "FEA_all_" + str_need_win_start + ".feather")
    #End of Function '__fun_set_FE_output_path'


    @classmethod
    def print_variable_debugging_only(cls):
        """Function: print variables to see if they are set correctly
           Input:    
           Output:   
        """
        #history window
        print("hist_win_start: ", cls.hist_win_start)
        print("hist_win_end: ", cls.hist_win_end)        

        #event window
        print("event_win_start: ", cls.event_win_start)
        print("event_win_end: ", cls.event_win_end)      

        #needs window
        print("need_win_start: ", cls.need_win_start)
        print("need_win_end: ", cls.need_win_end)      
        
        print("is_FE_on_training_data: ", cls.is_FE_on_training_data)

        #print feature folder
        print("hist_fea_folder: ", cls.hist_folder_path)
        print("event_fea_folder: ", cls.event_fea_folder)

        ### 1) hist feature output paths
        print("hist_fea_IPE_demo: ", cls.hist_fea_IPE_demo)
        print("hist_fea_IPE_tran: ", cls.hist_fea_IPE_tran)
        print("hist_fea_IDEAL_cash: ", cls.hist_fea_IDEAL_cash)
        print("hist_fea_IPE_msg: ", cls.hist_fea_IPE_msg)
        print("hist_fea_Giro: ", cls.hist_fea_Giro)
        print("hist_fea_Chq_In: ", cls.hist_fea_Chq_In)
        print("hist_fea_Chq_Out: ", cls.hist_fea_Chq_Out)

    
        ### 2) Event feature
        print("event_fea_QMS: ", cls.event_fea_QMS)
        print("event_fea_IPE_tran: ", cls.event_fea_IPE_tran)
        print("event_fea_trade_tran: ", cls.event_fea_trade_tran)
        print("event_fea_IDEAL_cash: ", cls.event_fea_IDEAL_cash)
        print("event_fea_IPE_msg: ", cls.event_fea_IPE_msg)
        print("event_fea_Giro: ", cls.event_fea_Giro)
        print("event_fea_Chq_In: ", cls.event_fea_Chq_In)
        print("event_fea_Chq_Out: ", cls.event_fea_Chq_Out)

        ### 3) the big 4
        print("fea_hist_all: ", cls.fea_hist_all)
        print("QMS_hist_all: ", cls.QMS_hist_all)
        print("fea_event_all: ", cls.fea_event_all)
        print("Y_all: ", cls.Y_all)
        print("fea_all: ", cls.fea_all)
    #End of Function 'print_variable_debugging_only'

    #End of class Env_Config
        



