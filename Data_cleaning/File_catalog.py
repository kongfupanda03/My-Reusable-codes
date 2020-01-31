###############################################################################################################
# Project     : Customer Science for bizCare Call Reduction
#
# Coding      : CAO Jianneng
#
# Date        : Since 2019-11-15
#
# Description : file catalog for csv and feather
#               Purpose. access file by variable, not by full path, which changes with data update.
###############################################################################################################

import sys
sys.path.append('./SourceCode')

from Config import Env_Config
Env_Config.fun_set_cwd(".")

from Load_Package import *
from Config import *
from Data_cleaning.data_clean import DataClean

class File_Catalog(object):
    """File catalog for csv and feather. Access file by variable.
       1) 
       2) 
       3)        
    """
    # glob.glob(Env_Config.data_path + "**/*.csv")
    # os.listdir(Env_Config.data_path)

    DR_table_mapping = None
    xlsx_file_mapping = None

    @classmethod
    def fun_set_file_catalog_b4_FE(cls): 
        """Function: map xlsx file name to its path
           Input:    
           Output:   
        """       
        cls.__fun_xlsx_mapping()
        cls.__fun_DRfile_map2_csv_feather()
    #End of Function 'fun_set_file_catalog_b4_FE'


    @classmethod
    def __fun_xlsx_mapping(cls): 
        """Function: map xlsx file name to its path
           Input:    
           Output:  A mapping (pandas dataframe) between xlsx file name and file path
        """       
        xlsx_list = ["datatype_map", "LOV_Freq_above500_2019", "top_10_LOV"]     

        file_path = ['datatype_map.xlsx', 'prod_subarea_Freq_above500_2019.xlsx', 'top_10_lov.xlsx']
        file_path = [fun_path_join(Env_Config.data_path, x) for x in file_path]

        cls.xlsx_file_mapping = pd.DataFrame({"xlsx_file": xlsx_list, "xlsx": file_path})
        
        cls.xlsx_file_mapping = cls.xlsx_file_mapping.set_index('xlsx_file')
    #End of Function 'fun_xlsx_mapping'
    
    @classmethod
    def __fun_DRfile_map2_csv_feather(cls):
        """Function: Map each DR table to its extracted csv and feather output
           Input:    
           Output:   
        """
        #File names
        tb_name = ['table_names']
               

        np_array = np.empty((len(tb_name), 1,))
        np_array[:] = np.nan

        cls.DR_table_mapping = pd.DataFrame(data = np_array,
                                # index = tb_name,    
                                columns = ['feather'])  
        cls.DR_table_mapping['tb_name']  = tb_name


        #Add in csv file path
        cls.DR_table_mapping = cls.__fun_map_DRtable_to_csv(mapping_list = cls.DR_table_mapping)  
        if cls.DR_table_mapping is None:
            sys.exit("Add in required csv files and re-run program")

        #Add feather path
        cls.DR_table_mapping.feather = cls.DR_table_mapping.tb_name + "_" + cls.DR_table_mapping.date + "_layer2.feather"

        # #For debugging
        # print(cls.DR_table_mapping.tb_name)
        # print(cls.DR_table_mapping.date)
        # print(cls.DR_table_mapping) 

        cls.DR_table_mapping.feather = [fun_path_join(Env_Config.output_data_cleaning, x) for x in cls.DR_table_mapping.feather]


        #Check if date is ok. Use try... catch        
        cls.DR_table_mapping['date'] = pd.to_datetime(arg = cls.DR_table_mapping['date'], 
                                                    errors = 'coerce')
                                                   
                     
        invalid_tb_path = [cls.DR_table_mapping.loc[idx, "tb_name"] for idx in list(range(cls.DR_table_mapping.shape[0])) \
                        if pd.isnull(cls.DR_table_mapping.date[idx])]

        if len(invalid_tb_path) > 0:
            print("+++ Table name with invalid date ---")
            print(invalid_tb_path)
            sys.exit("Error: Fix the naming issue and re-run the program")      

        cls.DR_table_mapping = cls.DR_table_mapping.set_index('tb_name')  
    #End of Function 'fun_DRfile_map2_csv_feather'
    

    @classmethod
    def __fun_map_DRtable_to_csv(cls, mapping_list):
        """Function: map DR table to its csv file path
           Input:    1) mapping_list.                      
           Output:   a list of csv file path
        """
        #Find all csv files
        csv_file_list = os.listdir(Env_Config.data_path)
        csv_file_list = [x for x in csv_file_list if x.endswith(".csv")]

        DC = DataClean()
        
        #split csv file path into tb_name and date
        tb_name_date = [DC.extract_table_date(x) for x in csv_file_list]
        tb_file_info = pd.DataFrame(data = tb_name_date, columns = ['tb_name', 'date'])
        tb_file_info['csv'] = [fun_path_join(Env_Config.data_path, x) for x in csv_file_list]
        tb_file_info['csv_name']=csv_file_list# added!
        
        #NOTE. Every table name should have a csv file ready in Data folder
        tb_wo_csv_file = list(set(cls.DR_table_mapping['tb_name']) - set(tb_file_info['tb_name']))
        if len(tb_wo_csv_file) > 0:
            print("\nError: csv file incomplete. The following tables without csv files")
            print(tb_wo_csv_file)
            return None

        mapping_list = pd.merge(left = mapping_list, right = tb_file_info, how = "left", on = "tb_name")                

        return mapping_list               
    #End of Function '__fun_map_DRtable_to_csv'

    
    @classmethod
    def __fun_map_file_2_path(cls, file_name, file_type):
        """Function: 
           Input:    
           Output:   
        """
        if file_type == 'xlsx':
            DF = File_Catalog.xlsx_file_mapping
        else:
            DF = File_Catalog.DR_table_mapping

        if file_name in DF.index:
            return DF.loc[file_name, file_type]
        else:
            print("Error: input file_name NOT in list below")
            print(DF.index)
            return None        
    #Enf of Function '__fun_map_file_2_path'
    
    @classmethod
    def fun_get_xlsx_mapping_filepath(cls, file_name):
        """Function: 
           Input:    
           Output:   
        """
        return cls.__fun_map_file_2_path(file_name = file_name, file_type = 'xlsx')
    #End of Function 'fun_get_xlsx_mapping_filepath'

    @classmethod
    def fun_get_raw_csv_filepath(cls, file_name):
        """Function: 
           Input:    
           Output:   
        """
        return cls.__fun_map_file_2_path(file_name = file_name, file_type = 'csv')
    #End of Function 'fun_get_raw_csv_filepath'
    
    @classmethod
    def fun_get_feather_filepath(cls, file_name):
        """Function: 
           Input:    
           Output:   
        """
        return cls.__fun_map_file_2_path(file_name = file_name, file_type = 'feather')
    #End of Function 'fun_get_raw_csv_filepath'

    @classmethod
    def fun_get_raw_csv_name(cls, file_name):
        """Function: 
           Input:    
           Output:   
        """
        return cls.__fun_map_file_2_path(file_name=file_name,file_type= 'csv_name')

    @classmethod
    def fun_print_file_name(cls):
        """Function: 
           Input:    
           Output:   
        """
        print("\n+++ List of xlsx mapping files ---")
        print(cls.xlsx_file_mapping)

        print("\n+++ List of DR tables ---")
        print(cls.DR_table_mapping)
#End of class 'File_Catalog'

