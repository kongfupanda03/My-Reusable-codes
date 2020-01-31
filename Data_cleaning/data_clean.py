# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:21:11 2019

@author: yuyuxiong
"""
from Load_Package import *
from Data_cleaning.data_read import DataRead
from Config import Env_Config
from Data_cleaning.File_catalog import File_Catalog

class DataClean:
    """
    Class for data cleaning
    #COMMENT. List the important functions inside the class. 
    #         If some function is obsolete or not ready to be released, make it private by adding "__" at the beginning of function name.
    """

    def extract_table_date(self, filename):
        '''
        function: to extract table name and dates from file name
        input: file name
        output: table name and date
        '''
        file = filename
        file_trunc = file.split ('_')  # split file name
        date = file_trunc[-1].split ('.')[0]
        tablename = '_'.join (file_trunc[0:-1])
        return tablename, date

    def map_col_type(self, tbname, row_to_skip=0, feather=False, thresh=None,column_use=None):
        """
        Function:This function is to convert datatypes to desired datatypes and output to feather format.
        Input:   filename:filename(eg:'vw_dlrg_sg_ipe_m_rxn_txn_pr_full_2019-10-04.csv')
                 row_to_skip: number of rows to skip when reading in data, default 0
                 father: set whether output is feather formant or datafram; default False, True if output
                         as feather format.
                 Thresh: optional. User can define the threshold of missing values' proportion.thresh is none, dropna \
                         cols/rows with all NA values.
                 cols_to_use: only read in data from specified list of columns(columns in raw data file)
        Output: data.frame/feather file
        """
        # Extract table name and file name
        File_Catalog.fun_set_file_catalog_b4_FE()
        filename = File_Catalog.fun_get_raw_csv_name(file_name = tbname)
        print('+++Extract table name and file name---')
        tablename, date = self.extract_table_date (filename)

        print('+++ Importing datatype_map table ---')
        #map_table_path = fun_path_join (Env_Config.data_path, 'datatype_map.xlsx')
        map_table_path = File_Catalog.fun_get_xlsx_mapping_filepath(filename='datatype_map')
        sheet_name = tablename
        if len (sheet_name) >= 32:
            sheet_name = sheet_name[:31]
        map_table = pd.read_excel(map_table_path, sheet_name=sheet_name)  #  read in map table
        map_table = map_table.apply (
            lambda x: x.str.strip() if x.dtype == 'object' else x)  # remove all leading and tailing spaces

        filename = fun_path_join(Env_Config.data_path, filename)
        # Read in data
        print("Read in data: ", filename)
        read_object = DataRead(filename, row_to_skip=row_to_skip, num_column=300)
        if column_use!=None:
            df_in = read_object.fun_read_single(usecols=column_use)
        else:
            df_in = read_object.fun_read_single()
            
        df_in.drop(df_in.columns[df_in.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
        df_in = self.clean_missing(df_in, thresh=thresh)  # drop columns with na values
        print('+++NA columns dropped---')
        print(len(df_in.columns),' columns left.')
        df_in.columns = df_in.columns.str.strip().str.lower().str.split('.').str[1]  # clean up colunm names

        if df_in.columns.is_unique:
            print('No duplicated columns!')
        else:
            print('Remove duplicated column: ', df_in.columns[df_in.columns.duplicated (keep=False)])
            df_in = df_in.loc[:, ~df_in.columns.duplicated()]  # remove duplicated columns

        for col in df_in.columns:
            df_in[col] = df_in[col].str.strip().str.lower()# remove heading/tailing spaces and change all to lower case

        # Map data types
        print ("+++ Column type conversion ---")
        col_dt_map = list(
            map_table[map_table.Datatype == 'datetime64']['Source Field Name'])
        col_numeric_map = list(map_table[map_table.Datatype == 'float64']['Source Field Name'])
        col_numeric = list(set(df_in.columns).intersection(set(col_numeric_map)))  # columns to be converted to numeric
        col_dt = list(set(df_in.columns).intersection(set(col_dt_map)))  # find columns to be converted to datetime

        # Convert types
        if len(col_numeric) > 0:
            df_in[col_numeric] = df_in[col_numeric].apply(lambda x: pd.to_numeric (x, errors='coerce'))

        df_out = self.correct_datetime(df=df_in, cols=col_dt)
        print ('+++datatype conversion done!---')

        output = tablename + '_' + date + '.feather'
        output = fun_path_join (Env_Config.output_data_cleaning, output)

        print('+++Output cleaned data---')
        if feather:
            print ("Output cleaned data to: ", output)
            return df_out.to_feather(output)
        else:
            return df_out



    def correct_datetime(self, df, cols):
        """
        Function:This function is to convert columns to datetime datatype.
        Input:   df: dataframe
                 date: columns to be converted to datetime type
        Output: data.frame
        
        """
        for col in cols:
            print ('datetime conversion at column: ', col)
            if df[col].isnull ().all ():
                pass
            else:
                try:
                    df[col] = pd.to_datetime (df[col])
                except:
                    try:
                        # To deal with datatime out of bound issue
                        date_str = df[col].str[:10]  # Only consider date
                        # truncation on lower bound
                        idx_lower_bound = [True if str (x) < "1800-01-01" and not pd.isna (x) else False for x in
                                           date_str]
                        df.loc[idx_lower_bound, col] = "1800-01-01"
                        # truncation on upper bound
                        idx_upper_bound = [True if str (x) > "2200-01-01" and not pd.isna (x) else False for x in
                                           date_str]
                        df.loc[idx_upper_bound, col] = "2200-01-01"

                        df[col] = pd.to_datetime (df[col])
                    except:
                        # To extract dates only for datetime cannot be strange formats which unable to be parsed
                        df[col] = df[col].str[0:10]
                        df[col] = pd.to_datetime (df[col])
        return df




    def type_converter(self, df, to_object=None, to_category=None, to_dt=None, to_numeric=None,to_bool=None):
        """
        Function:This function converts user_defined columns to convert to desired data type
        Input: DF:pandas dataframe
               to_object: columns you want to get converted to object data type
               to_category:columns you want to get converted to category data type
               to_dt:columns you want to get converted to datetime data type
               to_numeric: columns you want to get converted to numeric data type
        Output: pandas dataframe
        """
        if to_object != None:
            df[to_object] = df[to_object].astype('object')

        if to_category != None:
            df[to_category] = df[to_category].astype('category')

        if to_dt != None:
            for col in to_dt:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        if to_numeric != None:
            for col in to_numeric:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        if to_bool !=None:
            df[to_bool]=df[to_bool].astype('bool')
        return df

    def unify_missing(self, df, col, missing):
        """
        Function: There are multiple forms of missing values in dataset, such as blanks,Null,na etc.
                  This function will translate all forms of user-identifed missing values to np.nan.
        Input:    df: data.frame
                  col: columns user want to unify missing values
                  missing: forms user consider as missing values
        Output:   data.frame with missing values aligned as np.nan
        """
        data = df.copy
        data[col] = data[col].replace (missing, np.nan, regex=True)

        return data

    def clean_missing(self, df, axis=1, cols=None, thresh=None):
        """
        Function: This function is to remove missing values.
        Input:    df: data.frame
                  axis: 0 or 1, default is 1;
                        0: Drop rows which contain missing values.
                        1: Drop columns which contain missing value.
                  cols: optional. User can define what cols the function apply on.
                        if None, the function will apply on all cols in df.
                  Thresh: optional. User can define the threshold of missing values' proportion. Rows or cols will  be removed
                          when the missing value proportion more than or equal to the thresh.If thresh is none, dropn \
                          cols/rows with all NA values.
                 
                          
        Output: data.frame
        """
        # To define whether to remove missing values across whole dataframe or just certain cols
        if cols is None:
            data = df
        else:
            data = df[cols]

        # To drop cols/rows based on thresh
        if thresh is None:
            return data.dropna (axis, how='all')
        else:
            if axis == 0:
                return df.drop (data.index[data.isnull ().mean (axis=1) > thresh], axis=0)
            else:
                return df.drop (data.columns[data.isnull ().mean () > thresh], axis=1)

    def impute_missing(self, df, cols, method='mean', fill_value=None):
        """
        function: this function is to impute missing values
        input: df: data.frame
               cols: columns to be imputed
               method: impute missing values with column {'mean','mode','median','constant'}; default
                       is 'mean'.
                fill_value: when method is 'constant', fill in fill_value.
        output: data.frame
        
        """
        if method == 'mean':
            df[cols] = df[cols].fillna (df[cols].mean (skipna=True))
        elif method == 'mode':
            df[cols] = df[cols].fillna (df[cols].mode (skipna=True).head (1))
        elif method == 'median':
            df[cols] = df[cols].fillna (df[cols].median (skipna=True))
        elif method == 'constant':
            df[cols] = df[cols].fillna (fill_value)
        else:
            print ('Error! Please select method from {mean,mode,median,constant}')

        return df
