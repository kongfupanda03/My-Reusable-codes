###############################################################################################################
# Project     : Feature Engineering
#
# Coding      : CAO Jianneng
#
# Date        : Since 2019-07-15
#
# Description : General functions for feature engineering
#               1) statistics (min, max, sd...) on numerical variables
#               2) Nominal attributes to dummary variables (top-k, or based on output template)
#               3) 1-hot
# Change Control: 1) 12-24 fun_norminal_2_dummy_variables_groupBy changed naming by adding agg_col as prefix
###############################################################################################################

from Load_Package import *

class FE_Groupby(object):
    """Class for feature engineering based on groupby. It includes functions:
       1) fun_compute_stat: Compute (count, min, max, mean, median, sd)
       2) fun_compute_DF_stat: groupby and then compute the statistics in each group
       3) fun_norminal_2_dummy_variables_groupBy: Transform nominal variables to dummy variables
                                                  Output only top-k frequent dummy variables
       4) fun_norminal_2_dummy_template_groupBy: Transform nominal variables to dummy variables 
                                                 based on an output template
       5) fun_one_hot: Transform nominal variables to one hot based on a template
    """       

    @classmethod    
    def fun_compute_stat(cls, x, x_len = None):
        """Function: compute(count, min, max, mean, median, sd)
           Input:    1) x. a list of numeric values
                     2) x_len. a value shows the expected length of x. 
                     If x is smaller, padded with 0
           Output:   list of statistics
        """
        rslt = [0]*6 #To store result
        
        V = [v for v in x if ~np.isnan(v)] #remove NA
        
        len_V = len(V)

        if len_V > 0:
            rslt[0] = len_V
          
            if(x_len != None):
                diff_len = x_len - len_V
                assert diff_len >= 0

                if diff_len > 0: #when there is fixed length requirement, padding by 0's
                    V.extend([0]*diff_len)
            
            #the statistics
            rslt[1] = np.min(V)
            rslt[2] = np.max(V)
            rslt[3] = np.mean(V)
            rslt[4] = np.median(V)
            
            if len_V > 1:
                rslt[5] = np.std(V)

        return rslt
    #end of Function 'fun_compute_stat'           
    
    @classmethod
    def fun_compute_DF_stat(cls, DF,  group_by_attrs, numerical_attrs, x_len = None):
        """Function: groupby and then compute the statistics in each group
           Input:    1) DF. a data.frame
                     2) group_by_attrs. a list
                     3) numerical_attrs. a list, statistics is apply on each of its element in each group
                     4) x_len. See Function 'fun_compute_stat'
           Output: data.frame
        """

        rslt = None
        #compute statistics on each numerical attr
        for num_attr in numerical_attrs:            
            gb = DF.groupby(group_by_attrs)[num_attr].agg(cls.fun_compute_stat, x_len=x_len).reset_index()
            
            #transform list into data.frame
            # stat = pd.DataFrame([v for v in gb[num_attr]]) #simpler code & possibly better efficiency
            stat = pd.DataFrame(gb[num_attr].values.tolist())
            stat_DF = pd.concat([gb[group_by_attrs], stat], axis = 1)
            
            #rename the columns
            stat_DF.columns = group_by_attrs + [num_attr + "__" + x for x in ["count", "min", "max", "mean", "median", "sd"]]
            
            #combine the results of multiple groupby
            if rslt is None:
                rslt = stat_DF
            else:
                rslt = rslt.merge(right = stat_DF, 
                                  how = "outer", 
                                  on = group_by_attrs)

        return rslt
    #End of Function 'fun_compute_DF_stat'

    @classmethod
    def fun_norminal_2_dummy_variables_groupBy(cls, DF, group_by_attrs, nominal_attrs, top_k = 100, \
            agg_col = None, agg_func = None, process_NA = True, OTH_value = "OTH"):
        """Description: transform nominal variables to dummy variables. It can also work as one hot.
           Input:       1. DF: data frame
                        2. group_by_attrs: its values act as rows in the output
                        # group-by attributes used to partition data frame into groups
                        3. nominal_attrs: its values act as columns in the output
                        # nomimal attributes, which are transformed to dummy variables after group-by
                        # the dummy variable values are the count
                        4. top-k: for each nominal attribute, keep its top-k frequent values                         
                        5. agg_col:
                        6. agg_func: for one hot, assign "lambda x: 1" to agg_func
                        7. process_NA: whether NA values are set to Not_available    
                        8. OTH_value: 
           output:     (group_by_attrs, dummy_vars_for_nominal_attr_1, dummy_vars_for_nominal_attr_2, ...)
        """
        if agg_col is None:
            loc_DF = DF.loc[:, group_by_attrs + nominal_attrs]
        else:
            loc_DF = DF.loc[:, group_by_attrs + nominal_attrs + [agg_col]]

        if agg_col is None:
            loc_DF['count_col'] = 1 #Additional column for counting purpose in pivot table only
            agg_col = 'count_col'        
        
        if agg_func is None:
            agg_func = len                

        rslt = None #To store result

        #Compute (groupby and then dummy) for nominal attributes one by one
        for attr in nominal_attrs:
            #Frequency of attr values           
            freq = loc_DF[attr].value_counts()
            freq = pd.DataFrame(freq)
            freq.columns = ["Count"]
            freq['Var'] = freq.index
            freq = freq.sort_values(by = ['Count'], ascending = False)
      
            #Find the top_k attr values
            if freq.shape[0] > top_k :
                top_k_values = freq.Var[:top_k]
            else:
                top_k_values = freq.Var
            
            #Set non-frequent values to OTH
            loc_DF.loc[~loc_DF[attr].isin(top_k_values), attr] = OTH_value

            if(process_NA == True):
                loc_DF.loc[loc_DF[attr].isna(), attr] = OTH_value  
                   

            #Key operation: groupby, put attr values into dummy, and count frequency of each dummy variable
            ptb = pd.pivot_table(loc_DF, 
                           values = agg_col, 
                           index = group_by_attrs, 
                           columns = [attr], 
                           aggfunc = agg_func, 
                           fill_value=0)
        
            ptb = ptb.reset_index()
            
            ###Rename columns
            fea_name = ptb.columns[len(group_by_attrs):] 
            #If column names have multiple levels, merge them in reverse order
            if fea_name.nlevels > 1:
                fea_name = [x[::-1] for x in fea_name]
                fea_name = list(map("_".join, fea_name))            
            
            #Add column name as the prefix
            ptb.columns = group_by_attrs + [agg_col+ "__" +attr + "__" + x for x in fea_name]
            
            #combine the results of multiple groupby
            if rslt is None:
                rslt = ptb
            else:
                rslt = rslt.merge(right = ptb, 
                                    how = "outer", 
                                    on = group_by_attrs)        
        
        return rslt
    #End of Function 'fun_norminal_2_dummy_variables_groupBy'
    
    @classmethod
    def fun_norminal_2_dummy_template_groupBy(cls, DF, group_by_attrs, nominal_attr, template, percentage_needed = False):                
        """Description: transform nominal variables to dummy variables based on a template
           Input:       1. DF: data frame
                        2. group_by_attrs: its values act as rows in the output
                        # group-by attributes used to partition data frame into groups
                        3. nominal_attr: its values (in the template) act as columns in the output
                        # nomimal attr, which is transformed to dummy variables after group-by
                        # the dummy variable values are the count
                        4. template: a subset of nominal_attr values  
                        5. percentage_needed: whether compute the percentage of frequency
           output:     (group_by_attrs, dummy_vars in template)
                       # Output contains every value in template, missing ones are padded by 0.
        """

        loc_DF = DF.loc[:, group_by_attrs + [nominal_attr]]        
        loc_DF['count_col'] = 1 #Additional column for counting purpose in pivot table only
        
        rslt = None #To store result
                            
        #Keep items in template only
        loc_DF = loc_DF.loc[loc_DF[nominal_attr].isin(template), ]        

        #Key operation: groupby, put attr values into dummy, and count frequency of each dummy variable
        ptb = pd.pivot_table(loc_DF, 
                        values = 'count_col', 
                        index = group_by_attrs, 
                        columns = [nominal_attr], 
                        aggfunc = len,
                        fill_value=0)
        
        ptb = ptb.reset_index()
                           
        #rename columns
        ptb.columns = group_by_attrs + [nominal_attr + "__" + x for x in ptb.columns[len(group_by_attrs):]]
                
        #Find the column(s) of ptb that are not in template
        required_col = [nominal_attr + "__" + x for x in template]
        set_diff = set(required_col).difference(ptb.columns)
        set_diff = list(set_diff)

        #pad ptb for the missing columns by 0
        if len(set_diff) > 0 :
            ptb[set_diff] = pd.DataFrame(data = 0, 
                                         index = ptb.index, 
                                         columns = np.arange(len(set_diff)))

        
        
        if percentage_needed == True:
            ptb_percentage = ptb.loc[:, required_col]
            row_sums = ptb_percentage.sum(axis=1)
            ptb_percentage = ptb_percentage / row_sums[:, np.newaxis]
            ptb = pd.concat([ptb, ptb_percentage], axis = 1)

        return ptb
    #End of Function 'fun_norminal_2_dummy_template_groupBy'

    @classmethod
    def fun_one_hot(cls, DF, group_by_attrs, nominal_attr, template):                
        """Description: transform nominal variables to one hot based on a template
           Input:       1. DF: data frame
                        2. group_by_attrs: its values act as rows in the output
                        # group-by attributes used to partition data frame into groups
                        3. nominal_attr: its values (in the template) act as columns in the output
                        # nomimal attr, which is transformed to dummy variables after group-by
                        # the dummy variable values are the count
                        4. template: a subset of nominal_attr values                         
           output:     (group_by_attrs, dummy_vars in template)
                       # Output contains every value in template, missing ones are padded by 0.
        """

        loc_DF = DF.loc[:, group_by_attrs + [nominal_attr]]        
        loc_DF['count_col'] = 1 #Additional column for counting purpose in pivot table only
        
        rslt = None #To store result
                            
        #Keep items in template only
        loc_DF = loc_DF.loc[loc_DF[nominal_attr].isin(template), ]        

        #Key operation: groupby, put attr values into dummy, and count frequency of each dummy variable
        ptb = pd.pivot_table(loc_DF, 
                        values = 'count_col', 
                        index = group_by_attrs, 
                        columns = [nominal_attr], 
                        aggfunc = lambda x: 1, 
                        fill_value=0)
        
        ptb = ptb.reset_index()
                           
        #rename columns
        ptb.columns = group_by_attrs + [nominal_attr + "__" + x for x in ptb.columns[len(group_by_attrs):]]
                
        #Find the column(s) of ptb that are not in template
        required_col = [nominal_attr + "__" + x for x in template]
        set_diff = set(required_col).difference(ptb.columns)
        set_diff = list(set_diff)

        #pad ptb for the missing columns by 0
        if len(set_diff) > 0 :
            ptb[set_diff] = pd.DataFrame(data = 0, 
                                         index = ptb.index, 
                                         columns = np.arange(len(set_diff)))
       

        return ptb
    #End of Function 'fun_one_hot'

    @classmethod
    def fun_norminal_2_dummy_variables_groupBy_OBS(cls, DF, group_by_attrs, nominal_attrs, top_k, process_NA = True):
        """Description: transform nominal variables to dummy variables
           Input:       1. DF: data frame
                        2. group_by_attrs: its values act as rows in the output
                        # group-by attributes used to partition data frame into groups
                        3. nominal_attrs: its values act as columns in the output
                        # nomimal attributes, which are transformed to dummy variables after group-by
                        # the dummy variable values are the count
                        4. top-k: for each nominal attribute, keep its top-k frequent values 
                        5. process_NA: whether NA values are set to Not_available                            
           output:     (group_by_attrs, dummy_vars_for_nominal_attr_1, dummy_vars_for_nominal_attr_2, ...)
        """
        loc_DF = DF.loc[:, group_by_attrs + nominal_attrs]
                
        loc_DF['count_col'] = 1 #Additional column for counting purpose in pivot table only
        
        rslt = None #To store result

        #Compute (groupby and then dummy) for nominal attributes one by one
        for attr in nominal_attrs:
            #Frequency of attr values           
            freq = loc_DF[attr].value_counts()
            freq = pd.DataFrame(freq)
            freq.columns = ["Count"]
            freq['Var'] = freq.index
            freq = freq.sort_values(by = ['Count'], ascending = False)
      
            #Find the top_k attr values
            if freq.shape[0] > top_k :
                top_k_values = freq.Var[:top_k]
            else:
                top_k_values = freq.Var
            
            #Set non-frequent values to OTH
            loc_DF.loc[~loc_DF[attr].isin(top_k_values), attr] = "OTH"        

            if(process_NA == True):
                loc_DF.loc[loc_DF[attr].isna(), attr] = "OTH"
            
            #Key operation: groupby, put attr values into dummy, and count frequency of each dummy variable
            ptb = pd.pivot_table(loc_DF, 
                           values = 'count_col', 
                           index = group_by_attrs, 
                           columns = [attr], 
                           aggfunc = len, 
                           fill_value=0)
        
            ptb = ptb.reset_index()
            
            #rename columns
            ptb.columns = group_by_attrs + [attr + "__" + x for x in ptb.columns[len(group_by_attrs):]]
            
            #combine the results of multiple groupby
            if rslt is None:
                rslt = ptb
            else:
                rslt = rslt.merge(right = ptb, 
                                    how = "outer", 
                                    on = group_by_attrs)        
        
        return rslt
    #End of Function 'fun_norminal_2_dummy_variables_groupBy_OBS'

#End of class 'FE_Groupby'
