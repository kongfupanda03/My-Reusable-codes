###############################################################################################################
# Project     : Customer Science for bizCare Call Reduction
#
# Coding      : Kelly Tay
#
# Date        : 11/02/2020
#
# Description : Main function to create/integrate all Training and Testing features across 4 weeks
#               
###############################################################################################################

import sys
from datetime import datetime as date

sys.path.append ('./SourceCode')

# Import Packages
from Config import Env_Config
from Global_fun import *

Env_Config.fun_set_cwd (".")

print('Model_Config')
from Model.model_config import *
print('auto_xgboost')
from Model.auto_Xgboost import *
print('model_preprocessing')#FE,label
from Model.model_preprocessing import *
import pickle
from FE.sample_integrate import cls_sample_integrate
#define if testing data has labels inside
label = True

print("+++ Extract common features ---\n")
# Read Training and Testing data
print('+++ Training Features ---\n')
# Read in Y_start_time_train.csv
y_start_test_dt = pd.read_csv(fun_path_join(Env_Config.source_code_path, "Y_multi_start_time_test.csv"))
fea_multi = 'FEA_multi_win_' + '2019-06-24' + '.feather'
training_all = pd.read_feather(fun_path_join(Env_Config.output_FE_train, fea_multi))

# COMMENT - read y_start_time_test use it to generate the full name of testing data and then read in
# UPDATE - used Y_start_time_test.csv to specify the testing file to use
print('+++ Testing Features ---\n')

# Get a list of test data directories
test_dirs = [x for x in os.listdir(Env_Config.output_FE_test) if 'partition' in x] #list of folder names
test_dirs.sort()# order folders
testing_all_path = min([fun_path_join(Env_Config.output_FE_test,x) for x in test_dirs])  
testing_all_dirs = [x for x in os.listdir(testing_all_path) if x.startswith('FEA')] #list of partition file names in each folder
testing_all = pd.read_feather(fun_path_join(testing_all_path, testing_all_dirs[0]))
print(fun_path_join(testing_all_path, testing_all_dirs[0]))   

cls_fea_integrate = cls_sample_integrate()
testing_all_bin = cls_fea_integrate.fun_model_binarisation(data=testing_all,hist_end_date = '2019-05-31',label=label)

print('training data shape: ' + str(training_all.shape))
print('testing data shape:' + str(testing_all_bin.shape))
#Common columns:
x_train, y_train, x_test, y_test = model_preprocessing.sliding_win_train_test_split(train_data = training_all, 
                                                                                    test_data = testing_all_bin)
x_train_columns = x_train.columns
x_test_columns = list(x_test.columns)
if set(x_train_columns)==set(x_test_columns):
    print('columns are aligned')
else:
    print('columns are not aligned')
      
print("+++ Load in Model ---\n")
model = joblib.load(fun_path_join(Env_Config.output_model, 'XGB_model_2020_03_12.pkl'))

#prediction and Evaluation

print("+++ Prediction across multiple windows ---\n")
start_time = time.monotonic()
for i in range(len(test_dirs)):
    folder_path = fun_path_join(Env_Config.output_FE_test,test_dirs[i])
    print(f"Prediction for window folder {folder_path}")
    partition_dirs = sorted([x for x in os.listdir(folder_path) if x.startswith('FEA')])
    pred_lst=[]
    y_start_date = y_start_test_dt.iloc[i,0]
    for i in range(len(partition_dirs)):
        test_data = pd.read_feather(fun_path_join(folder_path, partition_dirs[i]))
        print('Input partition is {}'.format(fun_path_join(folder_path, partition_dirs[i])))
        test_data_bin = cls_fea_integrate.fun_model_binarisation(data=test_data,hist_end_date = '2019-05-31',label=label)
        test_data_bin = test_data_bin.fillna(0)
        x_test_par = test_data_bin.loc[:,x_test_columns].values
        print(test_data.shape,test_data_bin.shape,x_test_par.shape)
        y_pred = cls_auto_Xgboost.fun_pred(model = model, x_test = x_test_par, pred_type = 'prob')
        
        y_pred_df = pd.DataFrame(y_pred[:,1], columns = ['pred_prob']) 
        y_pred_df.insert(loc=0, column='cin', value=test_data_bin['cin'])
        y_pred_df.insert(loc=1, column='cust_needs', value=test_data_bin['cust_needs'])
        y_pred_df['label_pred'] = 0
        temp =y_pred_df.groupby(['cin'])['pred_prob'].apply(lambda x : x.nlargest(3))
        temp.names=['cin','index']
        index = temp.index.get_level_values(1)
        y_pred_df.loc[index,'label_pred'] = 1
        if label:
            y_pred_df['label']= test_data_bin['label'].astype(int)
            
        pred_lst.append(y_pred_df)
        print(len(pred_lst))
        del test_data,x_test_par,test_data_bin,y_pred_df,temp
     
    pred_final = pd.concat(pred_lst)
    pred_final = pred_final.reset_index(drop=True)
    path = "Overall_Pred_"+y_start_date+".feather"
    output_path = fun_path_join(Env_Config.output_eval,path)
    pred_final.to_feather(output_path)
    print("Prediction result saved to {}".format(output_path))
    
    print("+++ Evaluation for windown {} ---".format(folder_path))
    #recall for each needs
    recall_needs = pred_final.groupby(['cust_needs'])['label_pred','label'].apply(lambda x: recall_score(y_true=x['label'],y_pred=x['label_pred'],pos_label=1,average='binary'))
    print("Recall for for each need: ")
    print(recall_needs)

    recall_overall = recall_score(pred_final['label'],pred_final['label_pred'])
    print("Overall recall: {}".format(recall_overall))
    
    del pred_final
        
end_time = time.monotonic()
print("Model Prediction and evaluation consumed time(sec): ", end_time - start_time)    
    
    

      
      
