##########################################################################################################################################

# Run the XGBoost model 

# import modules and packages
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

# Read Training and Testing data
print('+++ Training Features ---\n')
# Read in Y_start_time_train.csv
y_start_train_dt = pd.read_csv(fun_path_join(Env_Config.source_code_path, "Y_start_time_train.csv"))
fea_multi = 'FEA_multi_win_' + y_start_train_dt['Y_start_time'].max() + '.feather'
training_all = pd.read_feather(fun_path_join(Env_Config.output_FE_train, fea_multi))

# COMMENT - read y_start_time_test use it to generate the full name of testing data and then read in
# UPDATE - used Y_start_time_test.csv to specify the testing file to use
print('+++ Testing Features ---\n')
y_start_test_dt = pd.read_csv(fun_path_join(Env_Config.source_code_path,'Y_start_time_test.csv'))
# Get a list of test data directories
test_dir = os.listdir(Env_Config.output_FE_test_par)
test_dir=[x for x in test_dir if x.startswith('FEA')]
testing_all = pd.read_feather(fun_path_join(Env_Config.output_FE_test_par, test_dir[0]))

Y_start_date = Env_Config.need_win_start.strftime('%Y-%m-%d')
hist_win_end = Env_Config.hist_win_end.strftime('%Y-%m-%d')

cls_fea_integrate = cls_sample_integrate()
testing_all_bin = cls_fea_integrate.fun_model_binarisation(data=testing_all,hist_end_date = hist_win_end,label=label)

print('training data shape: ' + str(training_all.shape))
print('testing data shape:' + str(testing_all_bin.shape))

print('+++ Create training and testing data and respective labels  ---\n')
# make sure same features in model preprocessing
x_train, y_train, x_test, y_test = model_preprocessing.sliding_win_train_test_split(train_data = training_all, 
                                                                                    test_data = testing_all_bin)


# COMMENT - check for cin: shouldnt be there
# UPDATE - print will check for cin (cin - 0, no cin in df): 
print('X train is of type:{},  shape:{}, cin:{}'.format(type(x_train).__name__, 
                                                        x_train.shape,'cin' in list(x_train.columns)))
print('y train is of type:{},  shape:{}'.format(type(y_train).__name__,
                                                        y_train.shape ))
print('X test is of type:{},  shape{}, cin:{}'.format(type(x_test).__name__, 
                                                      x_test.shape, 'cin' in list(x_test.columns)))
print('y test is of type:{}, shape{}'.format(type(y_test).__name__,
                                                     y_test.shape))

# COMMENT - check if x train and x test are well align - used same set of common names to arrange columns
# UPDATE - make sure that columns are aligned
# check if the columns of training and testing set are aligned
x_train_columns = x_train.columns
x_test_columns = list(x_test.columns)
if set(x_train_columns)==set(x_test_columns):
    print('columns are aligned')
else:
    print('columns are not aligned')

    


# change dataframe to ndarray

x_train = x_train.values
x_test = x_test.values
y_train = y_train.values.astype(int)
y_test = y_test.values.astype(int)

del testing_all_bin,testing_all

#timer - start
start_time = time.monotonic() 

#object for XGB
print('+++ Initialise XGB ---\n')
XGB = cls_auto_Xgboost(X = x_train, y = y_train, n_jobs = -7, 
                       max_depth_flag = True, n_estimator_flag = False) 
                       # n_jobs changed to -7
                       # max_depth_flag and n_estimator_flag to be set to True in production run
# #print param
# XGB.fun_print_param()

#train model
print('+++ Train XGB ---\n')
grid_result = XGB.fun_train_model()
end_time = time.monotonic()

print("+++ Model Training Done ---")
print("Model Training consumed time(sec): ", end_time - start_time)

print(grid_result)

# Save the model as pkl extension
model_filename = ''.join(['XGB_model_',date.today().strftime("%Y_%m_%d")])
filename = model_filename + '.pkl'
print(filename)

# save the best model
joblib.dump(grid_result,fun_path_join(Env_Config.output_model, filename))

#Load model
#grid_result=joblib.load(fun_path_join(Env_Config.output_model, filename))
#XGB.fun_save(model = grid_result, file_name = filename) 

#prediction
start_time = time.monotonic()
pred_lst=[]
print('+++ Prediction ---')
for i in range(len(test_dir)):
    test_data = pd.read_feather(fun_path_join(Env_Config.output_FE_test_par, test_dir[i]))
    print('Input partition is {}'.format(fun_path_join(Env_Config.output_FE_test_par, test_dir[i])))
    test_data_bin = cls_fea_integrate.fun_model_binarisation(data=test_data,hist_end_date = hist_win_end,label=label)
    test_data_bin = test_data_bin.fillna(0)
    x_test_par = test_data_bin.loc[:,x_test_columns].values ##fillna(0)
    #y_test_par = test_data_bin['label']
    print(test_data.shape,test_data_bin.shape,x_test_par.shape)
    
    y_pred = cls_auto_Xgboost.fun_pred(model = grid_result, x_test = x_test_par, pred_type = 'prob')

    # COMMENT - add in cin here (take cin from testing_all data )
    # UPDATE - add cin to y_pred_np
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
    
end_time = time.monotonic()
print("Model Prediction consumed time(sec): ", end_time - start_time)

pred_final = pd.concat(pred_lst)
pred_final = pred_final.reset_index(drop=True)
path = "Overall_Pred_"+Y_start_date+".feather"
output_path = fun_path_join(Env_Config.output_eval,path)
pred_final.to_feather(output_path)
print("Prediction result saved to {}".format(output_path))

print("+++ Evaluation ---")
#recall for each needs
recall_needs = pred_final.groupby(['cust_needs'])['label_pred','label'].apply(lambda x: recall_score(y_true=x['label'],y_pred=x['label_pred'],pos_label=1,average='binary'))
print("Recall for for each need: ")
print(recall_needs)

recall_overall = recall_score(pred_final['label'],pred_final['label_pred'])
print("Overall recall: {}".format(recall_overall))


