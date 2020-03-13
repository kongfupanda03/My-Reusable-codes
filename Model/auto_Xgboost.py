#########################################################################################################################################
# Project     : Automation of Xgboost
#
# Coding      : Xiong Yuyu
#
# Date        : Since 2020-2-19
#
# Note        : Current class only works on single output
#
# Description : 
#               1) Automate the training of a Xgboost
#               2) Automate the evaluation of a Xgboost
#########################################################################################################################################
from Model.model_config import *
from Global_fun import *
import pickle
from Config import Env_Config


class cls_auto_Xgboost(object):
    '''Class to automate auto-training and auto-evaluation of Xgboost
       1) set parameters 
       2> model training
       3) model prediction
       4) model evaluation    
    '''
    def __init__(self, X, y,
                    learning_rate_flag=True,
                    n_estimator_flag=True,
                    max_depth_flag=True,
                    scoring='roc_auc',
                    cv=1,
                    test_size=0.8,
                    n_jobs=1,
                    random_state=123,
                    early_stopping_rounds=20):
        '''
        Function : set initial parameters
        Input : X_train, y_train
        Output : 
        '''
        self.X = X
        self.y = y
        self.learning_rate_flag = learning_rate_flag
        self.n_estimator_flag = n_estimator_flag
        self.max_depth_flag = max_depth_flag
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.early_stopping_rounds = early_stopping_rounds
        if cv >= 1 :
            self.cv =cv
        else:
            sys.exit("cv cannot be less than 1!")
        
        self.test_size = test_size
        self.__fun_set_params()
        ## End of init

    def __fun_set_params(self):
        '''
        Function : set tuning parameters for model
        '''
        if self.learning_rate_flag == True:
            self.learning_rate = [0.01, 0.1]
        else:
            self.learning_rate = 0.1

        if self.n_estimator_flag == True:
            self.n_estimators = [500,1000,3000]
        else: 
#             self.n_estimators = 1000
            self.n_estimators = [500]
        
        if self.max_depth_flag == True:
            self.max_depth = [2,3,4,5,6]
        else:
            self.max_depth = [4]
        
        
        if self.cv == 1:
            t_size = int(self.X.shape[0]*self.test_size)
            train_val_split = [-1]*t_size + [0]*(self.X.shape[0]-t_size)
            seed(self.random_state)
            shuffle(train_val_split)
            self.ps = PredefinedSplit(train_val_split)
        else:
            self.ps = self.cv
        
        ## End of fun_set params
    
    def fun_print_params(self):
        '''Function: to print out tuning variables
        '''
        print('learning rate: ', self.learning_rate_flag)
        print('Iterations: ',self.n_estimators)
        print('Max_depth: ', self.max_depth)
        print("cv: ", self.cv)
        if self.cv == 1:
            for train_index, validation_index in self.ps.split():
                print("training: ", len(train_index), "; validation: ", len(validation_index))
        ## End of fun_print_params
    
    def fun_train_model(self):
        '''
        Function: Train Xgboost model
        '''
        seed(self.random_state)

        self.fun_print_params()

        param_grid =dict(max_depth=self.max_depth,
                        learning_rate=self.learning_rate,
                        n_estimators=self.n_estimators)

        model = XGBClassifier(objective = 'binary:logistic',random_state=self.random_state) 

        for _, val_index in self.ps.split():
            X_val = self.X[val_index] #removed iloc
            y_val = self.y[val_index]
            
        
        fit_params={'early_stopping_rounds': 20,
                    'eval_set':[(X_val,y_val)],
                    'eval_metric':'auc'}
        

       

        gdsearch = GridSearchCV(estimator=model, 
                                param_grid=param_grid,
                                scoring=self.scoring,
                                n_jobs=self.n_jobs,
                                cv=self.ps)
    
        
        #fit model
        grid_result = gdsearch.fit(X = self.X, 
                                y = self.y,
                                verbose = 1,
                                **fit_params)
        '''
         #fit model
        grid_result = gdsearch.fit(X = self.X, 
                                y = self.y,
                                verbose = 1)
        '''

        # summarize results        
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

        return grid_result
    
    @classmethod
    def fun_pred(cls, model, x_test, pred_type = 'prob'):
        """Function: Generate prediction using trained model
           Input:    1) Gridsearch fit output or xgboost trained model
                     2) Numpy array for prediction
                     3) Prediction type ('prob' and None)
           Output:   1) Numpy array with the probability of each data example being of a given class
        """
        if pred_type == 'prob':
            pred = model.predict_proba(x_test)
        else:
            pred = model.predict(x_test)
        return pred
    #End of Function 'fun_pred'
    

    def fun_eval(self, y_pred, y_test):
        """Function: evaluate the results
           Input:    1) 
                     2) 
           Output:  
        """

        #y_pred_prob = model.predict_proba(x_test)
        #y_pred = model.predict(x_test)
        
        print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))
        print('Precision Score : ' + str(precision_score(y_test,y_pred,average = 'weighted', labels=np.unique(y_pred))))
        recall_rslt = recall_score(y_test,y_pred,average = 'weighted')
        print('Recall Score : ' + str(recall_rslt))
        print('F1 Score : ' + str(f1_score(y_test,y_pred,average = 'weighted')))
        print('ROC-AUC Score : ' + str(roc_auc_score(y_test,y_pred,average = 'weighted')))
        # Compute ROC curve and ROC area
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # Plot of a ROC curve for a specific class
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

        # Confusion matrix
#         titles_options = [("Confusion matrix, without normalization", None),
#                           ("Normalized confusion matrix", 'true')]
#         for title, normalize in titles_options:
#             disp = plot_confusion_matrix(model, x_test, y_test,
#                                                  cmap=plt.cm.Blues,
#                                                  normalize=normalize)
#             disp.ax_.set_title(title)
#         plt.show()
        return recall_rslt
    
    #End of Function 'fun_eval'

    def fun_create_feature_map(fmap_filename, feature_names):
        """Function: create fmap file for feature ranking
        """
        fmap_file = open(fmap_filename, 'w')
        for i, feat in enumerate(feature_names):
            fmap_file.write('{0}\t{1}\tq\n'.format(i, feat))
        fmap_file.close()
    
    def fun_fea_ranking(self, model, feature_names, importance_type = 'gain'): # generate feature importance for every Y
        """Function: output feature importance ranking to csv and plot top features
           Input:    1) 
                     2) 
           Output:  
        """

        if (type(model) == xgb.sklearn.XGBClassifier):
            model = model
        elif (type(model) == sklearn.model_selection._search.GridSearchCV):
            model = model.best_estimator_
        
        # create feature map file
#         fmap_file = fun_path_join(Env_Config.output_model,'XGB_features.fmap')
#         cls_auto_Xgboost.fun_create_feature_map(fmap_filename = fmap_file, feature_names = feature_names)
#         fea_ranking = model.get_booster().get_score(fmap = fmap_file, importance_type = importance_type)

        # output feature ranking
        model.get_booster().feature_names = feature_names
        fea_ranking = model.get_booster().get_score(importance_type = importance_type)
        fea_ranking_df = pd.DataFrame.from_dict(fea_ranking, orient="index", columns = ['importance'])
        fea_ranking_df.index.name='features'
        fea_ranking_df = fea_ranking_df.reset_index()

        # plot top 20 features
        xgb.plot_importance(model, max_num_features=20, importance_type = importance_type)
        plt.title('Top Features (XGBoost Feature Ranking)')
        plt.show()
        return fea_ranking_df
    #End of Function 'fun_fea_ranking'
    
    
    def fun_save(self, model, file_name):
        """Function: save trained model
           Input:    1) Gridsearch fit output or xgboost trained model
                     2) Model file name in string format
           Output:  
        """

        if (type(model) == xgb.sklearn.XGBClassifier):
            model = model
        elif (type(model) == sklearn.model_selection._search.GridSearchCV):
            model = model.best_estimator_
        pickle.dump(model, open(fun_path_join(Env_Config.output_model,file_name), "wb"))
    #End of Function 'fun_save'

    def fun_load(self, file_name):
        """Function: load saved model
           Input:    1) Model file name in string format
           Output:  
        """
        loaded_model = pickle.load(open(fun_path_join(Env_Config.output_model,file_name), "rb"))
        return loaded_model
    #End of Function 'fun_load'


                         
    
    




