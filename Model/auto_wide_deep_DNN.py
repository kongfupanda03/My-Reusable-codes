import sys
import json
import joblib
import pandas as pd
from keras.layers import Input, Dense, Dropout, concatenate, BatchNormalization, Activation, Dropout
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras.regularizers import l2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataclasses import dataclass, field
from typing import List, Any

sys.path.append ('./SourceCode')
from Global_fun import *
from Config import Env_Config
from Model.early_stop_recall import EarlyStopByRecall
from Model.metrics import change_top_n_0_1, get_recall

Env_Config.fun_set_cwd (".")
nl = "\n"

@dataclass
class hyperparameters:
    """
    Hyperparamters setting to use for training Wide and Deep Model
    """
    epochs: int = 50
    neurons: Any = field(default_factory=list)
    dropout_rate: List[int] = field(default_factory=list)
    batch_size: List[int] = field(default_factory=list)
    patience: int = 5
    seed: int = 2019
    model_name: str = "run1"
    wide_features_path: str = "wide_features.csv"
    train_date: str = "2019-01-01"
    training_win_prefix: str = "FEA_multi_win_"
        
        
def preprocessing(wide_features_path: str, training_data_path: str, scaler_path: str, train_date: str):
    """
    Preprocess to get normalised training data, location of wide features and ordered names of columns
    """
    wide_features = pd.read_csv(wide_features_path)
    training_data = pd.read_feather(training_data_path)
    
    training_data = training_data.fillna(0)
    
    all_col = training_data.columns # Index type

    y_col_name = all_col[all_col.str.contains(f"^{Env_Config.prefix_Y}", regex=True)] # Subset Index type with str.contains
    x_col_name = all_col[~all_col.isin(y_col_name.to_list() + ['cin'])]
    
    y_train = training_data.reindex(y_col_name, axis=1, fill_value=0)
    x_train = training_data.reindex(x_col_name, axis=1, fill_value=0)
    
    std_scale = StandardScaler().fit(x_train)
    x_train_norm = std_scale.transform(x_train)
    
    LOCATION_WIDE = [all_col.get_loc(feature) for feature in wide_features["Wide_Features"] if feature in all_col]
    # dump scaler out
    joblib.dump(std_scale, scaler_path)
    
    return x_train_norm, y_train, LOCATION_WIDE, x_col_name, y_col_name


@dataclass
class Wide_Deep(hyperparameters):
    """
    Wide and Deep model:
    1.) Process data using preprocessing function
    2.) Custom Early Stopping 
    3.) Run a manual grid with hyperparamters inherited from hyperparamters dataclass
    Output: Model trained on full train set
    """
    
    def __post_init__(self):
        # create a new folder in model dir with run name 
        self.model_dir = fun_path_join(Env_Config.output_model, self.model_name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        self.training_data_path = fun_path_join(Env_Config.output_FE_train, f"{self.training_win_prefix}{self.train_date}.feather")
        self.scaler_path = fun_path_join(self.model_dir, f"{self.model_name}_scaler.save")
        self.model_path = fun_path_join(self.model_dir, f"{self.model_name}.h5")
        self.hparams_json = fun_path_join(self.model_dir, f"{self.model_name}_hparams.json")
        
        self.x_train, self.y_train, self.loc_wide, self.x_col_name, self.y_col_name\
        = preprocessing(self.wide_features_path, self.training_data_path, self.scaler_path, self.train_date)
        
        self.x_train_split, self.x_val_split, self.y_train_split, self.y_val_split\
        = train_test_split(self.x_train, self.y_train, test_size=0.2, random_state=self.seed)
        
        self.deep_dim = self.x_train.shape[1]
        self.wide_dim = len(self.loc_wide)
        self.output_dim = self.y_train.shape[1]
        
    def create_deep_wide_model(self, neurons, dropout_rate, batch_size, validation=True):
        
        deep0 = Input(shape=(self.deep_dim, ))
        wide = Input(shape=(self.wide_dim, ))

        deep1 = Dense(neurons[0],
                kernel_initializer="random_uniform",
                kernel_regularizer=l2(0.01),
                bias_regularizer=l2(0.01))(deep0)
        normalisation1 = BatchNormalization()(deep1)
        activation1 = Activation("relu")(normalisation1)
        dropout1 = Dropout(rate=dropout_rate, seed=self.seed)(activation1)
    
        deep2 = Dense(neurons[1],
                kernel_initializer="random_uniform",
                kernel_regularizer=l2(0.01),
                bias_regularizer=l2(0.01))(dropout1)
        normalisation2 = BatchNormalization()(deep2)
        activation2 = Activation("relu")(normalisation2)
        dropout2 = Dropout(rate=dropout_rate, seed=self.seed)(activation2)

        deep_out = dropout2

        out_layer = concatenate([deep_out, wide])
        out_layer1 = Dense(self.output_dim,
                          kernel_initializer="random_uniform",
                          activation="sigmoid")(out_layer)

        model = Model(inputs=[deep0] + [wide], outputs=out_layer1)

        model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["acc"])
        
        if validation:
            
            x_wide = self.x_train_split[:, self.loc_wide]
            x_val_wide = self.x_val_split[:, self.loc_wide]
            
            model.fit(x=[self.x_train_split] + [x_wide] , 
                    y=self.y_train_split, 
                    validation_data=([self.x_val_split] + [x_val_wide], self.y_val_split),
                    batch_size=batch_size,
                    epochs=self.epochs, 
                    verbose=2,
                    callbacks=[EarlyStopByRecall(
                        ([self.x_val_split] + [x_val_wide], self.y_val_split),
                        patience=self.patience)])

            return model 
        else:
            x_wide = self.x_train[:, self.loc_wide]
            
            model.fit(x=[self.x_train] + [x_wide], 
                    y=self.y_train, 
                    batch_size=batch_size,
                    epochs=self.epochs,
                    verbose=2,
                    callbacks=[EarlyStopByRecall(
                        ([self.x_train] + [x_wide], self.y_train), 
                        patience=self.patience)])

            return model
    
    def manual_grid(self):
        
        trial_num = 1
        best_set = {}
        
        x_val_wide = self.x_val_split[:, self.loc_wide]
        
        for neurons_hp in self.neurons:
            for dropout_rate_hp in self.dropout_rate:
                for batch_size_hp in self.batch_size:
                    
                    run_name = f"run-{trial_num}"
                    print(f"{nl}*** Starting Trial: {run_name}")
                    print(f"neurons: {neurons_hp}, dropout_rate_hp: {dropout_rate_hp}, batch_size: {batch_size_hp}")
                    
                    model_hp = self.create_deep_wide_model(neurons_hp, dropout_rate_hp, batch_size_hp, validation=True)
                    
                    score = get_recall(model_hp.predict([self.x_val_split] + [x_val_wide]),
                                       self.y_val_split,
                                       top_n=3, 
                                       avg_method="weighted")
                    
                    best_set.update({score: [neurons_hp, dropout_rate_hp, batch_size_hp]})
                    
                    trial_num += 1
        
        neurons_best, dropout_rate_best, batch_size_best = best_set.get(max(best_set))
        
        print(f"{nl}TRAIN ON BEST SET: NEURONS: {neurons_best}, DROPOUT: {dropout_rate_best}, BATCH SIZE: {batch_size_best}")
              
        best_model = self.create_deep_wide_model(neurons_best, dropout_rate_best, batch_size_best, validation=False)
              
        best_model.save(self.model_path)
        
        with open(self.hparams_json, "w") as hp_write:
            hp = {"BEST PARAMS": {"Neurons": neurons_best, 
                                  "Dropout": dropout_rate_best,
                                  "Batch Size": batch_size_best},
                  "PARMAS GRID": {"Neurons": self.neurons,
                                  "Dropout": self.dropout_rate,
                                  "Batch Size": self.batch_size}}
            json.dump(hp, hp_write, indent=4)
            
        return best_model
    
    def model_input_meta(self):

        return self.x_col_name, self.y_col_name, self.loc_wide, self.model_dir, self.scaler_path
        