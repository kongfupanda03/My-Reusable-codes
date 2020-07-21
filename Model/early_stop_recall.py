import sys
sys.path.append ('./SourceCode/Model')
import numpy as np
from metrics import change_top_n_0_1, get_recall
from keras.callbacks import EarlyStopping, Callback

class EarlyStopByRecall(Callback):
    """
    Early Stop Function for Recall, reverts to best weight
    
    Inputs:
    1.) validation_data: [x validation, y validation]
    2.) patience: How many epoch to run without improvement before stopping
    
    """
    def __init__(self, validation_data, patience=0):
        super(EarlyStopByRecall, self).__init__()
        
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.stopped_epoch = 0
        self.patience = patience
        self.best_weights = None
        
    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        predict = self.model.predict(self.x_val)
        target = self.y_val
        score = get_recall(predict, target, top_n=3, avg_method="weighted")
        
        if score > self.best:
            self.best = score
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print('Restoring model weights from the end of the best epoch.')
                self.model.set_weights(self.best_weights)
        
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
              print(f'Epoch {self.stopped_epoch + 1}: early stopping with best recall score of {self.best}')
                