import numpy as np
#import settings
from sklearn.model_selection import train_test_split
import glob
from keras.callbacks import ModelCheckpoint, LambdaCallback,LearningRateScheduler,TensorBoard,EarlyStopping,CSVLogger, ReduceLROnPlateau
from keras.utils import Sequence
import os
import random
import skimage.transform as stf
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import rotate
import keras.backend as tf
from keras.models import Model, load_model
from keras.layers import LeakyReLU, Input, Dense, Activation, Conv3D, Dropout, BatchNormalization,MaxPooling3D, Flatten,concatenate
from keras.utils import Sequence, plot_model, multi_gpu_model
from keras import regularizers
from keras.activations import sigmoid
from keras.optimizers import Adam
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from sklearn.metrics import roc_auc_score as roc_auc
import pandas as pd
import math
from keras_preprocessing import image as IM
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from lifelines.statistics import logrank_test

test_cohort = 'berlin'
i = 1
train_test_set_filename = './../input_dict_true_tumor.npy'
model_filename = '/labs/gevaertlab/data/radiology/Lung_AI_data/trained_models/final_model_all/best_model_no_clinical.h5'
LOGDIR = './logs'
checkpoint_dir = '/labs/gevaertlab/data/radiology/Lung_AI_data/checkpoints/cyclic/'
image_size = 64
cube_size = 60
num_gpus= 1
steps = 50
batch_size = 1
#clinical_features = ['stage','age','gender','histology']
clinical_features =[]
noise_variance = 0.01
reg_weight = 0.01

class data(Sequence):

    def __init__(self,params , batch_size,is_train,is_val):
        self.batch_size = batch_size
        self.is_train = is_train
        if is_train:
            prefix = 'train'
            self.indices = params['train_indices']
        elif is_val:
            prefix = 'train'
            self.indices = params['val_indices']
        else:
            prefix = 'test'
            self.indices = np.arange(len(params['test_labels']))
        self.images = params[prefix+'_data'][self.indices]
        self.labels = params[prefix+'_labels'][self.indices]
        self.is_dead = params[prefix+'_labels_is_dead'][self.indices]
        if len(clinical_features)!=0:
            for k in clinical_features:
                if params[prefix+'_labels_'+k].ndim==1:
                    params[prefix+'_labels_'+k] = np.expand_dims(params[prefix+'_labels_'+k],axis=1)
            self.clinical_feats = np.concatenate([params[prefix+'_labels_'+k] for k in clinical_features],axis=1)[self.indices]


    def __len__(self):
        return np.ceil(len(self.is_dead) / float(self.batch_size)).astype(np.int32)

    def __getitem__(self, idx):
        self.batch_images = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        self.batch_is_dead = self.is_dead[idx * self.batch_size:(idx + 1) * self.batch_size]
        self.batch_labels = self.labels[idx* self.batch_size:(idx+1)*self.batch_size]
        if len(clinical_features)>0:
            self.batch_clinical_feats = self.clinical_feats[idx * self.batch_size:(idx + 1) * self.batch_size]
        distorted_images = self.augmentate_tf(self.batch_images,self.is_train,noise_variance=noise_variance)
        if len(clinical_features)>0:
            return [distorted_images,self.batch_clinical_feats], [self.batch_is_dead,self.batch_labels]
        else:
            return distorted_images, [self.batch_is_dead,self.batch_labels]
    def augmentate_tf(self,x, is_train, noise_variance):
        yy = []; indices = []
        batch_size = x.shape[0]
        delta = (image_size-cube_size)//2
        for i in range(batch_size):
            indices.append(i)
            if (is_train):
                cropped_img = self.crop3d(x[i],cube_size)
                flipped_img = self.flip3d(cropped_img)
                random_brightness_img = self.random_brightness(flipped_img,[0.5,1])
                distorted_img = self.standardize(random_brightness_img)
            else:
                distorted_img = x[i,delta:delta+cube_size,delta:delta+cube_size,delta:delta+cube_size,:]
                distorted_image = self.standardize(distorted_img)
            yy.append(distorted_img)
        y = np.stack(yy,axis=0)
        return y

    def crop3d(self,image,size):
        offset = image.shape[0]-size
        random_offset = np.random.choice(offset,size=3)
        return image[random_offset[0]:random_offset[0]+size,random_offset[1]:random_offset[1]+size,random_offset[2]:random_offset[2]+size,:]

    def flip3d(self,image):
        is_flip_lr, is_flip_ud = np.random.choice([True, False],2)
        if is_flip_lr:
            image = np.flip(image,axis=1)
        if is_flip_ud:
            image = np.flip(image,axis=0)      
        return image

    def standardize(self,image):
        im = image[:,:,:,0]
        mask = image[:,:,:,1]
        im = (im-im.min())/(im.max()-im.min())
        return np.stack([im,mask],axis=3)

    def _apply_brightness(self,image,brightness):
        return np.stack([IM.apply_brightness_shift(np.expand_dims(image[i],axis=2),brightness) for i in range(image.shape[0])],axis=0)
    
    def random_brightness(self,image,brightness_range):
        img = image[:,:,:,0]
        mask = image[:,:,:,1]
        u = np.random.uniform(brightness_range[0],brightness_range[1])
        img = self._apply_brightness(img,u)
        return np.stack([np.squeeze(img),mask],axis=3)


def get_model(cube_size, clincal_features_size,kernel_size = (1,1,1), load_weight_path=None, gpu_num = 1):
    images = Input(shape = (cube_size,cube_size,cube_size,2), name = 'images')
    if len(clinical_features)>0:
        clinical_data = Input(shape=(clincal_features_size,), name = 'clinical_features')
    x = Conv3D(16,kernel_size,strides=(1,1,1),padding='same',activation='relu',kernel_regularizer=regularizers.l2(reg_weight),name = "Conv1")(images)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2,2,2), padding='same', data_format='channels_last', name = "MaxPool1")(x)
    x = Conv3D(16,kernel_size,strides=(1,1,1),padding='same',activation='relu',kernel_regularizer=regularizers.l2(reg_weight),name = "Conv2")(x)
    x = Conv3D(16,kernel_size,strides=(1,1,1),padding='same',activation='relu',kernel_regularizer=regularizers.l2(reg_weight),name = "Conv3")(x)
    x = Flatten()(x)
    x = Dense(128, activation = 'relu', kernel_regularizer=regularizers.l2(reg_weight), name ="Dense1")(x)
    x = Dense(64, activation = 'relu', kernel_regularizer=regularizers.l2(reg_weight), name ="Dense2")(x)
    x = Dense(64, activation = 'relu', kernel_regularizer=regularizers.l2(reg_weight), name ="Dense3")(x)   
    if len(clinical_features)>0:
        x = concatenate([x, clinical_data])
    hazard_ratio = Dense(1, activation = None, kernel_regularizer=regularizers.l2(reg_weight))(x)  
    prediction = Activation("sigmoid",name='class_prediction')(hazard_ratio)
    hazard_ratio = LeakyReLU(alpha = -1, name ="hazard_ratio")(hazard_ratio)
    if len(clinical_features)>0:
        model = Model(inputs = [images,clinical_data],outputs = [hazard_ratio,prediction])
    else:
        model = Model(inputs = images, outputs=[hazard_ratio,prediction])
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    if load_weight_path is not None:
        model.load_weights(load_weight_path, by_name=False)
    losses = {"hazard_ratio": cox_regression_loss,"class_prediction": "binary_crossentropy"}
    lossWeights = {"hazard_ratio": 1.0, "class_prediction": 0.0}
    model.compile(optimizer=opt,loss=losses, loss_weights=lossWeights)
    return model


def cox_regression_loss(are_dead,logits):
    # negative log likelihood:
    model_risk = logits
    # model_risk = logits
    hazard_ratio = tf.exp(model_risk)
    log_risk = tf.log(tf.cumsum(hazard_ratio))
    uncensored_likelihood = model_risk - log_risk
    censored_likelihood = uncensored_likelihood * are_dead
    neg_likelihood = -tf.sum(censored_likelihood)
#    return neg_likelihood
    num_events = tf.sum(are_dead)
    if num_events ==0:
        return 0
    return neg_likelihood/num_events

         
if __name__ == "__main__" :
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
   # clinical_features_size = 10
    meta_parameters_dictionary = np.load(train_test_set_filename).item()
    count = 0
    for f in clinical_features:
        if meta_parameters_dictionary['test_labels_{}'.format(f)].ndim ==1:
            count+=1
        else:
            count+=meta_parameters_dictionary['test_labels_{}'.format(f)].shape[1]
    clinical_features_size=count
    model = get_model(cube_size, clinical_features_size,kernel_size = (3,3,3))
    model= load_model(model_filename,custom_objects = {'cox_regression_loss':cox_regression_loss})
#    model.load_weights(checkpoint_dir+'cyclic_{}_{}.h5'.format(test_cohort,i))                                                
    test_generator = data(meta_parameters_dictionary,batch_size,False,False)
    preds = model.predict_generator(test_generator,verbose=1)
    preds = np.squeeze(preds)
    df = pd.DataFrame()
    df = df.assign(hazard = preds[0])
    df = df.assign(pred_class = (preds[1]))
    df = df.assign(actual_months = meta_parameters_dictionary['test_labels_months'])
    df = df.assign(is_dead = meta_parameters_dictionary['test_labels_is_dead'])
    df.to_csv(test_cohort+'_preds.csv')
    print(concordance_index(df.actual_months,-df.hazard,df.is_dead))
    fpr,tpr,thresholds = roc_curve(meta_parameters_dictionary['test_labels'],np.array(df.pred_class))
    opt_threshold = thresholds[np.argmax(tpr - fpr)]
    class_pred = np.zeros_like(np.array(df.pred_class))
    class_pred[np.where(np.array(df.pred_class)>opt_threshold)]=1
    df = df.assign(class_preds=class_pred)
    T = df['actual_months']
    E = df['is_dead']
#    ix = (df.class_preds==1)
    thres = np.median(df.hazard)
    ix = df.hazard < thres
    kmf = KaplanMeierFitter()
    kmf.fit(T[~ix],E[~ix],label='high-risk')
    ax = kmf.plot()
    kmf.fit(T[ix],E[ix],label='low-risk')
    kmf.plot(ax=ax)
    plt.savefig(test_cohort+'_km.png')


