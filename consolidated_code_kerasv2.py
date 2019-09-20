import numpy as np
#import settings
from sklearn.model_selection import train_test_split
import glob
from keras.callbacks import ModelCheckpoint, LambdaCallback,LearningRateScheduler,TensorBoard,EarlyStopping,CSVLogger, ReduceLROnPlateau
from ema import ExponentialMovingAverage
from keras.utils import Sequence
import os
import random
import skimage.transform as stf
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import rotate
import keras.backend as tf
from keras.models import Model, load_model
from keras.layers import LeakyReLU,Input, Dense, Activation, Conv3D, Dropout, BatchNormalization,MaxPooling3D, Flatten,concatenate
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
import matplotlib.pyplot as plt
from clr_callback import CyclicLR
from matplotlib.font_manager import FontProperties
from lifelines.statistics import logrank_test
import tensorflow as k
###################################
# TensorFlow wizardry
config = k.ConfigProto()
 
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
 
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 1
 
# Create a session with the above options specified.
tf.tensorflow_backend.set_session(k.Session(config=config))
###################################



test_cohort = 'nsclc'
train_test_set_filename = '/labs/gevaertlab/data/radiology/Lung_AI_data/Lung_AI_data/train_test_set_{}.npy'.format(test_cohort)
model_dir = '/labs/gevaertlab/data/radiology/Lung_AI_data/trained_models/{}'.format(test_cohort)
plot_dir = './plots/cyclic/{}/with_clinical'.format(test_cohort)
checkpoint_dir='/labs/gevaertlab/data/radiology/Lung_AI_data/checkpoints/cyclic/with_clinical/'
LOGDIR = './logs'
image_size = 64
cube_size = 60
num_gpus= 1
steps = 120
batch_size = 60
clinical_features = ['stage','age','gender','histology']
#clinical_features =[]
noise_variance = 0.01
itercount = 40
#itercount = settings.itercount
#WC = settings.window_center
#WW = settings.window_width
#max_angle = settings.MAX_ANGLE_TO_ROTATE
reg_weight = 0.01
early = True
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
            self.indices = params['test_indices']
        self.images = params[prefix+'_data'][self.indices]
        self.labels = params[prefix+'_labels'][self.indices]
        self.is_dead = params[prefix+'_labels_is_dead'][self.indices]
        if len(clinical_features)!=0:
            for k in clinical_features:
                if params[prefix+'_labels_'+k].ndim==1:
                    params[prefix+'_labels_'+k] = np.expand_dims(params[prefix+'_labels_'+k],axis=1)
           # print(params[prefix+'_labels_'+k].shape)
            self.clinical_feats = np.concatenate([params[prefix+'_labels_'+k] for k in clinical_features],axis=1)[self.indices]
        if len(np.where(np.isnan(self.clinical_feats)))>0:
            print(np.where(np.isnan(self.clinical_feats)))

    def __len__(self):
        return np.ceil(len(self.is_dead) / float(self.batch_size)).astype(np.int32)

    def __getitem__(self, idx):
        self.batch_images = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        self.batch_is_dead = self.is_dead[idx * self.batch_size:(idx + 1) * self.batch_size]
        self.batch_labels = self.labels[idx* self.batch_size:(idx+1)*self.batch_size]
        if len(clinical_features)>0:
            self.batch_clinical_feats = self.clinical_feats[idx * self.batch_size:(idx + 1) * self.batch_size]
        distorted_images = self.augmentate_tf(self.batch_images,self.is_train,noise_variance=noise_variance)
        #print(distorted_images.shape,self.batch_clinical_feats.shape)
        if len(clinical_features)>0:
            return [distorted_images,self.batch_clinical_feats], [self.batch_is_dead,self.batch_labels]
        else:
            return distorted_images, [self.batch_is_dead,self.batch_labels]
    def augmentate_tf(self,x, is_train, noise_variance):
        yy = []; indices = []
        batch_size = x.shape[0]
        #seeds = np.random.randint(low=1,high=1000,size=batch_size)
        delta = (image_size-cube_size)//2
        for i in range(batch_size):
            indices.append(i)
            if (is_train):
                cropped_img = self.crop3d(x[i],cube_size)
                flipped_img = self.flip3d(cropped_img)
                random_brightness_img = self.random_brightness(flipped_img,[0.5,1])
                #random_contrast_img = self.random_contrast(random_brightness_img,lower=0.2,upper=1.8,seed=seeds[i])
                distorted_img = self.standardize(random_brightness_img)
            else:
                if test_cohort=='stanford':
                    distorted_img = x[i,0:cube_size,delta:delta+cube_size,delta:delta+cube_size,:]
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
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False, clipnorm=1)
    if load_weight_path is not None:
        model.load_weights(load_weight_path, by_name=False)
    losses = {"hazard_ratio": cox_regression_loss,"class_prediction": "binary_crossentropy"}
    lossWeights = {"hazard_ratio": 1.0, "class_prediction": 0.0}
    model.compile(optimizer=opt,loss=losses, loss_weights=lossWeights)
    return model

#def step_decay(epoch,lr):
#    initial_lrate = 0.001
#    drop = 0.5
#    #k = 0.1
#    epochs_drop = 5
#    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
#   # lrate = initial_lrate*math.exp(-1*k*epoch)
#    return lrate

def get_plot(df, early_flag): 
    fontname='serif'
    T = df.actual_months
    E = df.is_dead
    ix = df.hazard > np.median(df.hazard)
    fig = plt.figure(figsize=[10,8])
    kmf= KaplanMeierFitter()
    kmf.fit(T[ix], E[ix],label='high_risk')
    ax = fig.gca()
    kmf.plot(ax=ax,show_censors=True)
    kmf.fit(T[~ix], E[~ix],label='low_risk')
    kmf.plot(ax=ax,show_censors=True)
    c_index = concordance_index(T, -df.hazard, E)
    results = logrank_test(T[ix], T[~ix], event_observed_A=E[ix], event_observed_B=E[~ix])
    print(c_index, results)
    plt.text(0.1,0.1,'c-index: {:.2f}\np-value: {:.2e}'.format(c_index,results.p_value),fontname=fontname,fontsize='x-large',transform=ax.transAxes)
    plt.xlabel('timeline',fontname=fontname,fontsize='x-large' )
    prop=FontProperties(family=fontname,size='x-large')
    plt.legend(['_','high_risk','_','low_risk'],loc=0,prop=prop)
    if early_flag:
        plt.savefig(os.path.join(plot_dir,'early_{}_{}_{:.2f}.png'.format(test_cohort,i,c_index)))
    else:
        plt.savefig(os.path.join(plot_dir,'{}_{}_{:.2f}.png'.format(test_cohort,i, c_index)))


         
if __name__ == "__main__" :
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
   # clinical_features_size = 10
    meta_parameters_dictionary = np.load(train_test_set_filename).item()
    count = 0
    for f in clinical_features:
        if meta_parameters_dictionary['train_labels_{}'.format(f)].ndim ==1:
            count+=1
        else:
            count+=meta_parameters_dictionary['train_labels_{}'.format(f)].shape[1]
    clinical_features_size=count
    c_scores = np.zeros(itercount)
    for i in range(itercount):
        train_indices, val_indices = train_test_split(np.arange(len(meta_parameters_dictionary['train_labels'])), test_size = 0.1)
        if early:
            early_test_indices = []
            for j in range(len(meta_parameters_dictionary['test_labels'])):
                if meta_parameters_dictionary['test_labels_stage'][j,0]==1 or meta_parameters_dictionary['test_labels_stage'][j,1]==1:
                    early_test_indices.append(j)
        test_indices = range(len(meta_parameters_dictionary['test_labels']))
        meta_parameters_dictionary['train_indices']=train_indices
        meta_parameters_dictionary['val_indices']=val_indices
        meta_parameters_dictionary['test_indices'] = np.array(test_indices)
        training_generator = data(meta_parameters_dictionary,batch_size,True,False)
        val_generator = data(meta_parameters_dictionary,batch_size,False,True)
        test_generator = data(meta_parameters_dictionary,batch_size,False,False)
        csv_logger = CSVLogger(os.path.join(LOGDIR,'training_{}.log'.format(test_cohort)))
        lrate = CyclicLR(base_lr=0.001,max_lr= 0.01,step_size=100,mode='triangular2')
        checkpointer = ExponentialMovingAverage(filepath=checkpoint_dir+'cyclic_{}_{}.h5'.format(test_cohort,i),save_best_only=True, save_weights_only=True,custom_objects={'cox_regression_loss':cox_regression_loss},verbose=1)
        lr_monitor = LambdaCallback(on_epoch_begin=lambda epoch, logs:print(tf.eval(model.optimizer.lr)))
        lr_callback = ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=2, min_lr = 0.00001)
        model = get_model(cube_size, clinical_features_size,kernel_size = (3,3,3))
        history = model.fit_generator(training_generator, verbose =2, epochs=steps, callbacks=[lr_callback,lr_monitor,lrate,csv_logger,checkpointer],validation_data= val_generator,workers=8, use_multiprocessing=True, shuffle=True)
        print(i)
        try:
            model.load_weights(checkpoint_dir+'cyclic_{}_{}.h5'.format(test_cohort,i))
        except OSError:
            print('Could not find checkpoint:'+ checkpoint_dir+'cyclic_{}_{}.h5'.format(test_cohort,i))
            continue

        #tensorboard_callback = TensorBoard(log_dir=LOGDIR, histogram_freq=0, write_graph=True)
        #early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=10)
        #weights = model.get_weights()
        #start_time = time.time()
        preds = model.predict_generator(test_generator)
        #np.save(test_cohort+'preds.npy',preds)
        preds = np.squeeze(preds)
        df = pd.DataFrame()
        df = df.assign(hazard = preds[0])
        df = df.assign(pred_class = (preds[1]))
        df = df.assign(actual_months = meta_parameters_dictionary['test_labels_months'][test_indices])
        df = df.assign(is_dead = meta_parameters_dictionary['test_labels_is_dead'][test_indices])
        get_plot(df, False)
    #    df.to_csv(test_cohort+'_preds.csv')
    #   # print(roc_auc(meta_parameters_dictionary['test_labels'][test_indices],np.array(df.pred_class)))
    #   # print(concordance_index(df.actual_months,-df.hazard,df.is_dead))
    #    c_scores[i] = concordance_index(df.actual_months,-df.hazard,df.is_dead)
    #    fpr,tpr,thresholds = roc_curve(meta_parameters_dictionary['test_labels'][test_indices],np.array(df.pred_class))
    #    opt_threshold = thresholds[np.argmax(tpr - fpr)]
    #   # print(opt_threshold,preds.min(),preds.max())
    #    class_pred = np.zeros_like(np.array(df.pred_class))
    #    class_pred[np.where(np.array(df.pred_class)>opt_threshold)]=1
    #    df = df.assign(class_preds=class_pred)
    #    T = df['actual_months']
    #    E = df['is_dead']
    #   # print(T,E)
    #    #ix = (df.class_preds==1)
    #   # print(ix)
    #    thres = np.median(df.hazard)
    #    ix = df.hazard<thres
    #    kmf = KaplanMeierFitter()
    #    kmf.fit(T[~ix],E[~ix],label='high-risk')
    #    ax = kmf.plot(show_censors=True)
    #    kmf.fit(T[ix],E[ix],label='low-risk')
    #    kmf.plot(ax=ax,show_censors=True)
    #    plt.text(0.1,0.1,'c-index:{:.2f}'.format(c_scores[i]),horizontalalignment='left',verticalalignment='bottom',transform = ax.transAxes)
    #    plotname = '_{:2.0f}_km_cyclic_v2.png'.format(c_scores[i]*100)
    #    plt.savefig(os.path.join(plot_dir,test_cohort+str(i)+plotname))
    #    print(c_scores[i])
        if early:
            early_df = df.iloc[early_test_indices,:]
            get_plot(early_df, True)
        #    early_df.to_csv('early_{}_preds.csv'.format(test_cohort))
        #    T= early_df['actual_months']
        #    E = early_df['is_dead']
        #    early_c = concordance_index(early_df.actual_months,-early_df.hazard,early_df.is_dead)
        #    thres = np.median(early_df.hazard)
        #    ix = early_df.hazard<thres
        #    kmf.fit(T[~ix],E[~ix],label='high-risk')
        #    plt.figure()
        #    ax = kmf.plot(show_censors=True)
        #    kmf.fit(T[ix],E[ix],label='low-risk')
        #    kmf.plot(ax=ax,show_censors=True)
        #    plt.text(0.1,0.1,'c-index:{:.2f}'.format(early_c),horizontalalignment='left',verticalalignment='bottom',transform = ax.transAxes)
        #    plotname = '_{:2.0f}_early_km_cyclic_v2.png'.format(early_c*100)
        #    plt.savefig(os.path.join(plot_dir,test_cohort+str(i)+plotname))
        #    print(early_c)
        del model
        tf.clear_session()
