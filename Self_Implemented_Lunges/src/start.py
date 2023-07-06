# This is the main file for the execution of training the Interaction Network model and RandomForest classifier

# Expects the input data (Excel files with each rep as a separate sheet) in the following directory structure
#                       root directory
#                        /          \
#                       /            \
#                     src            data
#                                   /    \
#                                  /      \
#                                test    train
#                                 |        |
#                                 |        |
#               <--- individiual excel file for each label --->

from pickle import TRUE
from re import L
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from BIN_module_dropout import BasicInteractionNetworkModule
from dataset_rep import LandmarksDataset
from utils_rep import *
import params
from classifier_best import *
from BIN import *


# sklearn for pre-processing
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, LeaveOneOut,RepeatedKFold,cross_val_score
import joblib
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix,classification_report
from sklearn.metrics import mean_squared_error

# classification
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.utils import shuffle
import random

# import sys
# file_n=sys.argv[1]
fl=open("PUSHUP_Classification","a+")


# important for reproducibility
# for seed in params.SEEDS[:1]:
params.savepath_ = params.savepath
for seed_ in [123,44,222,566,7000]:
    seed = seed_#params.seed_val
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    specs="../results/"
    from pathlib import Path
    Path(specs).mkdir(parents=True, exist_ok=True)
    params.savepath = params.savepath_ +"_"+str(seed)
    specs="../results/"+params.savepath
    print("Final savepath (including seed from start.py) = ", params.savepath)

    bin=BasicInteractionNetworks(params.n_objects,params.object_dim,file_n)
    if  1==2:#os.path.exists('../models/test_'+params.savepath+'.pth'):
        model = BasicInteractionNetworkModule(params.object_dim, params.relation_dim, params.effect_dim, params.external_effect_dim, params.output_dim)

        model.load_state_dict(torch.load( '../models/test_'+params.savepath+'.pth',map_location=torch.device(device)))
        model.to(device)
        model.eval()
        print("MODEL exits, loaded")
    else:
        model = bin.train()
        torch.save(model.state_dict(), '../models/test_'+params.savepath+'.pth')
    cor_loss_id,targets,predictions,_ = bin.test(model,bin.val_loader)


    plot_loss(cor_loss_id['total'],specs,cor_loss_id,True)
    ani_r=stick_animation(cor_loss_id,targets,predictions,name=file_n+"_",savepath=specs)

    ### at this point i can train multiple models and choose the model with best wighted f1 score on val dataset
    ### but this method doesnt checkk all variations in data, its always better to fix hyper prams and train multiple models using cv

    ### i have train_loader,test_loader and val loader for ccorect class noww along with their indices



    ## FROM HERE CLASSIFICATION part starts
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




    model = BasicInteractionNetworkModule(params.object_dim, params.relation_dim, params.effect_dim, params.external_effect_dim, params.output_dim)

    model.load_state_dict(torch.load( '../models/test_'+params.savepath+'.pth',map_location=torch.device(device)))
    model.to(device)
    model.eval()

    X=[]
    Y=[]
    X_test=[]
    Y_test=[]

    # correct_dataset = LandmarksDataset(xl_file='../data/'+'g1'+'.xlsx',
    #                                 root_dir='./')

    cor_train_loader = DataLoader(bin.train_indices + bin.val_indices,  shuffle=True)
    cor_loss_id_total,cor_targets,cor_predictions = test(model, cor_train_loader,bin.dataset  )
    # print("train cor mean total loss", np.mean(np.sum( [cor_loss_id['total'] for cor_loss_id in cor_loss_id_total]))) # this mean formula for nested list of diferent size is wrong
    cor_loss_id_testtotal,cor_testtargets,cor_testpredictions = test(model, bin.test_loader, bin.dataset        )


    # print("test cor mean total loss",np.mean(np.sum([cor_loss_id['total'] for cor_loss_id in cor_loss_id_testtotal])))
    # print("test cor mean total loss",np.mean(np.sum([cor_loss_id['total'] for cor_loss_id in cor_loss_id_testtotal])),file=fl)

    print("g1 Train len %d test len %d"%(len(cor_targets),len(cor_testtargets)),file=fl)
    print("g1 test indices:",bin.test_indices)
    # # import pdb;pdb.set_trace()
    for index,cor_loss_id in enumerate(cor_loss_id_total):
        # print(1,cor_loss_id,"\n")
        enc =dtft_conversion(cor_loss_id)
        # enc.extend(dtft_pos_conversion(cor_targets[index]))
        X.append(enc)
        Y.append(0)
    for index,cor_loss_id in enumerate(cor_loss_id_testtotal):
        enc =dtft_conversion(cor_loss_id)
        # enc.extend(dtft_pos_conversion(cor_testtargets[index]))

        X_test.append(enc)
        Y_test.append(0)

    inc_labels=params.inc_labels#
    for iter,inc in enumerate(inc_labels):
        incorrect_dataset = LandmarksDataset(xl_file='../data/'+inc+'.xlsx',
                                            root_dir='./')
        inc_train_indices, inc_test_indices,  = split_data(incorrect_dataset,1,remove_rear=100,train_size=.6)
        inc_train_loader = DataLoader(inc_train_indices,  shuffle=True)
        inc_test_loader = DataLoader(inc_test_indices, batch_size=1, shuffle=False)

        inc_loss_id_total,inc_targets,inc_predictions = test(model, inc_train_loader,incorrect_dataset )
        # print(inc,"train incor mean total loss",np.mean(np.sum([inc_loss_id['total'] for inc_loss_id in inc_loss_id_total])))

        inc_loss_id_testtotal,inc_testtargets,inc_testpredictions = test(model, inc_test_loader,incorrect_dataset )
        print(inc, " Train len %d test len %d"%(len(inc_targets),len(inc_testtargets)))

        print(inc, " Train len %d test len %d"%(len(inc_targets),len(inc_testtargets)),file=fl)
        # print(inc,"test inc mean total loss",np.mean(np.sum([inc_loss_id['total'] for inc_loss_id in inc_loss_id_testtotal])))
        # nose':[],'lh':[],'rh':[],'lk':[],'rk':[],'total':[]
        print(inc," test indices:",inc_test_indices)
        for index,inc_loss_id in enumerate(inc_loss_id_total):
            # print(0,inc_loss_id)
            enc = dtft_conversion(inc_loss_id)
            # enc.extend(dtft_pos_conversion(inc_targets[index]))

            X.append(enc)
            Y.append(iter+1)
            # Y.append(1)

        for index,inc_loss_id in enumerate(inc_loss_id_testtotal):
        
            enc = dtft_conversion(inc_loss_id)
            # enc.extend(dtft_pos_conversion(inc_testtargets[index]))

            X_test.append(enc)
            Y_test.append(iter+1)
            # Y_test.append(1)


    print("Train length %d, Test length %d, Cor count in Test %d" %(len(X),len(X_test),np.count_nonzero(np.array(Y_test)==0)))
    # print('Hyperparams lr %f window %d n_layers %d  epoch %d: ' %(params.learning_rate,params.dtft_width,params.n_layers,params.n_epoch),file=fl)

    file_str ="Hyperparams "
    for key,value in vars(params.args).items():
        file_str +=  key
        file_str += "="
        file_str += str(value)
    print(file_str,file=fl)
    # svm_classifier(X,Y,X_test,Y_test)
    print("Random forest results",file=fl)
    # rand_forest_classifier(copy.copy(X),copy.copy(Y),copy.copy(X_test),copy.copy(Y_test))
    rand_forest_classifier(X,Y,X_test,Y_test,fl)
 
    # print("SVM results",file=fl)
    # svm_classifier(X,Y,X_test,Y_test)
    # score,y_pred = svm_classifier(X,Y,X_test,Y_test)
    # print('F1 Score on test: ' + str(score) + " accruacy " + str(accuracy_score(y_true = Y_test, y_pred = y_pred,)))
    # print(Y_test)
    # # import pdb;pdb.set_trace()
    # print(confusion_matrix(Y_test,y_pred))
    # print(seed,classification_report(Y_test, y_pred, target_names=['g1', 'b1','b2','b3','b4','b5']),file=fl)
    # # print(classification_report(Y_test, y_pred, target_names=['g1', 'b1']),file=fl)

    # print(seed," F1=",score,"\n",cm)
    # print("THESE ARE RESULTS WITH 2 encodings my dtft encoder")
    # filename = './models/svm_classifier.joblib.pkl'
    # joblib.dump(clf, filename, compress=9)
    # clf2 = joblib.load(filename)

fl.close()
