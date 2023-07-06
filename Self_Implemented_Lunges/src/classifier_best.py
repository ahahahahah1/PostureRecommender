# THIS is CLASSIFIER ON restricted dataset, removing bad instance of shoulder press b2,b3,b4
#  5 fold CV results are provided here

from BIN_module_dropout import BasicInteractionNetworkModule
import params
import torch
import torch.nn as nn
from utils_rep import *
from dataset_rep import LandmarksDataset
import copy


#sklearn for pre-processing
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, LeaveOneOut,RepeatedKFold
import joblib
#scores
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix,classification_report
from sklearn.metrics import mean_squared_error

#classification
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.utils import shuffle
import random


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# important for reproducibility
# fl=open("cm_data_AdamW_rf","a+")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def kFoldResults(model,X,y,fl):
    X=np.array(X)
    y=np.array(y)
    # print(X,y)
    # print(np.shape(X),np.shape(y))
    # cv = RepeatedKFold(n_splits=10, n_repeats=len(X)//10,random_state=1)
    cv = KFold(n_splits=len(X), shuffle=True,random_state=1)

    # enumerate splits
    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    scores = []
    for train_ndx, test_ndx in cv.split(X):
        # print('train: %s, test: %s' % (train_ndx, test_ndx))
        train_X, train_y, test_X, test_y = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]
        # print(test_y)
        model.fit(train_X, train_y)
        pred_y = model.predict(test_X)
    
        actual_classes = np.append(actual_classes, test_y)
        predicted_classes = np.append(predicted_classes, model.predict(test_X))


        score = f1_score(y_true = test_y, y_pred = pred_y,average='weighted')
        scores.append(score)
        # print("Weighted f1 score",score)
    print( confusion_matrix(actual_classes, predicted_classes))
    target_nm= ['g1']
    target_nm.extend(params.inc_labels)
    
    print(classification_report(actual_classes, predicted_classes, target_names=target_nm))
    print(confusion_matrix(actual_classes, predicted_classes),"\n",classification_report(actual_classes, predicted_classes, target_names=target_nm),file=fl)
    # scores = cross_val_score(model, X, y, scoring='f1_weighted', cv=cv, n_jobs=-1)
    print(scores)
    print('F1 weighted: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


def rand_forest_classifier(x,y,x_test,y_test,fl):
    scaler = MinMaxScaler()
    scaler.fit(x)
    X_tr_norm = scaler.transform(x)
    X_ts_norm = scaler.transform(x_test)

    # n_val =[10, 100, 1000]# np.arange(10, 200, 5) # [10, 100, 1000]
    n_val = [5,50,100,200,400,800]
    min_samples_split_ =np.arange(2, 30, 4)
    max_val = np.arange(2, 20, 4)
    parameters = {'n_estimators':n_val,'max_features' : ['sqrt', 'log2'],'criterion':('gini', 'entropy'),'max_depth':max_val, 'min_samples_split':min_samples_split_}
    
    
    
    rf = RandomForestClassifier()
    clf = RandomizedSearchCV(rf, parameters,cv = 6,return_train_score = True, scoring = 'f1_weighted',verbose=0)
    clf.fit(X_tr_norm, y)
    # plot_search_results(clf)
    print('best parameters : '+str(clf.best_params_))
    print('RF best parameters : '+str(clf.best_params_),file=fl)

    rfc = RandomForestClassifier(n_estimators = clf.best_params_['n_estimators'],
                                criterion = clf.best_params_['criterion'],
                                max_depth = clf.best_params_['max_depth'])
    X_tr_norm = np.vstack((X_tr_norm,X_ts_norm))
    y.extend(y_test)
    kFoldResults(rfc,X_ts_norm,y_test,fl)
    # rfc.fit(X_tr_norm,y)
    # y_pred = rfc.predict(X_ts_norm)
    
    # score = f1_score(y_true = y_test, y_pred = y_pred,average='weighted')
    # print('F1 Score on test: ' + str(score))
    # return score,y_pred




def test(model,data_loader,dataset):
    # THIS TEST FUNCTION IS DIFFERENT
    #ELsewhere I dont need, seprate loss loss info, is just needs total loss info for each prediction to plot
    # but here i need each rep info for comparison among reps
        print("TESTIN\n")
        loss_id_all=[]
        
        losses=[]
        test_loss = 0
        criterion = nn.MSELoss()
        num_batches=len(data_loader)
        targets_all=[]
        predictions_all =[]
        max_loss=0
        ex_count=0
        empty_loss_id = params.loss_id

        #  this variation reduces speed as each data input is processed one at a time but we can get predictions with NO data now, test _data vals cane be removed for batch prediction with data
        with torch.no_grad():
            for batch_id, test_data_num in enumerate(data_loader):
                # print("sheetnum",test_data_num)
                loss_id = copy.deepcopy(empty_loss_id)
                test_data_vals=dataset[test_data_num]
                rep_loss=0
                targets = []
                predictions=[]
                hidden_=None
                # import pdb;pdb.set_trace()
                for steps,iter in enumerate(range(len(test_data_vals['time']))):
                    test_data= {"time": test_data_vals['time'][iter:iter+1], 'landmarks': test_data_vals['landmarks'][iter:iter+1], "output":test_data_vals['output'][iter:iter+1], "col_id":test_data_vals['col_id']} 
                    seq_size = len(test_data['landmarks'])

                    objects,  sender_relations, receiver_relations, relation_info,    external_effect_info, target = tranform_batch_BIN(test_data, params.n_objects,params.object_dim,params.USE_CUDA,0)
                    target = target.to(device)

                    if steps != 0: 
                        # print("1 step")
                        objects=prediction.reshape(-1,params.n_objects,params.object_dim)
                        # hidden_=hidden_.to(device)
                    # print(np.shape(objects),np.shape(sender_relations),np.shape(receiver_relations),np.shape(relation_info),np.shape(external_effect_info))
                    prediction = model(objects,  sender_relations, receiver_relations, relation_info, external_effect_info)

                    # prediction_temp = np.zeros((5,4))
                    # prediction_temp[:,2:4] = prediction
                    # prediction_temp[:,0:2]=objects[0,:,0:2] + prediction_temp[:,2:4]
                    # prediction=torch.from_numpy(prediction_temp)    

                    targets = np.hstack([np.reshape(target.cpu(),(params.n_objects,params.object_dim)),targets]) if len(targets) else np.reshape(target.cpu(),(params.n_objects,params.object_dim))
                    predictions= np.hstack([np.reshape(prediction.cpu(),(params.n_objects,params.object_dim)), predictions]) if len(predictions) else np.reshape(prediction.cpu(),(params.n_objects,params.object_dim))
                    # print("shape target",np.shape(targets))
                    prediction=torch.reshape(prediction,(params.n_objects,params.object_dim))
                    # print(target.is_cuda,prediction.is_cuda)
                    loss = torch.sum(torch.abs(torch.subtract(target,prediction)))
                    losses.append(loss)
                    # print(steps,"next pred loss",loss.item())
                    # if loss.item()>max_loss:
                    #     max_loss=loss.item()
                    #     print(max_loss,test_data_num,ex_count,objects,target,prediction)
                    ex_count+=1
                    rep_loss += loss.item()
                    loss_id = get_landmark_errors(target,prediction,loss_id)
                    # print("loss id",loss_id['total'])
                loss_id_all.append(loss_id)
                targets_all.append(targets)
                predictions_all.append(predictions)
                rep_loss /= (steps+1)
                # print("rep loss",rep_loss,(steps+1))
                test_loss+=rep_loss
                # break
            # print(num_batches,"batch id",(batch_id+1))
            test_loss /= (batch_id+1)
            print("Avg test loss per rep ",test_loss)
        return loss_id_all,targets_all,predictions_all

def svm_classifier(x,y,x_test,y_test):
    # for i in x:
    #     print(np.shape(i))
    
    scaler = MinMaxScaler()
    scaler.fit(x)
    X_tr_norm = scaler.transform(x)
    X_ts_norm = scaler.transform(x_test)

    c_val =[ 50,10,1.0, 0.1, 0.01]# np.arange(1, 100, 2) # [50, 10, 1.0, 0.1, 0.01]
    parameters = {'kernel':('linear', 'poly', 'rbf', 'sigmoid'), 'C':c_val}
    
    # svcc = svm.SVC(kernel = 'rbf', C = 1)
    
    svc = svm.SVC()
    clf = RandomizedSearchCV(svc, parameters,cv = 8,return_train_score = True,scoring = 'f1_weighted')
    clf.fit(X_tr_norm, y)
    # plot_search_results(clf)
    # print("SVM TRAINING RESULTS", clf.cv_results_)#['mean_test_accuracy'],clf.cv_results_['mean_train_accuracy'])
    print('best parameters : '+str(clf.best_params_))
    print('SVM best parameters : '+str(clf.best_params_),file=fl)



    svcc = svm.SVC(kernel = clf.best_params_['kernel'], C = clf.best_params_['C'])
    X_tr_norm = np.vstack((X_tr_norm,X_ts_norm))
    y.extend(y_test)
    kFoldResults(svcc,X_ts_norm,y_test)
    # svcc.fit(X_tr_norm,y)

    # filename = './models/svm_classifier.joblib.pkl'
# joblib.dump(clf, filename, compress=9)
    # svcc = joblib.load(filename)
    # y_pred = svcc.predict(X_ts_norm)
    # score = f1_score(y_true = y_test, y_pred = y_pred,average='weighted')    
    # return score,y_pred
# seed = params.seed_val
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)


# model = BasicInteractionNetworkModule(params.object_dim, params.relation_dim, params.effect_dim, params.external_effect_dim, params.output_dim)
# # model.load_state_dict(torch.load( './models/VISFULL_test_112.pth')) # test gives less corr error then less_train model
# # model.load_state_dict(torch.load( './models/VISFULL_FULLRELL_test_112.pth')) # test gives less corr error then less_train model
# # model.load_state_dict(torch.load( '../models/test_112.pth')) # test gives less corr error then less_train model
# model.load_state_dict(torch.load( '../models/test_'+params.savepath+'.pth',map_location=torch.device(device)))
# model.to(device)




# # model.load_state_dict(torch.load( './models/best.pth'))
# model.eval()

# X=[]
# Y=[]
# X_test=[]
# Y_test=[]

# correct_dataset = LandmarksDataset(xl_file='../data/'+'g1'+'.xlsx',
#                                 root_dir='./')


# cor_train_loader, cor_test_loader,  = split_data(correct_dataset,1,remove_rear=100,train_size=.6) # this split is needed to find best Hyper params for SVM
# cor_loss_id_total,cor_targets,cor_predictions = test(model, cor_train_loader,correct_dataset  )
# print("train cor mean total loss", np.mean(np.sum( [cor_loss_id['total'] for cor_loss_id in cor_loss_id_total]))) # this mean formula for nested list of diferent size is wrong
# cor_loss_id_testtotal,cor_testtargets,cor_testpredictions = test(model, cor_test_loader,correct_dataset  )
# print("test cor mean total loss",np.mean(np.sum([cor_loss_id['total'] for cor_loss_id in cor_loss_id_testtotal])))
# print("test cor mean total loss",np.mean(np.sum([cor_loss_id['total'] for cor_loss_id in cor_loss_id_testtotal])),file=fl)

# print("g1 Train len %d test len %d"%(len(cor_targets),len(cor_testtargets)))

# # # import pdb;pdb.set_trace()
# for index,cor_loss_id in enumerate(cor_loss_id_total):
#     # print(1,cor_loss_id,"\n")
#     enc =dtft_conversion(cor_loss_id)
#     # enc.extend(dtft_pos_conversion(cor_targets[index]))
#     X.append(enc)
#     Y.append(0)
# for index,cor_loss_id in enumerate(cor_loss_id_testtotal):
#     enc =dtft_conversion(cor_loss_id)
#     # enc.extend(dtft_pos_conversion(cor_testtargets[index]))

#     X_test.append(enc)
#     Y_test.append(0)

# inc_labels=params.inc_labels#
# for iter,inc in enumerate(inc_labels):
#     incorrect_dataset = LandmarksDataset(xl_file='../data/'+inc+'.xlsx',
#                                         root_dir='./')
#     inc_train_loader, inc_test_loader,  = split_data(incorrect_dataset,1,remove_rear=100,train_size=.6)


#     inc_loss_id_total,inc_targets,inc_predictions = test(model, inc_train_loader,incorrect_dataset )
#     print(inc,"train incor mean total loss",np.mean(np.sum([inc_loss_id['total'] for inc_loss_id in inc_loss_id_total])))

#     inc_loss_id_testtotal,inc_testtargets,inc_testpredictions = test(model, inc_test_loader,incorrect_dataset )

#     print(inc, " Train len %d test len %d"%(len(inc_targets),len(inc_testtargets)))
#     print(inc,"test inc mean total loss",np.mean(np.sum([inc_loss_id['total'] for inc_loss_id in inc_loss_id_testtotal])))
#     # nose':[],'lh':[],'rh':[],'lk':[],'rk':[],'total':[]

#     for index,inc_loss_id in enumerate(inc_loss_id_total):
#         # print(0,inc_loss_id)
#         enc = dtft_conversion(inc_loss_id)
#         # enc.extend(dtft_pos_conversion(inc_targets[index]))

#         X.append(enc)
#         Y.append(iter+1)
#         # Y.append(1)

#     for index,inc_loss_id in enumerate(inc_loss_id_testtotal):
     
#         enc = dtft_conversion(inc_loss_id)
#         # enc.extend(dtft_pos_conversion(inc_testtargets[index]))

#         X_test.append(enc)
#         Y_test.append(iter+1)
#         # Y_test.append(1)

# # X, Y = shuffle(X, Y, random_state=0)
# # KNN CLASSISIFER
# print("SEED,model = ",seed)
# # import pdb;pdb.set_trace()
# print("Train length %d, Test length %d, Test Cor %d" %(len(X),len(X_test),np.count_nonzero(np.array(Y_test)==0)))
# # print('Hyperparams lr %f window %d n_layers %d  epoch %d: ' %(params.learning_rate,params.dtft_width,params.n_layers,params.n_epoch),file=fl)

# file_str ="Hyperparams "
# for key,value in vars(params.args).items():
#     file_str +=  key
#     file_str += "="
#     file_str += str(value)
# print(file_str,file=fl)
# # svm_classifier(X,Y,X_test,Y_test)
# print("Random forest results",file=fl)
# # rand_forest_classifier(copy.copy(X),copy.copy(Y),copy.copy(X_test),copy.copy(Y_test))
# rand_forest_classifier(X,Y,X_test,Y_test)

# # print("SVM results",file=fl)
# # svm_classifier(X,Y,X_test,Y_test)
# # score,y_pred = svm_classifier(X,Y,X_test,Y_test)
# # print('F1 Score on test: ' + str(score) + " accruacy " + str(accuracy_score(y_true = Y_test, y_pred = y_pred,)))
# # print(Y_test)
# # # import pdb;pdb.set_trace()
# # print(confusion_matrix(Y_test,y_pred))
# # print(seed,classification_report(Y_test, y_pred, target_names=['g1', 'b1','b2','b3','b4','b5']),file=fl)
# # # print(classification_report(Y_test, y_pred, target_names=['g1', 'b1']),file=fl)

# # print(seed," F1=",score,"\n",cm)
# # print("THESE ARE RESULTS WITH 2 encodings my dtft encoder")
# # filename = './models/svm_classifier.joblib.pkl'
# # joblib.dump(clf, filename, compress=9)
# # clf2 = joblib.load(filename)

# fl.close()
