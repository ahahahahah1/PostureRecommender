# Currently using this in start.py
# When an object of BIN is instantiated, it initializes dimension parameters. It also loads the data from data directory.

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
import copy
from BIN_module_dropout import BasicInteractionNetworkModule
from dataset_rep import LandmarksDataset
from utils_rep import *

import params
class BasicInteractionNetworks:
    def __init__(self,n_obj, out_dim,file_n): #mode is either test/train
        self.n_objects = n_obj
        self.object_dim = params.object_dim
        self.output_dim = params.output_dim
        self.n_relations = params.n_relations
        self.relation_dim = params.relation_dim
        self.external_effect_dim = params.external_effect_dim
        self.effect_dim = params.effect_dim

        self.USE_CUDA = params.USE_CUDA
        
        #  for testing instead of handling split at code side i can also handle it at csv/xls side

        self.n_epoch = params.n_epoch
        self.batch_size = params.batch_size

        self.train_dataset = LandmarksDataset(xl_file='../data/train/' + file_n +'.xlsx',
                                    root_dir='./')
        # self.test_dataset = LandmarksDataset(xl_file='../data/13_2_SG_'+dttype+'.xlsx',
        #                             root_dir='./')
        self.test_dataset = LandmarksDataset(xl_file='../data/test/' + file_n + '.xlsx',
                                    root_dir='./')
        
        # self.test_dataset = self.train_dataset
        self.train_indices, self.val_indices, = split_data(self.train_dataset,self.batch_size,remove_rear=100,train_size=0.9,val_size=0)#this serves as train
        self.test_indices = list(range(0,len(self.test_dataset)))
        # self.train_loader, self.test_loader,  = split_data(self.test_dataset,self.batch_size,remove_rear=100,train_size=.6)# this serves as test
        self.train_loader = DataLoader(self.train_indices,  shuffle=True)
        self.test_loader = DataLoader(self.test_indices, batch_size=1, shuffle=False)
        self.val_loader = DataLoader(self.val_indices, batch_size=1, shuffle=False)
    
    def train(self,):
        interaction_network = BasicInteractionNetworkModule(self.object_dim, self.relation_dim, self.effect_dim, self.external_effect_dim, self.output_dim)
        if self.USE_CUDA:
            interaction_network = interaction_network.cuda()
        # optimizer = optim.Adam(interaction_network.parameters(), lr = params.learning_rate)
        optimizer = torch.optim.AdamW(interaction_network.parameters())
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=params.learning_rate, steps_per_epoch=len(self.train_loader), epochs=params.n_epoch)

        criterion = nn.MSELoss()
        noise =1
        last_loss=100
        for epoch in range(self.n_epoch):
            b_loss=0
            # if epoch > self.n_epoch*.6:
            #     noise=min(0,1 - 1.1*(epoch)/self.n_epoch)
            for batch_id, train_data_rep in enumerate(self.train_loader):
                # print(train_data)
                train_data=self.train_dataset[train_data_rep]
                # import pdb;pdb.set_trace()
                objects,  sender_relations, receiver_relations, relation_info,    external_effect_info, target = tranform_batch_BIN(train_data, self.n_objects,self.object_dim,self.USE_CUDA,noise)

                # print(np.shape(objects),np.shape(sender_relations),np.shape(receiver_relations),np.shape(relation_info),np.shape(external_effect_info))
                predicted = interaction_network(objects,  sender_relations, receiver_relations, relation_info, external_effect_info)
                
                loss = criterion(predicted, target)
               
                optimizer.zero_grad()
                loss.backward()
                clipping_value = 1. # arbitrary value of your choosing
                torch.nn.utils.clip_grad_norm_( interaction_network.parameters(), clipping_value )
                optimizer.step()
                scheduler.step()
                b_loss+=loss.item()
            print("Epoch",epoch,(b_loss/(1+batch_id)))
            # if epoch %30==0:
            #     _,_,_,current_loss = self.test(interaction_network,self.val_loader)
            #     # self.test(interaction_network,self.test_loader)
            #     interaction_network.train()

            if params.early_stopping :
                _,_,_,current_loss = self.test(interaction_network,self.val_loader)
                if current_loss > last_loss:
                    trigger_times += 1
           
                    if trigger_times >= params.patience:
                        print('Early stopping!\nStart to test process.')
                        interaction_network.load_state_dict(torch.load( '../models/ESmodels/ES_'+str(params.seed)+'.pth')) # test gives less corr error then less_train model

                        return interaction_network

                else:
                    print('trigger times: 0')
                    trigger_times = 0

                last_loss = current_loss
                print("CURRENT BEST LOSS ",current_loss)
                torch.save(interaction_network.state_dict(), '../models/test_'+str(params.seed)+'.pth')

        return interaction_network

    def test(self,interaction_network,data_loader,dataset): #data_type decides whether we are testing on validation data or testing data
        print("TESTING\n")
        interaction_network.eval()
        empty_loss_id = params.loss_id
        loss_id = copy.deepcopy(empty_loss_id)
        losses=[]
        test_loss = 0
        criterion = nn.MSELoss()
        num_batches=len(data_loader)
        targets = []
        predictions=[]
        max_loss=0
        ex_count=0
        #  this variation reduces speed as each data input is processed one at a time but we can get predictions with NO data now, test _data vals cane be removed for batch prediction with data
        with torch.no_grad():
            for batch_id, test_data_num in enumerate(data_loader):
                test_data_vals=dataset[test_data_num]
                rep_loss=0
                
                # import pdb;pdb.set_trace()
                for steps,iter in enumerate(range(len(test_data_vals['time']))):
                    test_data= {"time": test_data_vals['time'][iter:iter+1], 'landmarks': test_data_vals['landmarks'][iter:iter+1], "output":test_data_vals['output'][iter:iter+1], "col_id":test_data_vals['col_id']} 
                    objects,  sender_relations, receiver_relations, relation_info,    external_effect_info, target = tranform_batch_BIN(test_data, self.n_objects,self.object_dim,self.USE_CUDA,0)
                    if steps >0 : 
                        # print("1 step")
                        objects=prediction.reshape(-1,self.n_objects,self.object_dim)
                    # print(np.shape(objects),np.shape(sender_relations),np.shape(receiver_relations),np.shape(relation_info),np.shape(external_effect_info))
                    prediction = interaction_network(objects,  sender_relations, receiver_relations, relation_info, external_effect_info)
                    # print("vals test ",objects,prediction,target)
                    # print("steps",objects,target,prediction)

                    targets.extend(np.reshape(target.cpu(),(-1,self.n_objects,self.object_dim)))
                    predictions.extend(np.reshape(prediction.cpu(),(-1,self.n_objects,self.object_dim)))
                    # loss = criterion(prediction, target)
                    loss=torch.sum(torch.abs(torch.subtract(target,prediction)))
                    losses.append(loss)
                    # print(steps,"next pred loss",loss.item())
                    # if loss.item()>max_loss:
                    #     max_loss=loss.item()
                    #     print(max_loss,test_data_num,ex_count,objects,target,prediction)
                    ex_count+=1
                    rep_loss += loss.item()
                    loss_id = get_landmark_errors(target,prediction,loss_id)
                    # print("loss id",loss_id['total'])
                rep_loss /= (steps+1)
                print("rep loss",rep_loss,(steps+1))

                test_loss+=rep_loss
            print("Batch ID = ", batch_id)
            test_loss /= (batch_id+1)
            print("avg test loss BIN.py",test_loss)
        return loss_id,targets,predictions,test_loss
