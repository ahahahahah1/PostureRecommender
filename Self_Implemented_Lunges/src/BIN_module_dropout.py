# This module gives the declarations for a RelationalModel and ObjectModel, and combining them, resulting in the
# InteractionNetwork model. An additional change (from the original model described in the paper) is that Dropout
# layers have been added in the NN to improve accuracy



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import params

class RelationalModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(RelationalModel, self).__init__()
        
        self.output_size = output_size
        self.dropout_p = params.dropout_rate
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout( self.dropout_p),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout( self.dropout_p),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout( self.dropout_p),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )
    
    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_relations, input_size]
        Returns:
            [batch_size, n_relations, output_size]
        '''
        batch_size, n_relations, input_size = x.size()
        x = x.view(-1, input_size)
        x = self.layers(x)
        x = x.view(batch_size, n_relations, self.output_size)
        return x




class ObjectModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ObjectModel, self).__init__()
        self.dropout_p = 0.0

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout( self.dropout_p),

# addeed new layer
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout( self.dropout_p),

            nn.Linear(hidden_size, output_size), #xord, y cord, speedX and speedY
        )
        
    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_objects, input_size]
        Returns:
            [batch_size * n_objects, 2] speedX and speedY
        '''
        input_size = x.size(2)
        x = x.view(-1, input_size)
        return self.layers(x)




class BasicInteractionNetworkModule(nn.Module):
    def __init__(self, object_dim, relation_dim, effect_dim, external_effect_dim, output_dim):
        super(BasicInteractionNetworkModule, self).__init__()
        
        self.relational_model = RelationalModel(2*object_dim + relation_dim, effect_dim, 256 )
        self.object_model     = ObjectModel(object_dim + effect_dim + external_effect_dim, 256, output_dim)
    
    def forward(self, objects,  sender_relations, receiver_relations, relation_info, external_effect_info):
        # print(np.shape(sender_relations),np.shape(objects),np.shape(relation_info))
        senders   = sender_relations.permute(0, 2, 1).bmm(objects)
        receivers = receiver_relations.permute(0, 2, 1).bmm(objects) # for marshalling function m - the next concat
        effects = self.relational_model(torch.cat([senders, receivers, relation_info], 2)) # Phi_R
        effect_receivers = receiver_relations.bmm(effects)
        # print(effect_receivers)
        predicted = self.object_model(torch.cat([objects, external_effect_info, effect_receivers], 2))# phi_O
        return predicted
