###
# file name: model.py (Wavemotion-lightning)
# Author: Dr. Xiaoxian Guo @ Shanghai Jiao Tong University, China
# Email: xiaoxguo@sjtu.edu.cn
# Website: https://naoce.sjtu.edu.cn/teachers/9004.html
# Github: https://github.com/XiaoxG/waveMotion-lightning/
# Create date: 2021/10/08
# Wavemotion-lightning-- a demonstration code for wave motion prediction of an offshore platform. 
# Refers to preprint Arvix: "Probabilistic prediction of the heave motions of a semi-submersible by a deep learning model"
###

import torch
import torch.nn as nn
import sys

class NetworkRNN(nn.Module):
    def __init__(self, params, DEVICE):
        super().__init__()
        self.batch_size = params.training.batch_size      
        input_dim = params.model.input_dim
        output_size = params.data.forward_step
        hidden_dim = params.model.hidden_dim
        num_rnn_layers = params.model.num_rnn_layers
        num_FC_layers = params.model.num_FC_layers
        num_FC_neurons = params.model.num_FC_neurons
        dropout = params.model.dropout
        activation = params.model.activation
        self.rnn_type = params.model.rnn_type
        self.device = DEVICE
        # self.visualizer = None
        # self.xflow = None

        if num_FC_neurons == -1:
            num_FC_neurons = hidden_dim
        if self.rnn_type == 'gru':
            self.rnnInput = nn.GRU(input_dim, hidden_dim, batch_first = True)
            self.register_buffer("hidden_cell_input", torch.zeros(1,self.batch_size, hidden_dim))
            # self.hidden_cell_input = torch.zeros(1,self.batch_size, hidden_dim).type_as(ref)
            rnnlayers = []
            for _ in range(num_rnn_layers-1):
                rnnlayers.append(nn.GRU(hidden_dim, hidden_dim, batch_first = True))
            self.register_buffer("hidden_cell_layers", self._create_hiddens(rnnlayers, self.batch_size))

        elif self.rnn_type == 'lstm':
            self.rnnInput = nn.LSTM(input_dim, hidden_dim, batch_first = True)
            self.register_buffer("hidden_cell_input1", torch.zeros(1,self.batch_size,hidden_dim))
            self.register_buffer("hidden_cell_input2", torch.zeros(1,self.batch_size,hidden_dim))
            #self.hidden_cell_input = (torch.zeros(1,self.batch_size,hidden_dim),
            #                torch.zeros(1,self.batch_size,hidden_dim))
            rnnlayers = []
            for _ in range(num_rnn_layers-1):
                rnnlayers.append(nn.LSTM(hidden_dim, hidden_dim, batch_first = True))
            if rnnlayers:
                hidden_cells1, hidden_cells2 = self._create_hiddens(rnnlayers, self.batch_size)
                self.register_buffer("hidden_cell_layers1", hidden_cells1)
                self.register_buffer("hidden_cell_layers2", hidden_cells2)
            else:
                self.register_buffer("hidden_cell_layers", self._create_hiddens(rnnlayers, self.batch_size))

        else:
            print('rnn type unknown, please check.')
            sys.exit()

        self.rnn_net = nn.ModuleList(rnnlayers)
        
        
        self.decoder = nn.Sequential(
            self._densblock(hidden_dim, num_FC_neurons, dropout = dropout, activation = activation),
            *[self._densblock(num_FC_neurons,num_FC_neurons, dropout = dropout, activation = activation) for i in range(num_FC_layers-2)],
            nn.Linear(num_FC_neurons, output_size)
        )
        #self.dropoutlayer = nn.Dropout(0.1)

    def forward(self, x):
        if self.rnn_type == 'gru':
            x, self.hidden_cell_input = self.rnnInput(x, self.hidden_cell_input)
            for i, ilayer in enumerate(self.rnn_net):
                x_out, self.hidden_cell_layers[i,:,:,:] = ilayer(x, self.hidden_cell_layers[i,:,:,:])
                x = x + x_out # apply short cut
        if self.rnn_type == 'lstm':
            x, (self.hidden_cell_input1,self.hidden_cell_input2) = self.rnnInput(x, (self.hidden_cell_input1,self.hidden_cell_input2))
            for i, ilayer in enumerate(self.rnn_net):
                x_out, (self.hidden_cell_layers1[i,:,:,:],self.hidden_cell_layers2[i,:,:,:]) = ilayer(x, (self.hidden_cell_layers1[i,:,:,:],self.hidden_cell_layers2[i,:,:,:]))
                # x = x + x_out # apply short cut
                x = x_out
        x = self.decoder(x[:,-1,:])
        x = x.view(self.batch_size,-1)
        return x

    @staticmethod
    def _densblock(in_f, out_f, dropout = 0.0, activation = 'tanh'):
        activs = nn.ModuleDict([
            ['tanh', nn.Tanh()],
            ['relu', nn.ReLU()],
            ['lrelu', nn.LeakyReLU()]
        ])
        return nn.Sequential(
            nn.Linear(in_f, out_f),
            #nn.BatchNorm1d(out_f),
            activs[activation],
            nn.Dropout(dropout))

    @staticmethod 
    def _create_hiddens(rnn_layers, batch_size):
        n_layers = len(rnn_layers)
        if rnn_layers == []:
            hidden_cells = None
            return hidden_cells
        else:
            hidden_dim = rnn_layers[0].hidden_size
            mode = rnn_layers[0].mode
            if mode == 'GRU':
                hidden_cells = torch.zeros(n_layers, 1,batch_size, hidden_dim)
                return hidden_cells 
            elif mode == 'LSTM':
                hidden_cells1 = torch.zeros(n_layers, 1,batch_size, hidden_dim)
                hidden_cells2 = torch.zeros(n_layers, 1,batch_size, hidden_dim)
                return hidden_cells1, hidden_cells2
            else:
                print('rnn type unknow, please check.')
                sys.exit()
            
if __name__ == '__main__':
    pass