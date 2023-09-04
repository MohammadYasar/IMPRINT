import os, sys, datetime, glob, random, math, time
import torch, copy
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from models.attention_mechanisms import KeylessAttention

def to_np(x):
    return  x.data.cpu().numpy()

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()#cuda()
        if x.size(0) == 1:
            x = x.permute(1,0,2)
        else:
            x = x.squeeze(0).permute(1,0,2)

    return Variable(x).cuda()

#Encoder Network
class Q_net(nn.Module):

    def __init__(self, X_dim, N, z_dim, cat_dim=27, num_ftrs=1024,conditional=False, use_flow = True):
        super(Q_net, self).__init__()
        self.conditional = conditional
        self.feature_embed_size = 512#256#64#128
        self.lstm_hidden_size = 128

        self.encoder_1 = EncoderBlock(X_dim//2, N, self.feature_embed_size).cuda()
        self.encoder_2 = EncoderBlock(X_dim//2, N, self.feature_embed_size).cuda()

        #self.encoderattention = nn.MultiheadAttention(2*(N + X_dim//6), num_heads=5).cuda()

        self.encoderattention_agent1 = KeylessAttention(N//2) # nn.MultiheadAttention(N//2, num_heads=1).cuda()
        self.encoderattention_agent2 = KeylessAttention(N//2) # nn.MultiheadAttention(N//2, num_heads=1).cuda()
        att_dim = 2*N
            
        self.keyless_att_context = KeylessAttention(N//2)


        self.num_ftrs = 2048 #num_ftrs#512 # 2048

        # RGB
        self.batch_first = True
        self.lstm_dropout = 0.2
        self.dropout = 0.2
        self.fe_relu = nn.ReLU()
        self.fe_dropout = nn.Dropout(p=self.dropout)
        self.fe_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fe_fc1 = nn.Linear(self.num_ftrs, self.feature_embed_size*2)
        self.fe_fc2 = nn.Linear(self.feature_embed_size*2, self.feature_embed_size)
        self.lstm = nn.GRU(input_size=self.feature_embed_size,
                            hidden_size=N//2,
                            batch_first=self.batch_first,
                            dropout=self.lstm_dropout)
        

        # Flow
        self.flow_lstm_dropout = 0.2
        self.flow_dropout = 0.2
        self.flow_fe_fc1 = nn.Linear(self.num_ftrs, self.feature_embed_size*2)
        self.flow_fe_fc2 = nn.Linear(self.feature_embed_size*2, self.feature_embed_size)

        self.hidden_projection1 = nn.Linear(N, N//2)
        self.hidden_projection2 = nn.Linear(N, N//2)

        self.cos = torch.nn.CosineSimilarity(dim=2, eps=1e-16)

    def forward(self, x, vision_input, flow_input=None, train=True, dropout_pathway=0.5):
        agent_1 = 1
        agent_2 = 2
        div = 75
        #flow features
        """
        flow_contiguous = flow_input.view(flow_input.size(0), flow_input.size(1), -1)
        embed = self.flow_fe_fc1(flow_contiguous)
        embed = F.relu(self.fe_dropout((embed)))
        embed = self.flow_fe_fc2(embed)
        embed = F.relu(self.fe_dropout((embed)))
        """
        vision_contiguous = vision_input.view(-1, vision_input.size(-3), vision_input.size(-2), vision_input.size(-1)).contiguous()
        embed = self.fe_pool(vision_contiguous)
        embed = embed.view(vision_input.size(0), vision_input.size(1), -1)

        embed = self.fe_fc1(embed)
        embed = F.relu(self.fe_dropout((embed)))
        embed = self.fe_fc2(embed)
        embed = F.relu(self.fe_dropout((embed)))
        
        x = x.cuda()
        # agent 1 encoder
        x_pos = x[:,:, :75*agent_1]; x_vel = x[:,:, 150:225]; x_acc = x[:, :, 300:375]
        hidden_1, gru_1 = self.encoder_1(x_pos, x_vel, x_acc, embed)

        # agent 2 encoder
        x_pos = x[:,:, 75:75*agent_2]; x_vel = x[:,:, 225:150*agent_2]; x_acc = x[:, :, 375:]
        hidden_2, gru_2 = self.encoder_2(x_pos, x_vel, x_acc, embed)

        # agent 1/2 self
        hidden_1self, hidden_1selfweights = self.encoderattention_agent1(hidden_1)#[0] # + hidden_1
        hidden_2self, hidden_2selfweights = self.encoderattention_agent2(hidden_2)#[0] # + hidden_2

        # agent 1/2 interaction
        hidden_1cond, hidden1cond_weights = self.encoderattention_agent2(torch.cat((hidden_1, hidden_2), dim=0))#[0] # + hidden_1
        hidden_2cond, hidden2cond_weights  = self.encoderattention_agent1(torch.cat((hidden_2, hidden_1), dim=0))#[0] # + hidden_2
        # print ("hidden_1 cond :", hidden1cond_weights, "hidden_2 cond :",hidden2cond_weights)
        
        # summing over final dim
        hidden_1self = torch.sum(hidden_1self, dim=0).unsqueeze(0)
        hidden_2self = torch.sum(hidden_2self, dim=0).unsqueeze(0)
        hidden_1cond = torch.sum(hidden_1cond, dim=0).unsqueeze(0)
        hidden_2cond = torch.sum(hidden_2cond, dim=0).unsqueeze(0)

        # summing over different representation
        hiddenstate_1, hidden_1intweights = self.encoderattention_agent1(torch.cat((hidden_1self, hidden_1cond), dim=0))
        hiddenstate_2, hidden_2intweights = self.encoderattention_agent2(torch.cat((hidden_2self, hidden_2cond), dim=0)) #torch.add(hidden_2self, hidden_2cond)
        # print ("hidden_1 weights :", hidden_1intweights, "hidden_2 weights :",hidden_2intweights)

        
        vision_output, vision_features = self.lstm(embed) 
        
        hiddenstate_1 = torch.cat((hiddenstate_1, vision_features), dim=0)
        hiddenstate_1, hidden_1vizweights = self.keyless_att_context(hiddenstate_1) 
        hiddenstate_1 = hiddenstate_1 + vision_features
        
        hiddenstate_2 = torch.cat((hiddenstate_2, vision_features), dim=0)
        hiddenstate_2, hidden_2vizweights = self.keyless_att_context(hiddenstate_2) #+ vision_features
        hiddenstate_2 = hiddenstate_2 + vision_features
        
        # print ("viz_1 weights :",hidden_1vizweights, "viz_2 weights :",hidden_2vizweights)
        orthogonal_loss = 0# abs(self.cos(vision_features, hiddenstate_1))

        return hiddenstate_1, gru_1.cuda(), hiddenstate_2, gru_2.cuda(), orthogonal_loss

class EncoderBlock(nn.Module):
    def __init__(self, X_dim, N, context_size):
        super(EncoderBlock, self).__init__()
        self.lin1 = nn.GRU(X_dim//3, N//2, num_layers=1, dropout=0.1)
        self.lin1_vel = nn.GRU(X_dim//3, N//2, num_layers=1, dropout=0.1)
        self.lin1_acc = nn.GRU(X_dim//3, N//2, num_layers=1, dropout=0.1)
        self.vis_enc = nn.GRU(context_size, N//2, num_layers=1, dropout=0.7)

    def forward(self, x_pos, x_vel, x_acc, vis):
        vis = vis.permute(1,0,2)
        _, x_pos = self.lin1(x_pos)
        _, x_vel = self.lin1_vel(x_vel)
        _, x_acc = self.lin1_acc(x_acc)
        _, vis = self.vis_enc(vis)
        vis = vis + x_pos
        x_acc = x_acc + x_vel
        # concat along the first axis -> (3, B, C)
        x = torch.cat((x_pos, x_vel, x_acc, vis), 0)

        return x, self.lin1


#Decoder Network
class P_net(nn.Module):
    def __init__(self, X_dim, N, z_dim, c_dim, num_layers=1):
        super(P_net, self).__init__()
        self.num_layers = num_layers
        self.output_size = X_dim

        self.decoder1 = DecoderBlock(X_dim//2, N, z_dim, c_dim, num_layers=1).cuda()
        self.decoder2 = DecoderBlock(X_dim//2, N, z_dim, c_dim, num_layers=1).cuda()

        self.skel_dict = self.get_skelIndex()

    def forward(self, inputs, targets, hidden_1, encoder_rnn_1, hidden_2, encoder_rnn_2, temperature=0.5,test=True):
        self.decoder1 = self.decoder1.cuda(); self.decoder2 = self.decoder2.cuda()

        n_steps = inputs.size(0)#*2
        self.rnn_1 = encoder_rnn_1.cuda()
        self.rnn_2 = encoder_rnn_2.cuda()
        outputs = Variable(torch.zeros(n_steps, inputs.size(1), self.output_size), requires_grad=False)
        hidden = None

        input = inputs[-1,:,:self.output_size].unsqueeze(0).cuda() #Variable(torch.zeros(inputs.size(1), self.output_size), requires_grad=False).cuda()#inputs[0]
        input_1 = input[:,:, :75].cuda()
        input_2 = input[:,:, 75:].cuda()

        hidden_latent_1 = (hidden_1)
        hidden_latent_2 = (hidden_2)


        hidden = 0
        for i in range(n_steps):
            output_1, hidden_1 = self.decoder1(i, input_1, hidden_1, hidden_latent_1, self.skel_dict, self.rnn_1)
            output_2, hidden_2 = self.decoder2(i, input_2, hidden_2, hidden_latent_2, self.skel_dict, self.rnn_2)

            outputs[i] = torch.cat((output_1, output_2), -1)
            if test == True:
                input_1, input_2 = output_1, output_2
            else:
                input_1, input_2 = output_1, output_2
                #print ("inside teacher forcing ", targets[i].shape)
                use_teacher_forcing = random.random() <temperature# 0.5#0.85#0.9
                if use_teacher_forcing:
                    input = targets[i]#.unsqueeze(0)
                    #input_1, input_2 = targets[i:,:,:75].unsqueeze(0), targets[i:,:,75:].unsqueeze(0)
                    input_1, input_2 = targets[i,:,:75].unsqueeze(0), targets[i,:,75:].unsqueeze(0)

        return outputs.squeeze(1)

    def get_skelIndex(self):
        skel_dict = dict()
        skel_dict["head"] = 4; skel_dict["neck"] = 3; skel_dict["spine"] = 21; skel_dict["middleSpine"] = 2; skel_dict["baseSpine"] = 1
        skel_dict["leftShoulder"] = 5; skel_dict["leftElbow"] = 6; skel_dict["leftWrist"] = 7; skel_dict["leftHand"] = 8; skel_dict["leftThumb"] = 23; skel_dict["leftTip"] = 22
        skel_dict["rightShoulder"] = 9; skel_dict["rightElbow"] = 10; skel_dict["rightWrist"] = 11; skel_dict["rightHand"] = 12; skel_dict["rightThumb"] = 25; skel_dict["rightTip"] = 24
        skel_dict["leftHip"] = 13; skel_dict["leftKnee"] = 14; skel_dict["leftAnkle"] = 15; skel_dict["leftFoot"] = 16;
        skel_dict["rightHip"] = 17; skel_dict["rightKnee"] = 18; skel_dict["rightAnkle"] = 19; skel_dict["rightFoot"] = 20;

        for key in skel_dict.keys():
            skel_dict[key] -= 1

        return skel_dict


class DecoderBlock(nn.Module):
    def __init__(self, X_dim, N, z_dim, c_dim, num_layers=1):
        super(DecoderBlock, self).__init__()
        self.upperbody_dims = 12; self.hand_dims = 12; self.leg_dims = 12 #dims 
        hidden_units = 32; joint_output = 3; joint_input = 6

        self.decoderattention = nn.MultiheadAttention(N//2, num_heads=1) #KeylessAttention(N//2).cuda()
        self.ln1 = nn.LayerNorm(X_dim + z_dim + c_dim)
        self.hidden2latent = nn.Linear(X_dim*2, X_dim)
        self.lin1 = nn.Linear(N//2, X_dim)
        self.rnn =None# nn.GRU(X_dim, X_dim, num_layers=1, dropout=0.1)

    def forward(self,index, input, hidden, hidden_latent, skel_dict, rnn):
        self.rnn = rnn.cuda(); self.decoderattention = self.decoderattention.cuda()
        last_input = input

        hidden, _ = self.decoderattention(hidden, hidden_latent, hidden_latent)

        output, hidden = self.rnn(input, hidden)
        output = self.lin1(output)
        x = output + last_input#.squeeze(0)
        return x, hidden


class D_net_gauss(nn.Module):
    def __init__(self, N, z_dim):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, N).cuda()
        self.lin2 = nn.Linear(N, N).cuda()
        self.lin3 = nn.Linear(N, 1).cuda()
    def forward(self, x):
        x = x.cuda()
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        return torch.sigmoid(self.lin3(x))

class D_net_cat(nn.Module):
    def __init__(self, N, cat_dim):
        super(D_net_cat, self).__init__()
        self.lin1 = nn.Linear(cat_dim, N).cuda()
        self.lin2 = nn.Linear(N, N).cuda()
        self.lin3 = nn.Linear(N, 1).cuda()
    def forward(self, x):
        x = x.cuda()
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        return torch.sigmoid(self.lin3(x))
