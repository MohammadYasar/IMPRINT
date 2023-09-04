import sys
import pickle
import pandas as pd
import numpy as np
import argparse

from torch.utils.tensorboard import SummaryWriter
from dataset.ntu import DataReader
from models.model import *



video_wd = "/project/CollabRoboGroup/msy9an/data/ntu/fe_embed"

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('action', metavar='N', type=int, nargs='+', help='an integer for the accumulator')
args = parser.parse_args()


use_flow = False
warm_start = False
use_vision = True
conditional = False
orthogonal = False
use_i3d = False


ACTION = "A0{}".format(args.action[0])
hidden_sizes = [512] 
embed_sizes =  [512]
num_layers= [1]
n_epochs = 300
grad_clip = 1.0

n_characters = 75*3*2
INPUT_DIM = n_characters
BATCH_SIZE = 16#256#
log_every = 1
save_every = 30

data_wd = os.path.join(os.getenv("HOME"), "ntu","ntu_1")
print (data_wd)
annotations_dir = "" #relative
data_dir = "videos" #relative
dre = DataReader(data_wd,video_wd, annotations_dir, data_dir, batch_size=BATCH_SIZE, action=ACTION)
train_iterator, val_iterator, test_iterator = dre.iterateData("")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save(module, model_name):
    save_filename = '{}.pt'.format(model_name)
    torch.save(module, os.path.join(log_dir, save_filename))
    print('Saved as %s' % save_filename)

def restore_pose(actual_pose, predicted_pose, transform=True):
    """
    Loads scaler from pickle and restores the poses
    """
    #actual_pose = actual_pose.permute(1,0,2); predicted_pose = predicted_pose.permute(1,0,2)
    #print (actual_pose.shape, predicted_pose.shape)
    predicted_pose = predicted_pose.unsqueeze(1) if len(predicted_pose.size()) == 2 else predicted_pose
    actual_pose, predicted_pose =  actual_pose[:,:,:75*2], predicted_pose[:,:,:75*2]
    if transform == True:
        file = open(("norm_scaler{}.p".format(ACTION)), 'rb')
        norm_scaler = pickle.load(file)
        file.close()

        actual_pose = actual_pose.permute(1,0,2).cpu().detach().numpy(); predicted_pose = predicted_pose.permute(1,0,2).cpu().detach().numpy()
        for i in range(actual_pose.shape[0]):
            actual_pose[i] = norm_scaler.inverse_transform(actual_pose[i])
            predicted_pose[i] = norm_scaler.inverse_transform(predicted_pose[i])

        actual_pose, predicted_pose = torch.from_numpy(actual_pose), torch.from_numpy(predicted_pose)

    return actual_pose, predicted_pose

def eucdlidean(actual_pose, predicted_pose):
    euc = 0
    time_steps=  predicted_pose.shape[1]
    mse = (actual_pose-predicted_pose)**2
    mse = mse.reshape(-1, mse.size(-1))
    euc_tensor = torch.FloatTensor(mse.size(0), mse.size(-1)//3)
    for j in range(0,mse.size(-1), 3):
        euc_tensor[:,j//3] = (mse[:,j] + mse[:,j+1] + mse[:,j+2])

    euc_tensor = euc_tensor.reshape(-1, time_steps, euc_tensor.size(-1)).permute(1, 0, 2)
    euc_tensor = torch.mean(euc_tensor, [1,2], True).squeeze()

    return euc_tensor#.mean()



def readVid(video_file):
    
    #print (video_file, torch.cuda.is_available())
    torch.cuda.empty_cache()
    video_pt = torch.load(video_file, map_location='cuda:0')
    video_pt = video_pt.to('cpu')
    #print (video_file, video_pt.shape)
    
    return video_pt.detach().cpu().numpy()

    
def newtemporalizeVideo(video_file, video_indices):
    temporalized_video = list()
    # print(video_indices)
    previous_videofile = ""
    for i, video_index in enumerate(video_indices):
        # print (i, video_index)
        iter_local_start = (video_index[0].detach().numpy())
        iter_local_end = (video_index[1].detach().numpy()) +1
        # print (iter_local_start, iter_local_end)
        iter_local_start, iter_local_end = int(np.ceil(iter_local_start/3)), int(np.ceil(iter_local_end/3))
        # print (iter_local_start, iter_local_end)
        if video_file[i]!= previous_videofile:
            video_session = readVid(video_file[i])
        video_chunk = video_session[iter_local_start:iter_local_end]
        # print (iter_local_start, iter_local_end, video_session.shape, video_chunk.shape)
        temporalized_video.append(video_chunk)
        previous_videofile = video_file[i]

    temporalized_video = np.array(temporalized_video)#.reshape(BATCH_SIZE, 15, 4, video_session.shape[1], video_session.shape[2], video_session.shape[3])
    temporalized_video = torch.FloatTensor(temporalized_video)#.float()
    # print (temporalized_video.shape)
    return temporalized_video


def newtemporalizeVideoi3d(video_file, video_indices):
    temporalized_video = list()
    # print(video_indices)
    for i, video_index in enumerate(video_indices):
        # print (i, video_index)
        iter_local_start = int(video_index[0].detach().numpy())
        # print (video_file[i], iter_local_start)
        temp_video = str(video_file[i])
        temp_video = (temp_video).replace(".pt","_{}.pt".format(iter_local_start))
        video_session = readVid(temp_video)
        temporalized_video.append(video_session)

    temporalized_video = np.array(temporalized_video)#.reshape(BATCH_SIZE, 15, 4, video_session.shape[1], video_session.shape[2], video_session.shape[3])
    temporalized_video = torch.FloatTensor(temporalized_video)
    # print (temporalized_video.shape)
    return temporalized_video

def loadextractedFlow(video_files, video_indices):
    fe_embed = '/project/CollabRoboGroup/msy9an/data/mhad/flow_embed'
    flow_features = list(); previous_videofile = ""
    for i, video_index in enumerate(video_indices):
        
        video_filepath = video_files[i].replace(".pt", ".avi.pt").replace("/fe_embed/","/flow_embed/")
        if video_filepath != previous_videofile:
            video_pt = torch.load(video_filepath)
        
        iter_local_start = (video_index[0].detach().numpy())
        iter_local_end = (video_index[1].detach().numpy()) +1
        iter_local_start, iter_local_end = int(np.ceil(iter_local_start/3)), int(np.ceil(iter_local_end/3))
        
        flow_features.append(video_pt[iter_local_start:iter_local_end].squeeze(1))
        previous_videofile = video_filepath
        
    flow_features = torch.stack(flow_features, dim=0).float().permute(1,0,2,3,4)
    #print (flow_features.shape)
    return flow_features
    
def forward(iterator, P, Q, optim_P, optim_Q_enc, temperature=0.5, eval_flag=True):


    train_loss = 0; train_mse = 0; input_mse = 0; i =0; classifier_accuracy = 0; train_BCE = 0; final_t_mse= 0; D_loss=0

    for i, (observation, target, video_file, video_index) in enumerate(iterator): #for i, (observation, target) in enumerate(iterator):

        observation, target = to_var(observation), to_var(target); 
        video_file = video_file

        if use_i3d:
            video_file = newtemporalizeVideoi3d(video_file, (video_index))
        elif use_vision and use_flow:
            flow_file = loadextractedFlow(video_file, (video_index))
        else:
            video_file = newtemporalizeVideo(video_file, (video_index))
            
        
        video_file = video_file.cuda()
        #reconstruction loss
        if eval_flag == False:
            P.zero_grad()
            Q.zero_grad()
            
            Q.train()
            P.train()
        if use_vision and use_flow:
            hidden_1, encoder_rnn_agent1, hidden_2, encoder_rnn_agent2, orthogonal_loss = Q(observation, video_file, flow_file, train=False)
        elif use_vision == True:
            hidden_1, encoder_rnn_agent1, hidden_2, encoder_rnn_agent2, orthogonal_loss = Q(observation, video_file, train=True)
        else:
            z_sample, c_sample, hidden_1, encoder_rnn_agent1, hidden_2, encoder_rnn_agent2 = Q(observation)
            
        X_sample = P(observation, target, hidden_1, encoder_rnn_agent1, hidden_2, encoder_rnn_agent2, temperature=temperature, test=eval_flag)
        recon_loss = (X_sample.cuda()-target)**2
        recon_loss = recon_loss.mean()
        recon_loss = recon_loss/2
        if orthogonal:
            orthogonal_loss = orthogonal_loss.mean()
            recon_loss = recon_loss + orthogonal_loss
        else:
            recon_loss = recon_loss

        if eval_flag == False:
            recon_loss.backward()
            torch.nn.utils.clip_grad_norm_(P.parameters(), 10.0)
            torch.nn.utils.clip_grad_norm_(Q.parameters(), 10.0)

            optim_P.step()
            optim_Q_enc.step()
        
        transformed_target, transformed_decoded = restore_pose(target, X_sample)
        transformed_input, transformed_input = restore_pose(observation, observation)

        train_mse += (eucdlidean(transformed_target, transformed_decoded))
        input_mse += (eucdlidean(transformed_target, transformed_input[:,-1].repeat(transformed_target.shape[1],1,1).permute(1,0,2)))
        final_t_mse += ((transformed_target[:,-1] - transformed_decoded[:,-1])**2).mean()
        #print (transformed_target[3][3], transformed_decoded[3][3])

        train_BCE += recon_loss
        train_loss += orthogonal_loss
        del target
        del X_sample
        torch.cuda.empty_cache()
        transformed_target = transformed_target.cpu().detach().numpy()
        transformed_decoded = transformed_decoded.cpu().detach().numpy()
        del transformed_target
        del transformed_decoded
        torch.cuda.empty_cache()

    return train_loss, train_mse, input_mse, train_BCE, final_t_mse

def val_forward(iterator, P, Q, optim_P, optim_Q_enc, temperature=0.5, eval_flag=True):
    train_loss = 0; train_mse = 0; input_mse = 0; i =0; classifier_accuracy = 0; train_BCE = 0; final_t_mse= 0; D_loss=0
    with torch.no_grad():
        for i, (observation, target, video_file, video_index) in enumerate(iterator):#for i, (observation, target) in enumerate(iterator):
            observation, target = to_var(observation), to_var(target)
            video_file = video_file
            if use_i3d:
                video_file = newtemporalizeVideoi3d(video_file, (video_index))
            if use_vision and use_flow:
                flow_file = loadextractedFlow(video_file, (video_index))
                video_file = newtemporalizeVideo(video_file, (video_index))
            else:
                video_file = newtemporalizeVideo(video_file, (video_index))

            
            video_file = video_file.cuda()
            if use_vision and use_flow:
                hidden_1, encoder_rnn_agent1, hidden_2, encoder_rnn_agent2, _ = Q(observation, video_file, flow_file, train=False)
            elif use_vision == True:
                hidden_1, encoder_rnn_agent1, hidden_2, encoder_rnn_agent2, _ = Q(observation, video_file, train=False)
            else:
                z_sample, c_sample, hidden_1, encoder_rnn_agent1, hidden_2, encoder_rnn_agent2 = Q(observation)
                
            X_sample = P(observation, target, hidden_1, encoder_rnn_agent1, hidden_2, encoder_rnn_agent2, temperature=temperature, test=eval_flag)
           
            recon_loss = (X_sample.cuda()-target)**2
            recon_loss = recon_loss.mean()
            recon_loss = recon_loss/2
        
            transformed_target, transformed_decoded = restore_pose(target, X_sample)
            transformed_input, transformed_input = restore_pose(observation, observation)

            train_mse += (eucdlidean(transformed_target, transformed_decoded))
            input_mse += (eucdlidean(transformed_target, transformed_input[:,-1].repeat(transformed_target.shape[1],1,1).permute(1,0,2)))
            final_t_mse += ((transformed_target[:,-1] - transformed_decoded[:,-1])**2).mean()

            train_BCE += recon_loss

            del target
            del X_sample
            torch.cuda.empty_cache()
            transformed_target = transformed_target.cpu().detach().numpy()
            transformed_decoded = transformed_decoded.cpu().detach().numpy()
            del transformed_target
            del transformed_decoded
            torch.cuda.empty_cache()

    return train_loss, train_mse, input_mse, train_BCE, final_t_mse




for num_layer in num_layers:
    for hidden_size in hidden_sizes:
        for embed_size in embed_sizes:
            if embed_size > hidden_size:
                print ("not using this as embed_size > hidden_size")
                continue
            temperature = 0.0 #1.11
            EPS = 1e-15
            z_red_dims = embed_size; z_cat_dims = 20#7
            Q = Q_net(n_characters, hidden_size, z_red_dims, z_cat_dims, conditional).cuda()
            P = P_net(n_characters//3, hidden_size, z_red_dims, z_cat_dims, num_layer).cuda()
            print (Q)
            print (P)


            #Set learning rates
            gen_lr = 0.0001 
    
            #encode/decode optimizers
            optim_P = torch.optim.Adam(P.parameters(), lr=gen_lr)
            optim_Q_enc = torch.optim.Adam(Q.parameters(), lr=gen_lr)
            t0 = 20; t_mul =1
            if warm_start==True:
                scheduler_encoder = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim_Q_enc, T_0=t0, T_mult=t_mul) 
                scheduler_decoder = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim_P, T_0=t0, T_mult=t_mul)
                
            else:
                scheduler_encoder = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_Q_enc, 'min', patience=5,verbose=True) 
                scheduler_decoder = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_P, 'min', patience=5,verbose=True)
                
            best_val_mse = float("Inf")
            patience_counter = 0
            log_dir = os.path.join(os.getcwd(), "logs/vision_sensor{}+ws{}+cond{}+ortho{}{}512featuresie_scheduleddecay{}".format(ACTION, warm_start, conditional, orthogonal, hidden_size, temperature))
            writer = SummaryWriter(log_dir)
            for epoch in range(n_epochs):
                print (temperature)
                train_loss, train_mse, input_mse_train, train_BCE, finaltrain_mse = forward(train_iterator, P, Q, optim_P, optim_Q_enc, temperature, False)
                val_loss, val_mse, input_mse_val, val_BCE, finalval_mse = val_forward(val_iterator, P, Q, optim_P, optim_Q_enc, temperature, True)

                if warm_start==True:
                    scheduler_encoder.step() # val_BCE)
                    scheduler_decoder.step() # val_BCE)
                    
                else:
                    scheduler_encoder.step(val_BCE)
                    scheduler_decoder.step(val_BCE)
                    
                test_loss =0; test_mse=0; input_mse_test=0; test_BCE=0; diff_input = 0; perc_diff = 0; final_mse = 0; test_pck = 0; zv_pck = 0
                test_gt = list(); test_preds = list()
                with torch.no_grad():
                    for i, (observation, target, video_file, video_index) in enumerate(test_iterator):#for i, (observation, target) in enumerate(test_iterator):
                        observation, target = to_var(observation), to_var(target);
                        
                        video_file = video_file
                        if use_i3d:
                            video_file = newtemporalizeVideoi3d(video_file, (video_index))
                        elif use_vision and use_flow:
                            
                            flow_file = loadextractedFlow(video_file, (video_index))
                            video_file = newtemporalizeVideo(video_file, (video_index))
                        else:
                            video_file = newtemporalizeVideo(video_file, (video_index))

                        video_file = video_file.cuda()
                        
                        Q.eval(); P.eval(); 
                        #reconstruction loss
                        if use_vision and use_flow:
                            hidden_1, encoder_rnn_agent1, hidden_2, encoder_rnn_agent2, _ = Q(observation, video_file, flow_file, train=False)
                        elif use_vision == True:
                            hidden_1, encoder_rnn_agent1, hidden_2, encoder_rnn_agent2, _ = Q(observation, video_file, train=False)
                        else:
                            z_sample, c_sample, hidden_1, encoder_rnn_agent1, hidden_2, encoder_rnn_agent2 = Q(observation)
                            
                        X_sample = P(observation, None, hidden_1, encoder_rnn_agent1, hidden_2, encoder_rnn_agent2, temperature=temperature)

                        test_loss = (X_sample.cuda()-target)**2
                        test_loss = test_loss.mean()
                        test_loss = test_loss/2

                        transformed_target, transformed_decoded = restore_pose(target, X_sample)
                        transformed_input, transformed_input = restore_pose(observation, observation)

                        test_pck += 0
                        zv_pck += 0

                        test_mse += eucdlidean(transformed_target, transformed_decoded)

                        diff_input += eucdlidean(transformed_target, transformed_input[:,-1].repeat(transformed_target.shape[1],1,1).permute(1,0,2))

                        perc_diff += 100*(abs(transformed_target - transformed_decoded)/transformed_target).mean()
                        final_mse += (abs(transformed_decoded[:,-1] - transformed_target[:,-1])**2).mean()

                        test_gt.extend(((transformed_target.cpu().detach().numpy().reshape(-1, transformed_target.shape[-1]))))
                        test_preds.extend(((transformed_decoded.cpu().detach().numpy().reshape(-1, transformed_target.shape[-1]))))

                        del target
                        del X_sample
                        torch.cuda.empty_cache()
                        transformed_target = transformed_target.cpu().detach().numpy()
                        transformed_decoded = transformed_decoded.cpu().detach().numpy()
                        del transformed_target
                        del transformed_decoded
                        torch.cuda.empty_cache()


                if epoch % log_every == 0:
                        temperature = temperature*0.99

                        print('{} orthogonal_loss ={} train bce ={} (mse_target={}) (mse_input={}) ' .format(
                            epoch, train_loss/len(train_iterator), train_BCE/len(train_iterator), train_mse, input_mse_train
                        ))
                        print('[{}] orthogonal_loss ={} val bce ={} (mse_target={}), (mse_input={}) ' .format(
                            epoch, val_loss/len(val_iterator), val_BCE/len(val_iterator), val_mse, input_mse_val
                        ))
                        print('[{}] test_loss={} ( mse_target={}) avg = {}' .format(
                            epoch, test_loss/len(test_iterator), test_mse, test_mse.mean()/len(test_iterator)
                        ))

                        print('[{}] test_pck={} zv_pck = {} ( diff={}) avg = {}'.format(
                            epoch, test_pck/len(test_iterator), zv_pck/len(test_iterator), diff_input, diff_input.mean()/len(test_iterator)
                        ))

                        print('')

                        writer.add_hparams({'lr': gen_lr, 'batch_size': BATCH_SIZE, 'embedding_dim': embed_size,'hidden_dim': hidden_size, 'num_classes':n_characters},
                        {'hparam/train_mse': train_mse.mean()/len(train_iterator), 'hparam/test_mse': test_mse.mean()/len(test_iterator), 'hparam/train_loss': train_loss, 'hparam/test_loss': test_loss})
                        writer.add_scalar('Loss/train', train_loss/len(train_iterator), epoch)
                        writer.add_scalar('Loss/val', val_loss/len(val_iterator), epoch)
                        writer.add_scalar('Loss/test', test_loss//len(test_iterator), epoch)
                        writer.add_scalar('ADE/train', train_mse.mean()/len(train_iterator), epoch)
                        writer.add_scalar('ADE/val', val_mse.mean()/len(val_iterator), epoch)
                        writer.add_scalar('ADE/test', test_mse.mean()/len(test_iterator), epoch)
                        writer.add_scalar('ADE/train', train_mse.mean()/len(train_iterator), epoch)
                        for i in range(0,train_mse.shape[-1]):
                            writer.add_scalar('ADE{}/train'.format(i), train_mse[i], epoch)
                            writer.add_scalar('ADE{}/val'.format(i), val_mse[i], epoch)
                            writer.add_scalar('ADE{}/test'.format(i), test_mse[i], epoch)


                if epoch % save_every == 0:
                    save(Q,"encoder_1")
                    save(P,"decoder_1")


                """
                early stopping
                """

                if best_val_mse >= val_mse.mean():
                    best_val_mse = val_mse.mean()
                    patience_counter = 0

                else:
                    patience_counter +=1
                
                if  patience_counter >=20:
                    break
