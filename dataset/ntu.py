import os, sys, time, glob, scipy.io, torch, scipy, pickle
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
class DataReader:
    """
    Takes the directory where data is stored, along
    """
    def __init__(self, dir_name, video_wd, annotations_dir, data_dir, input_dim = 16, batch_size=16, action="A059"):
        self.base_dir = dir_name
        self.data_dir = os.path.join(dir_name, data_dir)
        self.video_dir = os.path.join(video_wd, "")
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.horizon = 15
        self.observed_horizon = 15
        self.forecasted_horizon = 15
        self.action = action
        #self.iterateData(annotations_dir)

    def iterateData(self, annotations_dir):
        """
        Uses glob to iterate over files and calls a helper function to read the data
        """
        annotations_dir = os.path.join(self.base_dir, annotations_dir)
        train_data, test_data, videotrain_list, videotest_list = self.readMat(annotations_dir)
        train_dataloader, val_dataloader, test_dataloader = self.prepareDataLoader(train_data, test_data, videotrain_list, videotest_list)

        return train_dataloader, val_dataloader, test_dataloader

    def readvideo(self, file_name):
        vid_reader = skvideo.io.vreader(file_name)

    def extractVelocity(self, position_tensor):
        """
        Extracts velocity values from temporal joints
        """
        zero_pad = np.zeros((1, 1, position_tensor.shape[2]))

        velocity_tensor = np.zeros((position_tensor.shape[0], position_tensor.shape[1], position_tensor.shape[2]))
        velocity_tensor[:, 0:-1] = position_tensor[:,1:] - position_tensor[:,:-1]

        acceleration_tensor = self.extractAcceleration(velocity_tensor)
        return velocity_tensor, acceleration_tensor

    def extractAcceleration(self, velocity_tensor):
        """
        Extracts acceleration from velocity values
        """
        acceleration_tensor = np.zeros((velocity_tensor.shape[0], velocity_tensor.shape[1], velocity_tensor.shape[2]))
        acceleration_tensor[:, 0:-2] =  velocity_tensor[:,1:-1] - velocity_tensor[:,0:-2,]

        return acceleration_tensor

    def readMat(self, file_dir):
        train_list = list(); test_list = list(); val_list = list()
        videotrain_list = list(); videoval_list = list(); videotest_list = list();
        train_file_list = list(); val_file_list = list(); test_file_list = list()

        # actions = ["*{}.skeleton.npy".format(self.action)]#"
        actions = ["*A05*.skeleton.npy", "*A060.skeleton.npy"] 
        print (actions)
        train_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15,16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
        test_subjects = list()
        shape_counter = 0
        target_num = {2}
        for action in actions:
            for file_ in glob.glob(os.path.join(file_dir, action)):
                subject = file_.split("/")[-1].split(".")[0]
                subject = subject.split("P")[-1].split("R")[0]
                demonstration_dictionary = np.load(file_, allow_pickle=True).item()
                current_subjects = demonstration_dictionary["nbodys"]
                #videotrain_list, videoval_list, videotest_list, vid_len = self.readVid(file_.replace(".skeleton.npy", "_rgb.pt"), videotrain_list, videoval_list, videotest_list)
                if (set(current_subjects)==target_num):
                    skel_data = demonstration_dictionary["skel_body0"]
                    skel_data = skel_data.reshape(skel_data.shape[0],-1)

                    skel_data1 = demonstration_dictionary["skel_body1"]
                    skel_data1 = skel_data1.reshape(skel_data1.shape[0],-1)

                    #self.plotResults(skel_data, skel_data1)
                    skel_data = np.concatenate((skel_data, skel_data1), axis=-1)
                    #self.checkdiscontinuity(skel_data, file_)
                    #self.plotResults(skel_data, skel_data1)
                    #print (skel_data.shape)
                    shape_counter +=skel_data.shape[0]

                    if not int(subject.replace("P","")) in train_subjects:
                        print (file_)
                        test_list.append(skel_data)
                        test_file_list.append(file_.replace(file_dir, self.video_dir))
                        if subject not in test_subjects:
                            test_subjects.append(subject)
                    else:
                        train_list.append(skel_data)
                        train_file_list.append(file_.replace(file_dir, self.video_dir))
                else:
                    continue
                    skel_data = demonstration_dictionary["skel_body0"] #scipy.io.loadmat(file_)["d_skel"]
                    if not int(subject.replace("P","")) in train_subjects:
                        test_list.append(skel_data)
                        if subject not in test_subjects:
                            test_subjects.append(subject)
                    else:
                        train_list.append(skel_data)
        return train_list, test_list, train_file_list, test_file_list

    def readVid(self, skel_file, videotrain_list, videoval_list, videotest_list):
        train_list = list(); test_list = list(); val_list = list()
        video_file = os.path.join(self.video_dir, skel_file.split("/")[-1])
        #print (video_file, torch.cuda.is_available())
        torch.cuda.empty_cache()
        video_pt = torch.load(video_file, map_location='cuda:0')
        video_pt = video_pt.to('cpu')
        #print (video_file, video_pt.shape)
        train_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15,16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
        subject = video_file.split("/")[-1].split(".")[0] #subject = skel_file.split("/")[-1].split("_")[1]
        subject = subject.split("P")[-1].split("R")[0]
        #print (video_file, subject, video_pt.shape)

        if not int(subject.replace("P","")) in train_subjects:
            videotest_list.append(video_pt)
        else:
            videotrain_list.append(video_pt)



        return videotrain_list, videoval_list, videotest_list, video_pt.shape[0]

    def checkdiscontinuity(self, skel12_data, file_name):
        """
        helper function for cleaning the A055 mess
        """
        for i in range(skel12_data.shape[0]-1):
            if abs(np.mean(skel12_data[i+1] - skel12_data[i])) >= 1e-1:
                print (file_name, abs(np.mean(skel12_data[i+1] - skel12_data[i])))
                skel_file = file_name.split("/")[-1]
                os.rename(file_name, "/home/huron/Documents/samin/corrupt_ntu/{}".format(skel_file))


    def readLabels(self, file_name):
        """
        Reads corresponding activities
        """
        mat = scipy.io.loadmat(file_name, squeeze_me=True)['IkeaDB']
        labels = mat['activity_labels']
        rec = np.random.choice(mat)
        dedup_activities = np.unique(
            [act for act in rec['activity_labels'] if len(act) > 0])
        print('Activity sequence for video:')
        print(' -> '.join(dedup_activities))
        print ("shape of labels :",labels.shape)
        print (rec['activity_labels'])
        return labels

    def getfileType(self, file_name):
        """
        uses the extension to get the file type
        """
        return file_name.split(".")[-1]

    def scaleData(self, data_array, scaler=None, test=False):
        """
        Normalizes data
        """
        if test==True:
            print ("loading provided scaler for normalizing data")
            local_scaler = scaler
        else:
            local_scaler = MinMaxScaler().fit(data_array) #thinking about normalizing
        #local_scaler = scaler if test==True else  StandardScaler().fit(data_array)
        data_array = local_scaler.transform(data_array)
        print (data_array.shape)

        return data_array, local_scaler

    def standardizeData(self, data_array):
        """
        Gets relative coordinates then downsamples, then standarizes
        """
        data_array = data_array
        data_array = data_array.reshape(data_array.shape[0],-1) #reshapes from (,2,14) to (,28)
        #self.plotResults(data_array[:,:75].reshape(-1,25,3),data_array[:,75:].reshape(-1,25,3))
        return data_array

    def readVid(self, video_file):

        #print (video_file, torch.cuda.is_available())
        torch.cuda.empty_cache()
        video_pt = torch.load(video_file)
        video_pt = video_pt.to('cpu')
        #print (video_file, video_pt.shape)

        return video_pt.detach().cpu().numpy()

    def temporalizeVideo(self, video_session, video_index):
        temporalized_video = list()
        video_chunk = np.zeros((4, video_session.shape[1], video_session.shape[2], video_session.shape[3]))
        iter_local = video_index

        for iter_local in range(video_index, video_index+15):
            print ("entering video chunk: ", video_chunk.shape, iter_local)
            if iter_local % 4 and iter_local < len(video_session)-4:
                video_chunk = video_session[iter_local:iter_local+(self.horizon*4//15)]
            print ("entering video chunk: ", video_chunk.shape, iter_local)
            temporalized_video.append(video_chunk)

        temporalized_video = np.array(temporalized_video).reshape(len(temporalized_video), 4, video_session.shape[1], video_session.shape[2], video_session.shape[3])
        temporalized_video = torch.FloatTensor(temporalized_video)
        return temporalized_video


    def prepareDataLoader(self, train_data, test_data, train_file_list, test_file_list):

        """
        receives pose information and creates DataLoader
        """
        train_index_split = 3*len(train_data)//4

        X_train = list(); X_test = list()
        y_train = list(); y_test = list()
        seqlen_train = list(); seqlen_test = list();
        train_file_list_extended = list(); val_file_list_extended = list(); test_file_list_extended = list()

        for i, value in enumerate(train_data):
            #value = value.transpose().astype(np.float32)
            value = value.astype(np.float32)
            value = self.standardizeData(value)
            X_train.extend(value)
            seqlen_train.append(len(value))
            sequence_counter  = [sequence_count for sequence_count in range(len(value)-00)]
            # if i < train_index_split:
            train_file_list_extended.extend("{}_{}".format(train_file_list[i],count) for count in sequence_counter)
            # else:
            val_file_list_extended.extend("{}_{}".format(train_file_list[i],count) for count in sequence_counter)

        max_seqlen_train = max(seqlen_train)  #needed for zero-padding
        #X_train_scaled, std_scaler  = self.meanscaleData(X_train)
        X_train_scaled, norm_scaler = self.scaleData(X_train)

        file = open(("norm_scaler{}.p".format(self.action)), 'wb')
        pickle.dump(norm_scaler, file)
        file.close()

        X_train_temporalized, y_train_temporalized, video_train_temporalized, train_count= self.temporalizeData(X_train_scaled, train_file_list_extended, seqlen_train, self.horizon, max_seqlen_train, 0, train_index_split)
        train_videolist = video_train_temporalized
        print ("train_videolist :", len(train_videolist), len(val_file_list_extended))
        X_valid_temporalized, y_valid_temporalized, video_val_temporalized, val_count= self.temporalizeData(X_train_scaled,  train_file_list_extended, seqlen_train, self.horizon, max_seqlen_train, train_index_split, len(seqlen_train), train_count)
        val_videolist = video_val_temporalized
        #file = open(("norm_scaler{}.p".format(self.action)), 'wb')
        #pickle.dump(norm_scaler, file)
        #file.close()

        for i, value in enumerate(test_data):
            #value = value.transpose().astype(np.float32)
            value = value.astype(np.float32)
            value = self.standardizeData(value)

            X_test.extend(value)
            seqlen_test.append(len(value))
            sequence_counter  = [sequence_count for sequence_count in range(len(value))]
            test_file_list_extended.extend("{}_{}".format(test_file_list[i],count) for count in sequence_counter)

        max_seqlen_test = max(seqlen_test) #needed for zero-padding
        X_test_scaled, norm_scaler = self.scaleData(X_test, norm_scaler, test=True)

        X_test_temporalized, y_test_temporalized, video_test_temporalized, _ = self.temporalizeData(X_test_scaled, test_file_list_extended, seqlen_test, self.horizon, max_seqlen_test, 0, len(seqlen_test))
        test_videolist = video_test_temporalized

        X_trainarray = torch.FloatTensor(X_train_temporalized)
        y_trainarray = torch.FloatTensor(y_train_temporalized)

        X_validarray = torch.FloatTensor(X_valid_temporalized)
        y_validarray = torch.FloatTensor(y_valid_temporalized)


        X_testarray = torch.FloatTensor(X_test_temporalized)
        y_testarray = torch.FloatTensor(y_test_temporalized)

        print ("train :{} val: {} test :{} ".format(X_trainarray.shape, X_validarray.shape, y_testarray.shape))
        train_dataset = SampleData(X_trainarray, y_trainarray, train_videolist)
        val_dataset = SampleData(X_validarray, y_validarray, val_videolist)
        test_dataset = SampleData(X_testarray, y_testarray, test_videolist)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,  shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        """
        for i, (observation, target, video_file, video_index) in enumerate(train_loader):
            video_file =video_file[0]
            print ((observation.shape),(video_file), video_index)
            # video_session = self.readVid(video_file)
            # self.temporalizeVideo(video_session, video_index)
        """
        return train_loader, val_loader, test_loader

    def temporalizeData(self, data_array, video_list, sequence_list, sequence, max_seqlen, start_index, end_index, count = 0):
        x_list_temporalized = list(); y_list_temporalized = list(); video_list_temporalized = list()
        velocity_list_temporalized = list(); acceleration_list_temporalized = list()
        overall_count1 = 0;
        overall_count2 = 0;
        video_iter = 0
        first_index = start_index
        for i in range(start_index, end_index):
            session = data_array[count:count+sequence_list[i]]
            video_file = video_list[count:count+sequence_list[i]]
            # print ("count {} sequence {} {} ".format(count, (count+sequence_list[i]), (count+sequence_list[i])-count))
            # print ("")
            temporalized_session = list(); temporalized_video = list(); forecasted_list = list()
            video_counter = 0
            if len(session) < (self.forecasted_horizon + self.observed_horizon):
                continue
            for iter_local in range(len(session)-(self.forecasted_horizon + self.observed_horizon)):
                temporalized_session.append(session[iter_local:iter_local+self.observed_horizon])
                temporalized_video.append(video_file[iter_local:iter_local+self.observed_horizon])

            for iter_local in range(self.observed_horizon, len(session)-self.forecasted_horizon):
                forecasted_list.append(session[iter_local:iter_local+self.forecasted_horizon])

            overall_count1 += (len(temporalized_session))
            forecasted_list_extended = forecasted_list#np.concatenate((np.asarray(temporalized_session[:len(forecasted_list)]),np.asarray(forecasted_list)), axis=1)
            # temporalized_video = temporalized_video[:len(forecasted_list)]

            velocity_tensor, acceleration_tensor = self.extractVelocity(np.asarray(temporalized_session))
            velocity_list_temporalized.extend(velocity_tensor); acceleration_list_temporalized.extend(acceleration_tensor)


            # velocity_tensor, acceleration_tensor = self.extractVelocity(np.asarray(temporalized_session[:len(forecasted_list)]))
            # velocity_list_temporalized.extend(velocity_tensor); acceleration_list_temporalized.extend(acceleration_tensor)


            t = np.concatenate((np.asarray(temporalized_session[:len(forecasted_list)]), velocity_tensor, acceleration_tensor), axis=2)
            x_list_temporalized.extend(t)
            y_list_temporalized.extend(forecasted_list_extended)
            video_list_temporalized.extend(temporalized_video)
            count += sequence_list[i]
            overall_count2 += (len(t))
            #print (count, (sequence_list[i]), overall_count1, overall_count2, np.asarray(temporalized_session[:len(forecasted_list)]).shape)

        print (len(temporalized_video),len(y_list_temporalized))
        assert (len(x_list_temporalized) == len(y_list_temporalized))
        assert (len(video_list_temporalized) == len(y_list_temporalized))
        #print (len(x_list_temporalized))
        return np.asarray(x_list_temporalized), np.asarray(y_list_temporalized), np.asarray(video_list_temporalized), count



class SampleData(DistributedSampler):
    def __init__ (self, x, y, video_list):
        self.x = x
        self.y = y
        self.video_list = video_list
        #print (self.x, self.video_list)
        """
        self.batch_indices = list(); start_index = 0
        self.batch_indices.append(start_index)
        for index in indices:
            index -=horizon*2
            self.batch_indices.append(start_index+index)
            start_index += index
        """

    def __len__(self):
        #return len(self.batch_indices) -1
        return len(self.x)
    def __getitem__(self, index):

        video_file = self.video_list[index]
        video_file_index = torch.FloatTensor([int(video_file[0].split("_")[-1]), int(video_file[-1].split("_")[-1])])
        video_file = video_file[0].replace(".skeleton.npy", "_rgb.pt").replace("_%s"%(int(video_file_index[0].detach().numpy())), "")
        # print (video_file, int(video_file_index[0].detach().numpy()))
        forecast = self.y[index]
        past_observations = self.x[index];
        # print ("video_file_index ", len(video_file_index))

        return past_observations,  forecast, video_file, video_file_index
"""
data_wd = os.path.join(os.getenv("HOME"), "ntu","ntu_1")
video_wd = "/home/samin/Documents/hello_world/src/ntu/fe_embed"
annotations_dir = "" #relative
data_dir = "videos" #relative
dre = DataReader(data_wd, video_wd,  annotations_dir, data_dir)
train_iterator, val_iterator, test_iterator = dre.iterateData("")
"""
