import os
import glob
import argparse
import imageio
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import time


class ResNet50Bottom(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
    def forward(self, x):
        x = self.features(x)
        return x
        
def iterate_videofile():
    rgb_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

    train_dir = (os.path.join(data_dir_base_path, "*A05*.avi"))
    
    print (train_dir)
    for video_file in sorted(glob.glob(train_dir)):
        print ("video file ", video_file)
        video_name = video_file.split("/")[-1].replace(".avi","")
        if not os.path.exists(f'{embed_dir_base_path}/{video_name}.pt'):
            
            seq, seq_len = extractVideo(video_file, rgb_transforms)
            clipped_seq = seq
            features = feature_extractor(clipped_seq)
            print (features.shape, seq_len)
            torch.save(features, f'{embed_dir_base_path}/{video_name}.pt')
            del features

def extractVideo(video_file, rgb_transforms):
    container = imageio.get_reader(video_file, 'ffmpeg')
    frames = list()
    samplinf_freq = 3
    for idx, frame in enumerate(container):
        if idx%samplinf_freq == 0:
            frame = Image.fromarray(frame)
            if rgb_transforms is not None:
                frame = rgb_transforms(frame)
            frames.append(frame)
        
    container.close()
    
    seq = torch.stack(frames, dim=0).float()
    seq_len = seq.size(0)
    print ("downsampled length {}, original length {} ".format(seq_len, idx))
    seq = seq.to(device)
    return seq, seq_len


parser = argparse.ArgumentParser()
parser.add_argument("-sfn", "--start_file_num", help="start_file_num",
                    type=int, default=-1)
parser.add_argument("-cdn", "--cuda_device_no", help="cuda device no",
                    type=int, default=0)
parser.add_argument("-ddbp", "--data_dir_base_path", help="data_dir_base_path",
                    default='<path_to_rgb_data>)
parser.add_argument("-edbp", "--embed_dir_base_path", help="data_dir_base_path",
                    default='<path_to_extracted_features>)
                    
args = parser.parse_args()
data_dir_base_path = args.data_dir_base_path
embed_dir_base_path = args.embed_dir_base_path

device = torch.device(f'cuda:{args.cuda_device_no}')
original_model = models.resnet50(pretrained=True)
feature_extractor = ResNet50Bottom(original_model)
feature_extractor.to(device)
feature_extractor.eval()        
    
iterate_videofile()


