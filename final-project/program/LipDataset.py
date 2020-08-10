from torch.utils.data import Dataset, DataLoader
import torch
import os
import cv2
import numpy as np

main_set_path = "data/main"
vocab_path = "vocab.txt"

MAX_FRAME_COUNT = 154


d1 = {}
d2 = {}
f = open(vocab_path, "r")
vs = f.readlines()
d1[""] = 0
d2[0] = ""
for i in range (len(vs)):
    d1[vs[i][:-1]] = i+1
    d2[i+1] = vs[i][:-1]

class LipDataset(Dataset):

    def __init__(self):
        self.__data = []
        self.word_to_index = d1
        self.index_to_word = d2
        dirs = os.listdir(main_set_path)
        for dir in dirs:
            path = os.path.join(main_set_path,dir)
            text_files = [name for name in os.listdir(path) if name.endswith(".txt")]
            video_files = [name for name in os.listdir(path) if name.endswith(".mp4")]
            if len(text_files) != len(video_files):
                print ("errorenous folder")
                continue
            length = len(text_files)
            for i in range (length):
                self.__data.append((os.path.join(path,video_files[i]),os.path.join(path, text_files[i])))
        
    def __getitem__(self,index):
        video_path, text_path = self.__data[index]
        cap = cv2.VideoCapture(video_path)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        buf = np.zeros((1, MAX_FRAME_COUNT, frameHeight, frameWidth), np.dtype('float32'))
        
        fc = 0
        ret = True

        # for i in range (MAX_FRAME_COUNT):
        while (fc < frameCount and ret):
            ret, frame  = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            buf[:,fc,:,:] = gray.reshape((1, frameHeight, frameWidth))
            fc += 1
            # else:
            #    buf[:,i,:,:] = np.zeros(shape=(1,frameHeight,frameWidth))

        cap.release()

        video_tensor = torch.from_numpy(buf)

        text_file = open(text_path, "r")
        text = text_file.readline().split(" ")[2:]
        text[-1] = text[-1][:-1]
        text_file.close()
        text_tensor = torch.tensor([self.word_to_index[w] for w in text] + [0] * (24- len(text)))

        return (video_tensor, text_tensor)
        

    def __len__(self):
        return len(self.__data)
