import cv2
import os

main_set_path = "data/main"

def max_video_length():
    max_length = 0
    dirs = os.listdir(main_set_path)
    for dir in dirs:
        path = os.path.join(main_set_path,dir)
        vs = [v for v in os.listdir(path) if ".mp4" in v]
        for v in vs:
            cap = cv2.VideoCapture(os.path.join(path, v))
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            max_length = max(max_length, length)

    print (max_length)
    #154

def max_sentence_length():
    max_length = (0, "")
    dirs = os.listdir(main_set_path)
    for dir in dirs:
        path = os.path.join(main_set_path,dir)
        vs = [v for v in os.listdir(path) if ".txt" in v]
        for v in vs:
            f = open((os.path.join(path, v)), "r")
            l = len(f.readline().split(" ")) - 2
            max_length = max(max_length, (l,(os.path.join(path, v))))
            
    print (max_length)
    #24

def vocab():
    vocab_set = set([])
    dirs = os.listdir(main_set_path)
    for dir in dirs:
        path = os.path.join(main_set_path,dir)
        vs = [v for v in os.listdir(path) if ".txt" in v]
        for v in vs:
            f = open((os.path.join(path, v)), "r")
            words = f.readline().split(" ")[2:]
            words[-1] = words[-1][:-1]
            f.close()
            print (words)
            for w in words:
                vocab_set.add(w)
            

    vocab_list = sorted(list(vocab_set))
    vocalfile = open("vocab.txt", "w")
    for w in vocab_list:
        vocalfile.write(w+"\n")

vocab()

            