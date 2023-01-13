import model
import torch
import utils
import csv

device = torch.device("cuda:1")

if __name__ == "__main__":
    # 加载模型
    bertmodel = model.BertClassfication()
    param = torch.load("./model.pth",map_location=device)
    bertmodel.load_state_dict(param)
    bertmodel.to(device)

    while True:
        sentence = input("请输入句子:")
        if sentence=="退出":
            break
        output = bertmodel([sentence], device)
        _, predict = torch.max(output, 1)
        a = predict.tolist()
        if a[0] == 1:
            print("happiness")
        elif a[0] == 2:
            print("Love")
        elif a[0] == 3:
            print("Sorrow")
        elif a[0] == 4:
            print("Fear")
        elif a[0] == 5:
            print("Disgust")
        elif a[0] == 6:
            print("None")


