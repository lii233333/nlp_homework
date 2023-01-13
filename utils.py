import pandas as pd
from sklearn.model_selection import train_test_split

def load_train_data(path,batch_size):
    train_set = pd.read_csv(path)
    sentences = train_set['Text'].values
    labels = train_set['Labels'].values
    sentences_splited = []
    labels_splited = []
    for i in range(len(sentences) - 1):
        z = sentences[i].split("__eou__")
        for j in range(len(z)):
            sentences_splited.append(z[j].strip(" "))
    for i in range(len(sentences) - 1):
        z = str(int(labels[i]))
        for j in range(len(z)):
            labels_splited.append(z[j])
    train_features, test_features, train_targets, test_targets = train_test_split(sentences_splited, labels_splited)
    batch_count = int(len(train_features) / batch_size)
    batch_train_inputs, batch_train_targets = [], []
    for i in range(len(train_targets)):
        train_targets[i] = int(train_targets[i])
    for i in range(len(test_targets)):
        test_targets[i] = int(test_targets[i])

    # 分段
    for i in range(batch_count):
        batch_train_inputs.append(train_features[i * batch_size: (i + 1) * batch_size])
        batch_train_targets.append(train_targets[i * batch_size: (i + 1) * batch_size])
    return batch_train_inputs, batch_train_targets,test_features,test_targets,batch_count

# 加载待分类的数据
def load_data(path):
    train_set = pd.read_csv(path)
    sentences = train_set['Text'].values
    input = []
    for i in range(len(sentences)):
        z = sentences[i].split("__eou__")
        input.append(z[-1].strip(" "))
    return input
