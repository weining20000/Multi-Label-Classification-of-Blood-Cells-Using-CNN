import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os, cv2
import matplotlib.pyplot as plt
FOLDER_DATASET = "./Metadata/train/"
plt.ion()

import torch.nn as nn
from sklearn.metrics import accuracy_score
from torchvision import datasets

#============================================================================
#                             Data Pre-processing
#============================================================================

class CellData(Dataset):
    __xs = []
    __json_info = []
    __ys = []

    def __init__(self, folder_path, transform=None):
        self.transform = transform
        # Open and load text file including the whole training data
        img_paths = []
        label_paths = []
        json_paths = []

        for file in os.listdir(folder_path):
            single_file_path = os.path.join(folder_path, file)
            if (single_file_path.endswith("png")):
                # image paths
                img_paths.append(single_file_path)

                # json_paths
                json_paths.append(single_file_path.replace(".png", ".json"))

                # label paths
                label_paths.append(single_file_path.replace(".png", ".txt"))

        self.__xs = img_paths
        self.__json_info = json_paths
        self.__ys = label_paths

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):

        img_path = self.__xs[index]

        # read raw img from path
        raw_img = cv2.imread(img_path)

        # resize img to quarter size
        dim = (400, 400) # (width, height)
        img = cv2.resize(raw_img, dim) # cv2 image: (W X H X C)
        img = img.transpose((2, 0, 1)) # torch image: (C X H X W)
        img = img.astype("float") / 255.0
        #image.astype("float") / 255.0

        # Convert image and label to torch tensors
        img = torch.from_numpy(np.asarray(img))

        raw_label = self.get_target(self.__ys[index])

        label = torch.from_numpy(np.asarray(raw_label))

        return img, label

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__xs)

    def get_target(self, label_path):
        dict = {"red blood cell": 0, "difficult": 1, "gametocyte": 2, "trophozoite": 3,
                "ring": 4, "schizont": 5, "leukocyte": 6}

        label_map = [0.0] * 7
        label_file = open(label_path)
        lines = [line.rstrip('\n') for line in label_file]
        label_file.close()
        # One-hot-encode each line into the form similar to [0,0,0,0,0,1,0]
        for i in range(len(lines)):
            item = lines[i]
            label_map[dict[item]] = 1
        # label_map = [dict[item] == 1 for item in lines]

        return label_map

def data_prep(data):
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True, num_workers=1)
    x_train, y_train = next(iter(train_loader))

    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=True, num_workers=1)
    x_test, y_test = next(iter(test_loader))

    return x_train, y_train, x_test, y_test

# Reference: https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/pytorch/dataloader-and-datasets.html

if __name__ == '__main__':
    full_dataset = CellData(FOLDER_DATASET)
    x_train, y_train, x_test, y_test = data_prep(full_dataset)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # %% --------------------------------------- Set-Up --------------------------------------------------------------------

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
    LR = 0.001
    N_EPOCHS = 18
    BATCH_SIZE = 33
    DROPOUT = 0.05


    # %% ----------------------------------- Helper Functions --------------------------------------------------------------
    def acc(x, y, return_labels=False):
        with torch.no_grad():
            logits = model(x)
            pred_labels = np.argmax(logits.cpu().numpy(), axis=1)
        if return_labels:
            return pred_labels
        else:
            return 100 * accuracy_score(y.cpu().numpy(), pred_labels)

    # %% -------------------------------------- CNN Class ------------------------------------------------------------------
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()

            self.conv1 = nn.Conv2d(3, 18, (3, 3))  # output (n_examples, 18, 396, 396) Output dim = (400-3)/1 + 1 = 398
            self.pool1 = nn.MaxPool2d((2, 2))  # output (n_examples, 18, 199, 199) Output dim = (398-2)/2 + 1 = 199
            self.conv2 = nn.Conv2d(18, 36, (3, 3))  # output (n_examples, 36, 197, 197) Output dim = (199-3)/1 + 1 = 197
            self.pool2 = nn.MaxPool2d((2, 2), ceil_mode=False)  # output (n_examples, 36, 98, 98) Output dim = (197 - 2)/2 + 1 = 98
            self.conv3 = nn.Conv2d(36, 72, (3, 3)) # output (n_examples, 72, 96, 96) Output dim = (98 - 3)/1 + 1 = 96
            self.pool3 = nn.MaxPool2d((2, 2)) # output (n_examples, 72, 48, 48) Output dim = (96 - 2)/2 + 1 = 48
            self.conv4 = nn.Conv2d(72, 72, (3,3)) # output (n_examples, 72, 46, 46) Output dim = (48 - 3)/1 + 1 = 46
            self.pool4 = nn.MaxPool2d((2,2)) # output (n_example, 72, 23, 23) Output dim = (46 - 2)/2 + 1 = 23
            self.linear1 = nn.Linear(72 * 23 * 23, 128)  # input will be flattened to (n_examples, 72 * 23 * 23)
            self.linear2 = nn.Linear(128, 64)
            self.linear3 = nn.Linear(64, 7)
            self.drop1 = nn.Dropout(DROPOUT)
            self.drop2 = nn.Dropout(DROPOUT)
            self.sigmoid = nn.Sigmoid()
            self.act = torch.relu

        def forward(self, x):
            x = self.drop1(self.pool1(self.act(self.conv1(x))))
            x = self.drop1(self.pool2(self.act(self.conv2(x))))
            x = self.drop1(self.pool3(self.act(self.conv3(x))))
            x = self.drop1(self.pool4(self.act(self.conv4(x))))
            x = self.linear3(self.drop2(self.act(self.linear2(self.drop2(self.act(self.linear1(x.view(len(x), -1))))))))
            return self.sigmoid(x)

    # %% -------------------------------------- Training Prep ----------------------------------------------------------
    # format the type from float to double
    model = CNN().type('torch.DoubleTensor').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCELoss()

    # %% -------------------------------------- Training Loop ----------------------------------------------------------
    print("Starting training loop...")
    for epoch in range(N_EPOCHS):

        loss_train = 0
        model.train()
        for batch in range(len(x_train) // BATCH_SIZE):
            inds = slice(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE)
            optimizer.zero_grad()
            logits = model(x_train[inds])
            loss = criterion(logits, y_train[inds])
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        with torch.no_grad():
            y_test_pred = model(x_test)
            print(y_test_pred)
            loss = criterion(y_test_pred, y_test)
            loss_test = loss.item()
        torch.save(model.state_dict(), "model_whu369.pt")
        model.eval()
        print("Perfect Result: Loss ---> "+str(loss_test))

    #model.load_state_dict(torch.load("model_whu369.pt"))
    print(model)



