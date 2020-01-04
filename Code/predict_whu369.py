# This script shows an example of how to test your predict function before submission
# 0. To use this, replace x_test (line 26) with a list of absolute image paths. And
# 1. Replace predict with your predict function
# or
# 2. Import your predict function from your predict script and remove the predict function define here
# Example: from predict_username (predict_ajafari) import predict
# %% -------------------------------------------------------------------------------------------------------------------
import numpy as np
import torch
import cv2
import torch.nn as nn

DROPOUT = 0.05

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 18, (3, 3))  # output (n_examples, 18, 398, 398) Output dim = (400-3)/1 + 1 = 398
        self.pool1 = nn.MaxPool2d((2, 2))  # output (n_examples, 18, 199, 199) Output dim = (398-2)/2 + 1 = 199
        self.conv2 = nn.Conv2d(18, 36, (3, 3))  # output (n_examples, 36, 197, 197) Output dim = (199-3)/1 + 1 = 197
        self.pool2 = nn.MaxPool2d((2, 2),
                                  ceil_mode=False)  # output (n_examples, 36, 98, 98) Output dim = (197 - 2)/2 + 1 = 98
        self.conv3 = nn.Conv2d(36, 72, (3, 3))  # output (n_examples, 72, 96, 96) Output dim = (98 - 3)/1 + 1 = 96
        self.pool3 = nn.MaxPool2d((2, 2))  # output (n_examples, 72, 48, 48) Output dim = (96 - 2)/2 + 1 = 48
        self.conv4 = nn.Conv2d(72, 72, (3, 3))  # output (n_examples, 72, 46, 46) Output dim = (48 - 3)/1 + 1 = 46
        self.pool4 = nn.MaxPool2d((2, 2))  # output (n_example, 72, 23, 23) Output dim = (46 - 2)/2 + 1 = 23
        self.linear1 = nn.Linear(72 * 23 * 23, 128)  # input will be flattened to (n_examples, 72 * 23 * 23)
        # self.linear1 = nn.Linear(72 * 48 * 48, 128)  # input will be flattened to (n_examples, 72 * 23 * 23)
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

# ---------------------------------------Helper Function----------------------------------------------------------------
def helper2(array, size):
    a = size[0]
    b = size[1]

    y_pred = []
    for i in range(a):
        list1 = []
        for j in range(b):
            if array[i][j] >= 0.5:
                num1 = 1
                list1.append(num1)
            else:
                num0 = 0
                list1.append(num0)
        y_pred.append(list1)
        y_pred_np = np.asarray(y_pred)
    return y_pred_np

# This predict is a dummy function, yours can be however you like as long as it returns the predictions in the right format
def predict(x):
    # On the exam, x will be a list of all the paths to the images of our held-out set
    images = []
    for img_path in x:
        # Here you would write code to read img_path and preprocess it
        raw_img = cv2.imread(img_path)
        # resize img to quarter size
        dim = (400, 400)  # (width, height)
        img = cv2.resize(raw_img, dim)  # cv2 image: (W X H X C)
        img = img.transpose((2, 0, 1))  # torch image: (C X H X W)
        img = img.astype("float") / 255.0
        # Convert image and label to torch tensors
        # img = torch.FloatTensor(np.asarray(img))
        # print(img.shape)
        images.append(np.asarray(img))  # I am using 1 as a dummy placeholder instead of the preprocessed image

    x = torch.FloatTensor(np.array(images))

    # Here you would load your model (.pt) and use it on x to get y_pred, and then return y_pred
    model = CNN()
    model.load_state_dict(torch.load("model_whu369.pt"))
    model.eval()

    y_pred = model(x)
    y_pred = y_pred.detach().clone()

    y_pred_shape = y_pred.shape
    y_pred_np = y_pred.numpy()
    y_predict = helper2(y_pred_np, y_pred_shape)
    y_predict_tensor = torch.from_numpy(y_predict).float()

    return y_predict_tensor

# # %% -------------------------------------------------------------------------------------------------------------------
# x_test = ["./Metadata/small_test/cells_1.png", "./Metadata/small_test/cells_2.png", "./Metadata/small_test/cells_0.png", "./Metadata/small_test/cells_7.png", "./Metadata/small_test/cells_8.png", "./Metadata/small_test/cells_12.png"]  # Dummy image path list placeholder
# y_test_pred = predict(x_test)
# print(y_test_pred)
#
# # %% -------------------------------------------------------------------------------------------------------------------
# assert isinstance(y_test_pred, type(torch.Tensor([1])))  # Checks if your returned y_test_pred is a Torch Tensor
# assert y_test_pred.dtype == torch.float  # Checks if your tensor is of type float
# assert y_test_pred.device.type == "cpu"  # Checks if your tensor is on CPU
# assert y_test_pred.requires_grad is False  # Checks if your tensor is detached from the graph
# assert y_test_pred.shape == (len(x_test), 7)  # Checks if its shape is the right one
# # Checks whether the your predicted labels are one-hot-encoded
# assert set(list(np.unique(y_test_pred))) in [{0}, {1}, {0, 1}]
# print("All tests passed!")
