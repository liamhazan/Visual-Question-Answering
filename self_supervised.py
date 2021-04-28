import random
import torchvision.transforms.functional as TF
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import pickle
from model import VQA_model
from preprocess import VQAdataset
import time
from cnn_v2 import I_encoder

BATCH_SIZE = 50
train_dataset = pickle.load( open( "vqa_train_dataset.pkl", "rb" ))
val_dataset = pickle.load( open( "vqa_val_dataset.pkl", "rb" ))

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


cnn = I_encoder().to(device)
cnn.load_state_dict(torch.load("epoch_5_self_cnn.pkl", map_location=torch.device(device)))
FC = nn.Sequential(
    nn.Linear(150*7*7,4),
    nn.LogSoftmax(dim=0)
    ).to(device)
EPOCHS = 35
LEARNING_RATE = 0.001
criterion = nn.CrossEntropyLoss()
transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4695836035981774, 0.44679106984101236, 0.40951072280388323)
                                             ,(0.24364206719957293, 0.2388205561041832, 0.24255008994787933))])


device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(cnn.parameters()) + list(FC.parameters()), lr=LEARNING_RATE)

angles_2_label_dict = {0:torch.tensor([0]).to(device), 90:torch.tensor([1]).to(device),
                       180:torch.tensor([2]).to(device), 270:torch.tensor([3]).to(device)}



for epoch in range(EPOCHS):
    print(f"start epoch: {epoch}")
    correct = 0.0
    total = 0
    t0 = time.time()
    for i , (image_filenames, questions, answers) in enumerate(train_dataloader):
        batch_size = len(image_filenames)
        if i%100 == 0 and i != 0:
            print(f"at {i*BATCH_SIZE}'th sample -- train")
            print(correct/(total))
        angles = random.choices([0,90,180,270],k=batch_size)
        for j,image_filename in enumerate(image_filenames):
            if j==0:
                images = TF.rotate(transform(pickle.load( open( f"./images/{image_filename}", "rb" ))),angles[j])
                continue
            images = torch.cat((images,TF.rotate(transform(pickle.load( open( f"./images/{image_filename}", "rb" ))),angles[j])))
        images = images.view((batch_size,3,224,224)).to(device)
        out = cnn(images)
        out = FC(out.view(batch_size,-1))
        labels = torch.tensor([angles_2_label_dict[angle] for angle in angles]).to(device)
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct += sum(torch.argmax(out, dim=1)==labels)
        total += batch_size
        t0=time.time()


        # print('number of parameters: ', sum(param.numel() for param in cnn.parameters()))




    torch.save(cnn.state_dict(),f'epoch_{6+epoch}_self_cnn.pkl')
