
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pickle
from model import VQA_model
from preprocess import VQAdataset
import time
import os
from PIL import Image
import matplotlib.pyplot as plt
from operator import itemgetter
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor()
                        ,transforms.Normalize((0.4695836035981774, 0.44679106984101236, 0.40951072280388323)
                                             ,(0.24364206719957293, 0.2388205561041832, 0.24255008994787933))])



train_dataset = pickle.load( open( "vqa_train_dataset.pkl", "rb" ))
val_dataset = pickle.load( open( "vqa_val_dataset.pkl", "rb" ))






inverse_word_dict = {v:k for k,v in train_dataset.word_dict.items()}


label2ans = {val : key for key,val in train_dataset.ans2label.items()}
label_counts = {x : 0 for x in label2ans.keys()}

for ans in train_dataset.train_answers:
    for i, label in enumerate(ans['labels']):
        if ans["scores"][i] != 0:
            label_counts[label] += 1

with open("class_weights.pkl", "wb") as file:
    pickle.dump(torch.tensor([1/v if v != 0 else 0 for v in label_counts.values()]), file)


BATCH_SIZE = 20

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

word_emb_dim = 75
vocab_len = len(train_dataset.word_dict)
lstm_hidden_dim = 100
region_emb_dim = 150
num_regions = 7*7
self_interaction_dim = 75
interaction_dim = 100
n_answers = 1276
EPOCHS = 35
LEARNING_RATE = 0.0003
model = VQA_model(word_emb_dim,num_regions,  vocab_len,lstm_hidden_dim, region_emb_dim, self_interaction_dim,
                  interaction_dim, n_answers)
# model.load_state_dict(torch.load("VQA_model_epoch:15.pkl", map_location=torch.device(device)))

optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.9), lr=LEARNING_RATE,
                                                       weight_decay=1e-5)

print(f"num of params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}")
if torch.cuda.is_available():
    model = model.cuda()
res_dict = {"train_loss":[],"val_loss":[], "train_acc":[],"val_acc":[]}
for epoch in range(EPOCHS):
    print(f"start epoch: {epoch}")
    correct = 0.0
    total = 0
    total_loss = 0.0
    label_predictions = {x : 0 for x in label2ans.values()}
    for i , (image_filenames, questions, answers) in enumerate(train_dataloader):
        batch_size = len(image_filenames)
        if i % 2000 == 0 and i!= 0:
            print(f"at {i*BATCH_SIZE}'th sample")
            print(f"current accuracy = {correct/total}")
            print(sorted(label_predictions.items(), key=itemgetter(1), reverse=True)[:15])

        t0 = time.time()
        for j,image_filename in enumerate(image_filenames):
            if j==0:
                images = transform(pickle.load( open( f"./images/{image_filename}", "rb" )))
                continue
            images = torch.cat((images,transform(pickle.load( open( f"./images/{image_filename}", "rb" )))))
        images = images.view((len(image_filenames),3,224,224)).to(device)
        questions = questions.to(device)
        # answers = answers.to(device)
        t1 = time.time()
        loss, predictions = model(images,questions,answers)
        total_loss += float(loss)
        t2 = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t3 = time.time()
        for labels, scores in zip(answers["labels"], answers["scores"]):
            for j in range(batch_size):
                if predictions[j] == labels[j]:
                    correct += float(scores[j])
                    label_predictions[label2ans[int(predictions[j])]] += 1
                    # print(label2ans[int(predictions[j])])
        total += batch_size

    print(f"epoch:{epoch} train accuracy:{correct/total}")
    print(f"epoch:{epoch} train loss:{total_loss/int(len(train_dataset)/BATCH_SIZE)}")
    res_dict["train_acc"].append(correct/total)
    res_dict["train_loss"].append(total_loss/int(len(train_dataset)/BATCH_SIZE))
    if torch.cuda.is_available():
        torch.save(model.state_dict(), f"VQA_model_epoch:{epoch}.pkl")
    # print(sorted(label_predictions.items(), key=itemgetter(1), reverse=True)[:15])


    # TEST
    label_predictions = {x : 0 for x in label2ans.values()}
    with torch.no_grad():
        correct = 0
        total = 0
        total_loss = 0
        for (image_filenames, questions, answers) in val_dataloader:
            batch_size = len(image_filenames)
            for i, image_filename in enumerate(image_filenames):
                if i == 0:
                    images = transform(pickle.load(open(f"./images/{image_filename}", "rb")))
                    continue
                images = torch.cat((images, transform(pickle.load(open(f"./images/{image_filename}", "rb")))))
            images = images.view((len(image_filenames), 3, 224, 224)).to(device)
            questions = questions.to(device)
            loss, predictions = model(images, questions, answers)
            total_loss += float(loss)
            for labels, scores in zip(answers["labels"], answers["scores"]):
                for j in range(batch_size):
                    if predictions[j] == labels[j]:
                        correct += float(scores[j])
                        label_predictions[label2ans[int(predictions[j])]] += 1

            total += batch_size
    res_dict["val_acc"].append(correct / total)
    res_dict["val_loss"].append(total_loss / int(len(val_dataset)/BATCH_SIZE))
    print(f"epoch:{epoch} val accuracy:{correct/total}")
    print(f"epoch:{epoch} val loss:{total_loss/int(len(val_dataset)/BATCH_SIZE)}")
    # print(sorted(label_predictions.items(), key=itemgetter(1), reverse=True)[:15])



    plt.plot(res_dict["train_acc"], c="blue", label="train Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.plot(res_dict["val_acc"], c="red", label="val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig('accuracy-epochs.png')

    plt.clf()

    plt.plot(res_dict["train_loss"], c="blue", label="train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()

    plt.plot(res_dict["val_loss"], c="red", label="val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig('loss-epochs.png')
    plt.clf()
