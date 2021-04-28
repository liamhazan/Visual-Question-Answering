#!/usr/bin/env python
# coding: utf-8


import torch
import json
from torch.utils.data import Dataset, DataLoader
import os
import re
from PIL import Image
import numpy as np
import argparse
import sys
import pickle
import torchvision.transforms as transforms



contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve":
    "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt":
    "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've":
    "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent":
    "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve":
    "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im":
    "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've":
    "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've":
    "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat":
    "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve":
    "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt":
    "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve":
    "shouldn't've", "somebody'd": "somebodyd", "somebodyd've":
    "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll":
    "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd":
    "something'd", "somethingd've": "something'd've", "something'dve":
    "something'd've", "somethingll": "something'll", "thats":
    "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres":
    "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve":
    "they'd've", "theyll": "they'll", "theyre": "they're", "theyve":
    "they've", "twas": "'twas", "wasnt": "wasn't", "wed've":
    "we'd've", "we'dve": "we'd've", "weve": "we've", "werent":
    "weren't", "whatll": "what'll", "whatre": "what're", "whats":
    "what's", "whatve": "what've", "whens": "when's", "whered":
    "where'd", "wheres": "where's", "whereve": "where've", "whod":
    "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl":
    "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve":
    "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll":
    "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd":
    "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll":
    "you'll", "youre": "you're", "youve": "you've"
}

manual_map = { 'none': '0',
              'zero': '0',
              'one': '1',
              'two': '2',
              'three': '3',
              'four': '4',
              'five': '5',
              'six': '6',
              'seven': '7',
              'eight': '8',
               'nine': '9',
              'ten': '10'}
articles = ['a', 'an', 'the']
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [';', r"/", '[', ']', '"', '{', '}',
                '(', ')', '=', '+', '\\', '_', '-',
                '>', '<', '@', '`', ',', '?', '!']



from torch.utils.data.dataset import Dataset
from torch.utils.data.dataset import Dataset


class VQAdataset(Dataset):
    def __init__(self, images_path, q_train_path, q_val_path, a_train_path, a_val_path, min_occurence, val=False):
        self.images_path = images_path
        self.transform = transforms.Compose([
            transforms.Resize((224, 224))])
        self.val = val
        self.q_train_path = q_train_path
        self.q_val_path = q_val_path
        self.a_train_path = a_train_path
        self.a_val_path = a_val_path
        self.min_occurence = min_occurence
        self.images_dict = {}
        self.max_q_len = 23

    def get_score(self, occurences):
        if occurences == 0:
            return 0
        elif occurences == 1:
            return 0.3
        elif occurences == 2:
            return 0.6
        elif occurences == 3:
            return 0.9
        else:
            return 1

    def process_punctuation(self, inText):
        outText = inText
        for p in punct:
            if (p + ' ' in inText or ' ' + p in inText) \
                    or (re.search(comma_strip, inText) != None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = period_strip.sub("", outText, re.UNICODE)
        return outText

    def process_digit_article(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = manual_map.setdefault(word, word)
            if word not in articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in contractions:
                outText[wordId] = contractions[word]
        outText = ' '.join(outText)
        return outText

    def preprocess_answer(self, answer):
        answer = self.process_digit_article(self.process_punctuation(answer))
        answer = answer.replace(',', '')
        return answer

    def filter_answers(self, answers_dset, min_occurence):
        """This will change the answer to preprocessed version
        """
        occurence = {}
        for ans_entry in answers_dset:
            gtruth = ans_entry['multiple_choice_answer']
            gtruth = self.preprocess_answer(gtruth)
            if gtruth not in occurence:
                occurence[gtruth] = set()
            occurence[gtruth].add(ans_entry['question_id'])
        answers_to_pop = []
        for answer in occurence.keys():
            if len(occurence[answer]) < min_occurence:
                answers_to_pop.append(answer)

        for answer in answers_to_pop:
            occurence.pop(answer)

        print('Num of answers that appear >= %d times: %d' % (
            min_occurence, len(occurence)))
        return occurence

    def create_ans2label(self, occurence, name, cache_root):
        """Note that this will also create label2ans.pkl at the same time

        occurence: dict {answer -> whatever}
        name: prefix of the output file
        cache_root: str
        """
        self.ans2label = {}
        label2ans = []
        label = 0
        for answer in occurence:
            label2ans.append(answer)
            self.ans2label[answer] = label
            label += 1

    def compute_target(self):
        """Augment answers_dset with soft score as label

        ***answers_dset should be preprocessed***
        """
        for answers_dset in [self.train_answers, self.val_answers]:
            target = []
            for ans_entry in answers_dset:
                answers = ans_entry['answers']
                answer_count = {}
                for answer in answers:
                    answer_ = answer['answer']
                    answer_count[answer_] = answer_count.get(answer_, 0) + 1

                labels = []
                scores = []
                for answer in answer_count:
                    if answer not in self.ans2label:
                        continue
                    labels.append(self.ans2label[answer])
                    score = self.get_score(answer_count[answer])
                    scores.append(score)

                label_counts = {}
                for k, v in answer_count.items():
                    if k in self.ans2label:
                        label_counts[self.ans2label[k]] = v

                ans_entry['labels'] = labels
                ans_entry['scores'] = scores

    def get_questions_for_answers(self, answers):
        qids = [ans['question_id'] for ans in answers]
        relevant_questions = [0] * len(qids)
        found = False
        questions = self.val_questions if self.val else self.train_questions
        for q in questions:
            if found and q['question_id'] not in qids:
                break
            if q['question_id'] in qids:
                found = True
                index = (i for i, qid in enumerate(qids) if q['question_id'] == qid)
                relevant_questions[next(index)] = q
        return relevant_questions

    def get_answers_for_image(self, image_id):
        relevant_answers = []
        found = False
        answers = self.val_answers if self.val else self.train_answers
        for answer in answers:
            if found and answer['image_id'] != image_id:
                break
            if answer['image_id'] == image_id:
                found = True
                relevant_answers.append(answer)

        return relevant_answers

    def get_word_dict(self):
        self.word_dict = {'<UNK>': 0, '<PAD>': 1}
        idx = 2
        for i, q in enumerate(self.train_questions):
            splitted_q = q["question"].lower().split()
            splitted_q[-1] = splitted_q[-1][:-1]
            for word in splitted_q:
                if word not in self.word_dict.keys():
                    self.word_dict[word] = idx
                    idx += 1

    def get_q_idxs(self, question_str):
        words = question_str.lower().split()
        words[-1] = words[-1][:-1]
        # words = words + ['<PAD>']*(self.max_q_len-len(words))
        words = torch.tensor([self.word_dict[word] if word in self.word_dict.keys() else self.word_dict["<UNK>"] for word in words])
        return torch.cat((words,torch.ones(self.max_q_len-len(words), dtype=torch.long)))

    def build_dataset(self):
        with open(self.q_train_path, "r") as file:
            self.train_questions = json.load(file)['questions']
        with open(self.a_train_path, "r") as file:
            self.train_answers = json.load(file)['annotations']
        with open(self.q_val_path, "r") as file:
            self.val_questions = json.load(file)['questions']
        with open(self.a_val_path, "r") as file:
            self.val_answers = json.load(file)['annotations']
        occurence = self.filter_answers(self.train_answers, self.min_occurence)
        self.create_ans2label(occurence, 'trainval', "data/cache")
        self.compute_target()
        self.get_word_dict()

        self.data = []
        for i, filename in enumerate(os.listdir(self.images_path)):
            print(f"at image {i}")
            image_filename = filename
            # image = Image.open(f"{self.images_path}/{image_filename}").convert("RGB")
            # image = self.transform(image)
            # with open(f"{filename}", "wb") as file:
            #     pickle.dump(image, file)
            image_id = int(filename.split('_')[2].split('.')[0])
            related_answers = self.get_answers_for_image(image_id)
            related_questions = self.get_questions_for_answers(related_answers)
            for question, answer in zip(related_questions, related_answers):
                q_idxs = self.get_q_idxs(question["question"])
                answer = {"labels" : answer["labels"], "scores":answer["scores"]}
                answer["labels"] = answer["labels"][:len(answer["labels"])] + [0] * (10 - len(answer["labels"]))
                answer["scores"] = answer["scores"][:len(answer["scores"])] + [0] * (10 - len(answer["scores"]))

                # ans = torch.zeros((2,10))
                # ans[0,:len(answer["labels"])] = torch.tensor(answer["labels"])
                # ans[1,:len(answer["labels"])] = torch.tensor(answer["scores"])
                self.data.append((filename, q_idxs, answer))
        self.train_questions = [0]
        # self.train_answers = [0]
        self.val_questions = [0]
        self.val_answers = [0]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def show(self, index):
        image, question, answer = self.data[index]
        img = Image.fromarray(image, 'RGB')
        img.show()
        print(question["question"])
        print(answer["multiple_choice_answer"])


# In[221]:
if __name__ == '__main__':
    q_train_path = "/datashare/v2_OpenEnded_mscoco_train2014_questions.json"
    q_val_path = "/datashare/v2_OpenEnded_mscoco_val2014_questions.json"
    a_train_path = "/datashare/v2_mscoco_train2014_annotations.json"
    a_val_path = "/datashare/v2_mscoco_val2014_annotations.json"
    train_images_path = "/datashare/train2014"
    val_images_path = "/datashare/val2014"
    # q_train_path = "v2_OpenEnded_mscoco_train2014_questions.json"
    # q_val_path = "v2_OpenEnded_mscoco_val2014_questions.json"
    # a_train_path = "v2_mscoco_train2014_annotations.json"
    # a_val_path = "v2_mscoco_val2014_annotations.json"
    # train_images_path = "./images"
    # val_images_path = "./images"

    vqa_train = VQAdataset(train_images_path, q_train_path, q_val_path, a_train_path, a_val_path, 20)
    vqa_val = VQAdataset(val_images_path, q_train_path, q_val_path, a_train_path, a_val_path, 20, val=True)







    vqa_train.build_dataset()

    with open("vqa_train_dataset.pkl", "wb") as file:
        pickle.dump(vqa_train,file)

    vqa_val.build_dataset()

    with open("vqa_val_dataset.pkl", "wb") as file:
        pickle.dump(vqa_val,file)


# In[222]:




