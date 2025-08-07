# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 20:11:43 2025

@author: banern
"""
#Use HuggingFace's interface for dataset loading
from datasets import load_dataset
from transformers import AutoTokenizer #Use Hugging Face's AutoTokenizer 
from collections import defaultdict
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, labels):
        self.tensorsAndMasks = dataset
        self.labels = labels
    def __len__(self):
        return len(self.tensorsAndMasks['input_ids'])
    def __getItem__(self, index):
        return {'input_ids': self.tensorsAndMasks['input_ids'][index], 'attention_mask': self.tensorsAndMasks['attention_mask'][index], 'label': torch.tensor(self.labels[index])}
        
        
    
# Load the GoEmotions dataset
dataset = load_dataset("go_emotions")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
#encoded = tokenizer("I love pizza", padding=True, truncation=True, return_tensors="pt")
#print(encoded)
# View a sample
print(dataset['train'][0])
print(type(dataset))
print(dataset.keys())
print(dataset['validation'][0])
print(type(dataset['train']))
print(type(dataset['validation']))
print(type(dataset['test']))
print(dataset['train'].features)
print(type(dataset['train'].features['labels']))
label_id = dataset['train'][0]['labels'][0]
label_name = dataset['train'].features['labels'].feature.names[label_id]
label_name = dataset['train'].features['labels'].feature.names
print(label_name)  # → e.g., 'contentment'
    
indicies_vocabulary = set({1, 2, 3, 5, 6, 7, 9, 17, 18, 20, 25, 27})
orig_indicies_to_new = dict()
for new_index, orig_index in enumerate(indicies_vocabulary):
    orig_indicies_to_new[orig_index] = new_index
indicies_to_emotions = dict()
emotion_counts = dict()
orig_indicies_keys = list(orig_indicies_to_new.keys())
for i in range(len(orig_indicies_keys)):
    new_index = orig_indicies_to_new[orig_indicies_keys[i]]
    indicies_to_emotions[new_index] = label_name[orig_indicies_keys[i]]
for index in list(indicies_to_emotions.keys()):
    print(indicies_to_emotions[index])
dictionary_data = dataset['train']
index = 0
messages = []
labels = []
for dictionary in dictionary_data:
    if len(dictionary['labels']) > 1 or not dictionary['labels'][0] in indicies_vocabulary: 
        continue
    messages.append(dictionary['text'])
    labels.append(orig_indicies_to_new[dictionary['labels'][0]])
tokenized_dataset = tokenizer(messages, padding=True, truncation=True, return_tensors="pt")
print(tokenized_dataset['input_ids'][0])
print(tokenized_dataset['attention_mask'][0])
split_idx = int(0.9 * len(messages))


#build a clean training dataset, one that does not feature messages with multiple emotions or messages with emotions we are not scanning for
#
    

    

#for dictionary in dictionary_data:"""
    


