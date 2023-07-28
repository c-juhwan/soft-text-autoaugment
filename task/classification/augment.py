# Standard Library Modules
import os
import sys
import pickle
import random
import argparse
# 3rd-party Modules
import bs4
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
# Pytorch Modules
import torch
# Huggingface Modules
from transformers import AutoTokenizer
from textaugment import EDA
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path, get_huggingface_model_name

def load_preprocessed_data(args: argparse.Namespace) -> dict:
    """
    Open preprocessed train pickle file from local directory.

    Args:
        args (argparse.Namespace): Arguments.

    Returns:
        train_data (dict): Preprocessed training data.
    """

    preprocessed_path = os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type, f'train_original_{args.data_subsample_size}.pkl')

    with open(preprocessed_path, 'rb') as f:
        train_data = pickle.load(f)

    return train_data

def augmentation(args: argparse.Namespace) -> None:
    train_data = load_preprocessed_data(args)

    augmented_data = {
        'input_ids': [],
        'attention_mask': [],
        'token_type_ids': [],
        'labels': [],
        'soft_labels': [],
        'num_classes': train_data['num_classes'],
        'vocab_size': train_data['vocab_size'],
        'pad_token_id': train_data['pad_token_id'],
        'augmentation_policy': train_data['augmentation_policy'], # None.
    }

    # Reconstruct sentence from input_ids using huggingface tokenizer
    model_name = get_huggingface_model_name(args.model_type)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for idx in tqdm(range(len(train_data['input_ids'])), desc=f'Augmenting with {args.augmentation_type}'):
        decoded_sent = tokenizer.decode(train_data['input_ids'][idx], skip_special_tokens=True)

        # Apply each augmentation
        if args.augmentation_type == 'hard_eda':
            augmented_sent = run_eda(decoded_sent, args)
            augmented_data['soft_labels'].append(train_data['soft_labels'][idx]) # Keep the soft labels as they are
        elif args.augmentation_type == 'soft_eda':
            augmented_sent = run_eda(decoded_sent, args)
            soft_labels = train_data['soft_labels'][idx] * (1 - args.softeda_smoothing) + (args.softeda_smoothing / train_data['num_classes']) # Apply label smoothing
            augmented_data['soft_labels'].append(soft_labels) # Apply label-smoothened soft labels
        elif args.augmentation_type == 'aeda':
            augmented_sent = run_aeda(decoded_sent, args)
            augmented_data['soft_labels'].append(train_data['soft_labels'][idx]) # Keep the soft labels as they are
        else:
            raise ValueError(f'Invalid augmentation type: {args.augmentation_type}')
        augmented_data['labels'].append(train_data['labels'][idx]) # Keep the hard labels as they are

        # Encode augmented sentence using huggingface tokenizer
        tokenized_sent = tokenizer(augmented_sent, padding='max_length', truncation=True,
                                   max_length=args.max_seq_len, return_tensors='pt')

        # Append augmented data
        augmented_data['input_ids'].append(tokenized_sent['input_ids'].squeeze())
        augmented_data['attention_mask'].append(tokenized_sent['attention_mask'].squeeze())
        if args.model_type in ['bert', 'albert', 'electra', 'deberta', 'debertav3']:
            augmented_data['token_type_ids'].append(tokenized_sent['token_type_ids'].squeeze())
        else: # roberta does not use token_type_ids
            augmented_data['token_type_ids'].append(torch.zeros(args.max_seq_len, dtype=torch.long))

    # Merge augmented data with original data
    total_dict = {
        'input_ids': train_data['input_ids'] + augmented_data['input_ids'],
        'attention_mask': train_data['attention_mask'] + augmented_data['attention_mask'],
        'token_type_ids': train_data['token_type_ids'] + augmented_data['token_type_ids'],
        'labels': train_data['labels'] + augmented_data['labels'],
        'soft_labels': train_data['soft_labels'] + augmented_data['soft_labels'],
        'num_classes': train_data['num_classes'],
        'vocab_size': train_data['vocab_size'],
        'pad_token_id': train_data['pad_token_id'],
        'augmentation_policy': train_data['augmentation_policy'], # None.
    }

    # Save total data as pickle file
    save_path = os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type)
    check_path(save_path)
    with open(os.path.join(save_path, f'train_{args.augmentation_type}_{args.data_subsample_size}.pkl'), 'wb') as f:
        pickle.dump(total_dict, f)

def run_eda(sentence: str, args: argparse.Namespace) -> str:
    augmenter = EDA()

    len_words = len(sentence.split(' '))
    n_to_modify = max(1, int(args.eda_alpha * len_words)) # Default value of alpha is 0.1 - Modify 10% of the words in the sentence

    augmentation_type = np.random.choice(['SR', 'RI', 'RS', 'RD'], p=[0.25, 0.25, 0.25, 0.25]) # Select augmentation type randomly

    if augmentation_type == 'SR':
        augmented_sentence = augmenter.synonym_replacement(sentence, n=n_to_modify)
    elif augmentation_type == 'RI':
        augmented_sentence = augmenter.random_insertion(sentence, n=n_to_modify)
    elif augmentation_type == 'RS':
        augmented_sentence = augmenter.random_swap(sentence, n=n_to_modify)
    elif augmentation_type == 'RD':
        augmented_sentence = augmenter.random_deletion(sentence, p=0.1)

    return augmented_sentence

def run_aeda(sentence: str, args: argparse.Namespace) -> str:
    """
    Main function to perform AEDA.
    Default value of alpha is 0.1 - Add 10% punctuation marks to the sentence

    Args:
        sentence (str): The sentence to be augmented.
        args (argparse.Namespace): The arguments passed to the program.

    Returns:
        str: The augmented sentence.
    """

    words = sentence.split(' ')
    words = [word for word in words if word != '']
    len_words = len(words)
    n_aeda = max(1, int(args.eda_alpha * len_words))

    new_words = aeda_random_insertion(words, n_aeda)

    # Join the words to form the sentence again
    augmented_sentence = ' '.join(words)
    return augmented_sentence

def aeda_random_insertion(words: list, n: int) -> list:
    """
    Following AEDA: An Easier Data Augmentation Technique for Text Classification
    Karimi et al., EMNLP Findings 2021, https://aclanthology.org/2021.findings-emnlp.234.pdf

    This function is also a random inserstion, but only inserts punctuation marks rather than synonyms.

    Args:
        words (list): The list of words in the sentence.
        n (int): The number of punctuation marks to be inserted.
    Return:
        list: The list of words in the sentence after insertion.
    """

    punct_marks = ['.', ';', '?', ':', '!', ','] # AEDA uses only these six punctuation marks
    new_words = words.copy()

    for _ in range(n):
        random_idx = random.randint(0, len(new_words)-1) # Pick a random index to insert the punctuation mark
        random_punct = punct_marks[random.randint(0, len(punct_marks)-1)] # Pick a random punctuation mark to insert

        new_words.insert(random_idx, random_punct) # Insert the punctuation mark at the random index

    return new_words
