# Standard Library Modules
import os
import sys
import pickle
import argparse
# 3rd-party Modules
import bs4
import pandas as pd
from tqdm.auto import tqdm
# Pytorch Modules
import torch
# Huggingface Modules
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path, get_huggingface_model_name

def load_data(args: argparse.Namespace) -> tuple: # (dict, dict, dict, int)
    """
    Load data from huggingface datasets.
    If dataset is not in huggingface datasets, takes data from local directory.

    Args:
        dataset_name (str): Dataset name.
        args (argparse.Namespace): Arguments.
        train_valid_split (float): Train-valid split ratio.

    Returns:
        train_data (dict): Training data. (text, label)
        valid_data (dict): Validation data. (text, label)
        test_data (dict): Test data. (text, label)
        num_classes (int): Number of classes.
    """

    name = args.task_dataset.lower()
    train_valid_split = args.train_valid_split

    train_data = {
        'text': [],
        'label': []
    }
    valid_data = {
        'text': [],
        'label': []
    }
    test_data = {
        'text': [],
        'label': []
    }

    if name == 'sst2':
        dataset = load_dataset('SetFit/sst2')

        train_df = pd.DataFrame(dataset['train'])
        valid_df = pd.DataFrame(dataset['validation'])
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    if name == 'sst5':
        dataset = load_dataset('SetFit/sst5')

        train_df = pd.DataFrame(dataset['train'])
        valid_df = pd.DataFrame(dataset['validation'])
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 5

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'cola':
        dataset = load_dataset('linxinyuan/cola')

        train_df = pd.DataFrame(dataset['train'])
        #valid_df = pd.DataFrame(dataset['validation']) # No pre-defined validation set
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        train_df = train_df.sample(frac=1).reset_index(drop=True) # Shuffle train data before split
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'imdb':
        dataset = load_dataset('imdb')

        train_df = pd.DataFrame(dataset['train'])
        #valid_df = pd.DataFrame(dataset['validation']) # No pre-defined validation set
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        # train-valid split
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'trec':
        dataset = load_dataset('trec')

        train_df = pd.DataFrame(dataset['train'])
        #valid_df = pd.DataFrame(dataset['validation']) # No pre-defined validation set
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 6

        # train-valid split
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['coarse_label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['coarse_label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['coarse_label'].tolist()
    elif name == 'subj':
        dataset = load_dataset('SetFit/subj')

        train_df = pd.DataFrame(dataset['train'])
        #valid_df = pd.DataFrame(dataset['validation']) # No pre-defined validation set
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        # train-valid split
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'agnews':
        dataset = load_dataset('ag_news')

        train_df = pd.DataFrame(dataset['train'])
        #valid_df = pd.DataFrame(dataset['validation']) # No pre-defined validation set
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 4

        # train-valid split
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'mr':
        # There is no MR dataset in HuggingFace datasets, so we use custom file - check dataset/MR
        train_path = os.path.join(args.data_path, 'MR', 'train.csv')
        test_path = os.path.join(args.data_path, 'MR', 'test.csv')
        train_df = pd.read_csv(train_path, header=None)
        test_df = pd.read_csv(test_path, header=None)
        num_classes = 2

        # train-valid split
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df[1].tolist()
        train_data['label'] = train_df[0].tolist()
        valid_data['text'] = valid_df[1].tolist()
        valid_data['label'] = valid_df[0].tolist()
        test_data['text'] = test_df[1].tolist()
        test_data['label'] = test_df[0].tolist()
    elif name == 'cr':
        dataset = load_dataset('SetFit/SentEval-CR')

        train_df = pd.DataFrame(dataset['train'])
        #valid_df = pd.DataFrame(dataset['validation']) # No pre-defined validation set
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        # train-valid split
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'proscons':
        # There is no MR dataset in HuggingFace datasets, so we use custom file - check dataset/ProsCons
        train_path = os.path.join(args.data_path, 'ProsCons', 'train.csv')
        test_path = os.path.join(args.data_path, 'ProsCons', 'test.csv')
        train_df = pd.read_csv(train_path, header=None)
        test_df = pd.read_csv(test_path, header=None)
        num_classes = 2

        # train-valid split
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df[1].tolist()
        train_data['label'] = train_df[0].tolist()
        valid_data['text'] = valid_df[1].tolist()
        valid_data['label'] = valid_df[0].tolist()
        test_data['text'] = test_df[1].tolist()
        test_data['label'] = test_df[0].tolist()
    elif name == 'dbpedia':
        dataset = load_dataset('dbpedia_14')

        train_df = pd.DataFrame(dataset['train'])
        #valid_df = pd.DataFrame(dataset['validation']) # No pre-defined validation set
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 14

        # train-valid split
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['content'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['content'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['content'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'yelp_polarity':
        dataset = load_dataset('yelp_polarity')

        train_df = pd.DataFrame(dataset['train'])
        #valid_df = pd.DataFrame(dataset['validation']) # No pre-defined validation set
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        # train-valid split
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'yelp_full':
        dataset = load_dataset('yelp_full')

        train_df = pd.DataFrame(dataset['train'])
        #valid_df = pd.DataFrame(dataset['validation']) # No pre-defined validation set
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 5

        # train-valid split
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'yahoo_answers_title':
        dataset = load_dataset('yahoo_answers_topics')

        train_df = pd.DataFrame(dataset['train'])
        #valid_df = pd.DataFrame(dataset['validation']) # No pre-defined validation set
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 10

        # train-valid split
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['question_title'].tolist()
        train_data['label'] = train_df['topic'].tolist()
        valid_data['text'] = valid_df['question_title'].tolist()
        valid_data['label'] = valid_df['topic'].tolist()
        test_data['text'] = test_df['question_title'].tolist()
        test_data['label'] = test_df['topic'].tolist()
    elif name == 'yahoo_answers_full':
        dataset = load_dataset('yahoo_answers_topics')

        train_df = pd.DataFrame(dataset['train'])
        #valid_df = pd.DataFrame(dataset['validation']) # No pre-defined validation set
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 10

        # train-valid split
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        for i in range(len(train_df)):
            train_data['text'].append(train_df['question_title'][i] + '[SEP]' + train_df['question_content'][i] + '[SEP]' + train_df['best_answer'][i])
            train_data['label'].append(train_df['topic'][i])
        for i in range(len(valid_df)):
            valid_data['text'].append(valid_df['question_title'][i] + '[SEP]' + valid_df['question_content'][i] + '[SEP]' + valid_df['best_answer'][i])
            valid_data['label'].append(valid_df['topic'][i])
        for i in range(len(test_df)):
            test_data['text'].append(test_df['question_title'][i] + '[SEP]' + test_df['question_content'][i] + '[SEP]' + test_df['best_answer'][i])
            test_data['label'].append(test_df['topic'][i])

    return train_data, valid_data, test_data, num_classes

def preprocessing(args: argparse.Namespace) -> None:
    """
    Main function for preprocessing.

    Args:
        args (argparse.Namespace): Arguments.
    """

    # Load data
    train_data, valid_data, test_data, num_classes = load_data(args)

    # Define tokenizer & config
    model_name = get_huggingface_model_name(args.model_type)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

    # Preprocessing - Define data_dict
    data_dict = {
        'train': {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'labels': [],
            'soft_labels': [],
            'num_classes': num_classes,
            'vocab_size': config.vocab_size,
            'pad_token_id': tokenizer.pad_token_id,
            'augmentation_policy': None
        },
        'valid': {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'labels': [],
            'soft_labels': [],
            'num_classes': num_classes,
            'vocab_size': config.vocab_size,
            'pad_token_id': tokenizer.pad_token_id,
            'augmentation_policy': None
        },
        'test': {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'labels': [],
            'soft_labels': [],
            'num_classes': num_classes,
            'vocab_size': config.vocab_size,
            'pad_token_id': tokenizer.pad_token_id,
            'augmentation_policy': None
        }
    }

    # Save data as pickle file
    preprocessed_path = os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type)
    check_path(preprocessed_path)

    for split_data, split in zip([train_data, valid_data, test_data], ['train', 'valid', 'test']):
        for idx in range(len(split_data['text'])):
            # Get text and label
            text = split_data['text'][idx]
            label = split_data['label'][idx]

            # Remove html tags
            clean_text = bs4.BeautifulSoup(text, 'lxml').text
            # Remove special characters
            clean_text = clean_text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
            # Remove multiple spaces
            clean_text = ' '.join(clean_text.split())

            # Tokenize
            tokenized = tokenizer(clean_text, padding='max_length', truncation=True,
                                  max_length=args.max_seq_len, return_tensors='pt')

            soft_label = [0.0] * num_classes
            if label != -1:
                soft_label[label] = 1.0

            # Append data
            data_dict[split]['input_ids'].append(tokenized['input_ids'].squeeze())
            data_dict[split]['attention_mask'].append(tokenized['attention_mask'].squeeze())
            if args.model_type in ['bert', 'albert', 'electra', 'deberta', 'debertav3']:
                data_dict[split]['token_type_ids'].append(tokenized['token_type_ids'].squeeze())
            else: # roberta does not use token_type_ids
                data_dict[split]['token_type_ids'].append(torch.zeros(args.max_seq_len, dtype=torch.long))
            data_dict[split]['labels'].append(torch.tensor(label, dtype=torch.long)) # Cross Entropy Loss
            data_dict[split]['soft_labels'].append(torch.tensor(soft_label, dtype=torch.float)) # Label Smoothing Loss

        # Save data as pickle file
        with open(os.path.join(preprocessed_path, f'{split}_original_full.pkl'), 'wb') as f:
            pickle.dump(data_dict[split], f)

    # Low-resource simulation for train data
    for each_size in [100, 500, 1000, 2000]:
        # Subsample train data
        subsampled_train_data = {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'labels': [],
            'soft_labels': [],
            'num_classes': num_classes,
            'vocab_size': config.vocab_size,
            'pad_token_id': tokenizer.pad_token_id,
            'augmentation_policy': None
        }

        # Randomly select data
        random_idx = torch.randperm(len(data_dict['train']['input_ids']))[:each_size]

        for idx in random_idx:
            subsampled_train_data['input_ids'].append(data_dict['train']['input_ids'][idx])
            subsampled_train_data['attention_mask'].append(data_dict['train']['attention_mask'][idx])
            subsampled_train_data['token_type_ids'].append(data_dict['train']['token_type_ids'][idx])
            subsampled_train_data['labels'].append(data_dict['train']['labels'][idx])
            subsampled_train_data['soft_labels'].append(data_dict['train']['soft_labels'][idx])

        # Save data as pickle file
        with open(os.path.join(preprocessed_path, f'train_original_{each_size}.pkl'), 'wb') as f:
            pickle.dump(subsampled_train_data, f)
