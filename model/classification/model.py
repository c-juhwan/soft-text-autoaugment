# Standard Library Modules
import os
import sys
import argparse
# Pytorch Modules
import torch
import torch.nn as nn
# Huggingface Modules
from transformers import AutoConfig, AutoModel
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import get_huggingface_model_name

class ClassificationModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(ClassificationModel, self).__init__()
        self.args = args

        if args.model_type == 'cnn':
            each_out_size = args.embed_size // 3

            self.embed = nn.Embedding(args.vocab_size, args.embed_size)
            self.conv = nn.ModuleList(
                [nn.Conv1d(in_channels=args.embed_size, out_channels=each_out_size,
                           kernel_size=kernel_size, stride=1, padding='same', bias=False)
                           for kernel_size in [3, 4, 5]]
            )

            self.classifier = nn.Sequential(
                nn.Linear(each_out_size * 3, args.hidden_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU(),
                nn.Linear(args.hidden_size, args.num_classes)
            )
        elif args.model_type in ['lstm', 'gru', 'rnn']:
            self.embed = nn.Embedding(args.vocab_size, args.embed_size)

            if args.model_type == 'lstm':
                self.rnn = nn.LSTM(input_size=args.embed_size, hidden_size=args.hidden_size, num_layers=args.num_layers_rnn, batch_first=True, bidirectional=args.rnn_isbidirectional)
            elif args.model_type == 'gru':
                self.rnn = nn.GRU(input_size=args.embed_size, hidden_size=args.hidden_size, num_layers=args.num_layers_rnn, batch_first=True, bidirectional=args.rnn_isbidirectional)
            elif args.model_type == 'rnn':
                self.rnn = nn.RNN(input_size=args.embed_size, hidden_size=args.hidden_size, num_layers=args.num_layers_rnn, batch_first=True, bidirectional=args.rnn_isbidirectional)

            self.linear_size = args.hidden_size * (2 if args.rnn_isbidirectional else 1) * args.num_layers_rnn
            self.classifier = nn.Sequential(
                nn.Linear(self.linear_size, args.hidden_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU(),
                nn.Linear(args.hidden_size, args.num_classes)
            )
        elif args.model_type == 'transformer_enc':
            self.embed = nn.Embedding(args.vocab_size, args.embed_size)
            self.pos_embed = nn.Embedding(args.max_seq_len, args.embed_size)

            encoder_layer = nn.TransformerEncoderLayer(d_model=args.embed_size, nhead=args.num_heads_transformer)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.num_layers_transformer)

            self.classifier = nn.Sequential(
                nn.Linear(args.embed_size, args.hidden_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU(),
                nn.Linear(args.hidden_size, args.num_classes)
            )
        else: # Huggingface models
            # Define model
            huggingface_model_name = get_huggingface_model_name(self.args.model_type)
            self.config = AutoConfig.from_pretrained(huggingface_model_name)
            if args.model_ispretrained:
                self.model = AutoModel.from_pretrained(huggingface_model_name)
            else:
                self.model = AutoModel.from_config(self.config)

            self.embed_size = self.model.config.hidden_size
            self.hidden_size = self.model.config.hidden_size
            self.num_classes = self.args.num_classes

            # Define classifier - custom classifier is more flexible than using BERTforSequenceClassification
            # For example, you can use soft labels for training, etc.
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Dropout(self.args.dropout_rate),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.num_classes),
            )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        if self.args.model_type == 'cnn':
            embed = self.embed(input_ids)
            embed = embed.transpose(1, 2) # (batch_size, embed_size, seq_len)

            conv_output = [conv(embed) for conv in self.conv] # [(batch_size, each_out_size, seq_len), ...]
            # Apply global max pooling to each conv output
            conv_output = [torch.max(conv, dim=2)[0] for conv in conv_output] # [(batch_size, each_out_size), ...]
            conv_output = torch.cat(conv_output, dim=1) # (batch_size, each_out_size * 3)

            classification_logits = self.classifier(conv_output) # (batch_size, num_classes)
        elif self.args.model_type in ['lstm', 'gru', 'rnn']:
            embed = self.embed(input_ids) # (batch_size, seq_len, embed_size)

            # Apply RNN
            # rnn_output, _ = self.rnn(embed) # (batch_size, seq_len, hidden_size * num_directions)
            # rnn_output = rnn_output[:, -1, :] # (batch_size, hidden_size * num_directions) # Use last output token as sentence representation
            # classification_logits = self.classifier(rnn_output) # (batch_size, num_classes)

            _, rnn_hidden = self.rnn(embed) # (num_layers * num_directions, batch_size, hidden_size)
            if self.args.model_type == 'lstm':
                rnn_hidden = rnn_hidden[0] # Discard cell state
            rnn_hidden = rnn_hidden.permute(1, 0, 2) # (batch_size, num_layers * num_directions, hidden_size)

            rnn_hidden = rnn_hidden.reshape(rnn_hidden.size(0), -1) # (batch_size, num_layers * num_directions * hidden_size)
            classification_logits = self.classifier(rnn_hidden) # (batch_size, num_classes)
        elif self.args.model_type == 'transformer_enc':
            word_embed = self.embed(input_ids) # (batch_size, seq_len, embed_size)
            pos_embed = self.pos_embed(torch.arange(input_ids.size(1), device=input_ids.device)) # (seq_len, embed_size)
            pos_embed = pos_embed.unsqueeze(0).repeat(input_ids.size(0), 1, 1) # (batch_size, seq_len, embed_size)
            embed = word_embed + pos_embed # (batch_size, seq_len, embed_size)
            embed = embed.permute(1, 0, 2) # (seq_len, batch_size, embed_size)
            src_key_padding_mask = (input_ids == self.args.pad_token_id) # (batch_size, seq_len)

            transformer_output = self.transformer_encoder(embed, src_key_padding_mask=src_key_padding_mask) # (seq_len, batch_size, embed_size)
            transformer_output = transformer_output.permute(1, 0, 2) # (batch_size, seq_len, embed_size)
            cls_output = transformer_output.mean(dim=1) # (batch_size, embed_size) - Global average pooling

            classification_logits = self.classifier(cls_output) # (batch_size, num_classes)
        else: # Huggingface models
            model_output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
            cls_output = model_output.last_hidden_state[:, 0, :] # (batch_size, hidden_size)
            classification_logits = self.classifier(cls_output) # (batch_size, num_classes)

        return classification_logits
