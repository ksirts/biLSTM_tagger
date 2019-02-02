import sys
sys.stderr.flush()

import torch.nn as nn
import torch

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CharacterEncoder(nn.Module):
    ''' The model with only character-based embeddings.
    Word embeddings are constructed by CharEmbeddings object and passed
    to the forward method.
    '''

    def __init__(self, logger, hidden_dim, tagset_size, embedding_dim):
        self.logger = logger
        super(CharacterEncoder, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.5)

        # The linear layer that maps from hidden state space to tag space
        linear_in = 2 * hidden_dim
        self.hidden2tag = nn.Linear(linear_in, tagset_size)

    def forward(self, lengths, char_embeddings):

        embeds = self.dropout(char_embeddings)

        lengths = lengths.reshape(-1)
        embeds_pack = pack_padded_sequence(embeds, lengths, batch_first=True)
        pack_lstm_out, _ = self.lstm(embeds_pack)
        lstm_out, _ = pad_packed_sequence(pack_lstm_out, batch_first=True)
        lstm_out = self.dropout(lstm_out)

        tag_space = self.hidden2tag(lstm_out)
        return tag_space


class WordEncoder(nn.Module):
    ''' The model with word embeddings and optional character embeddings.
       If character embeddings are used then they constructed by CharEmbeddings object and passed
       to the forward method.
    '''

    def __init__(self, logger, word_embedding_dim, vocab_size, embedding_dim, hidden_dim, tagset_size, fixed=False):
        super(WordEncoder, self).__init__()
        self.logger = logger

        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)
        if fixed:
            self.logger.info('# Freezing word embedding layer ...')
            self.word_embeddings.weight.requires_grad = False

        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.5)

        # The linear layer that maps from hidden state space to tag space
        linear_in = 2 * hidden_dim
        self.hidden2tag = nn.Linear(linear_in, tagset_size)

    def forward(self, words, lengths, char_embeddings=None):

        word_embeds = self.word_embeddings(words)
        word_embeds = self.dropout(word_embeds)


        if char_embeddings is not None:
            char_embeds = self.dropout(char_embeddings)
            embeds = torch.cat([word_embeds, char_embeds], dim=2)
        else:
            embeds = word_embeds

        lengths = lengths.reshape(-1)
        embeds_pack = pack_padded_sequence(embeds, lengths, batch_first=True)
        pack_lstm_out, _ = self.lstm(embeds_pack)
        lstm_out, _ = pad_packed_sequence(pack_lstm_out, batch_first=True)
        lstm_out = self.dropout(lstm_out)

        tag_space = self.hidden2tag(lstm_out)
        return tag_space


class FixedWordEncoder(nn.Module):
    ''' The model with word embeddings and optional character embeddings.
       If character embeddings are used then they constructed by CharEmbeddings object and passed
       to the forward method.
    '''

    def __init__(self, logger, word_embedding_dim, vocab_size, embedding_dim, hidden_dim, tagset_size, trained_vectors):
        super(FixedWordEncoder, self).__init__()
        self.logger = logger
        self.trained_vectors = trained_vectors
        self.word_embedding_dim = word_embedding_dim

        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)
        self.logger.info('# Freezing word embedding layer ...')
        self.word_embeddings.weight.requires_grad = False

        assert trained_vectors < vocab_size

        self.logger.info('# Creating the embedding layer for trainable embeddings')
        self.logger.info('# Number of trainable embedddings: {:d}'.format(vocab_size-trained_vectors))
        self.train_embeddings = nn.Embedding(vocab_size, word_embedding_dim)

        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.5)

        # The linear layer that maps from hidden state space to tag space
        linear_in = 2 * hidden_dim
        self.hidden2tag = nn.Linear(linear_in, tagset_size)

    def forward(self, words, lengths, char_embeddings=None):
        print(self.train_embeddings.weight.data[0])
        self.train_embeddings.weight.data[:self.trained_vectors] = torch.zeros(self.trained_vectors, self.word_embedding_dim)
        print(self.train_embeddings.weight.data[0])

        fixed_embeds = self.word_embeddings(words)
        trained_embeds = self.train_embeddings(words)
        word_embeds = fixed_embeds + trained_embeds
        word_embeds = self.dropout(word_embeds)


        if char_embeddings is not None:
            char_embeds = self.dropout(char_embeddings)
            embeds = torch.cat([word_embeds, char_embeds], dim=2)
        else:
            embeds = word_embeds

        lengths = lengths.reshape(-1)
        embeds_pack = pack_padded_sequence(embeds, lengths, batch_first=True)
        pack_lstm_out, _ = self.lstm(embeds_pack)
        lstm_out, _ = pad_packed_sequence(pack_lstm_out, batch_first=True)
        lstm_out = self.dropout(lstm_out)

        tag_space = self.hidden2tag(lstm_out)
        return tag_space


class LSTMTagger(nn.Module):

    def __init__(self, hidden_dim, tagset_size, embedding_dim, word_embedding_dim=0, vocab_size=0,
                 freeze=False, input_projection=False, special_symbols=None,
                 bos_index=None, eos_index=None, unk_index=None, pad_index=None):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        if word_embedding_dim > 0 and vocab_size > 0:
            print('# Creating word embedding layer ...', file=sys.stderr)
            self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)
            self.embeds_dropout = nn.Dropout(p=0.5)
            if freeze:
                print('# Freezing word embedding layer ...', file=sys.stderr)
                self.word_embeddings.weight.requires_grad = False

        if input_projection:
            assert word_embedding_dim > 0
            print('# Creating input projection layer ...', file=sys.stderr)
            self.embed2input = nn.Linear(word_embedding_dim, word_embedding_dim)
            self.input_dropout = nn.Dropout(p=0.5)

        if special_symbols is not None:
            assert type(special_symbols) == 'list'
            self.special_embeddings = nn.Embedding(len(special_symbols) + 1, word_embedding_dim,
                                                    padding_idx=len(special_symbols))
            self.special_pad = len(special_symbols)
            self.special_symbols = special_symbols

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.output_dropout = nn.Dropout(p=0.5)

        # The linear layer that maps from hidden state space to tag space
        linear_in = 2 * hidden_dim
        self.hidden2tag = nn.Linear(linear_in, tagset_size)

    def forward(self, lengths, words=None, char_embeddings=None, word_embeddings=None):

        # Char only model
        if char_embeddings is not None and words is None and word_embeddings is None:
            embeds = char_embeddings
            embeds = self.embeds_dropout(embeds)

        # Fine-tuned word model with or without char embeddings
        elif words is not None and word_embeddings is None:
            assert hasattr(self, 'word_embeddings')
            word_embeds = self.word_embeddings(words)
            word_embeds = self.embeds_dropout(word_embeds)
            if hasattr(self, 'embed2input'):
                word_embeds = self.embed2input(word_embeds)
                word_embeds = self.input_dropout(word_embeds)

            if char_embeddings is not None:
                char_embeds = self.embeds_dropout(char_embeddings)
                embeds = torch.cat([word_embeds, char_embeds], dim=2)
            else:
                embeds = word_embeds

        # Model with given word embeddings with or without char embeddings
        elif word_embeddings is not None and words is None:
            word_embeds = word_embeddings
            if hasattr(self, 'special_embeddings'):
                mask = words == self.special_symbols[0]
                for i in range(1, len(self.special_symbols)):
                    mask |= words == self.special_symbols[i]
                mask = ~mask
                words[mask] = self.special_pad

                special_embeds = self.special_embeddings(words)
                word_embeds += special_embeds

            if hasattr(self, 'embed2input'):
                word_embeds = self.embed2input(word_embeds)
                word_embeds = self.input_dropout(word_embeds)

            if char_embeddings is not None:
                char_embeds = self.embeds_dropout(char_embeddings)
                embeds = torch.cat([word_embeds, char_embeds], dim=2)
            else:
                embeds = word_embeds

        lengths = lengths.reshape(-1)
        embeds_pack = pack_padded_sequence(embeds, lengths, batch_first=True)
        pack_lstm_out, _ = self.lstm(embeds_pack)
        lstm_out, _ = pad_packed_sequence(pack_lstm_out, batch_first=True)
        lstm_out = self.output_dropout(lstm_out)

        tag_space = self.hidden2tag(lstm_out)
        return tag_space