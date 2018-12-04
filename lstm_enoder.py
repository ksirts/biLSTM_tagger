import sys
sys.stderr.flush()

import torch.nn as nn
import torch

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMTagger(nn.Module):

    def __init__(self, hidden_dim, tagset_size, embedding_dim, word_embedding_dim=0, vocab_size=0,
                 freeze=False, input_projection=False):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        if word_embedding_dim > 0 and vocab_size > 0:
            print('# Creating word embedding layer ...', file=sys.stderr)
            self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)
            if freeze:
                print('# Freezing word embedding layer ...', file=sys.stderr)
                self.word_embeddings.weight.requires_grad = False

        if input_projection:
            assert word_embedding_dim > 0
            print('# Creating input projection layer ...', file=sys.stderr, flush=True)
            self.embed2input = nn.Linear(word_embedding_dim, word_embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        linear_in = 2 * hidden_dim
        self.hidden2tag = nn.Linear(linear_in, tagset_size)

    def forward(self, lengths, words=None, char_embeds=None):

        # Char only model
        if words is None and char_embeds is not None:
            embeds = char_embeds

        # Word only model
        elif words is not None and char_embeds is None:

            if hasattr(self, 'word_embeddings'):
                words = self.word_embeddings(words)

            if hasattr(self, 'embed2input'):
                words = self.embed2input(words)
            embeds = words

        # Word and char model
        else:
            assert words is not None
            assert char_embeds is not None

            if hasattr(self, 'word_embeddings'):
                words = self.word_embeddings(words)

            if hasattr(self, 'embed2input'):
                words = self.embed2input(words)

            embeds = torch.cat([words, char_embeds], dim=2)


        lengths = lengths.reshape(-1)
        embeds_pack = pack_padded_sequence(embeds, lengths, batch_first=True)
        pack_lstm_out, _ = self.lstm(embeds_pack)
        lstm_out, _ = pad_packed_sequence(pack_lstm_out, batch_first=True)

        tag_space = self.hidden2tag(lstm_out)
        return tag_space