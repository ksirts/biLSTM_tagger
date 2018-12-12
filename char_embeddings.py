import torch.nn as nn
import torch

from torch.nn.utils.rnn import pack_padded_sequence


class CharEmbeddings(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(CharEmbeddings, self).__init__()
        self.hidden_dim = hidden_dim

        self.char_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, chars, lengths):
        chars_size = chars.size()
        # Aggregate sequence length and batch dimensions
        chars = chars.reshape(-1, chars_size[-1])
        # Embed characters
        embeds = self.char_embeddings(chars)
        embeds = self.dropout(embeds)

        # Sort and pack the embeddings
        lengths = lengths.reshape(-1)
        lengths_sort, idx_sort = torch.sort(lengths, dim=0, descending=True)
        # Replace 0 entries with 1s
        lengths_sort[lengths_sort == 0] = 1
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        embeds_sort = embeds.index_select(0, idx_sort)
        embeds_pack = pack_padded_sequence(embeds_sort, lengths_sort, batch_first=True)

        # Send the pack through LSTM
        _, hidden = self.lstm(embeds_pack)

        # Concatenate states to get word embeddings and
        # permute the batch to the beginning
        word_embeds = torch.cat(hidden, dim=2)
        word_embeds = word_embeds.permute(1, 0, 2)

        # Compute the last dim of the output
        dim = 4 * self.hidden_dim

        # Reshape back to (batch x sequence) x dimension
        word_embeds = word_embeds.reshape(-1, dim)

        # Restore the original index ordering
        word_embeds = word_embeds.index_select(0, idx_unsort)

        # Reshape back to original shape
        word_embeds = word_embeds.reshape(chars_size[0], chars_size[1], -1)

        return word_embeds
