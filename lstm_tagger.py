import sys

from argparse import ArgumentParser

import numpy as np

import torch
from torchtext import data
from torchtext import vocab

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchtext import datasets

from torch.nn.utils.rnn import pack_padded_sequence

SEED = 1234

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

parser = ArgumentParser(description="Sequence tagging model")
parser.add_argument('--load', action='store_true', help='Load existing model from file')
parser.add_argument('--chars', action='store_true', help='Use character level model for embeddings')
parser.add_argument('--bi-words', action='store_true', help='Use bidirectional word encoder')
parser.add_argument('--bi-chars', action='store_true', help='Use bidirectional char encoder')
parser.add_argument('--use-adam', action='store_true', help='Use Adam optimizer')
parser.add_argument('--pretrained', action='store_true', help='Use pretrained embeddings')
parser.add_argument('--freeze', action='store_true', help='Freeze pretrained embeddings')
parser.add_argument('--oov-embeddings', action='store_true', help='Load pretrained embeddings for all words')
parser.add_argument('--input-projection', action='store_true', help='Use input projection in the word model')
args = parser.parse_args()

print('# Load from file:\t\t', args.load, file=sys.stderr, flush=True)
print('# Use char model:\t\t', args.chars, file=sys.stderr, flush=True)
print('# BiLSTM for words:\t\t', args.bi_words, file=sys.stderr, flush=True)
print('# BiLSTM for chars:\t\t', args.bi_chars, file=sys.stderr, flush=True)
print('# Use Adam optimizer:\t\t', args.use_adam, file=sys.stderr, flush=True)
print('# Use pretrained embeddings:\t', args.pretrained, file=sys.stderr, flush=True)
print('# Freeze pretrained embeddings:\t', args.freeze, file=sys.stderr, flush=True)
print('# Use embeddings for OOV words:\t', args.oov_embeddings, file=sys.stderr, flush=True)
print('# Use input projection:\t\t', args.input_projection, file=sys.stderr, flush=True)

def sequence_accuracy(words, scores, targets):
    _, predict = torch.max(scores,1)

    mask = torch.ones(words.size())
    mask[words==WORD.vocab.stoi['<bos>']] = 0
    mask[words==WORD.vocab.stoi['<eos>']] = 0
    mask[words==WORD.vocab.stoi['<pad>']] = 0
    total = mask.sum().type(torch.FloatTensor)
    correct = (predict == targets).type(torch.FloatTensor)
    masked_correct = mask * correct
    acc = masked_correct.sum().type(torch.FloatTensor) / total
    return acc

def oov_accuracy(words, scores, targets):
    _, predictions = torch.max(scores, 1)
    data = [[pred, lab] for (word, pred, lab) in zip(words, predictions, targets) 
                    if WORD.vocab.itos[word] == '<unk>']
    if len(data) == 0:
        return None
    else:
        data = np.array(data)
        return np.mean(data[:,0] == data[:,1])

def train(model, iterator, optimizer, criterion, char_model=None, oov_embeds=False):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    if char_model is not None:
        char_model.train()
  
    for i, batch in enumerate(iterator):
        # Clear out gradients
        optimizer.zero_grad()

        # Run forward pass.
        
        words, lengths = batch.word
        char_embeddings = None
        
        if char_model:
            chars, _, char_lengths = batch.char
            char_embeddings = char_model(chars, char_lengths)

        if oov_embeds:
            words = words.cpu()
            words = F.embedding(words, WORD.vocab.vectors)
            words = words.cuda()

        predictions = model(words, char_embeddings)
        predictions = predictions.reshape(-1, predictions.size()[-1])
        labels = batch.udtag.reshape(-1)
        words = batch.word[0].reshape(-1)

        # Compute the loss, gradients, and update the parameters by
        #calling optimizer.step()
        loss = criterion(predictions, labels)
        acc = sequence_accuracy(words, predictions, labels)
        
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
        # if i % 10 == 0:
        #     print(f'| Batch: {i:02} | Batch Loss: {loss:.3f} | Batch Acc: {acc*100:.2f}%')
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, char_model=None, oov_embeds=False):
    
    epoch_loss = 0.0
    epoch_acc = 0.0
    epoch_oov_acc = 0.0
    oov_batches = 0
    
    model.eval()
    
    if char_model is not None:
        char_model.eval()
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):
            
            words, lengths = batch.word
            char_embeddings = None
        
            if char_model:
                chars, _, char_lengths = batch.char
                char_embeddings = char_model(chars, char_lengths)

            if oov_embeds:
                words = words.cpu()
                words = F.embedding(words, WORD.vocab.vectors)
                words = words.cuda()

            predictions = model(words, char_embeddings)
            predictions = predictions.reshape(-1, predictions.size()[-1])
            labels = batch.udtag.reshape(-1)
            words = batch.word[0].reshape(-1)
            
            loss = criterion(predictions, labels)
            acc = sequence_accuracy(words, predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            words = batch.word[0].reshape(-1)
            oov_acc = oov_accuracy(words, predictions, labels)
            if oov_acc:
                epoch_oov_acc += oov_acc
                oov_batches += 1
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_oov_acc / oov_batches


class CharEmbeddings(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size, bidirectional=False):
        super(CharEmbeddings, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional=bidirectional
        
        self.char_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        
    def forward(self, chars, lengths):
        chars_size = chars.size()
        # Aggregate sequence length and batch dimensions
        chars = chars.reshape(-1, chars_size[-1])
        # Embed characters
        embeds = self.char_embeddings(chars)

        # Sort and pack the embeddings
        lengths = lengths.reshape(-1)
        lengths_sort, idx_sort = torch.sort(lengths, dim=0, descending=True)
        # Replace 0 entries with 1s
        lengths_sort[lengths_sort==0] = 1
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        embeds_sort = embeds.index_select(0, idx_sort)
        embeds_pack= pack_padded_sequence(embeds_sort, lengths_sort, batch_first=True)

        # Send the pack through LSTM
        _, hidden = self.lstm(embeds_pack)
        
        # Concatenate states to get word embeddings
        word_embeds = torch.cat(hidden, dim=2)

        # Compute the last dim of the output
        dim = 2 * self.hidden_dim
        
        # If the encoder is bidirectional then permute the batch to the beginning
        # and compute the last dim of the output
        if self.bidirectional:
            word_embeds = word_embeds.permute(1, 0, 2)
            dim = 4 * self.hidden_dim
        
        # Reshape back to (batch x sequence) x dimension
        word_embeds = word_embeds.reshape(-1, dim)

        # Restore the original index ordering
        word_embeds = word_embeds.index_select(0, idx_unsort)
        # Reshape back to original shape
        word_embeds = word_embeds.reshape(chars_size[0], chars_size[1], -1)
        return word_embeds


class LSTMTagger(nn.Module):

    def __init__(self, word_embedding_dim, embedding_dim, hidden_dim, tagset_size, vocab_size=None, bidirectional=False, freeze=False, input_projection=False):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        if vocab_size is not None:
            print('# Creating word embedding layer ...', file=sys.stderr, flush=True)
            self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)
            if freeze:
                print('# Freezing word embedding layer ...', file=sys.stderr, flush=True)
                self.word_embeddings.weight.requires_grad = False

        if input_projection:
            print('# Creating input projection layer ...', file=sys.stderr, flush=True)
            self.embed2input = nn.Linear(word_embedding_dim, word_embedding_dim)


        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)

        # The linear layer that maps from hidden state space to tag space
        linear_in = hidden_dim
        if bidirectional:
            linear_in = 2 * hidden_dim
        self.hidden2tag = nn.Linear(linear_in, tagset_size)


    def forward(self, words, char_embeds=None):
        if hasattr(self, 'word_embeddings'):
            words = self.word_embeddings(words)

        if hasattr(self, 'embed2input'):
            words = self.embed2input(words)
        
        if char_embeds is not None:
            embeds = torch.cat([words, char_embeds], dim=2)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        return tag_space

WORD = data.Field(init_token="<bos>", eos_token="<eos>", include_lengths=True, lower=True, batch_first=True)
UD_TAG = data.Field(init_token="<bos>", eos_token="<eos>", unk_token=None, batch_first=True)

if args.chars:
    CHAR_NESTING = data.Field(tokenize=list, init_token="<bos>", eos_token="<eos>")
    CHAR = data.NestedField(CHAR_NESTING, init_token="<bos>", eos_token="<eos>", include_lengths=True)
    fields = ((('word', 'char'), (WORD, CHAR)), ('udtag', UD_TAG))
else:
    fields = (('word', WORD), ('udtag', UD_TAG))
print('# fields =', fields, file=sys.stderr, flush=True)

train_data, valid_data, test_data = datasets.UDPOS.splits(fields=fields)

WORD.build_vocab(train_data)

if args.pretrained:
    vectors=vocab.GloVe(name="6B", dim=300)

    if args.oov_embeddings:
        WORD.vocab.extend(vectors)
    WORD.vocab.load_vectors(vectors)
    

print('# Word vocab size:\t', len(WORD.vocab), file=sys.stderr, flush=True)
UD_TAG.build_vocab(train_data)
print('# Tag vocab size:\t', len(UD_TAG.vocab), file=sys.stderr, flush=True)
if args.chars:
    CHAR.build_vocab(train_data)
    print('# Char vocab size:\t', len(CHAR.vocab), file=sys.stderr, flush=True)


BATCH_SIZE = 64
WORD_EMBEDDING_DIM = 300
HIDDEN_DIM = 300
CHAR_EMBEDDING_DIM = 0
CHAR_HIDDEN_DIM = 0
if args.chars:
    CHAR_EMBEDDING_DIM = 100
    CHAR_HIDDEN_DIM = 100
EMBEDDING_DIM = WORD_EMBEDDING_DIM + 2 * CHAR_EMBEDDING_DIM
if args.bi_chars:
    EMBEDDING_DIM = WORD_EMBEDDING_DIM + 4 * CHAR_EMBEDDING_DIM

print('# batch size:\t\t', BATCH_SIZE, file=sys.stderr, flush=True)
print('# word embedding dim:\t', WORD_EMBEDDING_DIM, file=sys.stderr, flush=True)
print('# hidden dim:\t\t', HIDDEN_DIM, file=sys.stderr, flush=True)
print('# char embedding dim:\t', CHAR_EMBEDDING_DIM, file=sys.stderr, flush=True)
print('# char hidden dim:\t', CHAR_HIDDEN_DIM, file=sys.stderr, flush=True)
print('# embedding dim:\t', EMBEDDING_DIM, file=sys.stderr, flush=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('# device:', device, file=sys.stderr)

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_data), 
            batch_size=BATCH_SIZE,
            sort_within_batch=True,
            repeat=False,
            device=device)

char_model = None
if args.chars:
    print('# Creating char model ...', file=sys.stderr, flush=True)
    char_model = CharEmbeddings(CHAR_EMBEDDING_DIM, CHAR_HIDDEN_DIM, len(CHAR.vocab), bidirectional=args.bi_chars)

print('# Creating word model ...', file=sys.stderr, flush=True)
vocab_len = len(WORD.vocab)
if args.oov_embeddings:
    vocab_len = None
model = LSTMTagger(WORD_EMBEDDING_DIM, EMBEDDING_DIM, HIDDEN_DIM, len(UD_TAG.vocab), vocab_size=vocab_len, bidirectional=args.bi_words, freeze=args.freeze, 
                   input_projection=args.input_projection)

if args.load:
    print('# Loading model from file ...', file=sys.stderr, flush=True)
    model.load_state_dict(torch.load('.models/best_model'))
    if args.chars:
        print('# Loading char embedding model from file ...', file=sys.stderr, flush=True)
        char_model.load_state_dict(torch.load('.models/best_char_model'))
else:
    print('# Creating new parameters ...', file=sys.stderr, flush=True)


if args.pretrained and not args.oov_embeddings:
    print('# Copying pretrained embeddings ...', file=sys.stderr, flush=True)
    pretrained_embeddings = WORD.vocab.vectors
    model.word_embeddings.weight.data.copy_(pretrained_embeddings)

print('# Creating loss function ...', file=sys.stderr, flush=True)
loss_function = nn.CrossEntropyLoss(ignore_index=WORD.vocab.stoi['<pad>'])

print('# Creating the parameter optimizer ...', file=sys.stderr, flush=True)
params = list(filter(lambda p: p.requires_grad, model.parameters()))
if args.chars:
    params += list(char_model.parameters())

if args.use_adam:
    optimizer = optim.Adam(params)
else:
    optimizer = optim.SGD(params, lr=0.1)

print('# Copying objects to device ...', file=sys.stderr, flush=True)
model = model.to(device)
if args.chars:
    char_model.to(device)
loss_function = loss_function.to(device)


N_EPOCHS = 400
LIMIT = 40

best_epoch = 0
best_loss = 0
best_acc = 0

print('# Start training ...', file=sys.stderr, flush=True)
for epoch in range(N_EPOCHS):
    if epoch - best_epoch > LIMIT:
        break

    train_loss, train_acc = train(model, train_iterator, optimizer, loss_function, char_model, args.oov_embeddings)
    valid_loss, valid_acc, valid_oov_acc = evaluate(model, valid_iterator, loss_function, char_model, args.oov_embeddings)
    if valid_acc > best_acc:
        print(f'Epoch {epoch+1:01}: saving the best model ...', file=sys.stderr, flush=True)
        best_acc = valid_acc
        best_epoch = epoch
        best_loss = valid_loss
        torch.save(model.state_dict(), '.models/best_model')
        if args.chars:
            torch.save(char_model.state_dict(), '.models/best_char_model')
    
    print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Valid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc*100:.2f}% | Valid OOV Acc: {valid_oov_acc*100:.2f}%', 
          file=sys.stdout, flush=True)

print(f'Best epoch: {best_epoch+1:02}, Best Acc: {best_acc*100:.2f}%', file=sys.stdout, flush=True)

model.load_state_dict(torch.load('.models/best_model'))
if args.chars:
    char_model.load_state_dict(torch.load('.models/best_char_model'))
t_loss, t_acc, t_oov_acc  = evaluate(model, test_iterator, loss_function, char_model, args.oov_embeddings)
print(f'Best epoch: {best_epoch+1:02}, Dev loss: {best_loss:.3f}, Dev acc: {best_acc*100:.2f}%, Test loss: {t_loss:.3f}, Test acc: {t_acc*100:.2f}%, Test OOV acc: {t_oov_acc*100:.2f}%', 
      file=sys.stdout, flush=True)
