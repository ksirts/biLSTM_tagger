import sys

from argparse import ArgumentParser

import torch
from torchtext import data

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchtext import datasets

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
args = parser.parse_args()

print('# Load from file:\t', args.load, file=sys.stderr, flush=True)
print('# Use char model:\t', args.chars, file=sys.stderr, flush=True)
print('# BiLSTM for words:\t', args.bi_words, file=sys.stderr, flush=True)
print('# BiLSTM for chars:\t', args.bi_chars, file=sys.stderr, flush=True)
print('# Use Adam optimizer:\t', args.use_adam, file=sys.stderr, flush=True)

def sequence_accuracy(scores, targets, lengths):
    _, predict = torch.max(scores,1)
    batch_len = lengths.max()
    mask = torch.Tensor([[0] + [1 for i in range(l-2)] + [0 for i in range(batch_len - l + 1)] for l in lengths])
    mask = mask.permute(1, 0).reshape(-1)
    total = sum(mask)
    correct = predict == targets
    correct = correct.type(torch.FloatTensor)
    masked_correct = mask * correct
    acc = masked_correct.sum() / total
    return acc

def train(model, iterator, optimizer, criterion, char_model=None):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    if char_model:
        char_model.train()
  
    for i, batch in enumerate(iterator):
        # Clear out gradients
        optimizer.zero_grad()

        # Run forward pass.
        
        words, lengths = batch.word
        char_embeddings = None
        
        if char_model:
            chars = batch.char
            char_embeddings = char_model(chars)

        predictions = model(words, char_embeddings)
        predictions = predictions.reshape(-1, predictions.size()[-1])
        labels = batch.udtag.reshape(-1)

        # Compute the loss, gradients, and update the parameters by
        #calling optimizer.step()
        loss = criterion(predictions, labels)
        acc = sequence_accuracy(predictions, labels, lengths)
        
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
        # if i % 10 == 0:
        #     print(f'| Batch: {i:02} | Batch Loss: {loss:.3f} | Batch Acc: {acc*100:.2f}%')
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, char_model=None):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    if char_model:
        char_model.train()
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):
            
            words, lengths = batch.word
            char_embeddings = None
        
            if char_model:
                chars = batch.char
                char_embeddings = char_model(chars)

            predictions = model(words, char_embeddings)
            predictions = predictions.reshape(-1, predictions.size()[-1])
            labels = batch.udtag.reshape(-1)
            # print(predictions.size(), labels.size())
            
            loss = criterion(predictions, labels)
            acc = sequence_accuracy(predictions, labels, lengths)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


class CharEmbeddings(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size, bidirectional=False):
        super(CharEmbeddings, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional=bidirectional
        
        self.char_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional)
        
    def forward(self, chars):
        chars_size = chars.size()
        # Aggregate sequence length and batch dimensions
        chars = chars.view(-1, chars_size[-1])
        # Move batch to second dimension
        chars = chars.permute(1, 0)
        # Embed characters
        embeds = self.char_embeddings(chars)
        # Send through LSTM
        lstm_out, hidden = self.lstm(embeds)
        # Concatenate states to get word embeddings
        word_embeds = torch.cat(hidden, dim=2)

        # Compute the last dim of the output
        dim = 2 * self.hidden_dim
        
        # If the encoder is bidirectional then permute the batch to the beginning
        # and compute the last dim of the output
        if self.bidirectional:
            word_embeds = word_embeds.permute(1, 0, 2)
            dim = 4 * self.hidden_dim
        
        # Reshape back to batch x sequence x dimension
        word_embeds = word_embeds.reshape(chars_size[:2] + (dim,))
        # Permute axes to sequence x batch x dimension
        word_embeds = word_embeds.permute(1, 0, 2)
        return word_embeds


class LSTMTagger(nn.Module):

    def __init__(self, word_embedding_dim, embedding_dim, hidden_dim, vocab_size, tagset_size, bidirectional=False):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional)

        # The linear layer that maps from hidden state space to tag space
        linear_in = hidden_dim
        if bidirectional:
            linear_in = 2 * hidden_dim
        self.hidden2tag = nn.Linear(linear_in, tagset_size)


    def forward(self, sentence, char_embeds=None):
        embeds = self.word_embeddings(sentence)
        if char_embeds is not None:
            embeds = torch.cat([embeds, char_embeds], dim=2)
        lstm_out, hidden = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        return tag_space

WORD = data.Field(init_token="<bos>", eos_token="<eos>", include_lengths=True)
UD_TAG = data.Field(init_token="<bos>", eos_token="<eos>", unk_token=None)

if args.chars:
    CHAR_NESTING = data.Field(tokenize=list, init_token="<bos>", eos_token="<eos>")
    CHAR = data.NestedField(CHAR_NESTING, init_token="<bos>", eos_token="<eos>")
    fields = ((('word', 'char'), (WORD, CHAR)), ('udtag', UD_TAG))
else:
    fields = (('word', WORD), ('udtag', UD_TAG))
print('# fields =', fields, file=sys.stderr, flush=True)

train_data, valid_data, test_data = datasets.UDPOS.splits(fields=fields)


WORD.build_vocab(train_data)
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
    char_model = CharEmbeddings(CHAR_EMBEDDING_DIM, CHAR_HIDDEN_DIM, len(CHAR.vocab), bidirectional=args.bi_chars)
    print('# Creating char model ...', file=sys.stderr, flush=True)
model = LSTMTagger(WORD_EMBEDDING_DIM, EMBEDDING_DIM, HIDDEN_DIM, len(WORD.vocab), len(UD_TAG.vocab), bidirectional=args.bi_words)
print('# Creating word model ...', file=sys.stderr, flush=True)

if args.load:
    print('# Loading model from file ...', file=sys.stderr, flush=True)
    model.load_state_dict(torch.load('.models/best_model'))
    if args.chars:
        print('# Loading char embedding model from file ...', file=sys.stderr, flush=True)
        char_model.load_state_dict(torch.load('.models/best_char_model'))
else:
    print('# Creating new parameters ...', file=sys.stderr, flush=True)


loss_function = nn.CrossEntropyLoss(ignore_index=WORD.vocab.stoi['<pad>'])

params = model.parameters()
if args.chars:
    params = list(char_model.parameters()) + list(model.parameters())

if args.use_adam:
    optimizer = optim.Adam(params)
else:
    optimizer = optim.SGD(params, lr=0.1)

model = model.to(device)
if args.chars:
    char_model.to(device)
loss_function = loss_function.to(device)

N_EPOCHS = 400
LIMIT = 40

best_epoch = 0
best_loss = 0
best_acc = 0

for epoch in range(N_EPOCHS):
    if epoch - best_epoch > LIMIT:
        break

    train_loss, train_acc = train(model, train_iterator, optimizer, loss_function, char_model)
    valid_loss, valid_acc = evaluate(model, valid_iterator, loss_function, char_model)
    if valid_acc > best_acc:
        print(f'Epoch {epoch+1:01}: saving the best model ...', file=sys.stderr, flush=True)
        best_acc = valid_acc
        best_epoch = epoch
        best_loss = valid_loss
        torch.save(model.state_dict(), '.models/best_model')
        if args.chars:
            torch.save(char_model.state_dict(), '.models/best_char_model')
    
    print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |', file=sys.stdout, flush=True)

print(f'Best epoch: {best_epoch+1:02}, Best Acc: {best_acc*100:.2f}%', file=sys.stdout, flush=True)

model.load_state_dict(torch.load('.models/best_model'))
if args.chars:
    char_model.load_state_dict(torch.load('.models/best_char_model'))
t_loss, t_acc = evaluate(model, test_iterator, loss_function, char_model)
print(f'Best epoch: {best_epoch+1:02}, Dev loss: {best_loss:.3f}, Dev acc: {best_acc*100:.2f}%, Test loss: {t_loss:.3f}, Test acc: {t_acc*100:.2f}%', file=sys.stdout, flush=True)
