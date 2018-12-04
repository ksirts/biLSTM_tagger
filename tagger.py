import sys
sys.stderr.flush()
sys.stdout.flush()

from argparse import ArgumentParser

import torch
from torchtext import data
from torchtext import vocab

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchtext import datasets

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .evaluation import sequence_accuracy
from .evaluation import oov_accuracy

from .char_embeddings import CharEmbeddings
from .lstm_enoder import LSTMTagger
# SEED = 1234

# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True


def parse_arguments():
    parser = ArgumentParser(description="Sequence tagging model")
    parser.add_argument('--load', action='store_true', help='Load existing model from file')
    parser.add_argument('--chars', action='store_true', help='Use character level model for embeddings')
    parser.add_argument('--chars-only', action='store_true', help='Use character level model only for embeddings')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained embeddings')
    parser.add_argument('--freeze', action='store_true', help='Freeze pretrained embeddings')
    parser.add_argument('--oov-embeddings', action='store_true', help='Load pretrained embeddings for all words')
    parser.add_argument('--input-projection', action='store_true', help='Use input projection in the word model')
    args = parser.parse_args()

    print('# Load from file:\t\t', args.load, file=sys.stderr)
    print('# Use char model:\t\t', args.chars, file=sys.stderr)
    print('# Use pretrained embeddings:\t', args.pretrained, file=sys.stderr)
    print('# Freeze pretrained embeddings:\t', args.freeze, file=sys.stderr)
    print('# Use embeddings for OOV words:\t', args.oov_embeddings, file=sys.stderr)
    print('# Use input projection:\t\t', args.input_projection, file=sys.stderr)

    return args


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

        predictions = model(words, lengths, char_embeddings)
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

            predictions = model(words, lengths, char_embeddings)
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


if __name__ == '__main__':

    args = parse_arguments()

    # Set hyper-parameters
    BATCH_SIZE = 64
    WORD_EMBEDDING_DIM = 0
    HIDDEN_DIM = 300
    CHAR_EMBEDDING_DIM = 0
    CHAR_HIDDEN_DIM = 0

    # Char only model
    if args.char_only:
        CHAR_EMBEDDING_DIM = 100
        CHAR_HIDDEN_DIM = 100

    # Word and char model
    elif args.chars:
        WORD_EMBEDDING_DIM = 300
        CHAR_EMBEDDING_DIM = 100
        CHAR_HIDDEN_DIM = 100

    # Word only model
    else:
        WORD_EMBEDDING_DIM = 300

    EMBEDDING_DIM = WORD_EMBEDDING_DIM + 4 * CHAR_EMBEDDING_DIM

    print('# batch size:\t\t', BATCH_SIZE, file=sys.stderr)
    print('# word embedding dim:\t', WORD_EMBEDDING_DIM, file=sys.stderr)
    print('# hidden dim:\t\t', HIDDEN_DIM, file=sys.stderr)
    print('# char embedding dim:\t', CHAR_EMBEDDING_DIM, file=sys.stderr)
    print('# char hidden dim:\t', CHAR_HIDDEN_DIM, file=sys.stderr)
    print('# embedding dim:\t', EMBEDDING_DIM, file=sys.stderr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('# device:', device, file=sys.stderr)

    # Create fields
    WORD = data.Field(init_token="<bos>", eos_token="<eos>", include_lengths=True, lower=True, batch_first=True)
    UD_TAG = data.Field(init_token="<bos>", eos_token="<eos>", unk_token=None, batch_first=True)
    CHAR_NESTING = data.Field(tokenize=list, init_token="<bos>", eos_token="<eos>")
    CHAR = data.NestedField(CHAR_NESTING, init_token="<bos>", eos_token="<eos>", include_lengths=True)


    # Word and char model
    if args.chars:
        fields = ((('word', 'char'), (WORD, CHAR)), ('udtag', UD_TAG))

    # Word only model
    else:
        fields = (('word', WORD), ('udtag', UD_TAG))
    print('# fields =', fields, file=sys.stderr)

    train_data, valid_data, test_data = datasets.UDPOS.splits(fields=fields)

    WORD.build_vocab(train_data)

    if args.pretrained:
        vectors=vocab.GloVe(name="6B", dim=300)

        if args.oov_embeddings:
            WORD.vocab.extend(vectors)
        WORD.vocab.load_vectors(vectors)


    UD_TAG.build_vocab(train_data)
    CHAR.build_vocab(train_data)

    print('# Word vocab size:\t', len(WORD.vocab), file=sys.stderr)
    print('# Tag vocab size:\t', len(UD_TAG.vocab), file=sys.stderr, flush=True)
    print('# Char vocab size:\t', len(CHAR.vocab), file=sys.stderr, flush=True)


    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                (train_data, valid_data, test_data),
                batch_size=BATCH_SIZE,
                sort_within_batch=True,
                repeat=False,
                device=device)

    ## Creating models
    # Char only model
    if args.char_only:
        print('# Creating char model ...', file=sys.stderr)
        char_model = CharEmbeddings(CHAR_EMBEDDING_DIM, CHAR_HIDDEN_DIM, len(CHAR.vocab))
        print('# Creating encoder ...', file=sys.stderr)
        model = LSTMTagger(HIDDEN_DIM, len(UD_TAG.vocab), EMBEDDING_DIM)

    # Char and word model
    elif args.chars:
        print('# Creating char model ...', file=sys.stderr)
        char_model = CharEmbeddings(CHAR_EMBEDDING_DIM, CHAR_HIDDEN_DIM, len(CHAR.vocab))

        print('# Creating encoder ...', file=sys.stderr)
        vocab_len = len(WORD.vocab)
        if args.oov_embeddings:
            vocab_len = 0
        model = LSTMTagger(HIDDEN_DIM, len(UD_TAG.vocab), EMBEDDING_DIM, word_embedding_dim=WORD_EMBEDDING_DIM,
                           vocab_size=vocab_len, freeze=args.freeze, input_projection=args.input_projection)

    # Word model only
    else:
        char_model = None
        print('# Creating encoder ...', file=sys.stderr)
        vocab_len = len(WORD.vocab)
        if args.oov_embeddings:
            vocab_len = 0
        model = LSTMTagger(HIDDEN_DIM, len(UD_TAG.vocab), WORD_EMBEDDING_DIM, word_embedding_dim=WORD_EMBEDDING_DIM,
                           vocab_size=vocab_len, freeze=args.freeze, input_projection=args.input_projection)


    # Load previous model from file
    if args.load:
        print('# Loading model from file ...', file=sys.stderr)
        model.load_state_dict(torch.load('.models/best_model'))
        if args.chars:
            print('# Loading char embedding model from file ...', file=sys.stderr)
            char_model.load_state_dict(torch.load('.models/best_char_model'))
    else:
        print('# Creating new parameters ...', file=sys.stderr)


    # Copy the word embeddings to the model
    if args.pretrained and not args.oov_embeddings:
        print('# Copying pretrained embeddings ...', file=sys.stderr)
        pretrained_embeddings = WORD.vocab.vectors
        model.word_embeddings.weight.data.copy_(pretrained_embeddings)

    print('# Creating loss function ...', file=sys.stderr, flush=True)
    loss_function = nn.CrossEntropyLoss(ignore_index=WORD.vocab.stoi['<pad>'])

    print('# Creating the parameter optimizer ...', file=sys.stderr)
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if args.chars:
        params += list(char_model.parameters())

    optimizer = optim.Adam(params)


    print('# Copying objects to device ...', file=sys.stderr)
    model = model.to(device)
    if args.chars:
        char_model.to(device)
    loss_function = loss_function.to(device)


    N_EPOCHS = 400
    LIMIT = 40

    best_epoch = 0
    best_loss = 0
    best_acc = 0

    print('# Start training ...', file=sys.stderr)
    for epoch in range(N_EPOCHS):
        if epoch - best_epoch > LIMIT:
            break

        train_loss, train_acc = train(model, train_iterator, optimizer, loss_function, char_model, args.oov_embeddings)
        valid_loss, valid_acc, valid_oov_acc = evaluate(model, valid_iterator, loss_function, char_model, args.oov_embeddings)
        if valid_acc > best_acc:
            print(f'Epoch {epoch+1:01}: saving the best model ...', file=sys.stderr)
            best_acc = valid_acc
            best_epoch = epoch
            best_loss = valid_loss
            torch.save(model.state_dict(), '.models/best_model')
            if args.chars:
                torch.save(char_model.state_dict(), '.models/best_char_model')

        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Valid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc*100:.2f}% | Valid OOV Acc: {valid_oov_acc*100:.2f}%',
              file=sys.stdout, flush=True)

    print(f'Best epoch: {best_epoch+1:02}, Best Acc: {best_acc*100:.2f}%', file=sys.stdout)

    model.load_state_dict(torch.load('.models/best_model'))
    if args.chars:
        char_model.load_state_dict(torch.load('.models/best_char_model'))
    t_loss, t_acc, t_oov_acc  = evaluate(model, test_iterator, loss_function, char_model, args.oov_embeddings)
    print(f'Best epoch: {best_epoch+1:02}, Dev loss: {best_loss:.3f}, Dev acc: {best_acc*100:.2f}%, Test loss: {t_loss:.3f}, Test acc: {t_acc*100:.2f}%, Test OOV acc: {t_oov_acc*100:.2f}%',
          file=sys.stdout)
