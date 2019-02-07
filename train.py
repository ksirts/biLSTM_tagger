import sys
import time

from argparse import ArgumentParser
from collections import Counter

import numpy as np

import torch
from torchtext import data
from torchtext import vocab

import torch.nn as nn
import torch.optim as optim

from ud_data import UDPOS, UDPOSMorph
import utils

from model import Model


# SEED = 1234

# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True


def parse_arguments():
    parser = ArgumentParser(description="Sequence tagging model")
    parser.add_argument('--load', help='Load existing model from the given file path')
    parser.add_argument('--chars', action='store_true', help='Use character level model for embeddings')
    parser.add_argument('--words', default=None, choices=['random', 'fixed', 'tuned'], help='How to use word embeddings')
    parser.add_argument('--oov-embeddings', action='store_true', help='Load pretrained embeddings for all words')
    parser.add_argument('--transform', action='store_true', help='Use input transformation in the word model')
    parser.add_argument('--lang', default='en', help='Language of the dataset')
    parser.add_argument('--model-name', required=True, help='Name for the model')
    parser.add_argument('--log-level', default='info', choices=['info', 'debug'], help='Logging level')
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size')
    parser.add_argument('--word-emb', default=300, type=int, help='Word embedding dimension')
    parser.add_argument('--hidden-dim', default=300, type=int, help='Encoder hidden dimension')
    parser.add_argument('--char-emb', default=75, type=int, help='Char embedding dimension')
    parser.add_argument('--char-hidden', default=75, type=int, help='Char encoder hidden dimension')
    parser.add_argument('--test', action='store_true', help='Evaluate the model on the test set after training')
    parser.add_argument('--index', type=int, help='If given then indexes the runs of the same model')
    parser.add_argument('--task', default='pos', choices=['pos', 'posmorph'])
    parser.add_argument('--train-unk', action='store_true', help='Train unk vector')
    parser.add_argument('--decode', action='store_true', help='Decode only')
    args = parser.parse_args()

    return args



class Trainer(object):

    def __init__(self, model_name, logger, device, args):
        self.logger = logger
        self.model_type = self._get_model_type(args)
        self.model_fn = model_name
        self.train_unk = args.train_unk
        self.device = device

        self.params = {}
        self.params['batch_size'] = args.batch_size
        self.params['hidden_dim'] = args.hidden_dim
        self.params['char_emb'] = args.char_emb
        self.params['char_hidden'] = args.char_hidden
        self.params['word_emb'] = args.word_emb

        self.train_iterator = None
        self.dev_iterator = None
        self.test_iterator = None
        self.model = None
        self.optimizer = None
        self.loss_function = None

    def load_data(self, lang, task, pretrained=False, oov_embeddings=False):
        # Create fields
        WORD = data.Field(init_token="<bos>", eos_token="<eos>", include_lengths=True, lower=True, batch_first=True)
        LABEL = data.Field(init_token="<bos>", eos_token="<eos>", unk_token=None, batch_first=True)
        CHAR_NESTING = data.Field(tokenize=list, init_token="<bos>", eos_token="<eos>")
        CHAR = data.NestedField(CHAR_NESTING, init_token="<bos>", eos_token="<eos>", include_lengths=True)

        if task == 'pos':
            fields = ((None, None), (('word', 'char'), (WORD, CHAR)), (None, None), ('label', LABEL))
            self.logger.info('# fields: {}'.format(fields))

            train_data, valid_data, test_data = UDPOS.splits(root='data', fields=fields,
                                                                  train='{}-ud-train.conllu'.format(lang),
                                                                  validation='{}-ud-dev.conllu'.format(lang),
                                                                  test='{}-ud-test.conllu'.format(lang),
                                                                  lang=lang,
                                                                  logger=self.logger)
        else:
            assert task == 'posmorph'
            fields = ((('word', 'char'), (WORD, CHAR)),('label', LABEL))
            self.logger.info('# fields: {}'.format(fields))

            train_data, valid_data, test_data = UDPOSMorph.splits(root='data', fields=fields,
                                                             train='{}-ud-train.conllu'.format(lang),
                                                             validation='{}-ud-dev.conllu'.format(lang),
                                                             test='{}-ud-test.conllu'.format(lang),
                                                             lang=lang,
                                                             logger=self.logger)

        # Create vocabularies
        WORD.build_vocab(train_data, min_freq=1)
        LABEL.build_vocab(train_data)
        CHAR.build_vocab(train_data)

        self.word_field = WORD
        self.num_initial_words = len(WORD.vocab)  # for computing OOV accuracy
        self.logger.info('# Initial word vocab size: {}'.format(self.num_initial_words))

        if pretrained:
            self.logger.info("# Loading word vectors from file")
            vectors = vocab.FastText(language=lang)

            if oov_embeddings:
                self.logger.info("# Copying all pretrained word embeddings into vocabulary")
                WORD.vocab.extend(vectors)

            WORD.vocab.load_vectors(vectors)

        self.logger.info('# Word vocab size: {}'.format(len(WORD.vocab)))
        self.logger.info('# Tag vocab size: {}'.format(len(LABEL.vocab)))
        self.logger.info('# Char vocab size: {}'.format(len(CHAR.vocab)))
        self.params['num_words'] = len(WORD.vocab)
        self.params['num_tags'] = len(LABEL.vocab)
        self.params['num_chars'] = len(CHAR.vocab)

        train_iterator, dev_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size=self.params['batch_size'],
            sort_within_batch=True,
            repeat=False,
            device=-1)
        self.train_iterator = train_iterator
        self.dev_iterator = dev_iterator
        self.test_iterator = test_iterator

    def create_model(self, args, device):
        model_type = self.model_type
        logger = self.logger
        params = self.params
        unk_id = self.word_field.vocab.stoi['<unk>']
        self.model = Model(model_type, logger, params, vocab=self.word_field.vocab, device=self.device,
                           train_unk=args.train_unk, unk_id=unk_id)

        if model_type.startswith('tune'):
            self.logger.info('# Copying pretrained embeddings to model...')
            pretrained_embeddings = self.word_field.vocab.vectors
            self.model.copy_embeddings(pretrained_embeddings)

        self.logger.info('# Copying o to device ...')
        self.model.to_device(device)

    def create_loss(self, device):
        self.logger.info('# Creating loss function ...')
        loss_function = nn.CrossEntropyLoss(ignore_index=self.word_field.vocab.stoi['<pad>'])
        self.logger.info('# Copying loss to device ...')
        self.loss_function = loss_function.to(device)


    def create_optimizer(self, device):
        self.logger.info('# Creating the parameter optimizer ...')
        model_params = self.model.params
        self.optimizer = optim.Adam(model_params)


    def load_model(self, fn):
        #TODO: Check the existence of the file path
        self.model.load(fn)


    def _get_model_type(self, args):
        # Character only model
        if args.words is None:
            assert args.chars
            return 'char'

        # Model with randomly initialized word embeddings
        if args.words == 'random':
            if args.chars:
                return 'rnd+char'
            else:
                return 'rnd'

        # Model with pretrained fixed embeddings
        if args.words == 'fixed' and args.oov_embeddings is False and args.transform is False:
            if args.chars:
                return 'fix+char'
            else:
                return 'fix'

        # Model with fine-tuned word embeddings
        if args.words == 'tuned':
            if args.chars:
                return 'tune+char'
            else:
                return 'tune'

        # Model with pretrained fixed embedings that are also used during testing
        if args.words == 'fixed' and args.oov_embeddings is True and args.transform is False:
            if args.chars:
                return 'fix-oov+char'
            else:
                return 'fix-oov'

        # Model with pretrained fixed embeddings and input transformation matrix
        if args.words == 'fixed' and args.oov_embeddings is False and args.transform is True:
            if args.chars:
                return 'trans+char'
            else:
                return 'trans'

        # Model with pretrained fixed embeddings that are also used during testing and input transformation matrix
        if args.words == 'fixed' and args.oov_embeddings is True and args.transform is True:
            if args.chars:
                return 'trans-oov+char'
            else:
                return 'trans-oov'

    def train(self, es_limit=40):
        ''' Train the model
        :param max_epoch: maximum number of epochs
        :param es_limit: early stopping limit - stop training if the model has not improved
            on the development set for this number of epochs
        '''

        best_epoch = 0
        best_acc = 0
        best_oov_acc = 0

        self.logger.info('# Start training ...')
        for epoch in range(400):
            if epoch - best_epoch > es_limit:
                self.logger.info('# Finished training after {:d} epochs\n'.format(epoch-1))
                break

            train_loss, train_acc = self._train_epoch()
            valid_loss, valid_acc, valid_oov_acc = self.evaluate()

            fmt = '| Epoch {:d} | Train loss: {:.3f} | Train Acc: {:.2%} | Valid Loss: {:.3f} | Valid Acc: {:.2%} | Valid OOV Acc: {:.2%}'
            self.logger.info(fmt.format(epoch+1, train_loss, train_acc, valid_loss, valid_acc, valid_oov_acc))

            if valid_acc > best_acc:
                self.logger.info(f'Epoch {epoch+1:01}: saving the best model ...')
                best_acc = valid_acc
                best_epoch = epoch
                best_oov_acc = valid_oov_acc
                self.model.save(self.model_fn)

        self.logger.info(f'Best epoch: {best_epoch+1:02}, Best Acc: {best_acc*100:.3f}%, Best OOV Acc: {best_oov_acc*100:.3f}%')

    def _train_epoch(self):

        epoch_loss = 0
        epoch_acc = 0

        self.model.train()

        for i, batch in enumerate(self.train_iterator):
            # Clear out gradients
            self.optimizer.zero_grad()

            labels = batch.label.reshape(-1)
            words = batch.word[0].reshape(-1)

            # Run forward pass
            predictions = self.model.get_predictions(batch, train=True)


            # Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss = self.loss_function(predictions, labels)
            acc = self.sequence_accuracy(words, predictions, labels)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            if i % 10 == 0:
                self.logger.debug(f'| Batch: {i:02} | Batch Loss: {loss:.3f} | Batch Acc: {acc*100:.2f}%')

        return epoch_loss / len(self.train_iterator), epoch_acc / len(self.train_iterator)

    def sequence_accuracy(self, words, predictions, labels):
        _, predict = torch.max(predictions, 1)

        bos_index = self.word_field.vocab.stoi['<bos>']
        eos_index = self.word_field.vocab.stoi['<eos>']
        pad_index = self.word_field.vocab.stoi['<pad>']

        mask = torch.ones(words.size())
        mask[words == bos_index] = 0
        mask[words == eos_index] = 0
        mask[words == pad_index] = 0

        total = mask.sum().type(torch.FloatTensor)
        correct = (predict == labels).type(torch.FloatTensor)
        masked_correct = mask * correct
        acc = masked_correct.sum().type(torch.FloatTensor) / total
        return acc

    def evaluate(self, test=False):

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_oov_acc = 0.0
        oov_batches = 0

        self.model.eval()

        if test == True:
            iterator = self.test_iterator
        else:
            iterator = self.dev_iterator

        with torch.no_grad():
            for i, batch in enumerate(iterator):

                predictions = self.model.get_predictions(batch, train=False)
                labels = batch.label.reshape(-1)
                words = batch.word[0].reshape(-1)

                loss = self.loss_function(predictions, labels)
                acc = self.sequence_accuracy(words, predictions, labels)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

                oov_acc = self.oov_accuracy(words, predictions, labels)
                if oov_acc > 0:
                    epoch_oov_acc += oov_acc
                    oov_batches += 1
        return epoch_loss / len(iterator), epoch_acc / len(iterator), utils.safe_division(epoch_oov_acc, oov_batches)

    def oov_accuracy(self, words, predictions, labels):
        _, predictions = torch.max(predictions, 1)
        unk_index = self.word_field.vocab.stoi['<unk>']
        data = [[pred, lab] for (word, pred, lab) in zip(words, predictions, labels)
                if (word == unk_index or word >= self.num_initial_words)]
        self.logger.debug('# Number of OOV words: {:d}'.format(len(data)))
        if len(data) == 0:
            return 0
        data = np.array(data)
        return np.mean(data[:, 0] == data[:, 1])

    def test(self):
        self.model.load('.models/' + self.model_fn)
        t_loss, t_acc, t_oov_acc = self.evaluate(test=True)
        self.logger.info(f'Test loss: {t_loss:.3f}, Test acc: {t_acc*100:.3f}%, Test OOV acc: {t_oov_acc*100:.3f}%')



def main():

    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_name = utils.get_model_fn(args.model_name, args.index)
    logger = utils.get_logger(model_name + '.log', level=args.log_level)
    for key, val in vars(args).items():
        logger.info('## {}: {}'.format(key, val))

    trainer = Trainer(model_name, logger, device, args)

    pretrained = args.words in ('fixed', 'tuned')
    trainer.load_data(args.lang, args.task, pretrained=pretrained, oov_embeddings=args.oov_embeddings)
    trainer.create_model(args, device)

    # Load previous model from file
    if args.load is not None:
        logger.info('# Loading model from file ...')
        trainer.load_model(fn=args.load)
    else:
        logger.info('# Creating new parameters ...')

    trainer.create_loss(device=device)

    # Only evaluate the model
    if args.decode:
        valid_loss, valid_acc, valid_oov_acc = trainer.evaluate()
        fmt = '| Valid Loss: {:.3f} | Valid Acc: {:.3%} | Valid OOV Acc: {:.3%}'
        logger.info(fmt.format(valid_loss, valid_acc, valid_oov_acc))
    else:
        trainer.create_optimizer(device=device)
        trainer.train()

    if args.test:
        trainer.test()


if __name__ == '__main__':
    main()
