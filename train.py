import sys
sys.stderr.flush()
sys.stdout.flush()

from argparse import ArgumentParser

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
    parser.add_argument('--words', default=None, choices=['random', 'fixed'], help='How to use word embeddings')
    parser.add_argument('--freeze', action='store_true', help='Freeze pretrained embeddings')
    parser.add_argument('--oov-embeddings', action='store_true', help='Load pretrained embeddings for all words')
    parser.add_argument('--input-projection', action='store_true', help='Use input projection in the word model')
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
    args = parser.parse_args()

    return args



class Trainer(object):

    def __init__(self, model_name, logger, args):
        self.logger = logger
        self.model_type = self._get_model_type(args)
        self.model_fn = model_name

        self.params = {}
        self.params['batch_size'] = args.batch_size
        self.params['hidden_dim'] = args.hidden_dim

        self.train_iterator = None
        self.dev_iterator = None
        self.test_iterator = None
        self.model = None
        self.optimizer = None
        self.loss_function = None

    def load_data(self, lang, device, task, pretrained=False):
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
        WORD.build_vocab(train_data)
        LABEL.build_vocab(train_data)
        CHAR.build_vocab(train_data)

        self.word_field = WORD

        if pretrained:
            self.logger.info("# Loading word vectors from file")
            vectors = vocab.FastText(language=lang)
            WORD.vocab.load_vectors(vectors)
        #
        #    if args.oov_embeddings:
        #        WORD.vocab.extend(vectors)
        #    WORD.vocab.load_vectors(vectors)

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
            device=device)
        self.train_iterator = train_iterator
        self.dev_iterator = dev_iterator
        self.test_iterator = test_iterator

    def create_model(self, args):
        self._set_params(args)
        model_type = self.model_type
        logger = self.logger
        params = self.params
        unk_id = self.word_field.vocab.stoi['<unk>']
        self.model = Model(model_type, logger, unk_id, params)

    def create_optimizer(self, device):
        self.logger.info('# Creating loss function ...')
        loss_function = nn.CrossEntropyLoss(ignore_index=self.word_field.vocab.stoi['<pad>'])

        self.logger.info('# Creating the parameter optimizer ...')
        model_params = self.model.params
        self.optimizer = optim.Adam(model_params)

        self.logger.info('# Copying objects to device ...')
        self.model.to_device(device)
        self.loss_function = loss_function.to(device)

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

    def _set_params(self, args):
        if self.model_type in ('char', 'rnd+char'):
            self.params['char_emb'] = args.char_emb
            self.params['char_hidden'] = args.char_hidden
        if self.model_type in ('rnd', 'rnd+char'):
            self.params['word_emb'] = args.word_emb

    def train(self, max_epoch=400, es_limit=40):
        ''' Train the model
        :param max_epoch: maximum number of epochs
        :param es_limit: early stopping limit - stop training if the model has not improved
            on the development set for this number of epochs
        '''

        best_epoch = 0
        best_acc = 0
        best_oov_acc = 0

        self.logger.info('# Start training ...')
        for epoch in range(max_epoch):
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
    # def train(model, iterator, optimizer, criterion, char_model=None, oov_embeds=False):

        epoch_loss = 0
        epoch_acc = 0

        self.model.train()

        for i, batch in enumerate(self.train_iterator):
            # Clear out gradients
            self.optimizer.zero_grad()

            # Run forward pass
            predictions = self.model.get_predictions(batch, train=True)

            labels = batch.label.reshape(-1)
            words = batch.word[0].reshape(-1)

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
                # print('UNK vector:', self.model.unk_vector)

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
            #model, iterator, criterion, char_model=None, oov_embeds=False):

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
                if word == unk_index]
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

    trainer = Trainer(model_name, logger, args)

    pretrained = args.words != 'random'
    trainer.load_data(args.lang, device, args.task, pretrained=pretrained)
    trainer.create_model(args)

    # Load previous model from file
    if args.load is not None:
        logger.info('# Loading model from file ...')
        trainer.load_model(fn=args.load)
    else:
        logger.info('# Creating new parameters ...')

    trainer.create_optimizer(device=device)

    trainer.train()

    if args.test:
        trainer.test()









    # Char and word model
    # elif args.chars:
    #     print('# Creating char model ...', file=sys.stderr)
    #     char_model = CharEmbeddings(CHAR_EMBEDDING_DIM, CHAR_HIDDEN_DIM, len(CHAR.vocab))

    #     print('# Creating encoder ...', file=sys.stderr)
    #     vocab_len = len(WORD.vocab)
    #     if args.oov_embeddings:
    #         vocab_len = 0
    #     model = LSTMTagger(HIDDEN_DIM, len(UD_TAG.vocab), EMBEDDING_DIM, word_embedding_dim=WORD_EMBEDDING_DIM,
    #                        vocab_size=vocab_len, freeze=args.freeze, input_projection=args.input_projection)

    # Word model only
    # else:
    #     char_model = None
    #     print('# Creating encoder ...', file=sys.stderr)
    #     vocab_len = len(WORD.vocab)
    #     if args.oov_embeddings:
    #         vocab_len = 0
    #     model = LSTMTagger(HIDDEN_DIM, len(UD_TAG.vocab), WORD_EMBEDDING_DIM, word_embedding_dim=WORD_EMBEDDING_DIM,
    #                        vocab_size=vocab_len, freeze=args.freeze, input_projection=args.input_projection)




    # Copy the word embeddings to the model
    # if args.pretrained and not args.oov_embeddings:
    #     print('# Copying pretrained embeddings ...', file=sys.stderr)
    #     pretrained_embeddings = WORD.vocab.vectors
    #     model.word_embeddings.weight.data.copy_(pretrained_embeddings)




if __name__ == '__main__':
    main()
