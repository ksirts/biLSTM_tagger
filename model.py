import torch
import torch.nn.functional as F

from char_embeddings import CharEmbeddings
from lstm_encoder import CharacterEncoder
from lstm_encoder import WordEncoder
from lstm_encoder import FixedWordEncoder

import utils

class Model(object):

    def __init__(self, model_type, logger, params, device='cpu', vocab=None, train_unk=False, unk_id=False):
        self.char_encoder = None
        self.model = None
        self.logger = logger
        self.model_type = model_type
        self.train_unk = train_unk
        self.unk_id = unk_id
        self.unk_th = 0.9
        self.vocab=vocab
        self.device = device

        if model_type == 'char':
            self._create_char_only_model(params)
        elif model_type.startswith('rnd'):
            self._create_trainable_embedding_model(params)
        elif model_type.startswith('fix'):
            self._create_fixed_embedding_model(params)
        elif model_type.startswith('tune'):
            self._create_trainable_embedding_model(params)
        elif model_type.startswith('tune'):
            self._create_fixed_embedding_model(params)


    def _create_char_only_model(self, params):
        ## Creating models
        # Char only model
        self.logger.info('# Creating char model ...')
        char_emb = params['char_emb']
        char_hidden = params['char_hidden']
        char_vocab = params['num_chars']
        self.char_encoder = CharEmbeddings(self.logger, embedding_dim=char_emb, hidden_dim=char_hidden,
                                        vocab_size=char_vocab)
        self.logger.info('# Creating encoder ...')
        embedding_dim = 4 * char_hidden
        hidden_dim = params['hidden_dim']
        output_dim = params['num_tags']
        self.model = CharacterEncoder(self.logger, hidden_dim=hidden_dim, tagset_size=output_dim,
                                     embedding_dim=embedding_dim)

    def _create_trainable_embedding_model(self, params):
        ## Creating models
        # Char only model
        self.logger.info('# Creating a model with trainable embeddings ...')
        char_output = 0

        if self.model_type in ('rnd+char', 'tune+char'):
            self.logger.info('# Creating char model ...')
            char_emb = params['char_emb']
            char_hidden = params['char_hidden']
            char_vocab = params['num_chars']
            self.char_encoder = CharEmbeddings(self.logger, embedding_dim=char_emb, hidden_dim=char_hidden,
                                            vocab_size=char_vocab)
            char_output = 4 * char_hidden

        self.logger.info('# Creating encoder ...')
        word_emb = params['word_emb']
        vocab_size = params['num_words']
        embedding_dim = word_emb + char_output
        hidden_dim = params['hidden_dim']
        output_dim = params['num_tags']
        self.model = WordEncoder(self.logger, word_embedding_dim=word_emb, vocab_size=vocab_size,
                                 embedding_dim=embedding_dim,
                                 hidden_dim=hidden_dim, tagset_size=output_dim)

    def _create_fixed_embedding_model(self, params):
        self.logger.info('# Creating a model with fixed pretrained embeddings ...')
        char_output = 0

        if 'char' in self.model_type:
            self.logger.info('# Creating char model ...')
            char_emb = params['char_emb']
            char_hidden = params['char_hidden']
            char_vocab = params['num_chars']
            self.char_encoder = CharEmbeddings(self.logger, embedding_dim=char_emb, hidden_dim=char_hidden,
                                            vocab_size=char_vocab)
            char_output = 4 * char_hidden

        self.logger.info('# Creating encoder with fixed embeddings ...')
        word_emb = params['word_emb']
        embedding_dim = word_emb + char_output
        hidden_dim = params['hidden_dim']
        output_dim = params['num_tags']
        self.model = FixedWordEncoder(self.logger,
                                 embedding_dim=embedding_dim,
                                 hidden_dim=hidden_dim, tagset_size=output_dim)

    def load(self, fn):
        # TODO: check the existence of the file path
        if self.char_encoder is not None:
            self.logger.info('# Loading char embedding model from file ...')
            try:
                self.char_encoder.load_state_dict(torch.load(fn + '.char'))
            except:
                self.logger.error('# Could not load char model from file: {}'.format(fn + '.char'))

        assert self.model is not None
        self.model.load_state_dict(torch.load(fn + '.word'))


    @property
    def params(self):
        params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        if self.char_encoder is not None:
            params += list(self.char_encoder.parameters())
        return params

    # @property
    # def unk_vector(self):
    #     ind = torch.Tensor([self.unk_id], device='cpu').long()
    #     return self.model.word_embeddings(ind)

    def to_device(self, device):
        self.model = self.model.to(device)
        if self.char_encoder is not None:
            self.char_encoder = self.char_encoder.to(device)

    def train(self):
        self.model.train()
        if self.char_encoder is not None:
            self.char_encoder.train()

    def eval(self):
        self.model.eval()
        if self.char_encoder is not None:
            self.char_encoder.eval()

    def save(self, save_fn):
        torch.save(self.model.state_dict(), '.models/' + save_fn + '.word')
        if self.char_encoder is not None:
            torch.save(self.char_encoder.state_dict(), '.models/' + save_fn + '.char')


    def get_predictions(self, batch, train=True):
        predictions = None
        if self.model_type == 'char':
            predictions = self._get_char_model_predictions(batch, train)

        elif self.model_type in ('rnd', 'tune'):
            predictions = self._get_word_model_predictions(batch, train)

        elif self.model_type in ('fix', 'fix-oov'):
            predictions = self._get_fixed_word_model_predictions(batch, train)

        elif self.model_type in ('rnd+char', 'tune+char'):
            predictions = self._get_word_and_char_model_predictions(batch, train)

        elif self.model_type in ('fix+char', 'fix-oov+char'):
            predictions = self._get_fixed_word_and_char_model_predictions(batch, train)
        # words, lengths = batch.word
        # char_embeddings = None

        # if self.char_encoder is not None:
        #     chars, _, char_lengths = batch.char
        #     char_embeddings = self.char_encoder(chars, char_lengths)

        # word_embeddings = None
        # if oov_embeds:
        #     word_embeddings = F.embedding(words.cpu(), WORD.vocab.vectors)
        #     word_embeddings = word_embeddings.cuda()

        # predictions = model(words, lengths, char_embeddings=char_embeddings, word_embeddings=word_embeddings)
        predictions = predictions.reshape(-1, predictions.size()[-1])
        return predictions

    def _get_char_model_predictions(self, batch, train=True):
        assert self.char_encoder is not None
        chars, _, char_lengths = batch.char
        char_embeddings = self.char_encoder(chars, char_lengths)
        _, lengths = batch.word
        predictions = self.model(lengths=lengths, char_embeddings=char_embeddings)
        return predictions

    def _get_word_model_predictions(self, batch, train=True):
        assert self.char_encoder is None
        words, lengths = batch.word

        if train and self.train_unk:
            words = utils.sample_unks(words, self.unk_id, self.unk_th)
        predictions = self.model(words=words, lengths=lengths)
        return predictions

    def _get_fixed_word_model_predictions(self, batch, train=True):
        assert self.char_encoder is None
        assert self.vocab is not None
        words, lengths = batch.word

        if train and self.train_unk:
            words = utils.sample_unks(words, self.unk_id, self.unk_th)

        word_embeddings = F.embedding(words.cpu(), self.vocab.vectors)
        word_embeddings = word_embeddings.to(device=self.device)

        predictions = self.model(word_embeddings=word_embeddings, lengths=lengths)
        return predictions


    def _get_word_and_char_model_predictions(self, batch, train=True):
        assert self.char_encoder is not None
        chars, _, char_lengths = batch.char
        char_embeddings = self.char_encoder(chars, char_lengths)

        words, lengths = batch.word
        if train and self.train_unk:
            words = utils.sample_unks(words, self.unk_id, self.unk_th)
        predictions = self.model(words=words, lengths=lengths, char_embeddings=char_embeddings)
        return predictions

    def _get_fixed_word_and_char_model_predictions(self, batch, train=True):
        assert self.char_encoder is not None
        chars, _, char_lengths = batch.char
        char_embeddings = self.char_encoder(chars, char_lengths)

        words, lengths = batch.word
        if train and self.train_unk:
            words = utils.sample_unks(words, self.unk_id, self.unk_th)

        word_embeddings = F.embedding(words.cpu(), self.vocab.vectors)
        word_embeddings = word_embeddings.to(device=self.device)
        predictions = self.model(word_embeddings=word_embeddings, lengths=lengths, char_embeddings=char_embeddings)
        return predictions

    def copy_embeddings(self, pretrained_embeddings):
        self.model.word_embeddings.weight.data.copy_(pretrained_embeddings)