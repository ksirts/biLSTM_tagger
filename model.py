import torch

from char_embeddings import CharEmbeddings
from lstm_encoder import CharacterEncoder

class Model(object):

    def __init__(self, model_type, logger, params):
        self.char_encoder = None
        self.model = None
        self.logger = logger
        self.model_type = model_type

        if model_type == 'char':
            self._create_char_only_model(char_emb=params['char_emb'],
                                         char_hidden=params['char_hidden'],
                                         char_vocab=params['num_chars'],
                                         hidden_dim=params['hidden_dim'],
                                         output_dim=params['num_tags'])


    def _create_char_only_model(self, char_emb, char_hidden, char_vocab, hidden_dim, output_dim):
        ## Creating models
        # Char only model
        self.logger.info('# Creating char model ...')
        self.char_encoder = CharEmbeddings(embedding_dim=char_emb, hidden_dim=char_hidden,
                                        vocab_size=char_vocab)
        self.logger.info('# Creating encoder ...')
        embedding_dim = 4 * char_emb
        self.model = CharacterEncoder(hidden_dim=hidden_dim, tagset_size=output_dim,
                                     embedding_dim=embedding_dim)

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


    def get_predictions(self, batch):
        predictions = None
        if self.model_type == 'char':
            predictions = self._get_char_model_predictions(batch)
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

    def _get_char_model_predictions(self, batch):
        assert self.char_encoder is not None
        chars, _, char_lengths = batch.char
        char_embeddings = self.char_encoder(chars, char_lengths)
        _, lengths = batch.word
        predictions = self.model(lengths=lengths, char_embeddings=char_embeddings)
        return predictions