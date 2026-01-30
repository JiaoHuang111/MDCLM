# dmlm/tokenizers/crysta_tokenizer.py

import pickle
from byprot.crystallm._tokenizer import CIFTokenizer
from byprot import utils

log = utils.get_logger(__name__)

class CrystaTokenizerWrapper:
    """Wrapper around CrystaLLM's CIFTokenizer or meta.pkl vocab."""

    def __init__(self, meta=None):
        """
        Args:
            meta: dict from meta.pkl, containing vocab + special tokens.
        """
        log.info(f'Function CrystaTokenizerWrapper.__init__ Start!')
        if meta is not None:
            log.info(f'Function CrystaTokenizerWrapper.__init__: use pre-built vocab from meta.pkl!')

            # use pre-built vocab from meta.pkl
            self._id_to_token = meta["itos"]  # list: index -> token
            self._token_to_id = meta["stoi"]  # dict: token -> index

            # special tokens
            self.pad_token_id = meta.get("pad_token_id")
            self.eos_token_id = meta.get("eos_token_id")
            self.bos_token_id = meta.get("bos_token_id")
            self.mask_token_id = meta.get("mask_token_id")

            self.pad_token = self._id_to_token[self.pad_token_id] if self.pad_token_id is not None else "<pad>"
            self.eos_token = self._id_to_token[self.eos_token_id] if self.eos_token_id is not None else "<eos>"
            self.bos_token = self._id_to_token[self.bos_token_id] if self.bos_token_id is not None else "<bos>"
            self.mask_token = self._id_to_token[self.mask_token_id] if self.mask_token_id is not None else "<mask>"
            self.unk_token = "<unk>"

        else:
            log.info(f'Function CrystaTokenizerWrapper.__init__: build vocab from CIFTokenizer!')
            # fallback: build vocab from CIFTokenizer
            self.base_tokenizer = CIFTokenizer()

            # inherit vocab
            self._token_to_id = self.base_tokenizer.token_to_id
            self._id_to_token = self.base_tokenizer.id_to_token

            # define special tokens
            self.pad_token = "<pad>"
            self.eos_token = "<eos>"
            self.bos_token = "<bos>"
            self.mask_token = "<mask>"
            self.unk_token = "<unk>"
            #  self.x_token = "X"

            # assign ids (append to vocab)
            max_id = max(self._id_to_token.keys())
            self.unk_token_id = max_id
            self.pad_token_id = max_id + 1
            self.eos_token_id = max_id + 2
            self.bos_token_id = max_id + 3
            self.mask_token_id = max_id + 4
            #  self.x_token_id = max_id + 5

            # extend mappings
            self._token_to_id[self.unk_token] = self.unk_token_id
            self._token_to_id[self.pad_token] = self.pad_token_id
            self._token_to_id[self.eos_token] = self.eos_token_id
            self._token_to_id[self.bos_token] = self.bos_token_id
            self._token_to_id[self.mask_token] = self.mask_token_id
            #  self._token_to_id[self.x_token] = self.x_token_id

            self._id_to_token[self.unk_token_id] = self.unk_token
            self._id_to_token[self.pad_token_id] = self.pad_token
            self._id_to_token[self.eos_token_id] = self.eos_token
            self._id_to_token[self.bos_token_id] = self.bos_token
            self._id_to_token[self.mask_token_id] = self.mask_token
            #  self._id_to_token[self.x_token_id] = self.x_token

        log.info(f'Function CrystaTokenizerWrapper.__init__ Done!')

    @property
    def vocab_size(self):
        return len(self._id_to_token)

    def __len__(self):
        return self.vocab_size  # ✅ 用于网络自动识别 vocab size
    @property
    def token_to_id(self):
        return dict(self._token_to_id)

    @property
    def id_to_token(self):
        return dict(self._id_to_token)

    def encode(self, tokens):
        return [self._token_to_id.get(t, self._token_to_id.get(self.unk_token)) for t in tokens]

    def decode(self, ids):
        return ''.join([self._id_to_token.get(i, self.unk_token) for i in ids])

    def batch_decode(self, batches):
        return [self.decode(batch) for batch in batches]
