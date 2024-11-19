import pyewts
from typing import List, Union
from abc import ABC, abstractmethod
from pyctcdecode import build_ctcdecoder
from botok import tokenize_in_stacks, normalize_unicode
from OCR.Utils import preprocess_unicode, postprocess_wylie_label, preprocess_unicode


class LabelEncoder(ABC):
    def __init__(self, charset: Union[str, List[str]], name: str):
        self.name = name
        
        if isinstance(charset, str):
            self._charset = [x for x in charset]
        elif isinstance(charset, list):
            self._charset = charset
        else: 
            raise TypeError("charset must be a string or a list of strings")
        
        self.ctc_vocab = self._charset.copy()
        self.ctc_vocab.insert(0, " ")
        self.ctc_decoder = build_ctcdecoder(self.ctc_vocab)
        
    @abstractmethod
    def read_label(self, label_path: str):
        raise NotImplementedError
    
    @property
    def charset(self) -> List[str]:
        return self._charset
    
    @property
    def num_classes(self) -> int:
        return len(self._charset)
        
    def encode(self, label: str):
        return [self._charset.index(x) + 1 for x in label]
    
    def decode(self, inputs: List[int]) -> str:
        return "".join(self._charset[x - 1] for x in inputs)
    
    def ctc_decode(self, logits):
        return self.ctc_decoder.decode(logits).replace(" ", "")


class TibetanStackEncoder(LabelEncoder):
    def __init__(self, charset: List[str]):
        super().__init__(charset, "stack")

    def read_label(self, label_path: str, normalize: bool = True):
        with open(label_path, "r", encoding="utf-8") as f:
            label = f.readline()

        if normalize:
            label = normalize_unicode(label)

        label = label.replace(" ", "")
        label = preprocess_unicode(label)
        stacks = tokenize_in_stacks(label)

        return stacks
    
    @property
    def num_classes(self) -> int:
        return len(self._charset) + 1


class TibetanWylieEncoder(LabelEncoder):
    def __init__(self, charset: str):
        super().__init__(charset, "wylie")
        self.converter = pyewts.pyewts()

    def read_label(self, label_path: str):
        with open(label_path, "r", encoding="utf-8") as f:
            label = f.readline()
        label = preprocess_unicode(label)
        label = self.converter.toWylie(label)
        label = postprocess_wylie_label(label)

        return label

    @property
    def num_classes(self) -> int:
        return len(self._charset) + 1
