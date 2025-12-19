from transformers import AutoTokenizer,AutoModelForImageTextToText
import torch

model_path = "google/gemma-3n-E4B-it"


class ModelManager:
    _instance = None
    _model = None
    _tokenizer = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._load_model()
        return cls._instance
    
    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls.get_instance()
        return cls._model

    @classmethod
    def get_tokenizer(cls):
        if cls._tokenizer is None:
            cls.get_instance()
        return cls._tokenizer
    
    @classmethod
    def _load_model(cls):
        """ Load model and tokenizer"""
        cache_key = f"text::{model_path}"

        if cls._model is None or cls._tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map= torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            cls._model = model
            cls._tokenizer = tokenizer

    

model_cache = ModelManager()
