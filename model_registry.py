# model_registry.py
from model import CandleModel

MODELS = {
    "1": CandleModel(),
    "2": CandleModel(),
    "5": CandleModel(),
    "10": CandleModel(),
}

def get_model(tf: str):
    return MODELS.get(tf, MODELS["1"])  # fallback на 1