# model_registry.py
from model import CandleModel

# Создаём модели при старте
MODELS = {
    "1": CandleModel("1"),
    "2": CandleModel("2"),
    "5": CandleModel("5"),
    "10": CandleModel("10"),
}

def get_model(tf: str):
    return MODELS.get(tf, MODELS["1"])
