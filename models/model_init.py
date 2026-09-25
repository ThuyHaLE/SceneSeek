# models/model_init.py

import json
import torch
from transformers import AutoModel
from sentence_transformers import SentenceTransformer

# Configure logging to output to the notebook
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CONFIG_PATH = 'config/models.json'

def load_model_configs(config_path=CONFIG_PATH):
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading model config from {config_path}: {e}")
        raise

def load_model(model_name='jina-clip-v2', config_path=CONFIG_PATH):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Load model
    model_configs = load_model_configs(config_path)

    if model_name not in model_configs:
        raise ValueError(
            f"Unsupported model name '{model_name}'. "
            f"Choose one of: {list(model_configs.keys())}"
        )

    # Extract the actual model identifier/path from the config entry
    model_path = model_configs[model_name]["model_name"]

    if model_name == "jina-clip-v2":
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=False if device == "cuda" else True,
        )
        model = model.to(device)
        model.eval()

    elif model_name == "dangvantuan":
        model = SentenceTransformer(model_path, trust_remote_code=True)

    else:
        # Guards against configs that list a model but have no loader branch
        # implemented for it, avoiding an UnboundLocalError on `model`.
        raise ValueError(
            f"No loader implemented for model '{model_name}'"
        )

    logger.info(f"The model ({model_name}) is ready!!!")

    return device, model