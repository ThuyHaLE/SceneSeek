# tools/query_encoding.py

import torch
import model_state

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

# Models this module knows how to encode with. Kept local (not sourced from
# config/models.json) because the encode branch below is hardcoded per model —
# the two APIs (encode_text vs encode) aren't interchangeable, so this tuple
# must always match the if/elif branches exactly, not whatever happens to be
# listed in the model-loading config.
SUPPORTED_TEXT_MODELS = ("jina-clip-v2", "dangvantuan")


def _chunks(lst, chunk_size):
    """Split a list into smaller lists, each with a maximum of chunk_size elements."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def encode_texts(
    texts,
    model_name='jina-clip-v2',
    batch_size=32,
    truncate_dim=None,
    chunk_size=500,
    sort_by_length=True,
    show_progress=False,
):
    """Encode a list of texts using the specified text embedding model.

    :param texts: List of strings to be encoded.
    :param model_name: Which model to use — 'jina-clip-v2' (text-image, shared
        embedding space with images, use for Type 1/2 frame search) or
        'dangvantuan' (Vietnamese text-text, use for Type 3 transcript/event-mention search).
    :param batch_size: Number of texts to process in a single batch.
    :param truncate_dim: Dimension to truncate embeddings to. Only used by
        jina-clip-v2; must match the dimension used when encoding images. Ignored for dangvantuan.
    :param chunk_size: Number of texts to process in a single chunk, to avoid memory issues on large inputs.
    :param sort_by_length: Whether to sort texts by length before encoding, to improve batching efficiency.
    :param show_progress: Whether to display a progress bar during encoding.
    :return: A torch.Tensor of shape (N, dim) containing L2-normalized embeddings,
        in the same order as the input texts.
    """

    if model_name not in SUPPORTED_TEXT_MODELS:
        raise ValueError(
            f"Unsupported model name '{model_name}'. "
            f"Choose one of: {list(SUPPORTED_TEXT_MODELS)}"
        )

    n = len(texts)
    if sort_by_length:
        order = sorted(range(n), key=lambda i: len(texts[i]))
    else:
        order = list(range(n))
    sorted_texts = [texts[i] for i in order]

    all_features = []
    for chunk in _chunks(sorted_texts, chunk_size):
        with torch.no_grad():
            if model_name == "jina-clip-v2":
                embeddings = model_state.JINACLIPV2_MODEL.encode_text(
                    chunk, batch_size=batch_size,
                    truncate_dim=truncate_dim,
                )
                feats = torch.tensor(embeddings, dtype=torch.float32)
                feats = feats / feats.norm(dim=-1, keepdim=True)

            elif model_name == "dangvantuan":
                embeddings = model_state.DANGVANTUAN_MODEL.encode(
                    chunk, batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=show_progress,
                )
                feats = torch.tensor(embeddings, dtype=torch.float32)

            else:
                # Guards against SUPPORTED_TEXT_MODELS listing a model with no
                # encoding branch implemented, avoiding an UnboundLocalError on `feats`.
                raise ValueError(
                    f"No encoding logic implemented for model '{model_name}'"
                )

        all_features.append(feats.cpu())

    all_features = torch.cat(all_features, dim=0)
    inverse_order = torch.argsort(torch.tensor(order))
    all_features = all_features[inverse_order]
    return all_features


def encode_query(query_text, model_name='jina-clip-v2', truncate_dim=None):
    """Convenience wrapper for encoding a single query string.

    :param query_text: The query string to encode.
    :param model_name: 'jina-clip-v2' or 'dangvantuan' — see encode_texts().
    :param truncate_dim: Only used by jina-clip-v2, see encode_texts().
    :return: A torch.Tensor of shape (1, dim).
    """
    return encode_texts(
        [query_text],
        model_name=model_name,
        truncate_dim=truncate_dim,
        sort_by_length=False,
    )