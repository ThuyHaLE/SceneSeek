# database/db_init.py

import json
import pickle
import torch
import sqlite3
import faiss
import multiprocessing

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CONFIG_PATH = 'config/databases.json'

def load_database_configs(config_path=CONFIG_PATH):
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading database config from {config_path}: {e}")
        raise

def load_annotation(info_dict_path):
    try:
        with open(info_dict_path, 'r') as openfile:
            return json.load(openfile)
    except Exception as e:
        logger.error(f"Error loading annotation from {info_dict_path}: {e}")
        return None

def faiss_database_processing(database_name='hnsw_jinaclipv2', config_path=CONFIG_PATH):
    num_threads = multiprocessing.cpu_count()
    logger.info(f"Number of threads: {num_threads}")

    db_configs = load_database_configs(config_path)

    if database_name not in db_configs:
        raise ValueError(
            f"Unsupported database name '{database_name}'. "
            f"Choose one of: {list(db_configs.keys())}"
        )

    config = db_configs[database_name] 
    database_path = config['index_path']
    info_dict_path = config['info_path']

    # Load metadata
    image_info_dict = load_annotation(info_dict_path)
    logger.info(f"Load annotation {info_dict_path}: DONE!")

    # Load FAISS index
    index = faiss.read_index(database_path)

    if config.get('index_type') == 'hnsw':
        ef_search = config.get('hnsw_ef_search', 128)
        index.hnsw.efSearch = ef_search
        logger.info(f"Set HNSW efSearch={ef_search}")

    logger.info(f"Load database {database_name}: DONE!")

    logger.info(f'Index loaded: ntotal={index.ntotal}, dimension={index.d}')
    logger.info(f'Metadata loaded: {len(image_info_dict)} entries')
    assert index.ntotal == len(image_info_dict), 'Mismatch between index entries and metadata entries!'

    _sample_key = next(iter(image_info_dict))
    logger.info(f"Sample metadata entry (key={_sample_key}):")

    logger.info(f"The index for {database_name} is ready!!!")

    return index, image_info_dict
    
def load_jinaclipv2_encoded_frames(device, database_name='jinaclipv2_encoded_frames', config_path=CONFIG_PATH):
    db_configs = load_database_configs(config_path)

    if database_name not in db_configs:
        raise ValueError(
            f"Unsupported database name '{database_name}'. "
            f"Choose one of: {list(db_configs.keys())}"
        )

    encoded_frames_path = db_configs[database_name]['encoded_frames_path']
    encoded_frames = torch.load(encoded_frames_path, map_location=device, weights_only=True)
    logger.info(f"Load encoded frames {encoded_frames_path}: DONE!")
    return encoded_frames

def load_event_transcripts(database_name='all_event_transcripts', config_path=CONFIG_PATH):
    db_configs = load_database_configs(config_path)

    if database_name not in db_configs:
        raise ValueError(
            f"Unsupported database name '{database_name}'. "
            f"Choose one of: {list(db_configs.keys())}"
        )

    event_transcripts_path = db_configs[database_name]['event_transcripts_path']
    event_transcripts = load_annotation(event_transcripts_path)
    logger.info(f"Load event transcripts {event_transcripts_path}: DONE!")
    return event_transcripts


def load_keyword_vocabulary(vocab_path: str) -> set:
    with open(vocab_path, encoding="utf-8") as f:
        vocab_list = json.load(f)
    return set(vocab_list)


def init_query_cache_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_keywords (
            query_id            TEXT PRIMARY KEY,
            raw_query           TEXT NOT NULL,
            normalized_query    TEXT NOT NULL,
            generated_keywords  TEXT NOT NULL,
            gen_method          TEXT NOT NULL,
            gen_model           TEXT,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            hit_count           INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_normalized_query
        ON query_keywords(normalized_query)
    """)
    conn.commit()
    conn.close()

def load_keywords_resources(database_name='bm25_flatip_dangvantuan', config_path=CONFIG_PATH) -> dict:
    db_configs = load_database_configs(config_path)

    if database_name not in db_configs:
        raise ValueError(
            f"Unsupported database name '{database_name}'. "
            f"Choose one of: {list(db_configs.keys())}"
        )
    
    query_cache_path = db_configs[database_name]['query_cache_path']
    vocab_path = db_configs[database_name]['keyword_vocab_path']

    init_query_cache_db(query_cache_path)
    keyword_vocab = load_keyword_vocabulary(vocab_path)

    logger.info(f"Load keyword vocabulary {vocab_path}: DONE!")
    logger.info(f"Init query cache DB {query_cache_path}: DONE!")

    return {
        "keyword_vocab": keyword_vocab,
        "query_cache_db_path": query_cache_path,
    }

def load_bm25_flatip_dangvantuan(database_name='bm25_flatip_dangvantuan', config_path=CONFIG_PATH) -> tuple:
    """Read embeddings.pt (generated by build-database.ipynb) and 
    build FAISS index from scratch (do not save a separate .index file, 
    since the FlatIP index can be built from the vector in seconds even for hundreds of thousands of vectors, 
    so no need to maintain an extra file).
    """
    db_configs = load_database_configs(config_path)
    
    if database_name not in db_configs:
        raise ValueError(
            f"Unsupported database name '{database_name}'. "
            f"Choose one of: {list(db_configs.keys())}"
        )

    embeddings_path = db_configs[database_name]['embeddings_path']

    saved = torch.load(embeddings_path, map_location="cpu")
    embeddings = saved["embeddings"].numpy().astype("float32")
    chunks = saved["chunks"]

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    info_dict = {i: chunks[i] for i in range(len(chunks))}
    logger.info(f"Load bm25 flatip index {embeddings_path}: DONE! ({index.ntotal} vector (dim={dim}), original model: {saved.get("model_name", "unknown")}).")
    return index, info_dict

def load_bm25_database(database_name='bm25_flatip_dangvantuan', config_path=CONFIG_PATH) -> tuple:
    """Read BM25Okapi object + chunks without re-fitting, just deserialize from the .pkl file."""
    db_configs = load_database_configs(config_path)
    
    if database_name not in db_configs:
        raise ValueError(
            f"Unsupported database name '{database_name}'. "
            f"Choose one of: {list(db_configs.keys())}"
        )

    bm25_path = db_configs[database_name]['bm25_index_path']
    with open(bm25_path, "rb") as f:
        data = pickle.load(f)
    logger.info(f"Load BM25 index {bm25_path}: DONE! ({len(data['chunks'])} chunk).")
    return data["bm25"], data["chunks"]