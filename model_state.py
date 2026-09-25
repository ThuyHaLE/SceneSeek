# model_state.py

"""
Load model & search index once at import time.

With the same pattern as state.py (dataset):
everything heavy (model, FAISS index, encoded_frames) is loaded once at the module-level
when this module is first imported — DO NOT load again per request,
DO NOT attach to app.state (so that the router does not need `Request` to access them).
Router gets these objects via Depends() in deps.py, e.g.:
    from deps import get_model, get_clipv0_index
    def route(model = Depends(get_model)): ...
"""

from dataclasses import dataclass
from typing import Any, Dict
import state

from models.model_init import load_model
from database.db_init import (
    load_bm25_flatip_dangvantuan,
    load_jinaclipv2_encoded_frames,
    faiss_database_processing,
    load_event_transcripts,
    load_keywords_resources,
    load_bm25_database
)

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import psutil, os
_process = psutil.Process(os.getpid())
def _log_mem(label):
    rss = _process.memory_info().rss / 1e9
    logger.info(f"[MEM] after {label}: {rss:.2f} GB")
_log_mem("start")

@dataclass
class VectorDB:
    index: Any
    info_dict: Any  # HNSW: Dict[str, dict] (key = row index string); FLATIP: List[dict] (list of events)

logger.info("Loading jinaclipv2 model...")
DEVICE, JINACLIPV2_MODEL = load_model(model_name='jina-clip-v2')
_log_mem("jinaclipv2 model loaded")

logger.info("Loading dangvantuan model...")
_DANGVANTUAN_DEVICE, DANGVANTUAN_MODEL = load_model(model_name='dangvantuan')
_log_mem("dangvantuan model loaded")

if _DANGVANTUAN_DEVICE != DEVICE:
    logger.warning(
        "Device mismatch between models: jina-clip-v2 on '%s', dangvantuan on '%s'. "
        "Using '%s' as the shared DEVICE.",
        DEVICE, _DANGVANTUAN_DEVICE, DEVICE,
    )

logger.info("Loading jinaclipv2 encoded frames...")
JINACLIPV2_ENCODED_FRAMES = load_jinaclipv2_encoded_frames(DEVICE, database_name='jinaclipv2_encoded_frames')
_log_mem("encoded_frames loaded")

logger.info("Loading keywords resources...")
KEYWORDS_RESOURCES = load_keywords_resources(database_name='bm25_flatip_dangvantuan')
_log_mem("keywords_resources loaded")

logger.info("Loading bm25 database...")
BM25, BM25_CHUNKS = load_bm25_database(database_name='bm25_flatip_dangvantuan')
_log_mem("bm25 loaded")

logger.info("Loading all event transcripts...")
EVENT_TRANSCRIPTS = load_event_transcripts(database_name='all_event_transcripts')
_log_mem("event_transcripts loaded")

logger.info("Loading hnsw_jinaclipv2 index...")
_hnsw_index, _hnsw_info_dict = faiss_database_processing("hnsw_jinaclipv2")
HNSW_JINACLIPV2 = VectorDB(index=_hnsw_index, info_dict=_hnsw_info_dict)
_log_mem("hnsw index loaded")

logger.info("Loading flatip_dangvantuan index...")
_flatip_index, _flatip_info_dict = faiss_database_processing("flatip_dangvantuan")
FLATIP_DANGVANTUAN = VectorDB(index=_flatip_index, info_dict=_flatip_info_dict)
_log_mem("flatip index loaded")

logger.info("Loading bm25_flatip_dangvantuan index...")
_bm25_flatip_index, _bm25_flatip_info_dict = load_bm25_flatip_dangvantuan(database_name='bm25_flatip_dangvantuan')
BM25_FLATIP_DANGVANTUAN = VectorDB(index=_bm25_flatip_index, info_dict=_bm25_flatip_info_dict)
_log_mem("bm25_flatip index loaded")

logger.info("All model state loaded successfully.")
_log_mem("all state loaded")

# ---------------------------------------------------------------------------
# ALL_FRAMES / FRAME_PATH — derived from HNSW_JINACLIPV2.info_dict (dict of events)
# ---------------------------------------------------------------------------

logger.info("Building ALL_FRAME_PATHS / FRAME_PATH from hnsw_jinaclipv2...")

# row index (0..N-1) in JINACLIPV2_ENCODED_FRAMES <-> frame_path
# Assumption: HNSW_JINACLIPV2.info_dict[idx_str]["frame_path"] as same format
# with state.ALL_FRAMES[i]["frame_path"].
_missing_frame_path = [
    idx_str for idx_str, info in HNSW_JINACLIPV2.info_dict.items()
    if "frame_path" not in info
]
if _missing_frame_path:
    raise ValueError(
        f"Missing 'frame_path' field in HNSW_JINACLIPV2.info_dict entries "
        f"(e.g. keys: {_missing_frame_path[:5]}{'...' if len(_missing_frame_path) > 5 else ''})"
    )

FRAME_PATH_TO_ROW: Dict[str, int] = {
    info["frame_path"]: int(idx_str)
    for idx_str, info in HNSW_JINACLIPV2.info_dict.items()
}

# ---------------------------------------------------------------------------
# ALL_EVENTS / EVENT_INDEX — derived from EVENT_TRANSCRIPTS (list of events)
# ---------------------------------------------------------------------------

logger.info("Building ALL_EVENTS / EVENT_INDEX from all event transcripts...")

if not isinstance(EVENT_TRANSCRIPTS, list):
    raise TypeError(
        f"Expected EVENT_TRANSCRIPTS to be a list of events, "
        f"got {type(EVENT_TRANSCRIPTS)}"
    )

_raw_events = sorted(EVENT_TRANSCRIPTS, key=lambda e: (e.get("video_id") or "", e.get("start") or 0))

ALL_EVENTS: list = []
_skipped_no_video_id = 0
_skipped_no_frames = 0

# event_id in the raw data is NOT local per video — it can start above 0
# and have gaps (e.g. observed for L21_V014: [0, 6, 16, 26, 44, ...] instead
# of [0, 1, 2, ...]), since the transcripts were assembled from multiple
# sources/batches. We reassign a local, 0-based, gap-free event_id per video
# here so downstream consumers (event_router.py filtering, frontend range
# inputs) can safely assume "event_id resets to 0 per video" — this is the
# single place that assumption is made true, so it doesn't leak elsewhere.
#
# Renumbering happens AFTER frame-resolution filtering below (skipped events
# don't consume a local id), and follows the original list order per video,
# which is assumed chronological (raw event_id increases monotonically per
# video in the source data).
_local_event_counter: Dict[str, int] = {}

for _event in _raw_events:
    # Lưu ý field-name casing: event-level dùng "video_id" (lowercase);
    # bên trong "keyframes" của mỗi event lại dùng "video_ID" (uppercase) — không nhầm 2 field này.
    _vid = _event.get("video_id")
    if _vid is None:
        logger.warning("Event missing 'video_id' field, skipped: event_id=%s", _event.get("event_id"))
        _skipped_no_video_id += 1
        continue

    _frames = []
    for _kf in _event.get("keyframes", []):
        _frame = state.FRAME_BY_PATH.get(_kf.get("frame_path"))
        if _frame is not None:
            _frames.append(_frame)

    if not _frames:
        logger.warning(
            "Event has no resolvable frames, skipped: video_id=%s event_id=%s (raw keyframes=%d)",
            _vid, _event.get("event_id"), len(_event.get("keyframes", [])),
        )
        _skipped_no_frames += 1
        continue

    _local_id = _local_event_counter.get(_vid, 0)
    _local_event_counter[_vid] = _local_id + 1

    ALL_EVENTS.append({
        "video_id":            _vid,
        "event_id":            _local_id,               # local, 0-based, gap-free — used everywhere downstream
        "source_event_id":     _event.get("event_id"),   # original raw id, kept for debugging/traceability only
        "start":               _event.get("start"),
        "end":                 _event.get("end"),
        "text":                _event.get("text"),
        "frames":              _frames,
        "frame_count":         len(_frames),
    })

EVENT_INDEX: Dict[str, list] = {}
for _ev in ALL_EVENTS:
    EVENT_INDEX.setdefault(_ev["video_id"], []).append(_ev)

logger.info(
    "Loaded %d events across %d videos (skipped: %d no video_id, %d no resolvable frames).",
    len(ALL_EVENTS), len(EVENT_INDEX), _skipped_no_video_id, _skipped_no_frames,
)

# ---------------------------------------------------------------------------
# Design assumption check: HNSW_JINACLIPV2 (frame-level) is expected to cover
# every frame referenced inside FLATIP_DANGVANTUAN events. Not enforced (raise)
# because this is a data-completeness assumption, not a schema violation —
# log loudly instead so degraded coverage is visible without hard-crashing boot.
# ---------------------------------------------------------------------------

_event_frame_paths = {
    kf["frame_path"]
    for _event in ALL_EVENTS
    for kf in _event.get("keyframes", [])
}
_missing_in_hnsw = _event_frame_paths - FRAME_PATH_TO_ROW.keys()
if _missing_in_hnsw:
    logger.warning(
        "%d frame_path(s) referenced by FLATIP_DANGVANTUAN events are missing from "
        "HNSW_JINACLIPV2 FRAME_PATH_TO_ROW (design assumption 'HNSW covers FLATIP' violated). "
        "Example paths: %s",
        len(_missing_in_hnsw), list(_missing_in_hnsw)[:5],
    )