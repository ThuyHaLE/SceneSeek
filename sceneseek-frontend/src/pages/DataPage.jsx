// sceneseek-frontend/src/pages/DataPage.jsx

import { useState, useEffect, useRef } from "react";
import { fetchKeyframes } from "../api/client";
import GalleryItem from "../components/results/GalleryItem";
import Pagination from "../components/results/Pagination";
import SkeletonGrid from "../components/results/SkeletonGrid";
import VideoIDSelector from "../components/common/VideoIDSelector";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Fallback when NOT able to fetch /api/videos/l-options (or network error).
const FALLBACK_L_OPTIONS = Array.from({ length: 24 }, (_, i) =>
  String(i + 1).padStart(2, "0")
); // ["01", "02", ..., "24"]

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function parseTimestamp(ts) {
  try {
    const parts = ts.trim().split(":");
    const h = parseInt(parts[0], 10);
    const m = parseInt(parts[1], 10);
    const s = parts.length > 2 ? parseFloat(parts[2]) : 0;
    return h * 3600 + m * 60 + s;
  } catch {
    return null;
  }
}

const TIMESTAMP_RE = /^\d+:[0-5]\d(:[0-5]\d(\.\d+)?)?$/;

// Debounce for the "fetch end-of-video timestamp" request. VideoIDSelector
// emits a new videoId on every keystroke in the V field (1 -> V001, 12 -> V012, ...),
// so without this we'd fire requests per keystroke.
const END_FALLBACK_DEBOUNCE_MS = 300;

const DEFAULT_END_FALLBACK = "00:00:00";

// Timestamp of the latest frame in a list of keyframes.
function lastTimestampOf(keyframes) {
  if (!keyframes || keyframes.length === 0) return DEFAULT_END_FALLBACK;
  const last = keyframes.reduce((max, r) =>
    (r.timestamp_sec ?? 0) > (max.timestamp_sec ?? 0) ? r : max
  );
  return last.timestamp ?? DEFAULT_END_FALLBACK;
}

function validateTimestamp(ts) {
  if (!ts || !ts.trim()) return true;
  return TIMESTAMP_RE.test(ts.trim());
}

// ---------------------------------------------------------------------------
// TimestampInput
// ---------------------------------------------------------------------------

function TimestampInput({ id, label, value, onChange, placeholder = "hh:mm:ss", fallback = "00:00:00", disabled = false }) {
  function shift(delta) {
    if (disabled) return;
    const base = parseTimestamp(value);
    const startFrom = (base === null || isNaN(base))
      ? (parseTimestamp(fallback) ?? 0)
      : base;
    const next = Math.max(0, startFrom + delta);
    const h = Math.floor(next / 3600);
    const m = Math.floor((next % 3600) / 60);
    const s = Math.floor(next % 60);
    onChange([h, m, s].map((n) => String(n).padStart(2, "0")).join(":"));
  }

  function handlePaste(e) {
    if (disabled) return;
    e.preventDefault();
    onChange((e.clipboardData.getData("text") || "").trim());
  }

  return (
    <div className={`ss-form-group ss-form-group--inline${disabled ? " ss-form-group--disabled" : ""}`}>
      <label htmlFor={id}>{label}</label>
      <div className="ss-timestamp-row">
        <input
          id={id}
          type="text"
          value={value}
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value.trimStart())}
          onPaste={handlePaste}
          autoComplete="off"
          spellCheck={false}
          disabled={disabled}
        />
        <button type="button" className="ss-ts-btn" onClick={() => shift(1)}  title="+1s" disabled={disabled}>▲</button>
        <button type="button" className="ss-ts-btn" onClick={() => shift(-1)} title="-1s" disabled={disabled}>▼</button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// DataPage
// ---------------------------------------------------------------------------

export default function DataPage() {
  // --- Form state (what the user is currently typing/selecting) ---
  const [videoId,     setVideoId]     = useState("");
  const [tsStart,     setTsStart]     = useState("");
  const [tsEnd,       setTsEnd]       = useState("");
  const [filterError, setFilterError] = useState(null);

  // --- Applied filters (what the current results were actually fetched with) ---
  // null = user hasn't pressed Apply yet -> show a hint instead of "no results".
  // Pagination reads from here (NOT from the form state) so that editing the
  // form without pressing Apply doesn't silently change what page 2, 3... return.
  const [appliedFilters, setAppliedFilters] = useState(null);

  // --- Results ---
  const [page,       setPage]       = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [keyframes,  setKeyframes]  = useState([]);
  const [loading,    setLoading]    = useState(false);
  const [loadError,  setLoadError]  = useState(null);

  // Monotonic id of the latest results request. A response is only applied if
  // its id still matches, so slow/out-of-order responses can't overwrite newer
  // ones (and Clear can invalidate anything still in flight).
  const requestIdRef = useRef(0);

  // Bump to force VideoIDSelector remount (reset its internal lPart/vPart) when clearing filter —
  // because that component only reads videoId prop on mount, doesn't auto-sync when prop changes.
  const [selectorResetKey, setSelectorResetKey] = useState(0);

  // List of "L" for dropdown — fetch dynamically from backend (based on real video_IDs in JSON),
  // fallback to FALLBACK_L_OPTIONS if fetch fails.
  const [lOptions, setLOptions] = useState(FALLBACK_L_OPTIONS);

  const isExactVideo = /^L\d+_V\d+$/.test(videoId.trim());

  useEffect(() => {
    if (!isExactVideo && (tsStart || tsEnd)) {
      setTsStart("");
      setTsEnd("");
    }
  }, [isExactVideo]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    let cancelled = false;
    fetch("/api/videos/l-options")
      .then((res) => (res.ok ? res.json() : Promise.reject(res.status)))
      .then((data) => {
        if (!cancelled && Array.isArray(data?.lOptions) && data.lOptions.length > 0) {
          setLOptions(data.lOptions);
        }
      })
      .catch(() => {
        // keep the FALLBACK_L_OPTIONS if fetch fails
      });
    return () => { cancelled = true; };
  }, []);

  // Fallback for the end timestamp (start point of the ▲▼ buttons when the field
  // is empty): timestamp of the LAST frame of the selected video.
  //
  // It used to be derived from the currently displayed `keyframes`, which is only
  // one page (50 items) of results — so it was only right on the last page, and
  // stale/wrong before Apply or when browsing other videos. Now it's fetched
  // independently of the results list, from the last page of the selected video.
  const [endFallback, setEndFallback] = useState(DEFAULT_END_FALLBACK);

  // Reset to default immediately on every videoId change so we never shift from
  // the previous video's end time while the new request is in flight. `cancelled`
  // + the debounce timer drop stale/superseded requests. Results list is untouched.
  useEffect(() => {
    setEndFallback(DEFAULT_END_FALLBACK);
    if (!isExactVideo) return;

    let cancelled = false;
    const timer = setTimeout(async () => {
      try {
        const base = {
          perPage: 50,
          videoId: videoId.trim(),
          timestamp: "",
          timestamp_end: "",
        };
        const first = await fetchKeyframes({ ...base, page: 1 });
        let list = first.keyframes;
        // Short video: page 1 is already everything. Otherwise jump to the last page.
        if ((first.totalPages ?? 1) > 1) {
          const last = await fetchKeyframes({ ...base, page: first.totalPages });
          list = last.keyframes;
        }
        if (!cancelled) setEndFallback(lastTimestampOf(list));
      } catch {
        if (!cancelled) setEndFallback(DEFAULT_END_FALLBACK);
      }
    }, END_FALLBACK_DEBOUNCE_MS);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [videoId]); // eslint-disable-line react-hooks/exhaustive-deps

  // -------------------------------------------------------------------------
  // Validation
  // -------------------------------------------------------------------------

  function validate() {
    if (tsStart && !validateTimestamp(tsStart))
      return "Định dạng thời điểm bắt đầu không hợp lệ (hh:mm:ss[.SSS]).";
    if (tsEnd && !validateTimestamp(tsEnd))
      return "Định dạng thời điểm kết thúc không hợp lệ (hh:mm:ss[.SSS]).";
    if (tsStart && tsEnd) {
      const s = parseTimestamp(tsStart);
      const e = parseTimestamp(tsEnd);
      if (s !== null && e !== null && e <= s)
        return "Thời điểm kết thúc phải sau thời điểm bắt đầu.";
    }
    if ((tsStart || tsEnd) && !isExactVideo)
      return "Vui lòng chọn đúng 1 Video (cả L và V) khi lọc theo timestamp.";
    return null;
  }

  // -------------------------------------------------------------------------
  // Data loading
  // -------------------------------------------------------------------------

  // `filters` is passed explicitly (never read from form state) so results
  // always correspond to what was applied, and there's no stale-closure issue.
  async function load(targetPage, filters) {
    const reqId = ++requestIdRef.current;
    setLoading(true);
    setLoadError(null);
    try {
      const res = await fetchKeyframes({
        page: targetPage,
        perPage: 50,
        videoId: filters.videoId.trim(),
        timestamp: filters.tsStart.trim(),
        timestamp_end: filters.tsEnd.trim(),
      });
      if (reqId !== requestIdRef.current) return; // superseded by a newer request / Clear
      setKeyframes(res.keyframes);
      setTotalPages(res.totalPages);
      setPage(res.page ?? targetPage);
    } catch (e) {
      if (reqId !== requestIdRef.current) return;
      setLoadError("Không thể tải dữ liệu. Vui lòng thử lại.");
    } finally {
      if (reqId === requestIdRef.current) setLoading(false);
    }
  }

  // -------------------------------------------------------------------------
  // Handlers
  // -------------------------------------------------------------------------

  function handleSubmit(e) {
    e.preventDefault();
    const err = validate();
    if (err) { setFilterError(err); return; }
    setFilterError(null);

    const filters = { videoId, tsStart, tsEnd };
    setAppliedFilters(filters);
    load(1, filters);
  }

  function handlePageChange(targetPage) {
    if (!appliedFilters) return;
    load(targetPage, appliedFilters);
  }

  // Clear returns the page to its initial empty state (no fetch).
  function handleClear() {
    requestIdRef.current++; // invalidate any in-flight results request
    setVideoId("");
    setTsStart("");
    setTsEnd("");
    setFilterError(null);
    setSelectorResetKey((k) => k + 1); // force VideoIDSelector remount → delete selected lPart/vPart

    setAppliedFilters(null);
    setKeyframes([]);
    setPage(1);
    setTotalPages(1);
    setLoadError(null);
    setLoading(false);
  }

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  const hasApplied = appliedFilters !== null;

  return (
    <div className="ss-data-page">
      <h2>Data Overview</h2>

      <form className="ss-search-form" onSubmit={handleSubmit}>

        <VideoIDSelector key={selectorResetKey} videoId={videoId} onChange={setVideoId} lOptions={lOptions} />

        <div className="ss-form-row">
          <TimestampInput
            id="ts_start"
            label="Bắt đầu tại"
            value={tsStart}
            onChange={setTsStart}
            placeholder="hh:mm:ss[.SSS]"
            fallback="00:00:00"
            disabled={!isExactVideo}
          />
          <TimestampInput
            id="ts_end"
            label="Kết thúc tại"
            value={tsEnd}
            onChange={setTsEnd}
            placeholder="hh:mm:ss[.SSS]"
            fallback={endFallback}
            disabled={!isExactVideo}
          />
        </div>

        <p className="ss-form-hint">
          {isExactVideo
            ? <>Paste trực tiếp timestamp từ video player.
                {tsStart && !tsEnd && " Chỉ nhập Bắt đầu → frame gần nhất từ thời điểm đó."}
                {tsStart &&  tsEnd && " Hiển thị frame trong khoảng đã chọn."}
              </>
            : "Chọn đúng 1 Video (cả L và V) để lọc theo timestamp."}
        </p>

        {filterError && <p className="ss-form-error">{filterError}</p>}
        {loadError && <p className="ss-form-error">{loadError}</p>}

        <div className="ss-form-actions">
          <button type="submit" className="ss-btn ss-btn--primary" disabled={loading}>
            {loading ? "Đang tải..." : "Apply Filter"}
          </button>
          <button type="button" className="ss-btn ss-btn--ghost" onClick={handleClear}>
            Xoá filter
          </button>
        </div>
      </form>

      {!hasApplied ? (
        <p className="ss-results-status">Chọn bộ lọc rồi bấm "Apply Filter" để xem frame.</p>
      ) : loading ? (
        <SkeletonGrid count={50} />
      ) : keyframes.length === 0 ? (
        <p className="ss-results-status">Không tìm thấy frame nào phù hợp.</p>
      ) : (
        <div className="ss-gallery">
          {keyframes.map((item, i) => (
            <GalleryItem
              key={item.db_idx}
              item={item}
              allItems={keyframes}
              indexInList={i}
              showFeedback={false}
              readOnly
            />
          ))}
        </div>
      )}

      {hasApplied && <Pagination page={page} totalPages={totalPages} onChange={handlePageChange} />}
    </div>
  );
}