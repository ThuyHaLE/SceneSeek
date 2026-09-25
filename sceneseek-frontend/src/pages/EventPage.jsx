// sceneseek-frontend/src/pages/EventPage.jsx

import { useState, useEffect, useRef } from "react";
import { fetchEvents } from "../api/client";
import VideoGroupItem from "../components/results/VideoGroupItem";
import Pagination from "../components/results/Pagination";
import SkeletonGrid from "../components/results/SkeletonGrid";
import { groupClustersByVideo } from "../utils/clusterGrouping";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Fallback when NOT able to fetch /api/videos/l-options (or network error).
// Same endpoint/fallback as DataPage — the L dropdown is shared across pages.
const FALLBACK_L_OPTIONS = Array.from({ length: 24 }, (_, i) =>
  String(i + 1).padStart(2, "0")
); // ["01", "02", ..., "24"]

// Exact "L13_V001" format — event_id is local per video, so range filtering
// only makes sense when exactly one video is selected (not an "L13" prefix).
const EXACT_VIDEO_ID_RE = /^L\d+_V\d+$/;

// Debounce for the "fetch max event_id" request. VideoIDSelector emits a new
// videoId on every keystroke in the V field (1 -> V001, 12 -> V012, ...), so
// without this we'd fire one request per keystroke.
const BOUNDS_DEBOUNCE_MS = 300;

// ---------------------------------------------------------------------------
// EventIdInput — integer input with ▲▼ shift buttons, mirrors TimestampInput's
// mechanism but for event_id instead of hh:mm:ss.
//
// Clamping behaviour (min/max are optional; omit either to leave that side
// unbounded — used while the real max_event_id for the selected video hasn't
// loaded yet):
//   - Typing:  free-form, NOT clamped on every keystroke (so typing multi-
//              digit numbers isn't fought character-by-character).
//   - Blur:    value is clamped into [min, max] once the user leaves the field.
//   - ▲ / ▼:   always clamped — shifting can never produce an out-of-range
//              value, it just stops at the boundary.
// ---------------------------------------------------------------------------

function EventIdInput({ id, label, value, onChange, fallback, placeholder, disabled = false, min, max }) {
  function clamp(n) {
    let v = n;
    if (min !== undefined && v < min) v = min;
    if (max !== undefined && v > max) v = max;
    return v;
  }

  function shift(delta) {
    if (disabled) return;
    const trimmed = value.trim();
    const base = trimmed === "" ? fallback : parseInt(trimmed, 10);
    const startFrom = isNaN(base) ? fallback : base;
    onChange(String(clamp(startFrom + delta)));
  }

  function handleChange(e) {
    if (disabled) return;
    const val = e.target.value;
    // allow empty (-> uses fallback), optional leading '-', digits only
    if (val === "" || /^-?\d*$/.test(val)) {
      onChange(val);
    }
  }

  function handleBlur() {
    if (disabled || value.trim() === "") return;
    const n = parseInt(value.trim(), 10);
    if (!isNaN(n)) {
      const clamped = clamp(n);
      if (clamped !== n) onChange(String(clamped));
    }
  }

  return (
    <div className={`ss-form-group ss-form-group--inline${disabled ? " ss-form-group--disabled" : ""}`}>
      <label htmlFor={id}>{label}</label>
      <div className="ss-timestamp-row">
        <input
          id={id}
          type="text"
          inputMode="numeric"
          value={value}
          placeholder={placeholder}
          onChange={handleChange}
          onBlur={handleBlur}
          autoComplete="off"
          spellCheck={false}
          disabled={disabled}
        />
        <button type="button" className="ss-ts-btn" onClick={() => shift(1)} title="+1" disabled={disabled}>▲</button>
        <button type="button" className="ss-ts-btn" onClick={() => shift(-1)} title="-1" disabled={disabled}>▼</button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// VideoIDSelector — L select + V text input → "L13_V001"
// (identical to DataPage's — kept local/duplicated rather than shared to avoid
// coupling two pages through one file; extract to a shared component if it
// needs to change in more than one place.)
// ---------------------------------------------------------------------------

function VideoIDSelector({ videoId, onChange, lOptions }) {
  const match = videoId.match(/^L(\d+)_V(\d+)$/);
  const [lPart, setLPart] = useState(match ? match[1] : "");
  const [vPart, setVPart] = useState(match ? match[2] : "");

  useEffect(() => {
    if (lPart && vPart) {
      onChange(`L${lPart}_V${vPart.padStart(3, "0")}`);
    } else if (lPart) {
      onChange(`L${lPart}`);
    } else {
      onChange("");
    }
  }, [lPart, vPart]); // eslint-disable-line react-hooks/exhaustive-deps

  function handleVChange(e) {
    const val = e.target.value.replace(/\D/g, "").slice(0, 3);
    setVPart(val);
  }

  const previewLabel = lPart && vPart
    ? `→ L${lPart}_V${vPart.padStart(3, "0")}`
    : lPart
    ? `→ L${lPart}_V* (tất cả)`
    : null;

  return (
    <div className="ss-form-group">
      <label>Video ID</label>
      <div className="ss-videoid-row">
        <select
          value={lPart}
          onChange={(e) => setLPart(e.target.value)}
          className="ss-videoid-select"
          aria-label="Chọn L"
        >
          <option value="">-- L --</option>
          {lOptions.map((l) => (
            <option key={l} value={l}>L{l}</option>
          ))}
        </select>

        <span className="ss-videoid-sep">_</span>

        <div className="ss-videoid-v-wrap">
          <span className="ss-videoid-v-prefix">V</span>
          <input
            type="text"
            inputMode="numeric"
            value={vPart}
            onChange={handleVChange}
            placeholder="* (tất cả)"
            className="ss-videoid-v-input ss-videoid-v-input--wide"
            aria-label="Nhập số V (để trống = lấy tất cả V)"
            maxLength={3}
          />
        </div>

        {previewLabel && (
          <span className="ss-videoid-preview">{previewLabel}</span>
        )}

        {(lPart || vPart) && (
          <button
            type="button"
            className="ss-btn ss-btn--ghost"
            onClick={() => { setLPart(""); setVPart(""); }}
          >
            ×
          </button>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// EventPage
// ---------------------------------------------------------------------------

export default function EventPage() {
  // --- Form state (what the user is currently typing/selecting) ---
  const [videoId, setVideoId] = useState("");
  const [eventIdStart, setEventIdStart] = useState("");
  const [eventIdEnd, setEventIdEnd] = useState("");
  const [filterError, setFilterError] = useState(null);

  // --- Applied filters (what the current results were actually fetched with) ---
  // null = user hasn't pressed Apply yet -> show nothing, don't fetch anything.
  // Pagination reads from here (NOT from the form state) so that editing the
  // form without pressing Apply doesn't silently change what page 2, 3... return.
  const [appliedFilters, setAppliedFilters] = useState(null);

  // --- Results ---
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState(null);

  // Monotonic id of the latest results request. A response is only applied if
  // its id still matches, so slow/out-of-order responses can't overwrite newer
  // ones (and Clear can invalidate anything still in flight).
  const requestIdRef = useRef(0);

  // Bump to force VideoIDSelector remount (reset internal lPart/vPart) on clear —
  // same workaround as DataPage, see its comment for why.
  const [selectorResetKey, setSelectorResetKey] = useState(0);

  const [lOptions, setLOptions] = useState(FALLBACK_L_OPTIONS);

  const isExactVideo = EXACT_VIDEO_ID_RE.test(videoId.trim());

  // Real max event_id for the currently selected video, sourced from the
  // backend response (`maxEventId`), NOT derived from the loaded `events`
  // page — events is paginated (perPage=20) so its local max is unreliable.
  // Reset to 0 whenever videoId changes, then refilled by the bounds effect
  // below. Only meaningful once boundsReady is true.
  const [eventIdEndFallback, setEventIdEndFallback] = useState(0);

  const boundsReady = isExactVideo && eventIdEndFallback > 0;

  // Start and End share the same upper bound (start == end is a valid
  // single-event range; start <= end is checked separately in validate()).
  // End's lower bound is -1 because -1 means "last event" (see validate()).
  const startMax = boundsReady ? eventIdEndFallback : undefined;
  const endMin = boundsReady ? -1 : undefined;
  const endMax = boundsReady ? eventIdEndFallback : undefined;

  // Selecting anything other than one exact video clears the event range.
  useEffect(() => {
    if (!isExactVideo && (eventIdStart || eventIdEnd)) {
      setEventIdStart("");
      setEventIdEnd("");
    }
  }, [isExactVideo]); // eslint-disable-line react-hooks/exhaustive-deps

  // Fetch max event_id for the selected video — WITHOUT touching the results
  // list, so simply picking a video doesn't display any data. Only the
  // boundary is fetched (perPage: 1 keeps the payload tiny).
  //
  // Reset to 0 immediately on every videoId change so we never clamp against
  // the previous video's max while the new request is in flight. `cancelled`
  // + the debounce timer drop stale/superseded requests.
  useEffect(() => {
    setEventIdEndFallback(0);
    if (!isExactVideo) return;

    let cancelled = false;
    const timer = setTimeout(async () => {
      try {
        const res = await fetchEvents({
          page: 1,
          perPage: 1,
          videoId: videoId.trim(),
          eventIdStart: "",
          eventIdEnd: "",
        });
        if (!cancelled) setEventIdEndFallback(res.maxEventId ?? 0);
      } catch {
        if (!cancelled) setEventIdEndFallback(0);
      }
    }, BOUNDS_DEBOUNCE_MS);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [videoId]); // eslint-disable-line react-hooks/exhaustive-deps

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
        // keep FALLBACK_L_OPTIONS if fetch fails
      });
    return () => { cancelled = true; };
  }, []);

  // NOTE: no "load on mount" effect any more — the page starts empty and only
  // fetches results when the user presses Apply Filter.

  // -------------------------------------------------------------------------
  // Validation
  // -------------------------------------------------------------------------

  function validate() {
    const hasEventIdFilter = eventIdStart.trim() !== "" || eventIdEnd.trim() !== "";
    if (eventIdStart.trim() && !/^-?\d+$/.test(eventIdStart.trim()))
      return "Event ID bắt đầu phải là số nguyên.";
    if (eventIdEnd.trim() && !/^-?\d+$/.test(eventIdEnd.trim()))
      return "Event ID kết thúc phải là số nguyên (-1 = event cuối cùng).";
    // event_id is local per video (resets to 0 for each video), so the range
    // filter only makes sense against exactly one video, not an "L13" prefix.
    if (hasEventIdFilter && !EXACT_VIDEO_ID_RE.test(videoId.trim()))
      return "Vui lòng chọn đúng 1 Video ID (VD: L21_V001) khi lọc theo Event ID.";

    // Range check against the real max_event_id, only once it's known.
    if (boundsReady) {
      if (eventIdStart.trim()) {
        const s = parseInt(eventIdStart, 10);
        if (!isNaN(s) && (s < 0 || s > eventIdEndFallback))
          return `Event ID bắt đầu phải trong khoảng 0 – ${eventIdEndFallback}.`;
      }
      if (eventIdEnd.trim()) {
        const e = parseInt(eventIdEnd, 10);
        if (!isNaN(e) && e !== -1 && (e < 0 || e > eventIdEndFallback))
          return `Event ID kết thúc phải trong khoảng 0 – ${eventIdEndFallback} (hoặc -1 cho event cuối).`;
      }
    }

    if (eventIdStart.trim() && eventIdEnd.trim()) {
      const s = parseInt(eventIdStart, 10);
      const e = parseInt(eventIdEnd, 10);
      if (!isNaN(s) && !isNaN(e) && e !== -1 && e < s)
        return "Event ID kết thúc phải >= Event ID bắt đầu (hoặc -1 cho event cuối).";
    }
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
      const res = await fetchEvents({
        page: targetPage,
        perPage: 20,
        videoId: filters.videoId.trim(),
        eventIdStart: filters.eventIdStart.trim(),
        eventIdEnd: filters.eventIdEnd.trim(),
      });
      if (reqId !== requestIdRef.current) return; // superseded by a newer request / Clear
      setEvents(res.events);
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

    const filters = { videoId, eventIdStart, eventIdEnd };
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
    setEventIdStart("");
    setEventIdEnd("");
    setFilterError(null);
    setSelectorResetKey((k) => k + 1);
    setEventIdEndFallback(0);

    setAppliedFilters(null);
    setEvents([]);
    setPage(1);
    setTotalPages(1);
    setLoadError(null);
    setLoading(false);
  }

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  const groups = groupClustersByVideo(events);
  const hasApplied = appliedFilters !== null;

  return (
    <div className="ss-data-page">
      <h2>Event Overview</h2>

      <form className="ss-search-form" onSubmit={handleSubmit}>
        <VideoIDSelector key={selectorResetKey} videoId={videoId} onChange={setVideoId} lOptions={lOptions} />

        <div className="ss-form-row">
          <EventIdInput
            id="event_id_start"
            label="Bắt đầu tại"
            value={eventIdStart}
            onChange={setEventIdStart}
            fallback={0}
            placeholder="0 (event đầu tiên)"
            disabled={!isExactVideo}
            min={0}
            max={startMax}
          />
          <EventIdInput
            id="event_id_end"
            label="Kết thúc tại"
            value={eventIdEnd}
            onChange={setEventIdEnd}
            fallback={eventIdEndFallback}
            placeholder={boundsReady ? `${eventIdEndFallback} (event cuối cùng)` : "event cuối cùng"}
            disabled={!isExactVideo}
            min={endMin}
            max={endMax}
          />
        </div>

        <p className="ss-form-hint">
          {isExactVideo
            ? boundsReady
              ? `Để trống → 0 (đầu) và ${eventIdEndFallback} (cuối). Nhập -1 ở ô kết thúc = event cuối.`
              : "Để trống → từ event đầu tiên đến event cuối cùng."
            : "Chọn đúng 1 Video ID (VD: L21_V001) để lọc theo Event ID."}
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
        <p className="ss-results-status">Chọn bộ lọc rồi bấm "Apply Filter" để xem sự kiện.</p>
      ) : loading ? (
        <SkeletonGrid count={50} />
      ) : events.length === 0 ? (
        <p className="ss-results-status">Không tìm thấy sự kiện nào phù hợp.</p>
      ) : (
        <div className="ss-cluster-list">
          {groups.map((g) => (
            <VideoGroupItem
              key={g.video_id}
              videoId={g.video_id}
              events={g.events}
              // no search-similar action in browse mode, and no SearchProvider
              // wraps EventPage — FeedbackButtons needs SearchContext, so it must
              // stay off here (same convention as DataPage's GalleryItem readOnly/showFeedback={false})
              onSearchSimilar={undefined}
              showFeedback={false}
            />
          ))}
        </div>
      )}

      {hasApplied && <Pagination page={page} totalPages={totalPages} onChange={handlePageChange} />}
    </div>
  );
}