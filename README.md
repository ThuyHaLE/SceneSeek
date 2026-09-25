# SceneSeek

> Interactive video retrieval system — hỗ trợ 3 kiểu tìm kiếm (frame, chuỗi sự kiện theo thời gian, sự kiện được nhắc tới trong transcript) trên bộ dữ liệu 838 video (đa chủ đề: tin tức, thi đấu, dạy học, phim tài liệu...) của Hội thi AI Challenge HCMC 2026, kèm công cụ duyệt/xác nhận nhanh theo video.

![SceneSeek Demo](https://github.com/ThuyHaLE/SceneSeek/blob/main/SceneSeek-Demo-compressed.gif)

## ✨ Features

**Retrieval**
- 🖼️ **Frame Search (Type 1)** — tìm 1 khoảnh khắc cụ thể bằng mô tả văn bản (text-image search), dùng Jina CLIP v2 + FAISS HNSW.
- ⏱️ **Event Boundary Search (Type 2)** — nhập 2–5 mô tả cảnh theo thứ tự thời gian, hệ thống tìm chuỗi frame khớp đúng thứ tự trong cùng 1 video. Hỗ trợ chế độ "tìm chính xác" (khớp đủ mọi cảnh) và "tìm tương đối" (cho phép thiếu 1 cảnh).
- 💬 **Event Mention Search (Type 3)** — tìm sự kiện được nhắc tới trong transcript video (text-text search), dùng model dangvantuan. Tự động chuyển sang hybrid search (dense + BM25, merge bằng Reciprocal Rank Fusion) khi có từ khoá. **Đây là type duy nhất hỗ trợ nhập keyword ở MVP này** — vì text-text dễ thiết kế keyword matching hơn text-image (Type 1/2).
- 🔗 **Similar Frame Search** — từ 1 frame trong kết quả, tìm các frame khác có nội dung tương tự (cosine similarity trên embedding đã encode sẵn, không cần encode lại).

**Query Assist**
- 🔤 **Keyword Suggestion** — gợi ý từ khoá cho ô tìm kiếm dựa trên rule-based extraction (n-gram + vocab, fallback bằng POS tagging tiếng Việt), có cache bằng SQLite theo query đã normalize. Thiết kế sẵn điểm mở rộng sang LLM-based sau này.

**Browse & Verify**
- 📂 **Data Browser** — duyệt keyframe theo video_ID + khoảng thời gian, dùng để xác nhận nhanh 1 chuỗi frame trước khi nộp kết quả.
- 📝 **Event Browser** — duyệt theo video_ID + khoảng event_id (local theo từng video), dùng để xác nhận nhanh cặp transcript–chuỗi frame trả về từ Type 3.

## 🏗️ Architecture Overview
![SceneSeek Architecture](diagram/SceneSeek_diagram.png)

<details>
<summary>Text version</summary>

<pre>
                     ┌───────────────────────────────┐
                     │        Frontend (React)       │
                     │  SearchContext / GalleryItem  │
                     └───────────────┬───────────────┘
                                     │ REST (fallback mock nếu 501/network error)
                     ┌───────────────▼────────────────┐
                     │      FastAPI routers           │
                     │  search_router / data_router   │
                     │        / event_router          │
                     └───────────────┬────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                            ▼
 [Type 1: Frame]            [Type 2: Event Boundary]      [Type 3: Event Mention]
 Jina CLIP v2 (text)        Jina CLIP v2 (N queries)       dangvantuan (text-text)
 → HNSW index                → HNSW batch search           → FlatIP index
 → sort/group results        → cluster_by_video()          → optional hybrid: + BM25
                            (best_ordered_chain DP)        → RRF merge

        ┌─────────────────────────────────────────────────────────┐
        ▼                                                         ▼
 [Similar Frame]                                          [Browse & Verify]
 encoded_frames (preloaded)                          /api/data (by timestamp)
 → cosine similarity (torch.topk)                    /api/events (by event_id)
</pre>

</details>

Chi tiết thuật toán, schema, và các quyết định thiết kế xem tại [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## 🛠️ Tech Stack

| Thành phần | Công nghệ |
|---|---|
| Text-image embedding (Type 1, 2) | Jina CLIP v2 (`jinaai/jina-clip-v2`) |
| Text-text embedding (Type 3) | dangvantuan (`dangvantuan/vietnamese-embedding`) |
| Keyword search (Type 3 hybrid) | BM25 (rank_bm25) + `pyvi` (tokenize tiếng Việt) |
| Vector index | FAISS — HNSW (Type 1/2), FlatIP (Type 3) |
| Keyword suggestion | Rule-based (n-gram + vocab) + `pyvi` POS tagging (fallback), cache bằng SQLite |
| Backend | Python + FastAPI (dependency injection qua `deps.py`) |
| Frontend | React (fetch-with-mock-fallback pattern) |

## 📦 Dataset

- **Nguồn**: bộ dữ liệu 838 video dùng cho **Hội thi Thử thách Trí tuệ Nhân tạo (AI Challenge) Thành phố Hồ Chí Minh năm 2026**.
- **Nội dung**: đa dạng chủ đề — tin tức thời sự, cuộc thi múa lân, dạy nấu ăn, ôn thi online, phim tài liệu ngắn theo nhiều chủ đề khác nhau...
- Định dạng dữ liệu: keyframe theo video (`video_ID`, `frame_ID`, `frame_idx`, `time_in_seconds`...) + transcript đã segment thành event theo từng video (`event_id` được chuẩn hoá lại thành local/gap-free per video ngay khi khởi động — xem chi tiết tại `docs/ARCHITECTURE.md` §2.4).
- Cấu hình đường dẫn dữ liệu/model tại `config/databases.json` (FAISS index, encoded frames, BM25, vocab, query cache) và `config/models.json` (model identifier). Đổi dataset/model chỉ cần sửa 2 file JSON này, không cần sửa code loader.
- Pipeline tạo ra các file trong `static/databases/...` (build FAISS index, encode frames, tạo BM25 index, vocab): `__TODO: mô tả notebook/script build-database dùng để tạo các file này__`

## ⚙️ Yêu cầu hệ thống

- **Hệ điều hành**: Windows 10/11, macOS, hoặc Linux
- **CPU**: tối thiểu 4 nhân (khuyến nghị 8+ nhân để hiệu năng tốt hơn)
- **RAM**: tối thiểu 8GB (khuyến nghị 16GB+)
- **Dung lượng ổ đĩa**: tối thiểu 12GB cho database và cài đặt
- **Python**: 3.9 trở lên
- **Kết nối Internet**: cần thiết để tải các file database

> **Lưu ý**: SceneSeek được tối ưu để chạy trên CPU thông thường, không bắt buộc cần GPU.

## 🚀 Getting Started

### Cách 1 — Thử nhanh trên Google Colab
 
Không cần cài đặt gì, chạy thử ngay trên trình duyệt:
 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThuyHaLE/SceneSeek/blob/main/app_notebook.ipynb)
 
**Trước khi chạy notebook**, bạn cần chuẩn bị 1 ngrok authtoken để notebook có thể expose server ra 1 URL public truy cập được (vì Colab không cho truy cập trực tiếp vào localhost):
 
1. Tạo tài khoản free tại [ngrok.com](https://ngrok.com), vào Dashboard → **Your Authtoken** để lấy token.
2. Trong Colab, bấm icon 🔑 (Secrets) ở sidebar trái → **Add new secret**:
   - Name: `NGROK_API_KEY`
   - Value: dán authtoken vừa copy
   - **Bật toggle "Notebook access"** cho secret này (nếu quên bước này, notebook sẽ báo lỗi không đọc được secret dù đã tạo).
3. Chạy notebook như bình thường — cell cuối sẽ in ra 1 URL public (dạng `https://xxxx.ngrok-free.app`), đây là link để truy cập SceneSeek từ trình duyệt.
> **Lưu ý**: giữ cell đang chạy (không đóng tab/ngắt kết nối) — tunnel sẽ đóng ngay khi cell dừng.

### Cách 2 — Cài đặt local đầy đủ

```bash
# Clone SceneSeek repository
!git clone https://github.com/ThuyHaLE/SceneSeek.git
%cd SceneSeek

# Cài đặt backend
cd backend
pip install -r requirements.txt

# Cài đặt frontend
cd ../sceneseek-frontend
npm install
```

#### Build index (chạy 1 lần, hoặc khi dataset thay đổi)

Các file index/embedding (FAISS `.bin`, `encoded_frames.pt`, BM25 `.pkl`, vocab...) được tạo sẵn từ notebook/script build-database và đặt vào các đường dẫn khai báo trong `config/databases.json`. Khi backend khởi động (`uvicorn main:app`), `model_state.py` tự động load toàn bộ 1 lần duy nhất — không cần chạy thêm bước "build index" riêng trước khi start server, miễn là các file trong `config/databases.json` đã tồn tại đúng đường dẫn.

`__TODO: nếu có script/notebook riêng để TẠO các file index này từ video thô, ghi rõ lệnh chạy tại đây (VD: build-database.ipynb).__`

#### Chạy ứng dụng

```bash
# Terminal 1: backend
cd backend
uvicorn main:app --reload

# Terminal 2: frontend
cd sceneseek-frontend
npm run dev
```

Truy cập: `http://localhost:5173` (hoặc port frontend đang dùng). Nếu backend chưa chạy hoặc route chưa implement (trả 501), frontend tự động fallback sang mock data — vẫn xem được UI mà không cần backend sẵn sàng 100%.

## 💡 Usage

1. Chọn loại tìm kiếm: **Frame** (Type 1), **Event Boundary** (Type 2), hoặc **Event Mention** (Type 3).
2. Với Type 2: nhập 2–5 mô tả cảnh theo đúng thứ tự thời gian mong muốn; chọn "tìm chính xác" hoặc "tìm tương đối".
3. Với Type 3: có thể thêm từ khoá để kích hoạt hybrid search (dense + BM25).
4. Trên kết quả, click 🔍 ở 1 frame để tìm các frame tương tự (Similar Frame Search).
5. Dùng **Data Browser** hoặc **Event Browser** để xác nhận lại chuỗi frame/transcript trước khi chốt kết quả cuối.

## ⚠️ Known Limitations / Roadmap

- [ ] `displayOption` bị bỏ qua ở Type 2 (frontend hard-code `sort_by_frame_index`) — chưa cho user tuỳ chỉnh.
- [ ] `/api/search/{type}/refine` (relevance feedback re-rank) đang gọi từ frontend nhưng `__TODO: xác nhận đã implement ở backend chưa__`.
- [ ] Keyword suggestion hiện chỉ rule-based, giới hạn bởi độ phủ của `vocab` — chưa suy luận được từ đồng nghĩa/liên quan.
- [ ] **Keyword input cho Type 1/2** — backend Type 1 đã có sẵn logic (nối keyword vào query), nhưng chưa expose UI vì cần thiết kế cách "boost" keyword hợp lý trong không gian text-image (khác với text-text ở Type 3). Dự kiến phát triển sau khi có thời gian đánh giá chất lượng.
- [ ] `__TODO: liệt kê giới hạn khác nếu có__`

## 📄 License / Acknowledgement

- Models: Jina CLIP v2, dangvantuan (Vietnamese sentence embedding) — `__TODO: ghi rõ nguồn/license từng model__`
- Dataset: `__TODO__`
