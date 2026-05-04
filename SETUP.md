# SETUP — Deploy OE-VLM Shop trên VM

Hướng dẫn từ A → Z để dựng demo trên một VM Ubuntu sạch (đã test trên Ubuntu 22.04 / WSL2).
Stack: FastAPI + MongoDB + embedded Qdrant + FG-CLIP 2 + React/Vite.

> **GPU**: nếu VM có CUDA, FG-CLIP 2 sẽ tự dùng. Không có GPU thì rơi về CPU (chậm hơn ~5–10× khi seed; query vẫn chấp nhận được).

---

## 0. Yêu cầu hệ thống

| Thành phần | Phiên bản tối thiểu | Ghi chú |
|---|---|---|
| Python | 3.11 (khuyến nghị 3.12) | torch 2.11 yêu cầu ≥ 3.10 |
| Node.js | 18 LTS | cho Vite |
| Docker + docker compose | mới nhất | chạy MongoDB |
| Git | bất kỳ | |
| RAM | ≥ 8 GB | model FG-CLIP 2 base ≈ 2 GB |
| Disk | ≥ 10 GB trống | model cache + Mongo |
| GPU (tuỳ chọn) | CUDA 12.x + driver tương ứng | tăng tốc embedding |

Cài hệ thống nhanh trên Ubuntu:

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip git curl ca-certificates
# Node 18
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs
# Docker (nếu chưa có)
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker
```

---

## 1. Clone repo

```bash
git clone <repo-url> oe-vlm-demo
cd oe-vlm-demo
git checkout dev/fg-clip2-qdrant-for-demo
```

---

## 2. Khởi động MongoDB

```bash
docker compose up -d
docker compose ps          # mongodb phải ở trạng thái "running"
```

`docker-compose.yml` mở port `27017` ra host. Qdrant **không** cần Docker — chạy nhúng trong process backend (lưu file ở `backend/qdrant_storage/`).

---

## 3. Setup backend

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Lần đầu cài torch 2.11 + transformers sẽ tốn vài phút. Trên VM không có GPU, có thể cài bản CPU-only của torch để tiết kiệm dung lượng:

```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
```

### 3.1. Cấu hình `.env`

```bash
cp .env.example .env
nano .env       # hoặc vim/code
```

> ⚠️ **NHỚ ĐỔI `MONGODB_URL`** nếu Mongo **không** chạy ở localhost:
> - Mongo trong Docker cùng host: giữ `mongodb://localhost:27017`.
> - Mongo trên container khác / cùng compose network: `mongodb://mongodb:27017`.
> - Mongo Atlas / managed: dán nguyên connection string `mongodb+srv://user:pass@cluster…/`.
> - Mongo trên VM khác: `mongodb://<host-or-ip>:27017` (mở firewall port 27017).
>
> Các biến khác (`QDRANT_PATH`, `FGCLIP_MODEL_ID`) thường giữ mặc định.

### 3.2. (Tuỳ chọn) Test load FG-CLIP 2

```bash
python -c "from app.services.clip_service import load_clip_model, embed_text; load_clip_model(); print(embed_text('giày chạy bộ').shape)"
```

Lần đầu sẽ download checkpoint (≈ 2 GB) về `~/.cache/huggingface/`. Output mong đợi: `(768,)`.

---

## 4. Dữ liệu

Hai cách: **(A) tải snapshot Qdrant có sẵn** (nhanh, khuyến nghị nếu MongoDB đã trỏ Atlas dùng chung) hoặc **(B) seed lại từ CSV** (chậm hơn, dùng khi chạy Mongo local hoặc muốn rebuild embedding).

> 🛑 **Backend (uvicorn) phải đang TẮT** trước khi chạy bất kỳ thao tác nào với `backend/qdrant_storage/` — embedded Qdrant giữ file lock độc quyền.

### 4.A. Tải snapshot Qdrant từ Hugging Face (khuyến nghị)

Dùng khi `MONGODB_URL` trong `.env` đã trỏ tới Atlas (ObjectId stable → khớp với embedding trong snapshot).

```bash
pip install -U huggingface_hub        # nếu chưa có

# từ repo root
hf download ntAnh-dev/oe-vlm-qdrant bundle.tar.gz \
  --repo-type dataset \
  --local-dir .

# giải nén vào đúng vị trí backend/qdrant_storage/
tar -xzf bundle.tar.gz
rm -f backend/qdrant_storage/.lock    # phòng khi snapshot kèm lock cũ
```

Skip hoàn toàn FG-CLIP 2 download (~2 GB) và bước embedding 30–60 phút. Sang thẳng mục 5.

### 4.B. Seed lại từ CSV

CSV nguồn đã có sẵn ở `backend/data/products.csv` (3.4k dòng).

```bash
# vẫn đang ở backend/ với venv bật
python seed_data.py --csv ./data/products.csv
```

Quá trình:
1. Xoá collection Qdrant cũ rồi tạo lại.
2. Với mỗi dòng: tải ảnh → preprocess (alpha compositing nếu nền trong suốt) → FG-CLIP 2 image features → early fusion với prompt cố định (0.9 image + 0.1 text) → upsert Mongo + Qdrant.

Thời gian: ~30–60 phút trên CPU, ~5–10 phút trên 1× GPU consumer. Tải ảnh URL fail sẽ ghi vào `backend/failures.json`.

---

## 5. Chạy backend

```bash
# vẫn ở backend/ với venv bật
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Smoke test:

```bash
curl -s http://localhost:8000/api/products | head -c 300
```

Phải trả JSON `{"items":[…],"total":…}`.

---

## 6. Setup & chạy frontend

Mở terminal thứ 2:

```bash
cd frontend
npm install
```

### 6.1. Dev mode (hot reload)

Vite dev server proxy `/api/*` sang `http://localhost:8000` (xem `vite.config.ts`).

```bash
npm run dev -- --host 0.0.0.0 --port 5173
```

Mở browser: `http://<vm-ip>:5173`.

### 6.2. Production build (tĩnh)

```bash
npm run build      # output: frontend/dist/
```

Serve bằng nginx / caddy / `npx serve`. Reverse-proxy `/api/*` về backend port 8000. Ví dụ nginx tối thiểu:

```nginx
server {
    listen 80;
    server_name _;
    root /path/to/oe-vlm-demo/frontend/dist;
    index index.html;

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
    }

    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

---

## 7. Chạy như service (tuỳ chọn)

Để uvicorn không chết khi đóng SSH, dùng systemd. Tạo `/etc/systemd/system/oe-vlm-backend.service`:

```ini
[Unit]
Description=OE-VLM backend
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/oe-vlm-demo/backend
Environment="PATH=/home/ubuntu/oe-vlm-demo/backend/.venv/bin"
ExecStart=/home/ubuntu/oe-vlm-demo/backend/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now oe-vlm-backend
sudo journalctl -u oe-vlm-backend -f      # xem log
```

---

## 8. Smoke test cuối

1. Trang chủ `http://<vm-ip>` (hoặc :5173 nếu dev) load được, hiện grid sản phẩm.
2. Search "giày chạy bộ nam" → kết quả semantic xuất hiện, không có lỗi 500.
3. Click 1 sản phẩm → trang chi tiết load, "Sản phẩm liên quan" có dữ liệu.
4. Backend log không có traceback `IndexError` từ FG-CLIP 2.

---

## Troubleshooting

| Triệu chứng | Nguyên nhân & xử lý |
|---|---|
| `Cannot open qdrant_storage — backend still running?` | Backend đang giữ lock. Tắt uvicorn rồi chạy seed/mock. |
| `IndexError: index out of range in self` khi embed text | Code đã có patch `_repair_text_embeddings`. Nếu vẫn dính → check `torch==2.11.0` + `transformers>=4.56.0` đúng phiên bản. |
| `pymongo.errors.ServerSelectionTimeoutError` | Sai `MONGODB_URL`. Verify: `docker compose ps` và `mongosh "<MONGODB_URL>"`. |
| Frontend gọi `/api/...` 404 | Proxy chưa cấu hình. Dev mode dùng Vite proxy (mặc định OK); prod cần nginx như mục 6.2. |
| Seed quá chậm trên CPU | Chấp nhận, hoặc kiếm GPU. Có thể song song `--workers` (cần sửa seed_data, hiện chạy tuần tự). |
| Ổ đĩa đầy sau khi pip install | Torch CUDA ≈ 5 GB. Dùng bản CPU-only như mục 3 nếu không cần GPU. |

---

## Tóm tắt 1 dòng

```bash
docker compose up -d && \
cd backend && python3.12 -m venv .venv && source .venv/bin/activate && \
pip install -r requirements.txt && cp .env.example .env && \
# >>> SỬA MONGODB_URL TRONG .env NẾU CẦN <<< \
python seed_data.py --csv ./data/products.csv && \
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
cd ../frontend && npm install && npm run dev -- --host 0.0.0.0
```
