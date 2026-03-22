# HW01 — Your Own ChatGPT

> 作者：112550190 劉彥廷

---

## 原速影片

[Youtube_link](https://youtu.be/tL5YyFkqD6o)

---


---

## 專案簡介

本專案以 **FastAPI** 作為後端、**HTML + CSS + JavaScript** 作為前端，串接 NVIDIA NIM API，實作一個具備多種進階功能的自製 ChatGPT 介面。

---

## 主要技術

| 層級 | 技術 |
|------|------|
| 後端 | FastAPI + Python |
| 前端 | HTML / CSS / JavaScript |
| API  | NVIDIA NIM（OpenAI 相容介面）|
| 資料庫 | SQLite（本地持久化對話記錄）|

---

## 功能列表

### 核心功能
- 可挑選 LLM 模型（文字模型 / Vision 模型）
- 可自訂 System Prompt，並支援多組自訂模板管理
- 可調整 Temperature、Top-p、Max Tokens 等 API 參數
- 可切換 Streaming 串流輸出
- 可切換短期記憶，並自訂記憶視窗大小

### 進階功能
- **Thinking 模式**：讓 LLM 先進行深度推理再回答，可收折查看思考過程
- **圖片 / 檔案上傳**：支援 Vision 模型讀取圖片，亦可上傳文字檔、PDF 作為上下文
- **對話分支**：編輯訊息後可切換不同版本的問答，支援重新生成並比較結果
- **持久化對話記錄**：以本地 SQLite 儲存歷史對話，重啟後仍可繼續
- **對話匯出**：可將當前對話下載為 `.md` 或 `.json`
- **Token 用量顯示**：每則回答顯示 prompt / completion token 數與回應時間
- **Token 折線圖**：視覺化每次對話的 token 消耗
- **深色 / 淺色主題**：一鍵切換，設定自動保存
- **自訂 Accent 顏色**：六種預設色票 + 任意 hex 顏色選擇
- **字體大小調整**：側欄滑桿即時調整
- **語音輸入**：使用瀏覽器 Web Speech API 語音轉文字

---

## 專案結構

```
hw01-python-chatgpt/
├── app.py               # FastAPI 主程式
├── requirements.txt     # Python 套件清單
├── .env                 # 環境變數（不上傳 GitHub）
├── .env.example         # 環境變數範本
├── .gitignore
├── README.md
├── chat_history.db      # SQLite 資料庫（自動生成）
├── templates/
│   └── index.html       # 前端頁面
└── static/
    ├── style.css
    └── app.js
```

---

## 安裝與執行

### 1. 建立虛擬環境

```bash
python -m venv .venv
```

啟動虛擬環境：

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. 安裝套件

```bash
pip install -r requirements.txt
```

### 3. 設定環境變數

將 `.env.example` 複製為 `.env`，並填入 API 金鑰：

```env
NVIDIA_API_KEY=your_nvidia_api_key_here
```

### 4. 啟動伺服器

```bash
uvicorn app:app --reload
```

### 5. 打開瀏覽器

```
http://127.0.0.1:8000
```

---

## Demo 展示

1. 切換模型（文字模型 / Vision 模型）
2. 修改 System Prompt，使用模板快速切換
3. 調整 Temperature / Top-p / Max Tokens
4. 開關 Streaming 串流輸出
5. 開關短期記憶，展示有無上下文的差異
6. 編輯訊息並切換對話分支，比較不同版本回答
7. 上傳圖片，使用 Vision 模型進行圖片描述
8. 開啟 Thinking 模式，展示推理過程
9. 清空對話 / 停止生成

---

## 安全注意事項

- `.env` 檔案請勿上傳至 GitHub，已列入 `.gitignore`
- GitHub 上只放 `.env.example`（不含真實金鑰）
- API 金鑰只存放於後端，不寫入前端程式碼

---

> 禁止抄襲與複製貼上，抄襲是不誠信的行為。
