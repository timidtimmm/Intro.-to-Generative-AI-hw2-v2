# HW01 - Your own ChatGPT

## 主要概念
- 後端：FastAPI
- 前端：HTML + CSS
- 少量 JavaScript：送出訊息、接收 streaming、更新畫面

## 功能

- 可挑選 LLM 模型
- 可自訂 system prompt
- 可自訂常用 API 參數
- 可切換 streaming
- 可切換短期記憶

## 專案結構

```text
hw01-python-chatgpt/
├─ app.py
├─ requirements.txt
├─ .env
├─ .gitignore
├─ README.md
├─ templates/
│  └─ index.html
└─ static/
   ├─ style.css
   └─ app.js
```

## 安裝方式

1. 建立虛擬環境

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

2. 安裝套件

```bash
pip install -r requirements.txt
```

3. 設定環境變數

把 `.env.example` 複製成 `.env`，然後填入自己的 OpenAI API key。

```env
OPENAI_API_KEY=your_openai_api_key_here
```

4. 啟動伺服器

```bash
uvicorn app:app --reload
```

5. 打開瀏覽器

```text
http://127.0.0.1:8000
```

## Demo 建議展示

1. 切換模型
2. 修改 system prompt
3. 調整 temperature / top-p / max tokens
4. 開關 streaming
5. 開關短期記憶，展示上下文差異

## 交作業注意事項

- `.env` 不要上傳到 GitHub
- GitHub 只放 `.env.example`
- API key 不要寫死在前端或程式碼裡
