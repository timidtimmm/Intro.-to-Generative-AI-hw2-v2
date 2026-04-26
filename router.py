"""
router.py — Rule-based model router
Selects the best model and decides whether to use tools/memory
based on detected modality and task type.

FIXES:
  - image_gen pattern: added more Chinese trigger words (豬、幫我生成、生成一張)
  - datetime pattern: added more natural Chinese expressions
  - Widened search pattern to catch more implicit search queries
"""

import re
from dataclasses import dataclass

# ── Model catalogue ───────────────────────────────────────────────────────────

CHEAP_MODEL = "openai/gpt-oss-20b"
DEFAULT_MODEL = "openai/gpt-oss-120b"
STRONG_MODEL = "moonshotai/kimi-k2-instruct"
GEMINI_FLASH = "gemini/gemini-2.5-flash"
GEMINI_PRO = "gemini/gemini-2.5-pro"
GEMINI_IMAGE = "gemini/gemini-2.0-flash-exp"   # ✅ FIX: use real image-capable model
NVIDIA_VISION = "meta/llama-3.2-11b-vision-instruct"
VISION_ALT = "microsoft/phi-3.5-vision-instruct"

ALL_MODELS = [
    DEFAULT_MODEL,
    CHEAP_MODEL,
    STRONG_MODEL,
    GEMINI_FLASH,
    GEMINI_PRO,
    NVIDIA_VISION,
    VISION_ALT,
]

VISION_MODELS = {GEMINI_FLASH, GEMINI_PRO, NVIDIA_VISION, VISION_ALT}

# ── Decision object ───────────────────────────────────────────────────────────

@dataclass
class RouteDecision:
    model: str
    reason: str
    use_tools: bool
    use_memory: bool
    task_type: str = "general"
    confidence: float = 1.0


# ── Task detection ────────────────────────────────────────────────────────────

_TASK_PATTERNS = {
    # ✅ FIX: Much wider image_gen pattern to catch Chinese requests like:
    #   "幫我生一張豬的圖片", "生成一隻狗", "畫一個貓", "畫張圖", "生成圖片"
    "image_gen": re.compile(
        # English: draw/generate/create + image keyword
        r"(?:generate|create|draw|make|paint|render)\s*(?:an?\s+)?(?:image|picture|photo|illustration|art|drawing|painting)|"
        # Chinese: 生成/畫 + optional counter + any content + 圖片 keyword
        r"(?:生成|產生|幫我畫|幫我生|繪製|製作)\s*(?:一張|一個|一隻|一幅|一副|一些)?.*?(?:圖片|圖像|圖|照片|插圖|畫面|影像)|"
        # Chinese: 畫/生 + counter + subject (no 圖片 needed, e.g. "畫一隻狗", "生一隻貓")
        r"(?:畫|生|做)\s*(?:一張|一個|一隻|一幅|一副|一條|一頭|一朵|一棵|一座)\S|"
        # Standalone keywords
        r"(?:圖片生成|生成圖片|生成圖像|AI繪圖|AI作圖)|"
        r"(?:image\s+gen(?:eration)?|text.to.image)",
        re.I | re.DOTALL,
    ),
    "weather": re.compile(r"weather|forecast|temperature|天氣|氣溫|降雨|預報|幾度|下雨", re.I),
    # ✅ FIX: Expanded datetime pattern with more natural expressions
    "datetime": re.compile(
        r"what\s+time|what\s+day|today(?:'s)?\s+date|current\s+time|"
        r"現在(?:幾點|幾號|是幾|時間|時刻)|今天(?:幾號|是幾|星期|日期|是什麼|的日期)|"
        r"幾點了|幾號了|星期幾|日期是|今日日期|現在時間|現在幾點|"
        r"告訴我時間|現在是|今天是|查時間|時間是",
        re.I,
    ),
    "math": re.compile(r"calculate|solve|equation|math|integral|derivative|\d\s*[\+\-\*/\^]\s*\d|計算|解方程|微分|積分", re.I),
    "github": re.compile(r"github|repo|repository|開源|source\s*code|原始碼", re.I),
    "presentation": re.compile(r"ppt|powerpoint|slide|slides|deck|簡報|投影片|簡報檔", re.I),
    "search": re.compile(
        r"google|bing|search|look\s+up|find\s+(?:out|info)|what\s+is\s+(?:the\s+latest|current)|"
        r"查詢|查一下|幫我找|幫我查|幫我搜|搜尋|搜索|簡介|最新|官方|相容|支援機型|"
        r"現在(?:幾點|幾度|幾號|是誰|在哪|怎樣)|目前(?:狀況|情況|價格|股價|匯率)",
        re.I,
    ),
    "summarize": re.compile(r"summari[sz]e|tl;dr|重點整理|摘要|總結", re.I),
    "translation": re.compile(r"translate|translation|翻譯", re.I),
    "coding": re.compile(r"code|bug|debug|python|javascript|java|c\+\+|verilog|程式|除錯|修bug|函式|fastapi|html|css|js", re.I),
    "creative": re.compile(r"story|poem|novel|creative|brainstorm|故事|詩|創作|發想", re.I),
    "url_fetch": re.compile(r"fetch|scrape|read\s+url|open\s+url|visit|瀏覽|讀取網頁|抓取", re.I),
    "unit_converter": re.compile(r"convert|換算|轉換|單位|公里|英里|磅|公斤|華氏|攝氏|開爾文|公升|加侖", re.I),
    "note_manager": re.compile(r"note|memo|記事|筆記|備忘|記錄|寫下|儲存(?:這個|這段)", re.I),
    "qr_generator": re.compile(r"qr|qr\s*code|二維碼|條碼|掃描碼", re.I),
    "hash_generator": re.compile(r"hash|md5|sha|sha256|sha512|校驗碼|雜湊", re.I),
    "color_converter": re.compile(r"color|colour|hex|rgb|hsl|顏色|色碼|色值", re.I),
    "dictionary_lookup": re.compile(r"define|definition|what does .+ mean|dictionary|字典|查字|單字|詞義", re.I),
    "random_generator": re.compile(r"random|randomize|generate\s+(password|uuid|number)|隨機|亂數|密碼生成|產生UUID", re.I),
    "timezone_converter": re.compile(r"timezone|time\s+zone|convert\s+time|時區|時差|轉換時間", re.I),
    # 在 _TASK_PATTERNS 裡加：
    "youtube": re.compile(r"youtube|youtu\.be|影片摘要|幫我摘要.*影片|這部影片|字幕",re.I),
}


def detect_task_type(text: str) -> str:
    t = (text or "").strip()
    for name, pat in _TASK_PATTERNS.items():
        if pat.search(t):
            return name
    return "general"


def needs_tools(text: str, task_type: str) -> bool:
    return task_type in {"image_gen", "search", "weather", "datetime", "math", "github", "presentation", "url_fetch", "unit_converter", "note_manager", "qr_generator", "hash_generator", "color_converter", "dictionary_lookup", "random_generator", "timezone_converter", "youtube"}


# ── Main routing ──────────────────────────────────────────────────────────────

def route_request(
    text: str,
    has_image: bool = False,
    preferred_model: str | None = None,
    auto_route: bool = True,
    attachment_kind: str | None = None,
) -> RouteDecision:
    """Main routing logic."""
    task = detect_task_type(text)
    attachment_kind = (attachment_kind or "").lower().strip()

    # Auto-routing disabled → respect user's model choice
    if not auto_route:
        m = preferred_model if preferred_model in ALL_MODELS else DEFAULT_MODEL
        return RouteDecision(
            model=m,
            reason="手動選擇 (自動路由已關閉)",
            use_tools=needs_tools(text, task),
            use_memory=True,
            task_type=task,
        )
    # router.py 的 route_request() 裡，在現有邏輯前加入：
    # ★ 長文分析 → Gemini Pro（優先判斷，放最前面）
    _LONG_TEXT_ANALYSIS = re.compile(
        r"分析|分析以下|幫我分析|請分析|解析|評析|詮釋|閱讀分析|"
        r"analyze|analysis|interpret|review|examine",
        re.I
    )
    if _LONG_TEXT_ANALYSIS.search(text) and len(text) > 300:
        return RouteDecision(
            model=GEMINI_PRO,
            reason="長文分析 → Gemini 2.5 Pro（長上下文 + 理解力強）",
            use_tools=False,
            use_memory=True,
            task_type="analysis",
    )
    # ⑤ 長文摘要 / 長上下文 → Gemini Pro（context window 最大）
    # ★ 長文摘要 → Gemini Pro
    if task == "summarize" and len(text) > 300:
        return RouteDecision(
            model=GEMINI_PRO,
            reason="長文摘要 → Gemini 2.5 Pro（長上下文能力強）",
            use_tools=False,
            use_memory=True,
            task_type="summarize",
        )

    # ⑤ 短翻譯 → 輕量模型省成本
    if task == "translation" and len(text) < 200:
        return RouteDecision(
            model=CHEAP_MODEL,
            reason="短文翻譯 → 輕量模型（gpt-oss-20b）省成本",
            use_tools=False, use_memory=True, task_type=task,
        )

    # ⑥ 複雜推理 / 數學證明 → 強力模型
    _HARD_REASONING = re.compile(
        r"證明|推導|為什麼.*原理|解釋.*機制|分析.*原因|"
        r"prove|derive|explain why|analyze|reasoning",
        re.I
    )
    if _HARD_REASONING.search(text) and task not in {"coding", "math"}:
        return RouteDecision(
            model=STRONG_MODEL,
            reason="複雜推理 → kimi-k2（推理能力強）",
            use_tools=False, use_memory=True, task_type="reasoning",
        )

    # ⑦ 程式碼 → Gemini Flash（速度快 + 夠準）
    if task == "coding":
        return RouteDecision(
            model=GEMINI_FLASH,
            reason="程式撰寫 → Gemini 2.5 Flash（速度快）",
            use_tools=False, use_memory=True, task_type=task,
        )

    # ⑧ 創意寫作 → kimi-k2
    if task == "creative":
        return RouteDecision(
            model=STRONG_MODEL,
            reason="創意寫作 → kimi-k2（創作能力強）",
            use_tools=False, use_memory=True, task_type=task,
        )
        # ① Image input → Gemini multimodal by default
    if has_image:
        return RouteDecision(
            model=GEMINI_FLASH,
            reason="圖片輸入 → Gemini multimodal",
            use_tools=False,
            use_memory=True,
            task_type="vision",
        )

    # ② PDF / text attachment → Gemini long-context model
    if attachment_kind in {"text", "pdf", "document"}:
        return RouteDecision(
            model=GEMINI_FLASH,
            reason="文件 / PDF 輸入 → Gemini 長上下文 / multimodal",
            use_tools=False,
            use_memory=True,
            task_type="document",
        )

    # ③ Image generation → Gemini image tool
    if task == "image_gen":
        return RouteDecision(
            model=GEMINI_IMAGE,
            reason="圖片生成 → 呼叫 Gemini 圖片生成工具",
            use_tools=True,
            use_memory=False,
            task_type=task,
        )

    # ④ Search / GitHub / presentation / weather / datetime / math → tools
    if task in {"search", "github", "presentation", "weather", "datetime", "math"}:
        reason_map = {
            "search":       "需要外部資料 → 呼叫搜尋工具",
            "github":       "GitHub / repo 查詢 → 呼叫 GitHub 搜尋工具",
            "presentation": "簡報任務 → 呼叫簡報生成工具",
            "weather":      "天氣查詢 → 呼叫天氣工具",
            "datetime":     "時間 / 日期查詢 → 呼叫 get_datetime 工具",
            "math":         "數學計算 → 呼叫計算器工具",
        }
        return RouteDecision(
            model=DEFAULT_MODEL,
            reason=reason_map.get(task, f"任務類型 '{task}' → 工具使用"),
            use_tools=True,
            use_memory=task != "math",
            task_type=task,
        )

    # ④b — New MCP utility tools
    if task in {"url_fetch", "unit_converter", "note_manager", "qr_generator",
                "hash_generator", "color_converter", "dictionary_lookup",
                "random_generator", "timezone_converter"}:
        reason_map = {
            "url_fetch":          "網頁讀取 → url_fetch 工具",
            "unit_converter":     "單位換算 → unit_converter 工具",
            "note_manager":       "筆記管理 → note_manager 工具",
            "qr_generator":       "QR Code 生成 → qr_generator 工具",
            "hash_generator":     "雜湊計算 → hash_generator 工具",
            "color_converter":    "顏色轉換 → color_converter 工具",
            "dictionary_lookup":  "英文字典 → dictionary_lookup 工具",
            "random_generator":   "隨機生成 → random_generator 工具",
            "timezone_converter": "時區轉換 → timezone_converter 工具",
        }
        return RouteDecision(
            model=DEFAULT_MODEL,
            reason=reason_map.get(task, f"任務 {task} → 工具"),
            use_tools=True,
            use_memory=True,
            task_type=task,
        )

    # ⑤ Simple tasks → cheap model
    if task in {"summarize", "translation"}:
        return RouteDecision(
            model=CHEAP_MODEL,
            reason=f"簡單任務 '{task}' → 輕量模型",
            use_tools=False,
            use_memory=True,
            task_type=task,
        )
    
    if task == "youtube":
        return RouteDecision(
            model=DEFAULT_MODEL,
            reason="YouTube 字幕摘要 → youtube_transcript 工具",
            use_tools=True, use_memory=False, task_type=task,
        )

    # ⑥ Coding → default (good tool ecosystem and code ability)
    if task == "coding":
        return RouteDecision(
            model=DEFAULT_MODEL,
            reason="程式任務 → 預設文字模型",
            use_tools=False,
            use_memory=True,
            task_type=task,
        )

    # ⑦ Creative → stronger model
    if task == "creative":
        return RouteDecision(
            model=STRONG_MODEL,
            reason="創意寫作 → 強力模型",
            use_tools=False,
            use_memory=True,
            task_type=task,
        )

    # ⑧ Default general chat
    return RouteDecision(
        model=DEFAULT_MODEL,
        reason="一般對話 → 自動選擇預設文字模型",
        use_tools=False,
        use_memory=True,
        task_type=task,
    )