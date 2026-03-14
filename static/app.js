const PROMPT_PRESETS = {
  default: "You are a helpful assistant. Answer clearly and accurately.",
  teacher: "You are a patient teacher. Explain step by step with simple examples.",
  coder: "You are a senior software engineer. Write robust and maintainable code.",
  translator: "You are a professional translator. Translate accurately and preserve tone."
};

const messagesEl = document.getElementById("messages");
const errorBox = document.getElementById("errorBox");
const sendButton = document.getElementById("sendButton");
const chatInput = document.getElementById("chatInput");
const modelEl = document.getElementById("model");
const currentModelText = document.getElementById("currentModelText");

const systemPromptEl = document.getElementById("systemPrompt");
const temperatureEl = document.getElementById("temperature");
const topPEl = document.getElementById("topP");
const maxTokensEl = document.getElementById("maxTokens");
const streamingEl = document.getElementById("streaming");
const shortMemoryEl = document.getElementById("shortMemory");
const memoryWindowEl = document.getElementById("memoryWindow");

const temperatureValueEl = document.getElementById("temperatureValue");
const topPValueEl = document.getElementById("topPValue");
const maxTokensValueEl = document.getElementById("maxTokensValue");
const memoryWindowValueEl = document.getElementById("memoryWindowValue");

const streamingBadge = document.getElementById("streamingBadge");
const memoryBadge = document.getElementById("memoryBadge");

const clearChatButton = document.getElementById("clearChat");
const stopStreamingButton = document.getElementById("stopStreaming");

const presetDefault = document.getElementById("presetDefault");
const presetTeacher = document.getElementById("presetTeacher");
const presetCoder = document.getElementById("presetCoder");
const presetTranslator = document.getElementById("presetTranslator");

let messages = [
  {
    role: "assistant",
    content: "嗨，我是你的自製 ChatGPT。現在後端接的是 NVIDIA NIM。你可以切換模型、調整參數、改 system prompt，還能開關 streaming 與短期記憶。"
  }
];

let isLoading = false;
let abortController = null;

function renderMessages() {
  messagesEl.innerHTML = "";

  messages.forEach((msg) => {
    const row = document.createElement("div");
    row.className = `message-row ${msg.role === "user" ? "user-row" : "assistant-row"}`;

    const bubble = document.createElement("div");
    bubble.className = `message-bubble ${msg.role}`;

    const role = document.createElement("div");
    role.className = "message-role";
    role.textContent = msg.role === "user" ? "You" : "Assistant";

    const content = document.createElement("div");
    content.className = "message-content";
    content.textContent = msg.content;

    bubble.appendChild(role);
    bubble.appendChild(content);
    row.appendChild(bubble);
    messagesEl.appendChild(row);
  });

  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setError(text = "") {
  if (!text) {
    errorBox.textContent = "";
    errorBox.classList.add("hidden");
    return;
  }
  errorBox.textContent = text;
  errorBox.classList.remove("hidden");
}

function updateControls() {
  temperatureValueEl.textContent = Number(temperatureEl.value).toFixed(1);
  topPValueEl.textContent = Number(topPEl.value).toFixed(1);
  maxTokensValueEl.textContent = maxTokensEl.value;
  memoryWindowValueEl.textContent = memoryWindowEl.value;
  currentModelText.textContent = `目前模型：${modelEl.value}`;
  streamingBadge.textContent = streamingEl.checked ? "Streaming On" : "Streaming Off";
  memoryBadge.textContent = shortMemoryEl.checked ? `Memory ${memoryWindowEl.value}` : "Memory Off";
  memoryWindowEl.disabled = !shortMemoryEl.checked;
  sendButton.disabled = isLoading;
}

function getPayloadMessages(userText) {
  const memoryOn = shortMemoryEl.checked;
  const memoryWindow = Number(memoryWindowEl.value);

  const context = memoryOn
    ? messages.slice(-memoryWindow).map((m) => ({ role: m.role, content: m.content }))
    : [];

  return [...context, { role: "user", content: userText }];
}

async function sendMessage() {
  const text = chatInput.value.trim();
  if (!text || isLoading) return;

  setError("");
  isLoading = true;
  updateControls();

  messages.push({ role: "user", content: text });
  messages.push({ role: "assistant", content: "" });
  renderMessages();

  const assistantIndex = messages.length - 1;
  const useStreaming = streamingEl.checked;

  const payload = {
    model: modelEl.value,
    systemPrompt: systemPromptEl.value,
    temperature: Number(temperatureEl.value),
    top_p: Number(topPEl.value),
    max_tokens: Number(maxTokensEl.value),
    messages: getPayloadMessages(text)
  };

  chatInput.value = "";

  try {
    if (useStreaming) {
      abortController = new AbortController();

      const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload),
        signal: abortController.signal
      });

      if (!response.ok || !response.body) {
        const errText = await response.text();
        throw new Error(errText || "Streaming request failed.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      let assistantText = "";
      let done = false;

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        if (readerDone) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;

          const event = JSON.parse(line);

          if (event.type === "delta") {
            assistantText += event.content;
            messages[assistantIndex].content = assistantText;
            renderMessages();
          } else if (event.type === "done") {
            done = true;
            break;
          } else if (event.type === "error") {
            throw new Error(event.content || "Streaming error.");
          }
        }
      }
    } else {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(errText || "Chat request failed.");
      }

      const data = await response.json();
      messages[assistantIndex].content = data.output || "";
      renderMessages();
    }
  } catch (error) {
    if (error.name === "AbortError") {
      setError("已停止生成。");
    } else {
      messages[assistantIndex].content = `Error: ${error.message || "Unknown error"}`;
      renderMessages();
      setError(error.message || "Unknown error");
    }
  } finally {
    isLoading = false;
    abortController = null;
    updateControls();
  }
}

function clearChat() {
  if (abortController) {
    abortController.abort();
  }

  messages = [
    {
      role: "assistant",
      content: "對話已清空。你可以重新開始。"
    }
  ];
  setError("");
  isLoading = false;
  renderMessages();
  updateControls();
}

function stopStreaming() {
  if (abortController) {
    abortController.abort();
  }
}

temperatureEl.addEventListener("input", updateControls);
topPEl.addEventListener("input", updateControls);
maxTokensEl.addEventListener("input", updateControls);
memoryWindowEl.addEventListener("input", updateControls);
streamingEl.addEventListener("change", updateControls);
shortMemoryEl.addEventListener("change", updateControls);
modelEl.addEventListener("change", updateControls);

sendButton.addEventListener("click", sendMessage);
clearChatButton.addEventListener("click", clearChat);
stopStreamingButton.addEventListener("click", stopStreaming);

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

presetDefault.addEventListener("click", () => {
  systemPromptEl.value = PROMPT_PRESETS.default;
});

presetTeacher.addEventListener("click", () => {
  systemPromptEl.value = PROMPT_PRESETS.teacher;
});

presetCoder.addEventListener("click", () => {
  systemPromptEl.value = PROMPT_PRESETS.coder;
});

presetTranslator.addEventListener("click", () => {
  systemPromptEl.value = PROMPT_PRESETS.translator;
});

renderMessages();
updateControls();