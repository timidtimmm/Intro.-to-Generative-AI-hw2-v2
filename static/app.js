// ═══════════════════════════════════════════════════════════════════════════════
//  HW02 ChatGPT — app.js  (with RAG support)
// ═══════════════════════════════════════════════════════════════════════════════

// ── Constants ─────────────────────────────────────────────────────────────────
const VISION_MODELS = new Set([
  "gemini/gemini-2.5-flash",
  "gemini/gemini-2.5-pro",
  "meta/llama-3.2-11b-vision-instruct",
  "microsoft/phi-3.5-vision-instruct",
]);

const DEFAULT_PRESETS = {
  default:    "You are a helpful assistant. Answer clearly and accurately.",
  teacher:    "You are a patient teacher. Explain step by step with simple examples.",
  coder:      "You are a senior software engineer. Write robust and maintainable code.",
  translator: "You are a professional translator. Translate accurately and preserve tone.",
};

const ACCENT_PRESETS = [
  { name:"綠",   value:"#10a37f" },
  { name:"藍",   value:"#3b82f6" },
  { name:"紫",   value:"#8b5cf6" },
  { name:"橘",   value:"#f97316" },
  { name:"玫瑰", value:"#f43f5e" },
  { name:"青",   value:"#06b6d4" },
];

// ── DOM refs ──────────────────────────────────────────────────────────────────
const messagesEl           = document.getElementById("messages");
const errorBox             = document.getElementById("errorBox");
const sendButton           = document.getElementById("sendButton");
const chatInput            = document.getElementById("chatInput");
const modelEl              = document.getElementById("model");
const currentModelText     = document.getElementById("currentModelText");
const chatTitleEl          = document.getElementById("chatTitle");
const systemPromptEl       = document.getElementById("systemPrompt");
const temperatureEl        = document.getElementById("temperature");
const topPEl               = document.getElementById("topP");
const maxTokensEl          = document.getElementById("maxTokens");
const streamingEl          = document.getElementById("streaming");
const shortMemoryEl        = document.getElementById("shortMemory");
const memoryWindowEl       = document.getElementById("memoryWindow");
const thinkingModeEl       = document.getElementById("thinkingMode");
const thinkingBudgetEl     = document.getElementById("thinkingBudget");
const thinkingBudgetGroup  = document.getElementById("thinkingBudgetGroup");
const fontSizeEl           = document.getElementById("fontSize");
const temperatureValueEl    = document.getElementById("temperatureValue");
const topPValueEl           = document.getElementById("topPValue");
const maxTokensValueEl      = document.getElementById("maxTokensValue");
const memoryWindowValueEl   = document.getElementById("memoryWindowValue");
const thinkingBudgetValueEl = document.getElementById("thinkingBudgetValue");
const fontSizeValueEl       = document.getElementById("fontSizeValue");
const streamingBadge        = document.getElementById("streamingBadge");
const memoryBadge           = document.getElementById("memoryBadge");
const thinkingBadge         = document.getElementById("thinkingBadge");
const tokenBadge            = document.getElementById("tokenBadge");
const ragBadge              = document.getElementById("ragBadge");
const clearChatButton       = document.getElementById("clearChat");
const stopStreamingButton   = document.getElementById("stopStreaming");
const newChatBtn            = document.getElementById("newChatBtn");
const sessionListEl         = document.getElementById("sessionList");
const exportMdBtn           = document.getElementById("exportMd");
const exportJsonBtn         = document.getElementById("exportJson");
const themeToggleBtn        = document.getElementById("themeToggle");
const fileInput             = document.getElementById("fileInput");
const attachmentPreview     = document.getElementById("attachmentPreview");
const visionHint            = document.getElementById("visionHint");
const voiceBtn              = document.getElementById("voiceBtn");
const promptTemplatesList   = document.getElementById("promptTemplatesList");
const newTemplateNameEl     = document.getElementById("newTemplateName");
const newTemplateTextEl     = document.getElementById("newTemplateText");
const saveTemplateBtn       = document.getElementById("saveTemplateBtn");
const accentSwatchesEl      = document.getElementById("accentSwatches");
const accentCustomEl        = document.getElementById("accentCustom");
const tokenChartCanvas      = document.getElementById("tokenChart");
const chartToggleBtn        = document.getElementById("chartToggle");
const chartPanel            = document.getElementById("chartPanel");
const longMemoryEl          = document.getElementById("longMemory");
const autoRouteEl           = document.getElementById("autoRoute");
const toolsEnabledEl        = document.getElementById("toolsEnabled");
const toolMaxIterEl         = document.getElementById("toolMaxIter");
const toolMaxIterValueEl    = document.getElementById("toolMaxIterValue");
const routeBadge            = document.getElementById("routeBadge");
const longMemoryBadge       = document.getElementById("longMemoryBadge");
const refreshMemBtn         = document.getElementById("refreshMemBtn");
const memoryListEl          = document.getElementById("memoryList");
const toolsListEl           = document.getElementById("toolsList");
const agentModeEl           = document.getElementById("agentMode");
const ttsAutoEl             = document.getElementById("ttsAuto");
const shareChatBtn          = document.getElementById("shareChat");
const shareResultEl         = document.getElementById("shareResult");
const uploadBtn             = document.getElementById('uploadButton') || document.querySelector('.upload-btn');

// ── RAG DOM refs (NEW) ────────────────────────────────────────────────────────
const ragUploadBtn  = document.getElementById('ragUploadBtn');
const ragFileInput  = document.getElementById('ragFileInput');
const ragDocListEl  = document.getElementById('ragDocList');
const ragEnabledEl  = document.getElementById('ragEnabled');
const ragStatusEl   = document.getElementById('ragStatus');

// ── Upload always enabled ─────────────────────────────────────────────────────
function forceEnableUpload() {
  if (fileInput) fileInput.disabled = false;
  if (uploadBtn) {
    uploadBtn.classList.remove('disabled');
    uploadBtn.style.opacity = '1';
    uploadBtn.style.cursor = 'pointer';
    uploadBtn.title = '上傳圖片、PDF 或文字檔（需要時會自動路由到 Vision 模型）';
  }
}
forceEnableUpload();

// ── State ─────────────────────────────────────────────────────────────────────
let messages          = [];
let isLoading         = false;
let abortController   = null;
let currentSessionId  = null;
let sessions          = [];
let pendingAttachment = null;
let isDarkTheme       = true;
let totalPromptTokens     = 0;
let totalCompletionTokens = 0;
let tokenHistory = [];
let chartInstance = null;

// ── localStorage ──────────────────────────────────────────────────────────────
const lsGet = (k,fb) => { try { const v=localStorage.getItem(k); return v!==null?JSON.parse(v):fb; } catch { return fb; } };
const lsSet = (k,v) => { try { localStorage.setItem(k,JSON.stringify(v)); } catch {} };

// ── Accent color ──────────────────────────────────────────────────────────────
function applyAccent(hex) {
  const darken = (h, amt) => {
    let [r,g,b] = [1,3,5].map(i=>parseInt(h.slice(i,i+2),16));
    [r,g,b] = [r,g,b].map(c=>Math.max(0,c-amt));
    return '#'+[r,g,b].map(c=>c.toString(16).padStart(2,'0')).join('');
  };
  document.documentElement.style.setProperty('--accent', hex);
  document.documentElement.style.setProperty('--accent-hover', darken(hex, 25));
  document.documentElement.style.setProperty('--bubble-user', hex);
  lsSet('accent', hex);
  document.querySelectorAll('.accent-swatch').forEach(s => {
    s.classList.toggle('selected', s.dataset.color === hex);
  });
  if (accentCustomEl) accentCustomEl.value = hex;
}

function buildAccentPicker() {
  if (!accentSwatchesEl) return;
  accentSwatchesEl.innerHTML = '';
  ACCENT_PRESETS.forEach(({name, value}) => {
    const sw = document.createElement('button');
    sw.className = 'accent-swatch';
    sw.dataset.color = value;
    sw.title = name;
    sw.style.background = value;
    sw.onclick = () => applyAccent(value);
    accentSwatchesEl.appendChild(sw);
  });
}

// ── Token chart ───────────────────────────────────────────────────────────────
function addTokenRecord(label, prompt, completion) {
  tokenHistory.push({ label, prompt, completion });
  if (tokenHistory.length > 20) tokenHistory.shift();
  renderChart();
}

function renderChart() {
  if (!tokenChartCanvas || !window.Chart) return;
  const labels      = tokenHistory.map((r,i) => r.label || `#${i+1}`);
  const prompts     = tokenHistory.map(r => r.prompt);
  const completions = tokenHistory.map(r => r.completion);
  const accent  = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#10a37f';
  const isDark  = document.documentElement.getAttribute('data-theme') !== 'light';
  const gridColor = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';
  const textColor = isDark ? '#a0a0a0' : '#666';
  if (chartInstance) {
    chartInstance.data.labels = labels;
    chartInstance.data.datasets[0].data = prompts;
    chartInstance.data.datasets[1].data = completions;
    chartInstance.options.scales.x.ticks.color = textColor;
    chartInstance.options.scales.y.ticks.color = textColor;
    chartInstance.options.scales.x.grid.color  = gridColor;
    chartInstance.options.scales.y.grid.color  = gridColor;
    chartInstance.update('none');
    return;
  }
  chartInstance = new Chart(tokenChartCanvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: '↑ Prompt', data: prompts, backgroundColor: accent + 'aa', borderColor: accent, borderWidth: 1, borderRadius: 4 },
        { label: '↓ Completion', data: completions, backgroundColor: '#6366f1aa', borderColor: '#6366f1', borderWidth: 1, borderRadius: 4 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 300 },
      plugins: {
        legend: { labels: { color: textColor, font: { size: 11 } } },
        tooltip: { callbacks: { title: (items) => `回答 ${items[0].label}`, label: (item) => `${item.dataset.label}: ${item.raw} tok` } },
      },
      scales: {
        x: { stacked: false, ticks: { color: textColor, font: { size: 10 }, maxRotation: 0 }, grid: { color: gridColor } },
        y: { ticks: { color: textColor, font: { size: 10 } }, grid: { color: gridColor } },
      },
    },
  });
}

// ── Theme ─────────────────────────────────────────────────────────────────────
function applyTheme(dark) {
  isDarkTheme = dark;
  document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
  themeToggleBtn.textContent = dark ? '☀️ 淺色' : '🌙 深色';
  lsSet('theme', dark ? 'dark' : 'light');
  if (chartInstance) renderChart();
}
themeToggleBtn.addEventListener('click', () => applyTheme(!isDarkTheme));

// ── Font size ─────────────────────────────────────────────────────────────────
fontSizeEl.addEventListener('input', () => {
  const s = fontSizeEl.value;
  fontSizeValueEl.textContent = s + 'px';
  messagesEl.style.fontSize = s + 'px';
  lsSet('fontSize', s);
});

// ── Custom templates ──────────────────────────────────────────────────────────
let customTemplates = lsGet('customTemplates', {});
const saveCustomTemplates = () => lsSet('customTemplates', customTemplates);

function renderPromptTemplates() {
  promptTemplatesList.innerHTML = '';
  Object.entries({ ...DEFAULT_PRESETS, ...customTemplates }).forEach(([key, text]) => {
    const isCustom = key in customTemplates;
    const item = document.createElement('div'); item.className = 'template-item';
    const btn = document.createElement('button'); btn.className = 'template-use-btn';
    btn.textContent = key; btn.title = text.slice(0,80)+(text.length>80?'…':'');
    btn.onclick = () => { systemPromptEl.value = text; };
    item.appendChild(btn);
    if (isCustom) {
      const del = document.createElement('button'); del.className = 'template-del-btn'; del.textContent = '✕';
      del.onclick = () => { delete customTemplates[key]; saveCustomTemplates(); renderPromptTemplates(); };
      item.appendChild(del);
    }
    promptTemplatesList.appendChild(item);
  });
}

saveTemplateBtn.addEventListener('click', () => {
  const name = newTemplateNameEl.value.trim(), text = newTemplateTextEl.value.trim();
  if (!name || !text) return;
  if (name in DEFAULT_PRESETS) { alert('不能覆蓋內建模板名稱'); return; }
  customTemplates[name] = text; saveCustomTemplates(); renderPromptTemplates();
  newTemplateNameEl.value = ''; newTemplateTextEl.value = '';
});

// ── Markdown ──────────────────────────────────────────────────────────────────
const escapeHtml = str => String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

function inlineMarkdown(raw) {
  // ① Extract download links BEFORE escapeHtml so brackets aren't destroyed
  const links = [];
  raw = raw.replace(
    /\[([^\]]+)\]\((\/api\/(?:download|rag)[^\)]+)\)/g,
    (_, label, href) => {
      const idx = links.length;
      links.push(`<a href="${href}" download="${href.split('/').pop()}">${label}</a>`);
      return `\x00LINK${idx}\x00`;
    }
  );

  let result = escapeHtml(raw)
    .replace(/!\[([^\]]*)\]\((data:image\/[a-zA-Z0-9.+-]+;base64,[^)]+|https?:\/\/[^)]+)\)/g, '<img class="generated-image" alt="$1" src="$2">')
    .replace(/`([^`]+)`/g,'<code class="inline-code">$1</code>')
    .replace(/\*\*\*(.+?)\*\*\*/g,'<strong><em>$1</em></strong>')
    .replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>')
    .replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g,'<em>$1</em>')
    .replace(/~~(.+?)~~/g,'<del>$1</del>')
    .replace(/\[([^\]]+)\]\(((?:https?:\/\/|\/)[^\)]+)\)/g,'<a href="$2" download>$1</a>')
    .replace(/(^|[\s(])(\/api\/download\/[^\s)]+)/g,'$1<a href="$2" download>$2</a>');

  // ③ Restore extracted links
  result = result.replace(/\x00LINK(\d+)\x00/g, (_, i) => links[+i]);
  return result;
}

function renderMarkdown(raw) {
  if (!raw) return '';
  const lines = raw.split('\n'); let html = '', i = 0;
  const isRow = l => l.trim().startsWith('|') && l.trim().endsWith('|');
  const isSep = l => /^\|[\s|:\-]+\|$/.test(l.trim());
  while (i < lines.length) {
    const line = lines[i];
    const fm = line.match(/^```(\w*)/);
    if (fm) {
      const lang=fm[1]||'plaintext'; let code=''; i++;
      while(i<lines.length&&!lines[i].startsWith('```')){code+=lines[i]+'\n';i++;} i++;
      const uid='cb-'+Math.random().toString(36).slice(2,8);
      html+=`<pre><div class="code-block-header"><span class="code-lang">${escapeHtml(lang)}</span><button class="copy-btn" data-target="${uid}" onclick="copyCode(this)">複製</button></div><code id="${uid}" class="language-${escapeHtml(lang)}">${escapeHtml(code.trimEnd())}</code></pre>`;
      continue;
    }
    if (isRow(line)&&i+1<lines.length&&isSep(lines[i+1])) {
      const hds=line.trim().slice(1,-1).split('|').map(c=>c.trim()); i+=2;
      const al=lines[i-1].trim().slice(1,-1).split('|').map(c=>c.trim()).map(c=>c.startsWith(':')&&c.endsWith(':')? 'center':c.endsWith(':')? 'right':'left');
      let t='<div class="table-wrapper"><table><thead><tr>';
      hds.forEach((h,j)=>{t+=`<th style="text-align:${al[j]}">${inlineMarkdown(h)}</th>`;});
      t+='</tr></thead><tbody>';
      while(i<lines.length&&isRow(lines[i])){const cs=lines[i].trim().slice(1,-1).split('|').map(c=>c.trim());t+='<tr>';hds.forEach((_,j)=>{t+=`<td style="text-align:${al[j]}">${inlineMarkdown(cs[j]??'')}</td>`;});t+='</tr>';i++;}
      html+=t+'</tbody></table></div>'; continue;
    }
    const hm=line.match(/^(#{1,6})\s+(.*)/);
    if(hm){html+=`<h${hm[1].length} class="md-h${hm[1].length}">${inlineMarkdown(hm[2])}</h${hm[1].length}>`;i++;continue;}
    if(/^(\*{3,}|-{3,}|_{3,})\s*$/.test(line)){html+='<hr>';i++;continue;}
    if(line.startsWith('> ')){let bq='';while(i<lines.length&&lines[i].startsWith('> ')){bq+=lines[i].slice(2)+'\n';i++;}html+=`<blockquote>${renderMarkdown(bq.trim())}</blockquote>`;continue;}
    if(/^(\s*)[-*+] /.test(line)){html+='<ul>';while(i<lines.length&&/^(\s*)[-*+] /.test(lines[i])){html+=`<li>${inlineMarkdown(lines[i].replace(/^\s*[-*+] /,''))}</li>`;i++;}html+='</ul>';continue;}
    if(/^\d+\. /.test(line)){html+='<ol>';while(i<lines.length&&/^\d+\. /.test(lines[i])){html+=`<li>${inlineMarkdown(lines[i].replace(/^\d+\. /,''))}</li>`;i++;}html+='</ol>';continue;}
    if(line.trim()===''){html+='<br>';i++;continue;}
    html+=`<p>${inlineMarkdown(line)}</p>`;i++;
  }
  return html;
}

function copyCode(btn) {
  const el=document.getElementById(btn.getAttribute('data-target')); if(!el) return;
  navigator.clipboard.writeText(el.textContent).then(()=>{
    btn.textContent='已複製 ✓'; btn.classList.add('copied');
    setTimeout(()=>{btn.textContent='複製';btn.classList.remove('copied');},2000);
  });
}
window.copyCode = copyCode;

const parseThinking = text => {
  const m=text.match(/^<think>([\s\S]*?)<\/think>([\s\S]*)$/);
  return m?{thinkContent:m[1].trim(),answerContent:m[2].trim()}:{thinkContent:null,answerContent:text};
};

const formatTs  = iso => iso ? new Date(iso).toLocaleTimeString('zh-TW',{hour:'2-digit',minute:'2-digit'}) : '';
const formatDur = ms  => ms < 1000 ? `${ms}ms` : `${(ms/1000).toFixed(1)}s`;

// ── TTS ───────────────────────────────────────────────────────────────────────
function cleanForSpeech(text) {
  return String(text || '')
    .replace(/```[\s\S]*?```/g, '程式碼區塊略過。')
    .replace(/!\[[^\]]*\]\(data:image[^)]+\)/g, '圖片已生成。')
    .replace(/<think>[\s\S]*?<\/think>/g, '')
    .replace(/[#*_`>\[\]()]/g, '')
    .slice(0, 900);
}
function speakAssistant(text) {
  if (!ttsAutoEl?.checked || !('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(cleanForSpeech(text));
  utter.lang = 'zh-TW'; utter.rate = 1.0;
  window.speechSynthesis.speak(utter);
}

// ── Render messages ───────────────────────────────────────────────────────────
function renderMessages() {
  messagesEl.innerHTML = '';
  messages.forEach((msg, idx) => {
    const row = document.createElement('div');
    row.className = `message-row ${msg.role==='user'?'user-row':'assistant-row'}`;
    const bubble = document.createElement('div');
    bubble.className = `message-bubble ${msg.role}`;

    const hdr = document.createElement('div'); hdr.className = 'message-header';
    const roleEl = document.createElement('span'); roleEl.className = 'message-role';
    roleEl.textContent = msg.role==='user' ? 'You' : 'Assistant';
    const tsEl = document.createElement('span'); tsEl.className = 'message-ts';
    tsEl.textContent = formatTs(msg.ts);
    hdr.appendChild(roleEl); hdr.appendChild(tsEl); bubble.appendChild(hdr);

    if (msg.imagePreview) { const img=document.createElement('img'); img.src=msg.imagePreview; img.className='message-image'; bubble.appendChild(img); }
    else if (msg.filename) { const chip=document.createElement('div'); chip.style.cssText='font-size:12px;color:#9ef3df;margin-bottom:6px;'; chip.textContent='📎 '+msg.filename; bubble.appendChild(chip); }

    const contentEl = document.createElement('div'); contentEl.className = 'message-content';

    if (msg.role === 'assistant') {
      const activeContent = msg.branches ? (msg.branches[msg.branchIdx||0] || '') : (msg.content || '');
      const { thinkContent, answerContent } = parseThinking(activeContent);
      if (thinkContent) {
        const tb=document.createElement('div'); tb.className='thinking-block';
        tb.innerHTML=`<div class="thinking-header" onclick="this.parentElement.classList.toggle('open')"><span>🧠 思考過程</span><span class="thinking-toggle">▼</span></div><div class="thinking-content">${escapeHtml(thinkContent)}</div>`;
        contentEl.appendChild(tb);
      }
      if (msg.isStreaming && !answerContent) {
        const dots=document.createElement('div'); dots.className='typing-dots'; dots.innerHTML='<span></span><span></span><span></span>'; contentEl.appendChild(dots);
      } else {
        const ans=document.createElement('div'); ans.innerHTML=renderMarkdown(answerContent); contentEl.appendChild(ans);
        contentEl.querySelectorAll('pre code').forEach(b=>{if(window.hljs)hljs.highlightElement(b);});
      }

      // RAG sources badge
      if (msg.ragChunks && msg.ragChunks.length) {
        const ragBox = document.createElement('div');
        ragBox.className = 'rag-sources';
        ragBox.innerHTML = `<div class="rag-sources-title">📄 RAG 來源（${msg.ragChunks.length} 片段）</div>` +
          msg.ragChunks.map(c =>
            `<div class="rag-source-item"><span class="rag-source-score">${(c.score*100).toFixed(0)}%</span><span class="rag-source-file">${escapeHtml(c.filename)}</span><span class="rag-source-preview">${escapeHtml((c.content||'').slice(0,80))}…</span></div>`
          ).join('');
        contentEl.appendChild(ragBox);
      }

      const metaEl = document.createElement('div'); metaEl.className = 'msg-meta';
      if (msg.usage) metaEl.textContent = `↑ ${msg.usage.prompt_tokens} · ↓ ${msg.usage.completion_tokens} tokens`;
      if (msg.duration) metaEl.textContent += (msg.usage ? '  ·  ' : '') + `⏱ ${formatDur(msg.duration)}`;
      if (msg.route) metaEl.textContent += (metaEl.textContent ? '  ·  ' : '') + `🔀 ${msg.route.model || ''}${msg.route.reason ? ' — ' + msg.route.reason : ''}`;
      if (metaEl.textContent) contentEl.appendChild(metaEl);

      if (msg.agentSteps && msg.agentSteps.length) {
        const agentBox = document.createElement('div'); agentBox.className = 'agent-steps';
        agentBox.innerHTML = `<div class="agent-title">🧩 Agent Plan</div>` + msg.agentSteps.map((st, i) =>
          `<div class="agent-step"><span class="agent-dot">${i+1}</span><div><b>${escapeHtml(st.title||`Step ${i+1}`)}</b><p>${escapeHtml(st.detail||'')}</p></div></div>`
        ).join('');
        contentEl.appendChild(agentBox);
      }
      if (msg.toolIterProgress) {
        const badge = document.createElement('div'); badge.className = 'tool-iter-badge';
        badge.textContent = `⚙️ ${msg.toolIterProgress}`; contentEl.appendChild(badge);
      }
      if (msg.toolEvents && msg.toolEvents.length) {
        const toolBox = document.createElement('div'); toolBox.className = 'tool-events';
        toolBox.innerHTML = msg.toolEvents.map(ev => {
          if (ev.type === 'tool_start') return `<div>🔧 呼叫工具：<code>${escapeHtml(ev.name||'')}</code></div>`;
          if (ev.type === 'tool_end')   return `<div>✅ 工具完成：<code>${escapeHtml(ev.name||'')}</code> <span>${ev.latency_ms||0}ms</span></div>`;
          return `<div>${escapeHtml(JSON.stringify(ev))}</div>`;
        }).join('');
        contentEl.appendChild(toolBox);
      }

      if (!msg.isStreaming) {
        const bar=document.createElement('div'); bar.className='msg-actions';
        const regen=document.createElement('button'); regen.className='msg-action-btn';
        regen.textContent='🔄 重新生成'; regen.onclick=()=>regenerateFrom(idx);
        bar.appendChild(regen); contentEl.appendChild(bar);
      }
    } else {
      const activeContent = msg.branches ? (msg.branches[msg.branchIdx||0]||'') : msg.content;
      if (msg.branches && msg.branches.length > 1) {
        const nav=document.createElement('div'); nav.className='branch-nav';
        const prev=document.createElement('button'); prev.className='branch-btn'; prev.textContent='‹';
        prev.disabled=(msg.branchIdx||0)===0; prev.onclick=()=>switchBranch(idx,(msg.branchIdx||0)-1);
        const counter=document.createElement('span'); counter.className='branch-counter';
        counter.textContent=`${(msg.branchIdx||0)+1} / ${msg.branches.length}`;
        const next=document.createElement('button'); next.className='branch-btn'; next.textContent='›';
        next.disabled=(msg.branchIdx||0)===msg.branches.length-1; next.onclick=()=>switchBranch(idx,(msg.branchIdx||0)+1);
        nav.appendChild(prev); nav.appendChild(counter); nav.appendChild(next); contentEl.appendChild(nav);
      }
      const txt=document.createElement('div'); txt.style.whiteSpace='pre-wrap'; txt.textContent=activeContent; contentEl.appendChild(txt);
      const bar=document.createElement('div'); bar.className='msg-actions user-actions';
      const edit=document.createElement('button'); edit.className='msg-action-btn';
      edit.textContent='✏️ 編輯'; edit.onclick=()=>startEditMessage(idx);
      bar.appendChild(edit); contentEl.appendChild(bar);
    }

    bubble.appendChild(contentEl); row.appendChild(bubble); messagesEl.appendChild(row);
  });
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ── Branching ─────────────────────────────────────────────────────────────────
function startEditMessage(idx) {
  if (isLoading) return;
  const msg = messages[idx];
  chatInput.value = msg.branches ? (msg.branches[msg.branchIdx||0]||'') : msg.content;
  chatInput.dataset.editIdx = idx; chatInput.focus();
}

function switchBranch(userIdx, newBranchIdx) {
  const uMsg = messages[userIdx];
  if (!uMsg?.branches || newBranchIdx < 0 || newBranchIdx >= uMsg.branches.length) return;
  uMsg.branchIdx = newBranchIdx;
  const aMsg = messages[userIdx + 1];
  if (aMsg?.role === 'assistant' && aMsg.branches && newBranchIdx < aMsg.branches.length) {
    aMsg.branchIdx = newBranchIdx; aMsg.content = aMsg.branches[newBranchIdx];
  }
  renderMessages();
}

async function regenerateFrom(assistantIdx) {
  if (isLoading) return;
  const userIdx = assistantIdx - 1;
  if (userIdx < 0 || messages[userIdx]?.role !== 'user') return;
  const userText = messages[userIdx].branches ? (messages[userIdx].branches[messages[userIdx].branchIdx||0]||'') : messages[userIdx].content;
  await doSend(userText, userIdx, true);
}

// ── Error ─────────────────────────────────────────────────────────────────────
function setError(text='') { errorBox.textContent=text; errorBox.classList.toggle('hidden',!text); }

// ── Token badge ───────────────────────────────────────────────────────────────
function updateTokenBadge(usage, label) {
  if (!usage) return;
  totalPromptTokens     += usage.prompt_tokens || 0;
  totalCompletionTokens += usage.completion_tokens || 0;
  tokenBadge.textContent = `↑${totalPromptTokens} ↓${totalCompletionTokens} tok`;
  tokenBadge.classList.remove('hidden');
  addTokenRecord(label || `#${tokenHistory.length+1}`, usage.prompt_tokens||0, usage.completion_tokens||0);
}

// ── Controls ──────────────────────────────────────────────────────────────────
function updateControls() {
  temperatureValueEl.textContent    = Number(temperatureEl.value).toFixed(1);
  topPValueEl.textContent           = Number(topPEl.value).toFixed(1);
  maxTokensValueEl.textContent      = maxTokensEl.value;
  memoryWindowValueEl.textContent   = memoryWindowEl.value;
  thinkingBudgetValueEl.textContent = thinkingBudgetEl.value;
  fontSizeValueEl.textContent       = fontSizeEl.value+'px';
  if (toolMaxIterValueEl && toolMaxIterEl) {
    toolMaxIterValueEl.textContent = toolMaxIterEl.value;
    const iterGroup = document.getElementById('toolIterGroup');
    if (iterGroup) iterGroup.style.display = toolsEnabledEl?.checked ? 'flex' : 'none';
  }
  currentModelText.textContent = `目前模型：${modelEl.value}`;
  streamingBadge.textContent   = streamingEl.checked ? 'Streaming On' : 'Streaming Off';
  memoryBadge.textContent      = shortMemoryEl.checked ? `Memory ${memoryWindowEl.value}` : 'Memory Off';
  if (longMemoryBadge) {
    longMemoryBadge.textContent = longMemoryEl?.checked ? 'Long Memory On' : 'Long Memory Off';
    longMemoryBadge.classList.toggle('muted', !(longMemoryEl?.checked));
  }
  if (routeBadge) {
    routeBadge.textContent = autoRouteEl?.checked ? '🔀 Auto Route' : '🔀 Manual Model';
    routeBadge.classList.toggle('muted', !(autoRouteEl?.checked));
  }
  // RAG badge
  if (ragBadge) {
    ragBadge.textContent = '📄 RAG On';
    ragBadge.classList.toggle('hidden', !(ragEnabledEl?.checked));
  }
  memoryWindowEl.disabled = !shortMemoryEl.checked;
  thinkingBudgetGroup.style.display = thinkingModeEl.checked ? 'block' : 'none';
  thinkingBadge.classList.toggle('hidden', !thinkingModeEl.checked);
  const isVision   = VISION_MODELS.has(modelEl.value);
  const autoRouteOn = !!(autoRouteEl?.checked);
  if (visionHint) {
    if (isVision) { visionHint.textContent='👁 此模型支援圖片 / PDF 上傳'; visionHint.classList.remove('hidden'); }
    else if (autoRouteOn) { visionHint.textContent='🔀 已啟用自動路由：可直接上傳圖片 / PDF'; visionHint.classList.remove('hidden'); }
    else { visionHint.textContent='📎 可上傳檔案；若上傳圖片 / PDF，系統會自動改用 Vision 模型'; visionHint.classList.remove('hidden'); }
  }
  const ul = uploadBtn || document.querySelector('.upload-btn');
  if (ul) { ul.style.opacity='1'; ul.style.cursor='pointer'; if(fileInput) fileInput.disabled=false; }
  forceEnableUpload();
  sendButton.disabled = isLoading;
}

// ── Memory / Tools panels ─────────────────────────────────────────────────────
async function loadMemories() {
  if (!memoryListEl) return;
  try {
    const r = await fetch('/api/memories'); const data = await r.json();
    const memories = data.memories || [];
    memoryListEl.innerHTML = '';
    if (!memories.length) { memoryListEl.innerHTML='<div class="empty-small">尚無長期記憶</div>'; return; }
    memories.forEach(m => {
      const item=document.createElement('div'); item.className='memory-item';
      const tag=document.createElement('span'); tag.className='memory-tag'; tag.textContent=m.type||'memory';
      const content=document.createElement('span'); content.className='memory-content';
      content.textContent=(m.content||'').length>70?(m.content||'').slice(0,70)+'…':(m.content||''); content.title=m.content||'';
      const del=document.createElement('button'); del.className='memory-delete'; del.textContent='✕';
      del.onclick=async()=>{ await fetch('/api/memories/'+m.id,{method:'DELETE'}); await loadMemories(); };
      item.appendChild(tag); item.appendChild(content); item.appendChild(del); memoryListEl.appendChild(item);
    });
  } catch(e) { console.error(e); }
}

async function loadTools() {
  if (!toolsListEl) return;
  try {
    const r = await fetch('/api/tools'); const data = await r.json();
    const tools = data.tools || [];
    toolsListEl.innerHTML = '';
    tools.forEach(t => {
      const item=document.createElement('div'); item.className='tool-item';
      item.innerHTML=`<span class="tool-name">${escapeHtml(t.name)}</span><span class="tool-risk ${escapeHtml(t.risk_level||'low')}">${escapeHtml(t.risk_level||'low')}</span>`;
      item.title=t.description||''; toolsListEl.appendChild(item);
    });
  } catch(e) { console.error(e); }
}

// ── RAG Document Manager (NEW) ────────────────────────────────────────────────
function showRagStatus(msg, isError = false) {
  if (!ragStatusEl) return;
  ragStatusEl.textContent = msg;
  ragStatusEl.className = 'rag-status' + (isError ? ' error' : '');
  ragStatusEl.classList.remove('hidden');
  setTimeout(() => ragStatusEl.classList.add('hidden'), 4000);
}

async function loadRagDocs() {
  console.log("loadRagDocs called, session:", currentSessionId);
  if (!ragDocListEl || !currentSessionId) return;
  try {
    const r = await fetch(`/api/rag/docs?session_id=${currentSessionId}`);
    const data = await r.json();
    const docs = data.docs || [];
    ragDocListEl.innerHTML = '';
    if (!docs.length) {
      ragDocListEl.innerHTML = '<div class="empty-small">尚無文件，上傳後可以直接問答。</div>';
      return;
    }
    docs.forEach(doc => {
      const item = document.createElement('div'); item.className = 'rag-doc-item';
      item.innerHTML = `
        <span class="rag-doc-icon">📄</span>
        <span class="rag-doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</span>
        <span class="rag-doc-meta">${doc.chunk_count} 片段</span>
        <button class="rag-doc-del" data-id="${doc.id}" title="刪除">✕</button>
      `;
      item.querySelector('.rag-doc-del').addEventListener('click', async (e) => {
        const id = e.currentTarget.dataset.id;
        await fetch(`/api/rag/docs/${id}`, { method: 'DELETE' });
        loadRagDocs();
      });
      ragDocListEl.appendChild(item);
    });
  } catch(e) { console.error(e); }
}

if (ragUploadBtn && ragFileInput) {
  ragUploadBtn.addEventListener('click', () => { ragFileInput.value = ''; ragFileInput.click(); });
  ragFileInput.addEventListener('change', async () => {
    const file = ragFileInput.files[0];
    if (!file || !currentSessionId) return;
    showRagStatus(`正在處理 ${file.name}…`);
    ragUploadBtn.disabled = true;
    try {
      const fd = new FormData(); fd.append('file', file);
      const r = await fetch(`/api/rag/ingest?session_id=${currentSessionId}`, { method: 'POST', body: fd });
      const data = await r.json();
      if (data.error) {
        showRagStatus('上傳失敗：' + data.error, true);
      } else if (data.status === 'already_ingested') {
        showRagStatus(`${file.name} 已存在（${data.chunk_count} 片段）`);
      } else {
        showRagStatus(`✅ ${file.name} 已建立索引（${data.chunk_count} 片段）`);
        loadRagDocs();
      }
    } catch(e) { showRagStatus('上傳錯誤：' + e.message, true); }
    finally { ragUploadBtn.disabled = false; }
  });
}

// ★ RAG URL 抓取
const ragUrlBtn   = document.getElementById('ragUrlBtn');
const ragUrlInput = document.getElementById('ragUrlInput');
 
if (ragUrlBtn && ragUrlInput) {
  ragUrlBtn.addEventListener('click', async () => {
    const url = ragUrlInput.value.trim();
    if (!url || !currentSessionId) return;
    if (!url.startsWith('http')) { showRagStatus('請輸入完整的 http/https 網址', true); return; }
    showRagStatus(`正在抓取 ${url}…`);
    ragUrlBtn.disabled = true;
    try {
      const r = await fetch(
        `/api/rag/ingest?session_id=${currentSessionId}&url=${encodeURIComponent(url)}`,
        { method: 'POST' }
      );
      const data = await r.json();
      if (data.error) {
        showRagStatus('抓取失敗：' + data.error, true);
      } else if (data.status === 'already_ingested') {
        showRagStatus(`已存在（${data.chunk_count} 片段）`);
      } else {
        showRagStatus(`✅ 已建立索引（${data.chunk_count} 片段）`);
        loadRagDocs();
        ragUrlInput.value = '';
      }
    } catch(e) { showRagStatus('錯誤：' + e.message, true); }
    finally { ragUrlBtn.disabled = false; }
  });
 
  // 按 Enter 也能觸發
  ragUrlInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); ragUrlBtn.click(); }
  });
}
if (ragEnabledEl) ragEnabledEl.addEventListener('change', updateControls);

// ── Sessions ──────────────────────────────────────────────────────────────────
async function loadSessions() {
  try { const r=await fetch('/api/sessions'); sessions=(await r.json()).sessions||[]; renderSessionList(); } catch(e){console.error(e);}
}

function renderSessionList() {
  sessionListEl.innerHTML='';
  if(!sessions.length){sessionListEl.innerHTML='<div style="font-size:12px;color:var(--subtext);padding:4px 0">尚無歷史對話</div>';return;}
  sessions.forEach(s=>{
    const item=document.createElement('div'); item.className='session-item'+(s.id===currentSessionId?' active':'');
    const title=document.createElement('span'); title.className='session-item-title'; title.textContent=s.title||'New Chat';
    const del=document.createElement('button'); del.className='session-delete'; del.textContent='✕';
    del.onclick=async e=>{e.stopPropagation();await fetch(`/api/sessions/${s.id}`,{method:'DELETE'});if(currentSessionId===s.id)await startNewChat();else await loadSessions();};
    item.appendChild(title); item.appendChild(del); item.onclick=()=>loadSession(s.id);
    sessionListEl.appendChild(item);
  });
}

async function startNewChat() {
  totalPromptTokens=0; totalCompletionTokens=0; tokenHistory=[];
  tokenBadge.classList.add('hidden');
  if(chartInstance){chartInstance.destroy();chartInstance=null;}
  const r=await fetch('/api/sessions',{method:'POST'});
  currentSessionId=(await r.json()).session_id;
  messages=[{role:'assistant',content:'嗨！我是你的自製 ChatGPT。有什麼可以幫你的嗎？',ts:new Date().toISOString()}];
  chatTitleEl.textContent='Chat'; renderMessages(); await loadSessions(); await loadRagDocs();
}

async function loadSession(sid) {
  currentSessionId=sid; totalPromptTokens=0; totalCompletionTokens=0; tokenHistory=[];
  tokenBadge.classList.add('hidden');
  if(chartInstance){chartInstance.destroy();chartInstance=null;}
  const r=await fetch(`/api/sessions/${sid}/messages`); const d=await r.json();
  messages=d.messages.length?d.messages.map(m=>({...m,ts:m.created_at})):[{role:'assistant',content:'這是空的對話，開始輸入吧！',ts:new Date().toISOString()}];
  chatTitleEl.textContent=sessions.find(s=>s.id===sid)?.title||'Chat';
  renderMessages(); renderSessionList(); await loadRagDocs();
}

async function autoTitleSession(text) {
  if(!currentSessionId)return;
  if(sessions.find(s=>s.id===currentSessionId)?.title!=='New Chat')return;
  const title=text.slice(0,30)+(text.length>30?'…':'');
  await fetch(`/api/sessions/${currentSessionId}`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({title})});
  chatTitleEl.textContent=title; await loadSessions();
}

// ── Export ────────────────────────────────────────────────────────────────────
exportMdBtn.addEventListener('click',()=>{
  const title=chatTitleEl.textContent||'chat'; let md=`# ${title}\n\n`;
  messages.forEach(m=>{const c=m.branches?(m.branches[m.branchIdx||0]||''):m.content;md+=`**${m.role==='user'?'You':'Assistant'}**${m.ts?` _(${formatTs(m.ts)})_`:''}\n\n${c}\n\n---\n\n`;});
  dl(md,`${title}.md`,'text/markdown');
});
exportJsonBtn.addEventListener('click',()=>{
  const title=chatTitleEl.textContent||'chat';
  dl(JSON.stringify({title,messages,exported_at:new Date().toISOString()},null,2),`${title}.json`,'application/json');
});
function dl(content,filename,mime){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([content],{type:mime}));a.download=filename;a.click();URL.revokeObjectURL(a.href);}

// ── File upload ───────────────────────────────────────────────────────────────
forceEnableUpload();
fileInput.addEventListener('change',async()=>{
  const file=fileInput.files[0];if(!file)return;
  const fd=new FormData();fd.append('file',file);
  try{const r=await fetch('/api/upload',{method:'POST',body:fd});if(!r.ok){setError('上傳失敗');return;}pendingAttachment=await r.json();showAttachmentPreview(pendingAttachment);}
  catch(e){setError('上傳錯誤：'+e.message);}
  fileInput.value='';
});
function showAttachmentPreview(data){
  attachmentPreview.innerHTML='';attachmentPreview.classList.remove('hidden');
  const chip=document.createElement('div');chip.className='attachment-chip';
  if(data.type==='image'){const img=document.createElement('img');img.src=`data:${data.mime};base64,${data.b64}`;chip.appendChild(img);}
  else{const ic=document.createElement('span');ic.textContent=data.type==='pdf'?'📄':'📝';chip.appendChild(ic);}
  const nm=document.createElement('span');nm.textContent=data.filename;
  const rm=document.createElement('button');rm.className='attachment-remove';rm.textContent='✕';rm.onclick=clearAttachment;
  chip.appendChild(nm);chip.appendChild(rm);attachmentPreview.appendChild(chip);
}
function clearAttachment(){pendingAttachment=null;attachmentPreview.innerHTML='';attachmentPreview.classList.add('hidden');}

// ── Voice input ───────────────────────────────────────────────────────────────
let recognition=null,isRecording=false;
function initVoice(){
  const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
  if(!SR){voiceBtn.title='瀏覽器不支援語音輸入';voiceBtn.style.opacity='0.35';voiceBtn.style.cursor='not-allowed';return;}
  recognition=new SR();recognition.lang='zh-TW';recognition.continuous=false;recognition.interimResults=true;
  recognition.onresult=e=>{chatInput.value=Array.from(e.results).map(r=>r[0].transcript).join('');};
  recognition.onend=()=>{isRecording=false;voiceBtn.textContent='🎤';voiceBtn.classList.remove('recording');};
  recognition.onerror=()=>{isRecording=false;voiceBtn.textContent='🎤';voiceBtn.classList.remove('recording');};
}
voiceBtn.addEventListener('click',()=>{
  if(!recognition)return;
  if(isRecording){recognition.stop();}else{recognition.start();isRecording=true;voiceBtn.textContent='⏹';voiceBtn.classList.add('recording');}
});

// ── Send ──────────────────────────────────────────────────────────────────────
function getPayloadMessages(userText){
  const win=Number(memoryWindowEl.value);
  const ctx=shortMemoryEl.checked?messages.slice(-win).map(m=>({role:m.role,content:m.branches?(m.branches[m.branchIdx||0]||''):m.content})):[];
  return[...ctx,{role:'user',content:userText}];
}

async function sendMessage(){
  const text=chatInput.value.trim();
  if((!text&&!pendingAttachment)||isLoading)return;
  const editIdx = chatInput.dataset.editIdx !== undefined ? parseInt(chatInput.dataset.editIdx) : null;
  delete chatInput.dataset.editIdx; chatInput.value='';
  await doSend(text, editIdx, false);
}

async function doSend(overrideText, editIdx=null, isRegen=false){
  let text = overrideText;
  if (text === null) {
    const lu = [...messages].reverse().find(m => m.role==='user');
    text = lu?.branches ? (lu.branches[lu.branchIdx||0]||'') : (lu?.content||'');
  }
  setError(''); isLoading=true; updateControls();

  let fullText=text, imageB64=null, imageMime=null, userImagePreview=null, userFilename=null;
  if (pendingAttachment) {
    if (pendingAttachment.type==='image') {
      imageB64=pendingAttachment.b64; imageMime=pendingAttachment.mime;
      userImagePreview=`data:${pendingAttachment.mime};base64,${pendingAttachment.b64}`; userFilename=pendingAttachment.filename;
      if (!fullText) fullText='請描述這張圖片。';
    } else if (pendingAttachment.type==='text'&&pendingAttachment.text) {
      userFilename=pendingAttachment.filename;
      fullText=(text?text+'\n\n':'')+`以下是上傳的檔案「${pendingAttachment.filename}」內容：\n\`\`\`\n${pendingAttachment.text}\n\`\`\``;
    } else if (pendingAttachment.type==='pdf') {
      userFilename=pendingAttachment.filename; imageB64=pendingAttachment.b64; imageMime=pendingAttachment.mime;
      if (!fullText) fullText=`請分析這份 PDF：${pendingAttachment.filename}`;
    }
  }

  let userMsgIdx, assistantMsgIdx;
  if (editIdx !== null && messages[editIdx]?.role === 'user') {
    const uMsg = messages[editIdx]; const aMsg = messages[editIdx + 1];
    if (!uMsg.branches) { uMsg.branches=[uMsg.content]; uMsg.branchIdx=0; }
    if (!isRegen) uMsg.branches.push(fullText);
    uMsg.branchIdx = uMsg.branches.length - 1;
    if (aMsg && aMsg.role === 'assistant') {
      if (!aMsg.branches) { aMsg.branches=[aMsg.content]; aMsg.branchIdx=0; }
      aMsg.branches.push(''); aMsg.branchIdx=aMsg.branches.length-1; aMsg.content=''; aMsg.isStreaming=true; aMsg.ts=new Date().toISOString();
      userMsgIdx=editIdx; assistantMsgIdx=editIdx+1;
    } else {
      messages.splice(editIdx+1,0,{role:'assistant',content:'',branches:[''],branchIdx:0,isStreaming:true,ts:new Date().toISOString()});
      userMsgIdx=editIdx; assistantMsgIdx=editIdx+1;
    }
    messages = messages.slice(0, assistantMsgIdx + 1);
  } else {
    messages.push({role:'user',content:fullText,imagePreview:userImagePreview,filename:userFilename,ts:new Date().toISOString()});
    messages.push({role:'assistant',content:'',isStreaming:true,ts:new Date().toISOString()});
    userMsgIdx=messages.length-2; assistantMsgIdx=messages.length-1;
  }
  renderMessages();

  const payload = {
    model: modelEl.value, systemPrompt: systemPromptEl.value,
    temperature: Number(temperatureEl.value), top_p: Number(topPEl.value), max_tokens: Number(maxTokensEl.value),
    messages: getPayloadMessages(fullText), thinking: thinkingModeEl.checked,
    thinking_budget: Number(thinkingBudgetEl.value), session_id: currentSessionId,
    image_b64: imageB64||undefined, image_mime: imageMime||undefined,
    attachment_type: pendingAttachment?.type || undefined,
    auto_route: autoRouteEl?.checked ?? true,
    use_memory: longMemoryEl?.checked ?? true,
    tools_enabled: toolsEnabledEl?.checked ?? true,
    tool_max_iterations: Number(toolMaxIterEl?.value ?? 20),
    agent_mode: agentModeEl?.checked ?? false,
    // ★ RAG: send flag so backend knows to inject context
    use_rag: ragEnabledEl?.checked ?? true,
    user_id: 'default',
  };

  clearAttachment();
  if (!isRegen && editIdx === null) autoTitleSession(text||fullText);

  const t0 = Date.now();
  const finaliseAssistant = (finalText, usage, duration, ragChunks) => {
    const aMsg = messages[assistantMsgIdx]; if (!aMsg) return;
    aMsg.content=finalText; aMsg.isStreaming=false; aMsg.duration=duration;
    if (usage) aMsg.usage=usage;
    if (ragChunks) aMsg.ragChunks=ragChunks;
    if (aMsg.branches) aMsg.branches[aMsg.branchIdx]=finalText;
  };

  try {
    if (streamingEl.checked) {
      abortController = new AbortController();
      const resp = await fetch('/api/chat/stream', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify(payload), signal:abortController.signal
      });
      if (!resp.ok||!resp.body) throw new Error((await resp.text())||'Streaming failed.');
      const reader=resp.body.getReader(); const dec=new TextDecoder('utf-8');
      let buf='', assTxt='', done=false;
      while (!done) {
        const {value,done:rd}=await reader.read(); if(rd) break;
        buf+=dec.decode(value,{stream:true}); const ls=buf.split('\n'); buf=ls.pop()||'';
        for (const l of ls) {
          if (!l.trim()) continue; const ev=JSON.parse(l);
          if (ev.type==='delta') {
            assTxt += ev.content;
            const aMsg=messages[assistantMsgIdx];
            if(aMsg){aMsg.content=assTxt;aMsg.isStreaming=false;if(aMsg.branches)aMsg.branches[aMsg.branchIdx]=assTxt;}
            renderMessages();
          } else if (ev.type==='rag') {
            // Store RAG chunks for display
            const aMsg=messages[assistantMsgIdx];
            if(aMsg) aMsg.ragChunks=ev.chunks;
            renderMessages();
          } else if (ev.type==='routing') {
            const aMsg=messages[assistantMsgIdx];
            if(aMsg) aMsg.route={model:ev.model,reason:ev.reason,task_type:ev.task_type,use_tools:ev.use_tools};
            if(routeBadge&&ev.model) {
                routeBadge.textContent=`🔀 ${ev.model}`;
                // ★ 閃爍動畫
                routeBadge.style.animation='none';
                routeBadge.offsetHeight; // trigger reflow
                routeBadge.style.animation='routeFlash 0.6s ease';
            }
            if(currentModelText&&ev.model) currentModelText.textContent=`目前模型：${ev.model}`;
            renderMessages();
          } else if (ev.type==='memory') {
            if(memoryListEl) loadMemories();
          } else if (ev.type==='agent_step') {
            const aMsg=messages[assistantMsgIdx];
            if(aMsg){aMsg.agentSteps=aMsg.agentSteps||[];aMsg.agentSteps[ev.index??aMsg.agentSteps.length]={title:ev.title,detail:ev.detail};}
            renderMessages();
          } else if (ev.type==='tool_start'||ev.type==='tool_end') {
            const aMsg=messages[assistantMsgIdx];
            if(aMsg){aMsg.toolEvents=aMsg.toolEvents||[];aMsg.toolEvents.push(ev);}
            renderMessages();
          } else if (ev.type==='tool_iteration') {
            const aMsg=messages[assistantMsgIdx];
            if(aMsg) aMsg.toolIterProgress=`第 ${ev.iteration} / ${ev.max} 次工具呼叫`;
            renderMessages();
          } else if (ev.type==='usage') {
            updateTokenBadge(ev.usage, fullText.slice(0,16));
          } else if (ev.type==='done') {
            const aMsg=messages[assistantMsgIdx];
            finaliseAssistant(assTxt, ev.usage, Date.now()-t0, ev.rag_chunks);
            if(aMsg&&ev.routing) aMsg.route=ev.routing;
            if(aMsg&&ev.agent_steps) aMsg.agentSteps=ev.agent_steps;
            if(ev.stored_memory_id) loadMemories();
            if(ev.usage) updateTokenBadge(ev.usage, fullText.slice(0,16));
            renderMessages(); speakAssistant(assTxt); done=true; break;
          } else if (ev.type==='error') throw new Error(ev.content||'Streaming error.');
        }
      }
    } else {
      const resp=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
      if (!resp.ok) throw new Error((await resp.text())||'Chat failed.');
      const d=await resp.json();
      finaliseAssistant(d.output||'', d.usage, Date.now()-t0, d.rag_chunks);
      const aMsg=messages[assistantMsgIdx];
      if(aMsg){
        if(d.routing){
    aMsg.route=d.routing;
    if(currentModelText&&d.routing.model) currentModelText.textContent=`目前模型：${d.routing.model}`;
    if(routeBadge&&d.routing.model){
        routeBadge.textContent=`🔀 ${d.routing.model}`;
        routeBadge.style.animation='none';
        routeBadge.offsetHeight;
        routeBadge.style.animation='routeFlash 0.6s ease';
    }
}
        if(d.tool_events) aMsg.toolEvents=d.tool_events;
        if(d.agent_steps) aMsg.agentSteps=d.agent_steps;
      }
      speakAssistant(d.output||'');
      if(d.stored_memory_id) loadMemories();
      if(d.usage) updateTokenBadge(d.usage, fullText.slice(0,16));
      renderMessages();
    }
  } catch(err) {
    const aMsg=messages[assistantMsgIdx];
    if(err.name==='AbortError'){
      setError('已停止生成。');
      if(aMsg){aMsg.isStreaming=false;if(aMsg.branches)aMsg.branches[aMsg.branchIdx]=aMsg.content;}
    } else {
      if(aMsg){aMsg.content=`Error: ${err.message}`;aMsg.isStreaming=false;if(aMsg.branches)aMsg.branches[aMsg.branchIdx]=aMsg.content;}
      setError(err.message);
    }
    renderMessages();
  } finally {
    isLoading=false; abortController=null; updateControls();
  }
}

// ── Clear / stop / share ──────────────────────────────────────────────────────
function clearChat(){
  if(abortController)abortController.abort();
  setError('');isLoading=false;clearAttachment();
  if(currentSessionId)fetch(`/api/sessions/${currentSessionId}`,{method:'DELETE'});
  startNewChat();
}
function stopStreaming(){if(abortController)abortController.abort();}

async function shareCurrentChat() {
  if (!currentSessionId) { setError('目前沒有可分享的對話。'); return; }
  try {
    const resp = await fetch(`/api/sessions/${currentSessionId}/share`, { method:'POST' });
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    if (shareResultEl) {
      shareResultEl.classList.remove('hidden');
      shareResultEl.innerHTML = `<a href="${data.url}" target="_blank" rel="noopener">${data.url}</a><button class="mini-copy" type="button">複製</button>`;
      const btn = shareResultEl.querySelector('button');
      if (btn) btn.onclick = () => navigator.clipboard.writeText(data.url).then(()=>{btn.textContent='已複製';});
    }
  } catch(e) { setError('分享失敗：' + e.message); }
}

// ── Event wiring ──────────────────────────────────────────────────────────────
[temperatureEl,topPEl,maxTokensEl,memoryWindowEl,thinkingBudgetEl,fontSizeEl,toolMaxIterEl].forEach(el=>el&&el.addEventListener('input',updateControls));
[streamingEl,shortMemoryEl,thinkingModeEl,longMemoryEl,autoRouteEl,toolsEnabledEl,agentModeEl,ttsAutoEl].forEach(el=>{if(el)el.addEventListener('change',updateControls);});
modelEl.addEventListener('change',updateControls);
sendButton.addEventListener('click',sendMessage);
clearChatButton.addEventListener('click',clearChat);
stopStreamingButton.addEventListener('click',stopStreaming);
if(shareChatBtn) shareChatBtn.addEventListener('click',shareCurrentChat);
if(uploadBtn&&fileInput) uploadBtn.addEventListener('click',(e)=>{e.preventDefault();e.stopPropagation();fileInput.disabled=false;fileInput.removeAttribute('disabled');fileInput.click();});
if(fileInput) fileInput.disabled=false;
newChatBtn.addEventListener('click',startNewChat);
chatInput.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMessage();}});

if(chartToggleBtn&&chartPanel){
  chartToggleBtn.addEventListener('click',()=>{
    const open=chartPanel.classList.toggle('open');
    chartToggleBtn.textContent=open?'📊 隱藏圖表':'📊 Token 圖表';
    if(open)renderChart();
  });
}
if(accentCustomEl) accentCustomEl.addEventListener('input',()=>applyAccent(accentCustomEl.value));
if(refreshMemBtn) refreshMemBtn.addEventListener('click', loadMemories);

// ── Init ──────────────────────────────────────────────────────────────────────
(async()=>{
  applyTheme(lsGet('theme','dark')==='dark');
  const fs=lsGet('fontSize',15);fontSizeEl.value=fs;messagesEl.style.fontSize=fs+'px';
  buildAccentPicker();
  const savedAccent=lsGet('accent','#10a37f');applyAccent(savedAccent);
  initVoice();
  renderPromptTemplates();
  await loadTools();
  await loadMemories();
  await loadSessions();
  if(sessions.length>0) await loadSession(sessions[0].id); else await startNewChat();
  updateControls();
})();