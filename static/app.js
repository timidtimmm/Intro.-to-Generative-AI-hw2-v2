// ═══════════════════════════════════════════════════════════════════════════════
//  HW01 ChatGPT — app.js
//  NEW: token chart, accent color picker, response timer, conversation branches
// ═══════════════════════════════════════════════════════════════════════════════

// ── Constants ─────────────────────────────────────────────────────────────────
const VISION_MODELS = new Set([
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

// ── State ─────────────────────────────────────────────────────────────────────
// Messages: each entry is { role, content, ts, branches?, branchIdx?, ... }
// branches = array of content strings (assistant variants), branchIdx = active
let messages          = [];
let isLoading         = false;
let abortController   = null;
let currentSessionId  = null;
let sessions          = [];
let pendingAttachment = null;
let isDarkTheme       = true;
let totalPromptTokens     = 0;
let totalCompletionTokens = 0;

// Token chart data: array of { label, prompt, completion }
let tokenHistory = [];
let chartInstance = null;

// ── localStorage ──────────────────────────────────────────────────────────────
const lsGet = (k,fb) => { try { const v=localStorage.getItem(k); return v!==null?JSON.parse(v):fb; } catch { return fb; } };
const lsSet = (k,v) => { try { localStorage.setItem(k,JSON.stringify(v)); } catch {} };

// ── Accent color ──────────────────────────────────────────────────────────────
function applyAccent(hex) {
  // Derive hover = darken 10%
  const darken = (h, amt) => {
    let [r,g,b] = [1,3,5].map(i=>parseInt(h.slice(i,i+2),16));
    [r,g,b] = [r,g,b].map(c=>Math.max(0,c-amt));
    return '#'+[r,g,b].map(c=>c.toString(16).padStart(2,'0')).join('');
  };
  document.documentElement.style.setProperty('--accent', hex);
  document.documentElement.style.setProperty('--accent-hover', darken(hex, 25));
  document.documentElement.style.setProperty('--bubble-user', hex);
  lsSet('accent', hex);
  // Update swatch selection
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
  if (tokenHistory.length > 20) tokenHistory.shift(); // keep last 20
  renderChart();
}

function renderChart() {
  if (!tokenChartCanvas || !window.Chart) return;
  const labels     = tokenHistory.map((r,i) => r.label || `#${i+1}`);
  const prompts    = tokenHistory.map(r => r.prompt);
  const completions = tokenHistory.map(r => r.completion);

  const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#10a37f';
  const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
  const gridColor  = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';
  const textColor  = isDark ? '#a0a0a0' : '#666';

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
  if (chartInstance) renderChart(); // update chart colors
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
  return escapeHtml(raw)
    .replace(/`([^`]+)`/g,'<code class="inline-code">$1</code>')
    .replace(/\*\*\*(.+?)\*\*\*/g,'<strong><em>$1</em></strong>')
    .replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>')
    .replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g,'<em>$1</em>')
    .replace(/~~(.+?)~~/g,'<del>$1</del>')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\)]+)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>');
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

const formatTs = iso => iso ? new Date(iso).toLocaleTimeString('zh-TW',{hour:'2-digit',minute:'2-digit'}) : '';
const formatDur = ms => ms < 1000 ? `${ms}ms` : `${(ms/1000).toFixed(1)}s`;

// ── Render messages ───────────────────────────────────────────────────────────
function renderMessages() {
  messagesEl.innerHTML = '';
  messages.forEach((msg, idx) => {
    const row = document.createElement('div');
    row.className = `message-row ${msg.role==='user'?'user-row':'assistant-row'}`;
    const bubble = document.createElement('div');
    bubble.className = `message-bubble ${msg.role}`;

    // Header
    const hdr = document.createElement('div'); hdr.className = 'message-header';
    const roleEl = document.createElement('span'); roleEl.className = 'message-role';
    roleEl.textContent = msg.role==='user' ? 'You' : 'Assistant';
    const tsEl = document.createElement('span'); tsEl.className = 'message-ts';
    tsEl.textContent = formatTs(msg.ts);
    hdr.appendChild(roleEl); hdr.appendChild(tsEl); bubble.appendChild(hdr);

    // Image / file chip
    if (msg.imagePreview) { const img=document.createElement('img'); img.src=msg.imagePreview; img.className='message-image'; bubble.appendChild(img); }
    else if (msg.filename) { const chip=document.createElement('div'); chip.style.cssText='font-size:12px;color:#9ef3df;margin-bottom:6px;'; chip.textContent='📎 '+msg.filename; bubble.appendChild(chip); }

    const contentEl = document.createElement('div'); contentEl.className = 'message-content';

    if (msg.role === 'assistant') {
      // No branch navigator on assistant side — it's controlled by the user nav above
      const activeContent = msg.branches
        ? (msg.branches[msg.branchIdx||0] || '')
        : (msg.content || '');

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

      // Token info + response time
      const metaEl = document.createElement('div'); metaEl.className = 'msg-meta';
      if (msg.usage) metaEl.textContent = `↑ ${msg.usage.prompt_tokens} · ↓ ${msg.usage.completion_tokens} tokens`;
      if (msg.duration) metaEl.textContent += (msg.usage ? '  ·  ' : '') + `⏱ ${formatDur(msg.duration)}`;
      if (metaEl.textContent) contentEl.appendChild(metaEl);

      // Action bar
      if (!msg.isStreaming) {
        const bar=document.createElement('div'); bar.className='msg-actions';
        const regen=document.createElement('button'); regen.className='msg-action-btn';
        regen.textContent='🔄 重新生成'; regen.onclick=()=>regenerateFrom(idx);  // idx = assistantIdx
        bar.appendChild(regen); contentEl.appendChild(bar);
      }
    } else {
      // User: show active branch content
      const activeContent = msg.branches ? (msg.branches[msg.branchIdx||0]||'') : msg.content;

      // Branch navigator for user messages too
      if (msg.branches && msg.branches.length > 1) {
        const nav=document.createElement('div'); nav.className='branch-nav';
        const prev=document.createElement('button'); prev.className='branch-btn'; prev.textContent='‹';
        prev.disabled=(msg.branchIdx||0)===0;
        prev.onclick=()=>switchBranch(idx,(msg.branchIdx||0)-1);
        const counter=document.createElement('span'); counter.className='branch-counter';
        counter.textContent=`${(msg.branchIdx||0)+1} / ${msg.branches.length}`;
        const next=document.createElement('button'); next.className='branch-btn'; next.textContent='›';
        next.disabled=(msg.branchIdx||0)===msg.branches.length-1;
        next.onclick=()=>switchBranch(idx,(msg.branchIdx||0)+1);
        nav.appendChild(prev); nav.appendChild(counter); nav.appendChild(next);
        contentEl.appendChild(nav);
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
/**
 * Data model
 * ----------
 * Every exchange (user + assistant pair) is stored as TWO consecutive entries in
 * `messages[]`, BUT we use a "slot" concept to track paired branches:
 *
 *   messages[N]   = { role:'user', content, branches:['v1','v2',...], branchIdx, slotId, ... }
 *   messages[N+1] = { role:'assistant', content, branches:['r1','r2',...], branchIdx, slotId, ... }
 *
 * Both have the SAME slotId.  branchIdx is kept in sync so branch[i] of the user
 * always corresponds to branch[i] of the assistant.
 *
 * switchBranch(userIdx, newBranchIdx)
 *   → sets both messages[userIdx].branchIdx and messages[userIdx+1].branchIdx
 *   → re-renders
 *
 * When user edits message at userIdx:
 *   1. Save current state: no truncation yet — just note editIdx in chatInput
 *   2. On re-send: save the OLD assistant reply into aMsg.branches[currentBranchIdx]
 *      if not already stored, then push new versions to both branches arrays.
 */

function startEditMessage(idx) {
  if (isLoading) return;
  const msg = messages[idx];
  chatInput.value = msg.branches ? (msg.branches[msg.branchIdx||0]||'') : msg.content;
  chatInput.dataset.editIdx = idx;
  chatInput.focus();
}

function switchBranch(userIdx, newBranchIdx) {
  const uMsg = messages[userIdx];
  if (!uMsg?.branches || newBranchIdx < 0 || newBranchIdx >= uMsg.branches.length) return;

  uMsg.branchIdx = newBranchIdx;

  // Always sync the immediately following assistant message
  const aMsg = messages[userIdx + 1];
  if (aMsg?.role === 'assistant' && aMsg.branches && newBranchIdx < aMsg.branches.length) {
    aMsg.branchIdx = newBranchIdx;
    // Also sync content so streaming / rendering reads the right branch
    aMsg.content = aMsg.branches[newBranchIdx];
  }
  renderMessages();
}

async function regenerateFrom(assistantIdx) {
  if (isLoading) return;
  const userIdx = assistantIdx - 1;
  if (userIdx < 0 || messages[userIdx]?.role !== 'user') return;

  const uMsg = messages[userIdx];
  const aMsg = messages[assistantIdx];

  // Current user text (active branch)
  const userText = uMsg.branches ? (uMsg.branches[uMsg.branchIdx||0]||'') : uMsg.content;

  // Mark editIdx so doSend knows to branch
  await doSend(userText, userIdx, /* isRegen */ true);
}

// ── Error ─────────────────────────────────────────────────────────────────────
function setError(text='') { errorBox.textContent=text; errorBox.classList.toggle('hidden',!text); }

// ── Token badge + chart ───────────────────────────────────────────────────────
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
  currentModelText.textContent      = `目前模型：${modelEl.value}`;
  streamingBadge.textContent        = streamingEl.checked?'Streaming On':'Streaming Off';
  memoryBadge.textContent           = shortMemoryEl.checked?`Memory ${memoryWindowEl.value}`:'Memory Off';
  memoryWindowEl.disabled           = !shortMemoryEl.checked;
  thinkingBudgetGroup.style.display = thinkingModeEl.checked?'block':'none';
  thinkingBadge.classList.toggle('hidden',!thinkingModeEl.checked);
  const isVision=VISION_MODELS.has(modelEl.value);
  visionHint.classList.toggle('hidden',!isVision);
  const ul=document.querySelector('.upload-btn');
  if(ul){ul.title=isVision?'上傳圖片（Vision 模型）':'請切換至 Vision 模型才能上傳圖片'; ul.style.opacity=isVision?'1':'0.35'; ul.style.cursor=isVision?'pointer':'not-allowed'; fileInput.disabled=!isVision;}
  sendButton.disabled=isLoading;
}

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
  chatTitleEl.textContent='Chat'; renderMessages(); await loadSessions();
}

async function loadSession(sid) {
  currentSessionId=sid; totalPromptTokens=0; totalCompletionTokens=0; tokenHistory=[];
  tokenBadge.classList.add('hidden');
  if(chartInstance){chartInstance.destroy();chartInstance=null;}
  const r=await fetch(`/api/sessions/${sid}/messages`); const d=await r.json();
  messages=d.messages.length?d.messages.map(m=>({...m,ts:m.created_at})):[{role:'assistant',content:'這是空的對話，開始輸入吧！',ts:new Date().toISOString()}];
  chatTitleEl.textContent=sessions.find(s=>s.id===sid)?.title||'Chat';
  renderMessages(); renderSessionList();
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
  const title=chatTitleEl.textContent||'chat';
  let md=`# ${title}\n\n`;
  messages.forEach(m=>{const c=m.branches?(m.branches[m.branchIdx||0]||''):m.content;md+=`**${m.role==='user'?'You':'Assistant'}**${m.ts?` _(${formatTs(m.ts)})_`:''}\n\n${c}\n\n---\n\n`;});
  dl(md,`${title}.md`,'text/markdown');
});
exportJsonBtn.addEventListener('click',()=>{
  const title=chatTitleEl.textContent||'chat';
  dl(JSON.stringify({title,messages,exported_at:new Date().toISOString()},null,2),`${title}.json`,'application/json');
});
function dl(content,filename,mime){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([content],{type:mime}));a.download=filename;a.click();URL.revokeObjectURL(a.href);}

// ── File upload ───────────────────────────────────────────────────────────────
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

// ── Send / branching logic ────────────────────────────────────────────────────
function getPayloadMessages(userText){
  const win=Number(memoryWindowEl.value);
  // Use active branch content for context
  const ctx=shortMemoryEl.checked?messages.slice(-win).map(m=>({role:m.role,content:m.branches?(m.branches[m.branchIdx||0]||''):m.content})):[];
  return[...ctx,{role:'user',content:userText}];
}

async function sendMessage(){
  const text=chatInput.value.trim();
  if((!text&&!pendingAttachment)||isLoading)return;
  const editIdx = chatInput.dataset.editIdx !== undefined ? parseInt(chatInput.dataset.editIdx) : null;
  delete chatInput.dataset.editIdx;
  chatInput.value='';
  await doSend(text, editIdx, false);
}

async function doSend(overrideText, editIdx=null, isRegen=false){
  // overrideText = the new user text (or null to re-use last user msg for regen)
  let text = overrideText;
  if (text === null) {
    const lu = [...messages].reverse().find(m => m.role==='user');
    text = lu?.branches ? (lu.branches[lu.branchIdx||0]||'') : (lu?.content||'');
  }

  setError(''); isLoading=true; updateControls();

  let fullText=text, imageB64=null, imageMime=null, userImagePreview=null, userFilename=null;
  if (pendingAttachment) {
    if (pendingAttachment.type==='image') {
      if (!VISION_MODELS.has(modelEl.value)) {
        if (confirm(`目前模型「${modelEl.value}」不支援圖片。\n要切換到 llama-3.2-11b-vision 嗎？`)) { modelEl.value='meta/llama-3.2-11b-vision-instruct'; updateControls(); }
      }
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

  // ── Determine user message slot & assistant slot ──────────────────────────
  let userMsgIdx, assistantMsgIdx;

  if (editIdx !== null && messages[editIdx]?.role === 'user') {
    // EDIT or REGEN: branch the existing user+assistant pair
    const uMsg = messages[editIdx];
    const aMsg = messages[editIdx + 1]; // may or may not exist

    // Initialize branches array on user message if first edit
    if (!uMsg.branches) {
      uMsg.branches   = [uMsg.content];
      uMsg.branchIdx  = 0;
    }

    if (!isRegen) {
      // Only push a new user branch when text changed (edit, not regen)
      uMsg.branches.push(fullText);
    }
    uMsg.branchIdx = uMsg.branches.length - 1;

    // Initialize or extend assistant branches
    if (aMsg && aMsg.role === 'assistant') {
      if (!aMsg.branches) {
        aMsg.branches  = [aMsg.content];
        aMsg.branchIdx = 0;
      }
      // Push a placeholder for the incoming reply
      aMsg.branches.push('');
      aMsg.branchIdx = aMsg.branches.length - 1;
      aMsg.content   = '';
      aMsg.isStreaming = true;
      aMsg.ts        = new Date().toISOString();
      userMsgIdx      = editIdx;
      assistantMsgIdx = editIdx + 1;
    } else {
      // No assistant message yet — create one
      messages.splice(editIdx + 1, 0, {
        role:'assistant', content:'', branches:[''], branchIdx:0,
        isStreaming:true, ts:new Date().toISOString()
      });
      userMsgIdx      = editIdx;
      assistantMsgIdx = editIdx + 1;
    }

    // Drop any messages that came AFTER this pair (they belong to old branch)
    messages = messages.slice(0, assistantMsgIdx + 1);

  } else {
    // NORMAL send: push new user + assistant pair
    messages.push({
      role:'user', content:fullText,
      imagePreview:userImagePreview, filename:userFilename,
      ts:new Date().toISOString()
    });
    messages.push({ role:'assistant', content:'', isStreaming:true, ts:new Date().toISOString() });
    userMsgIdx      = messages.length - 2;
    assistantMsgIdx = messages.length - 1;
  }

  renderMessages();

  const payload = {
    model:modelEl.value, systemPrompt:systemPromptEl.value,
    temperature:Number(temperatureEl.value), top_p:Number(topPEl.value), max_tokens:Number(maxTokensEl.value),
    messages:getPayloadMessages(fullText), thinking:thinkingModeEl.checked,
    thinking_budget:Number(thinkingBudgetEl.value), session_id:currentSessionId,
    image_b64:imageB64||undefined, image_mime:imageMime||undefined,
  };

  clearAttachment();
  if (!isRegen && editIdx === null) autoTitleSession(text||fullText);

  const t0 = Date.now();

  // Helper: write completed assistant text into the right slot
  const finaliseAssistant = (finalText, usage, duration) => {
    const aMsg = messages[assistantMsgIdx];
    if (!aMsg) return;
    aMsg.content     = finalText;
    aMsg.isStreaming = false;
    aMsg.duration    = duration;
    if (usage) aMsg.usage = usage;
    // Keep branches in sync
    if (aMsg.branches) {
      aMsg.branches[aMsg.branchIdx] = finalText;
    }
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
            const aMsg = messages[assistantMsgIdx];
            if (aMsg) { aMsg.content=assTxt; aMsg.isStreaming=false; if(aMsg.branches) aMsg.branches[aMsg.branchIdx]=assTxt; }
            renderMessages();
          } else if (ev.type==='usage') {
            updateTokenBadge(ev.usage, fullText.slice(0,16));
          } else if (ev.type==='done') {
            finaliseAssistant(assTxt, ev.usage, Date.now()-t0);
            if (ev.usage) updateTokenBadge(ev.usage, fullText.slice(0,16));
            renderMessages(); done=true; break;
          } else if (ev.type==='error') throw new Error(ev.content||'Streaming error.');
        }
      }
    } else {
      const resp=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
      if (!resp.ok) throw new Error((await resp.text())||'Chat failed.');
      const d=await resp.json();
      finaliseAssistant(d.output||'', d.usage, Date.now()-t0);
      if (d.usage) updateTokenBadge(d.usage, fullText.slice(0,16));
      renderMessages();
    }
  } catch(err) {
    const aMsg = messages[assistantMsgIdx];
    if (err.name==='AbortError') {
      setError('已停止生成。');
      if (aMsg) { aMsg.isStreaming=false; if(aMsg.branches) aMsg.branches[aMsg.branchIdx]=aMsg.content; }
    } else {
      if (aMsg) { aMsg.content=`Error: ${err.message}`; aMsg.isStreaming=false; if(aMsg.branches) aMsg.branches[aMsg.branchIdx]=aMsg.content; }
      setError(err.message);
    }
    renderMessages();
  } finally {
    isLoading=false; abortController=null; updateControls();
  }
}

// ── Clear / stop ──────────────────────────────────────────────────────────────
function clearChat(){
  if(abortController)abortController.abort();
  setError('');isLoading=false;clearAttachment();
  if(currentSessionId)fetch(`/api/sessions/${currentSessionId}`,{method:'DELETE'});
  startNewChat();
}
function stopStreaming(){if(abortController)abortController.abort();}

// ── Event wiring ──────────────────────────────────────────────────────────────
[temperatureEl,topPEl,maxTokensEl,memoryWindowEl,thinkingBudgetEl,fontSizeEl].forEach(el=>el.addEventListener('input',updateControls));
[streamingEl,shortMemoryEl,thinkingModeEl].forEach(el=>el.addEventListener('change',updateControls));
modelEl.addEventListener('change',updateControls);
sendButton.addEventListener('click',sendMessage);
clearChatButton.addEventListener('click',clearChat);
stopStreamingButton.addEventListener('click',stopStreaming);
newChatBtn.addEventListener('click',startNewChat);
chatInput.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMessage();}});

// Chart toggle
if(chartToggleBtn&&chartPanel){
  chartToggleBtn.addEventListener('click',()=>{
    const open=chartPanel.classList.toggle('open');
    chartToggleBtn.textContent=open?'📊 隱藏圖表':'📊 Token 圖表';
    if(open)renderChart();
  });
}

// Accent color picker
if(accentCustomEl){
  accentCustomEl.addEventListener('input',()=>applyAccent(accentCustomEl.value));
}

// ── Init ──────────────────────────────────────────────────────────────────────
(async()=>{
  applyTheme(lsGet('theme','dark')==='dark');
  const fs=lsGet('fontSize',15);fontSizeEl.value=fs;messagesEl.style.fontSize=fs+'px';
  buildAccentPicker();
  const savedAccent=lsGet('accent','#10a37f');applyAccent(savedAccent);
  initVoice();
  renderPromptTemplates();
  await loadSessions();
  if(sessions.length>0)await loadSession(sessions[0].id);else await startNewChat();
  updateControls();
})();
