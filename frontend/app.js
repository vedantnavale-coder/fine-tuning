/* ── NeuralForge App ─────────────────────────────────────────────── */

const API = 'http://99.49.245.187:8000/api';

// ── State ──────────────────────────────────────────────────────────────────
let chatHistory = [];
let trainingPollInterval = null;
let downloadPollInterval = null;

// ── Tab switching ──────────────────────────────────────────────────────────
document.querySelectorAll('.nav-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    const id = tab.dataset.tab;
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(`tab-${id}`).classList.add('active');
    if (id === 'train')  { loadPdfsForTraining(); loadModelsForTraining(); }
    if (id === 'models') { loadModels(); }
    if (id === 'chat')   { loadModelsForChat(); }
  });
});

// ── Global status indicator ────────────────────────────────────────────────
function setGlobalStatus(state, text) {
  const dot  = document.getElementById('globalStatusDot');
  const span = document.getElementById('globalStatusText');
  dot.className  = `status-dot ${state}`;
  span.textContent = text.toUpperCase();
}

// ── Helpers ────────────────────────────────────────────────────────────────
async function apiFetch(path, opts = {}) {
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

function toast(msg, type = 'ok') {
  const el = document.createElement('div');
  el.style.cssText = `
    position:fixed; bottom:24px; right:24px; z-index:99999;
    padding:12px 20px; border-radius:8px; font-family:var(--mono);
    font-size:0.75rem; letter-spacing:0.08em; max-width:360px;
    animation: fadeUp 0.25s ease;
    ${type==='ok'    ? 'background:rgba(57,255,136,0.15);border:1px solid rgba(57,255,136,0.4);color:#39ff88;' : ''}
    ${type==='error' ? 'background:rgba(255,59,92,0.15);border:1px solid rgba(255,59,92,0.4);color:#ff3b5c;' : ''}
    ${type==='info'  ? 'background:rgba(0,229,255,0.1);border:1px solid rgba(0,229,255,0.3);color:#00e5ff;' : ''}
  `;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

function termLog(elId, text, cls = '') {
  const el = document.getElementById(elId);
  if (!el) return;
  const line = document.createElement('div');
  line.className = `log-line ${cls}`;
  line.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}

// ── SETUP: Base model download ─────────────────────────────────────────────
async function checkBaseModel() {
  try {
    const data = await apiFetch('/models');
    const base = data.models.find(m => m.id === 'base');
    const infoEl = document.getElementById('baseModelStatus');
    const btn    = document.getElementById('downloadBtn');

    if (base && base.exists) {
      infoEl.textContent = `✓ Qwen3-4B is available locally (${base.size_mb} MB)`;
      infoEl.className   = 'infobox ok';
      btn.textContent    = '✓ MODEL READY';
      btn.disabled       = true;
    } else {
      infoEl.textContent = '⚠ Qwen3-4B not found locally. Click to download (~8 GB).';
      infoEl.className   = 'infobox warn';
      btn.disabled       = false;
    }
  } catch (e) {
    document.getElementById('baseModelStatus').textContent = 'Could not check model status.';
    document.getElementById('downloadBtn').disabled = false;
  }
}

async function downloadBaseModel() {
  try {
    const res = await apiFetch('/models/download-base', { method: 'POST' });
    if (res.status === 'exists') {
      toast('Base model already downloaded!', 'ok');
      checkBaseModel();
      return;
    }
    toast('Download started — this may take a while', 'info');
    document.getElementById('downloadProgressWrap').classList.remove('hidden');
    setGlobalStatus('active', 'Downloading');
    pollDownload();
  } catch (e) {
    toast(e.message, 'error');
  }
}

function pollDownload() {
  clearInterval(downloadPollInterval);
  downloadPollInterval = setInterval(async () => {
    try {
      const s = await apiFetch('/models/download-status');
      const pct = s.progress || 0;
      document.getElementById('downloadProgress').style.width = pct + '%';
      document.getElementById('downloadLabel').textContent = pct + '%';
      document.getElementById('baseModelStatus').textContent = s.message;

      if (s.status === 'completed') {
        clearInterval(downloadPollInterval);
        document.getElementById('baseModelStatus').className = 'infobox ok';
        setGlobalStatus('success', 'Ready');
        toast('Model downloaded successfully!', 'ok');
        checkBaseModel();
      } else if (s.status === 'error') {
        clearInterval(downloadPollInterval);
        document.getElementById('baseModelStatus').className = 'infobox error';
        setGlobalStatus('error', 'Error');
        toast('Download failed: ' + s.message, 'error');
      }
    } catch (_) {}
  }, 2500);
}

// ── SETUP: PDF management ──────────────────────────────────────────────────
const dropZone  = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => handleFiles(e.target.files));
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag'); });
dropZone.addEventListener('dragleave', ()  => dropZone.classList.remove('drag'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag');
  handleFiles(e.dataTransfer.files);
});

async function handleFiles(files) {
  const pdfs = Array.from(files).filter(f => f.name.endsWith('.pdf'));
  if (!pdfs.length) { toast('Only PDF files are accepted', 'error'); return; }

  for (const file of pdfs) {
    const fd = new FormData();
    fd.append('file', file);
    try {
      await apiFetch('/upload', { method: 'POST', body: fd });
      toast(`Uploaded: ${file.name}`, 'ok');
    } catch (e) {
      toast(`Failed to upload ${file.name}: ${e.message}`, 'error');
    }
  }
  fileInput.value = '';
  loadPdfList();
  loadPdfsForTraining();
}

async function loadPdfList() {
  try {
    const data = await apiFetch('/pdfs');
    const el   = document.getElementById('pdfList');
    if (!data.pdfs.length) { el.innerHTML = '<span class="muted">No PDFs uploaded yet</span>'; return; }
    el.innerHTML = data.pdfs.map(pdf => `
      <div class="pdf-item">
        <span>📄 ${pdf}</span>
        <button class="del-btn" onclick="deletePdf('${pdf}')" title="Delete">✕</button>
      </div>
    `).join('');
  } catch (_) {}
}

async function deletePdf(name) {
  if (!confirm(`Delete "${name}"?`)) return;
  try {
    await apiFetch(`/pdfs/${encodeURIComponent(name)}`, { method: 'DELETE' });
    toast(`Deleted: ${name}`, 'ok');
    loadPdfList();
    loadPdfsForTraining();
  } catch (e) {
    toast(e.message, 'error');
  }
}

// ── TRAIN: Load PDFs & models ──────────────────────────────────────────────
async function loadPdfsForTraining() {
  try {
    const data = await apiFetch('/pdfs');
    const el   = document.getElementById('trainPdfList');
    if (!data.pdfs.length) {
      el.innerHTML = '<span class="muted">No PDFs — upload some in Setup</span>';
      return;
    }
    el.innerHTML = data.pdfs.map(pdf => `
      <div class="pdf-check-item">
        <input type="checkbox" id="train-${pdf}" value="${pdf}" checked />
        <label for="train-${pdf}">${pdf}</label>
      </div>
    `).join('');
  } catch (_) {}
}

async function loadModelsForTraining() {
  try {
    const data   = await apiFetch('/models');
    const sel    = document.getElementById('sourceModelSelect');
    const models = data.models.filter(m => m.exists);
    if (!models.length) {
      sel.innerHTML = '<option value="">— no models available —</option>';
      return;
    }
    sel.innerHTML = models.map(m =>
      `<option value="${m.id}">[${m.type.toUpperCase()}] ${m.id}</option>`
    ).join('');
  } catch (_) {}
}

async function startTraining() {
  const sourceId  = document.getElementById('sourceModelSelect').value;
  const outName   = document.getElementById('outputModelName').value.trim();
  const checked   = document.querySelectorAll('#trainPdfList input[type="checkbox"]:checked');
  const pdfs      = Array.from(checked).map(c => c.value);

  if (!sourceId)       { toast('Select a source model', 'error'); return; }
  if (!outName)        { toast('Enter an output model name', 'error'); return; }
  if (!pdfs.length)    { toast('Select at least one PDF', 'error'); return; }

  const btn = document.getElementById('trainBtn');
  btn.disabled = true;

  // Clear terminal
  const term = document.getElementById('trainingLog');
  term.innerHTML = '';
  termLog('trainingLog', `Starting training: ${sourceId} → ${outName}`, 'ok');

  try {
    await apiFetch('/train', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ pdfs, base_model_id: sourceId, output_model_name: outName })
    });
    pollTraining(btn);
  } catch (e) {
    toast(e.message, 'error');
    btn.disabled = false;
  }
}

function pollTraining(btn) {
  setGlobalStatus('active', 'Training');
  clearInterval(trainingPollInterval);
  trainingPollInterval = setInterval(async () => {
    try {
      const s = await apiFetch('/training/status');
      document.getElementById('trainProgress').style.width = (s.progress || 0) + '%';
      document.getElementById('trainProgressLabel').textContent = (s.progress || 0) + '%';
      document.getElementById('trainStatusBadge').textContent = s.status.toUpperCase().replace('_', ' ');
      document.getElementById('trainStatusBadge').className   = `status-badge ${s.status}`;

      termLog('trainingLog', s.message, getLogClass(s.status));

      if (s.status === 'completed') {
        clearInterval(trainingPollInterval);
        btn.disabled = false;
        setGlobalStatus('success', 'Done');
        toast('Training complete! ' + s.message, 'ok');
        loadModelsForTraining();
      } else if (s.status === 'error') {
        clearInterval(trainingPollInterval);
        btn.disabled = false;
        setGlobalStatus('error', 'Error');
        toast('Training error: ' + s.message, 'error');
      }
    } catch (_) {}
  }, 2000);
}

function getLogClass(status) {
  if (status === 'completed') return 'ok';
  if (status === 'error')     return 'err';
  if (status === 'training')  return 'warn';
  return '';
}

// ── MODELS: Registry ───────────────────────────────────────────────────────
async function loadModels() {
  const grid = document.getElementById('modelGrid');
  grid.innerHTML = '<span class="muted">Loading...</span>';
  try {
    const data = await apiFetch('/models');
    if (!data.models.length) {
      grid.innerHTML = '<span class="muted">No models registered yet</span>';
      return;
    }
    grid.innerHTML = data.models.map(m => `
      <div class="model-card ${m.type === 'base' ? 'base-card' : 'trained-card'}">
        <div class="model-card-head">
          <span class="model-card-type ${m.type === 'base' ? 'type-base' : 'type-trained'}">
            ${m.type.toUpperCase()}
          </span>
          <span class="model-card-id">${m.id}</span>
        </div>
        <div class="model-card-meta">
          Created: <span>${new Date(m.created_at).toLocaleDateString()}</span><br/>
          Size: <span>${m.size_mb} MB</span><br/>
          ${m.parent_id ? `Parent: <span>${m.parent_id}</span><br/>` : ''}
          ${m.pdfs_trained_on?.length ? `PDFs: <span>${m.pdfs_trained_on.length}</span>` : ''}
        </div>
        <div class="model-card-actions">
          ${m.deletable
            ? `<button class="btn btn-danger btn-sm" onclick="deleteModel('${m.id}')">DELETE</button>`
            : `<span class="muted" style="font-size:0.65rem">BASE — PROTECTED</span>`
          }
        </div>
      </div>
    `).join('');
  } catch (e) {
    grid.innerHTML = `<span class="muted" style="color:var(--red)">${e.message}</span>`;
  }
}

async function deleteModel(id) {
  if (!confirm(`Delete model "${id}"? This cannot be undone.`)) return;
  try {
    await apiFetch(`/models/${encodeURIComponent(id)}`, { method: 'DELETE' });
    toast(`Deleted: ${id}`, 'ok');
    loadModels();
    loadModelsForTraining();
    loadModelsForChat();
  } catch (e) {
    toast(e.message, 'error');
  }
}

// ── CHAT ───────────────────────────────────────────────────────────────────
async function loadModelsForChat() {
  try {
    const data = await apiFetch('/models');
    const sel  = document.getElementById('chatModelSelect');
    const models = data.models.filter(m => m.exists);

    if (!models.length) {
      sel.innerHTML = '<option value="">— no models available —</option>';
      return;
    }
    sel.innerHTML = models.map(m =>
      `<option value="${m.id}">[${m.type.toUpperCase()}] ${m.id}</option>`
    ).join('');

    // Check what's already loaded
    const loaded = await apiFetch('/chat/loaded');
    if (loaded.loaded_model) {
      sel.value = loaded.loaded_model;
      document.getElementById('loadedModelInfo').textContent = `Loaded: ${loaded.loaded_model}`;
      document.getElementById('loadedModelInfo').className   = 'infobox ok';
    }
  } catch (_) {}
}

async function loadChatModel() {
  const modelId = document.getElementById('chatModelSelect').value;
  if (!modelId) { toast('Select a model to load', 'error'); return; }

  const btn  = document.getElementById('loadModelBtn');
  const info = document.getElementById('loadedModelInfo');

  btn.disabled     = true;
  btn.textContent  = 'LOADING...';
  info.textContent = `Loading ${modelId} into memory...`;
  info.className   = 'infobox warn';
  setGlobalStatus('active', 'Loading model');

  try {
    await apiFetch(`/chat/load/${encodeURIComponent(modelId)}`, { method: 'POST' });
    info.textContent = `✓ Loaded: ${modelId}`;
    info.className   = 'infobox ok';
    setGlobalStatus('success', 'Model ready');
    toast(`Model ${modelId} loaded!`, 'ok');
  } catch (e) {
    info.textContent = `Error: ${e.message}`;
    info.className   = 'infobox error';
    setGlobalStatus('error', 'Error');
    toast(e.message, 'error');
  } finally {
    btn.disabled    = false;
    btn.textContent = 'LOAD INTO MEMORY';
  }
}

function handleChatKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function clearChat() {
  chatHistory = [];
  const msgs = document.getElementById('chatMessages');
  msgs.innerHTML = `
    <div class="chat-welcome">
      <div class="welcome-icon">⬡</div>
      <p>Chat cleared</p>
    </div>
  `;
}

function appendMessage(role, text, isThinking = false) {
  const msgs = document.getElementById('chatMessages');

  // Remove welcome if present
  const welcome = msgs.querySelector('.chat-welcome');
  if (welcome) welcome.remove();

  const id  = 'msg-' + Date.now() + Math.random().toString(36).slice(2);
  const icon = role === 'user' ? 'U' : '⬡';
  const bubble = isThinking ? '<em>Thinking...</em>' : escapeHtml(text).replace(/\n/g, '<br/>');

  const el = document.createElement('div');
  el.className = `msg ${role}`;
  el.id = id;
  el.innerHTML = `
    <div class="msg-avatar">${icon}</div>
    <div class="msg-bubble${isThinking ? ' thinking' : ''}">${bubble}</div>
  `;
  msgs.appendChild(el);
  msgs.scrollTop = msgs.scrollHeight;
  return id;
}

function updateMessage(id, text) {
  const el = document.getElementById(id);
  if (!el) return;
  const bubble = el.querySelector('.msg-bubble');
  bubble.className = 'msg-bubble';
  bubble.innerHTML = escapeHtml(text).replace(/\n/g, '<br/>');
}

function escapeHtml(t) {
  return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

async function sendMessage() {
  const input   = document.getElementById('chatInput');
  const message = input.value.trim();
  const modelId = document.getElementById('chatModelSelect').value;
  const temp    = parseFloat(document.getElementById('tempSlider').value);

  if (!message) return;
  if (!modelId) { toast('Select and load a model first', 'error'); return; }

  input.value = '';
  appendMessage('user', message);
  const thinkingId = appendMessage('bot', '', true);

  const sendBtn = document.getElementById('sendBtn');
  sendBtn.disabled = true;
  setGlobalStatus('active', 'Generating');

  // Add to history
  chatHistory.push({ role: 'user', content: message });

  try {
    const data = await apiFetch('/chat', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        message,
        model_id: modelId,
        history:  chatHistory.slice(0, -1),  // don't include current msg
      })
    });

    updateMessage(thinkingId, data.response);
    chatHistory.push({ role: 'assistant', content: data.response });
    setGlobalStatus('success', 'Ready');
  } catch (e) {
    updateMessage(thinkingId, `Error: ${e.message}`);
    document.getElementById(thinkingId)?.querySelector('.msg-bubble')?.style.setProperty('color', 'var(--red)');
    chatHistory.pop();
    setGlobalStatus('error', 'Error');
    toast(e.message, 'error');
  } finally {
    sendBtn.disabled = false;
    document.getElementById('chatMessages').scrollTop = 99999;
  }
}

// ── Init ───────────────────────────────────────────────────────────────────
(async function init() {
  await checkBaseModel();
  await loadPdfList();
})();