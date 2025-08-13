// ===== Helpers =====
const $ = sel => document.querySelector(sel);
const baseInput = $('#baseUrl');
const statusDot = $('#statusDot');

function getBase() { return localStorage.getItem('apiBase') || 'http://localhost:8000'; }
function setBase(v) { localStorage.setItem('apiBase', v); }

async function getJSON(path) {
  const r = await fetch(getBase() + path, { headers: { 'Accept': 'application/json' }});
  if (!r.ok) throw await r.json().catch(()=>({error:r.statusText}));
  return r.json();
}
async function postJSON(path, body) {
  const r = await fetch(getBase() + path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)});
  const text = await r.text();
  try {
    const data = text ? JSON.parse(text) : {};
    if (!r.ok) throw data;
    return data;
  } catch(e) {
    if (!r.ok) throw text || { error: r.statusText };
    throw e;
  }
}

function toast(title, body, ms=3000) {
  $('#toastTitle').textContent = title;
  $('#toastBody').textContent = typeof body === 'string' ? body : JSON.stringify(body);
  $('#toastTime').textContent = new Date().toLocaleTimeString();
  const t = new bootstrap.Toast($('#toast'), { delay: ms }); t.show();
}

function setStatus(ok) {
  statusDot.textContent = ok ? 'online' : 'offline';
  statusDot.className = 'badge rounded-pill ' + (ok? 'text-bg-success' : 'text-bg-secondary');
}

function pretty(obj) { return JSON.stringify(obj, null, 2); }

function renderRows(rows) {
  if (!rows || !rows.length) return '<div class="text-secondary">No rows</div>';
  const cols = Object.keys(rows[0]);
  const th = cols.map(c=>`<th>${c}</th>`).join('');
  const trs = rows.map(r=>`<tr>${cols.map(c=>`<td>${String(r[c])}</td>`).join('')}</tr>`).join('');
  return `<div class="mb-2"><span class="badge text-bg-secondary">Rows (${rows.length})</span></div>
          <div class="table-responsive">
            <table class="table table-sm table-dark table-striped align-middle">
              <thead><tr>${th}</tr></thead>
              <tbody>${trs}</tbody>
            </table>
          </div>`;
}

function clip(text) { navigator.clipboard?.writeText(text).then(()=>toast('Copied','Content copied to clipboard',1500)); }

// ===== Wire up =====
function loadBase(){ baseInput.value = getBase(); }
$('#saveBase').addEventListener('click', ()=> { setBase(baseInput.value.trim()); toast('Saved','Base URL updated'); });

// Health
$('#btnHealth').addEventListener('click', async ()=>{
  $('#outHealth').textContent = 'Loading…';
  try { const h = await getJSON('/health'); $('#outHealth').textContent = pretty(h); setStatus(!!h.ok); }
  catch(e){ $('#outHealth').textContent = typeof e==='string'? e : pretty(e); setStatus(false); toast('Health error', e.error||'Failed'); }
});

// Index
$('#btnIndexSample').addEventListener('click', ()=>{
  $('#idxText').value = 'Qdrant supports named vectors so you can store multiple embedding spaces per document.';
  $('#idxMeta').value = '{\n  "source": "ui",\n  "topic": "qdrant"\n}';
});
$('#btnIndex').addEventListener('click', async ()=>{
  const id = $('#idxId').value.trim();
  let meta = {}; const mtxt = $('#idxMeta').value.trim();
  if (mtxt) {
    try { meta = JSON.parse(mtxt); } catch(e){ return toast('Invalid JSON in metadata', String(e)); }
  }
  const text = $('#idxText').value.trim(); if (!text) return toast('Missing text','Please paste some text to index');
  $('#outIndex').textContent = 'Indexing…';
  try { const out = await postJSON('/index', { id: id || undefined, text, metadata: meta }); $('#outIndex').textContent = pretty(out); toast('Indexed', `id: ${out.id}`); }
  catch(e){ $('#outIndex').textContent = typeof e==='string'? e : pretty(e); toast('Index error', e.detail?.toString?.() || 'Failed'); }
});

// Search
$('#btnSearch').addEventListener('click', async ()=>{
  const query = $('#sQuery').value.trim(); if (!query) return toast('Missing query','Enter some text');
  const field = $('#sField').value; const limit = parseInt($('#sLimit').value||'5',10);
  const box = $('#outSearch'); box.innerHTML = '<div class="text-secondary">Searching…</div>';
  try {
    const out = await postJSON('/search', { query, field, limit });
    const list = (out.hits||[]).map(h=>
      `<li class="list-group-item bg-transparent text-light border-secondary">
        <div class="d-flex justify-content-between">
          <code class="text-info">${h.id}</code>
          <span class="badge text-bg-secondary">${h.score?.toFixed? h.score.toFixed(4): h.score}</span>
        </div>
        <div class="text-secondary mt-1">${(h.payload?.text||'').slice(0,180)}${(h.payload?.text||'').length>180?'…':''}</div>
      </li>`).join('');
    box.innerHTML = `<ul class="list-group list-group-flush">${list || '<li class="list-group-item bg-transparent text-light">No hits</li>'}</ul>`;
  } catch(e){ box.textContent = typeof e==='string'? e : pretty(e); toast('Search error', e.detail?.toString?.() || 'Failed'); }
});

// Ask (RAG)
$('#btnAskSample').addEventListener('click', ()=>{ $('#askQ').value = 'How can I keep two embedding spaces together?'; });
$('#btnAsk').addEventListener('click', async ()=>{
  const question = $('#askQ').value.trim(); if (!question) return toast('Missing question','Type a question');
  const top_k = parseInt($('#askK').value||'3',10);
  const min_score = parseFloat($('#askMinScore').value||'0.25');
  const max_context_chars = parseInt($('#askMaxCtx').value||'3000',10);
  const rerank = $('#askRerank').checked;
  $('#outAsk').textContent = 'Thinking…';
  try {
    const out = await postJSON('/ask', { question, top_k, min_score, rerank, max_context_chars });
    $('#outAsk').textContent = pretty(out);
  } catch(e){ $('#outAsk').textContent = typeof e==='string'? e : pretty(e); toast('Ask error', e.detail?.toString?.() || 'Failed'); }
});
$('#btnCopyAnswer').addEventListener('click', ()=>{ try { const o = JSON.parse($('#outAsk').textContent||'{}'); clip(o.answer||''); } catch { clip($('#outAsk').textContent); } });

// Text2SQL
$('#btnSQLSample').addEventListener('click', ()=>{ $('#sqlQ').value = 'Top users by total order amount'; });
$('#btnSQL').addEventListener('click', async ()=>{
  const question = $('#sqlQ').value.trim(); if (!question) return toast('Missing question','Type a question');
  const limit = parseInt($('#sqlLimit').value||'10',10);
  const execute = $('#sqlExec').checked;
  $('#outSQL').textContent = 'Generating…'; $('#outSQLRows').innerHTML = '';
  try {
    const out = await postJSON('/text2sql', { question, limit, execute });
    $('#outSQL').textContent = out.sql || '';
    if (Array.isArray(out.rows)) { $('#outSQLRows').innerHTML = renderRows(out.rows); }
  } catch(e){ $('#outSQL').textContent = typeof e==='string'? e : pretty(e); toast('Text2SQL error', e.detail?.toString?.() || 'Failed'); }
});
$('#btnCopySQL').addEventListener('click', ()=> clip($('#outSQL').textContent || ''));

// init
(function init(){
  baseInput.value = getBase();
  // enable Bootstrap tooltips
  document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => new bootstrap.Tooltip(el));
  (async()=>{ try{ const h = await getJSON('/health'); setStatus(!!h.ok);} catch{ setStatus(false);} })();
})();
