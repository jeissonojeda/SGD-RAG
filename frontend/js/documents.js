// ═══════════════════════════════════════════════
// documents.js — Gestión de documentos y carpetas
// ═══════════════════════════════════════════════

let allDocs = [];
let folderFiles = [];

// ── Cargar documentos ─────────────────────────────────────
async function loadDocs() {
  try {
    const res = await fetch(`${API}/api/documents`, {
      headers: { Authorization: `Bearer ${token}` }
    });
    allDocs = await res.json();
    renderDocsTable();
    renderDocFilter();
  } catch (e) { console.error(e); }
}

// ── Tabla de documentos ───────────────────────────────────
function renderDocsTable() {
  const tbody = document.getElementById('docs-tbody');
  if (!allDocs.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No hay documentos cargados aún</td></tr>';
    return;
  }
  tbody.innerHTML = allDocs.map(d => `
    <tr>
      <td>
        <div class="doc-name-cell">
          ${getFileIcon(d.file_type)}
          <div>
            <div style="color:var(--text);font-weight:600;font-size:13px">${d.original_name}</div>
            ${d.description ? `<div style="font-size:11px;color:var(--text3);margin-top:2px">${d.description.slice(0, 60)}${d.description.length > 60 ? '...' : ''}</div>` : ''}
          </div>
        </div>
      </td>
      <td>${d.area}</td>
      <td style="font-family:var(--mono);font-size:12px">${d.owner}</td>
      <td><span class="status-pill status-${d.status.toLowerCase()}">${d.status}</span></td>
      <td style="font-family:var(--mono);font-size:11px">${d.uploaded_at}</td>
      <td><button class="btn btn-danger" onclick="deleteDoc(${d.id})">🗑 Eliminar</button></td>
    </tr>
  `).join('');
}

// ── Filtro lateral (chat) ─────────────────────────────────
function renderDocFilter() {
  const list = document.getElementById('doc-filter-list');
  if (!allDocs.length) {
    list.innerHTML = '<p style="font-size:12px;color:var(--text3);padding:12px;text-align:center">Sin documentos</p>';
    return;
  }
  list.innerHTML = allDocs.map(d => `
    <div class="doc-filter-item selected" onclick="toggleDocFilter(${d.id}, this)">
      <input type="checkbox" id="cf-${d.id}" checked>
      ${getFileIcon(d.file_type)}
      <span class="doc-filter-name">${d.original_name}</span>
    </div>
  `).join('');
}

function toggleDocFilter(id, el) {
  const cb = document.getElementById(`cf-${id}`);
  cb.checked = !cb.checked;
  el.classList.toggle('selected', cb.checked);
}

function selectAllDocs() {
  document.querySelectorAll('#doc-filter-list input[type=checkbox]').forEach(cb => {
    cb.checked = true;
    cb.closest('.doc-filter-item').classList.add('selected');
  });
}

function getSelectedDocIds() {
  // Retorna null para buscar en todos
  return null;
}

// ── Eliminar documento ────────────────────────────────────
async function deleteDoc(id) {
  if (!confirm('¿Eliminar este documento? Se eliminará del índice.')) return;
  try {
    const res = await fetch(`${API}/api/documents/${id}`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${token}` }
    });
    if (res.ok) { toast('Documento eliminado'); loadDocs(); }
    else toast('Error al eliminar', 'error');
  } catch (e) { toast('Error de conexión', 'error'); }
}

// ── Upload drag & drop ────────────────────────────────────
function handleDragOver(e) {
  e.preventDefault();
  document.getElementById('upload-zone').classList.add('dragover');
}
function handleDragLeave() {
  document.getElementById('upload-zone').classList.remove('dragover');
}
function handleDrop(e) {
  e.preventDefault();
  document.getElementById('upload-zone').classList.remove('dragover');
  uploadFiles(e.dataTransfer.files);
}
function handleFileSelect(e) { uploadFiles(e.target.files); }

async function uploadFiles(files) {
  const area = document.getElementById('upload-area').value;
  const desc = document.getElementById('upload-desc').value;
  const progressWrap = document.getElementById('progress-wrap');
  const progressBar = document.getElementById('progress-bar');
  progressWrap.style.display = 'block';

  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    progressBar.style.width = `${Math.round((i / files.length) * 100)}%`;
    const form = new FormData();
    form.append('file', file);
    form.append('area', area);
    form.append('description', desc);
    try {
      const res = await fetch(`${API}/api/documents/upload`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: form
      });
      const data = await res.json();
      if (res.ok) toast(`✓ ${file.name} indexado`);
      else toast(`Error: ${data.detail}`, 'error');
    } catch (e) {
      toast(`Error subiendo ${file.name}`, 'error');
    }
  }

  progressBar.style.width = '100%';
  setTimeout(() => {
    progressWrap.style.display = 'none';
    progressBar.style.width = '0%';
  }, 800);
  document.getElementById('file-input').value = '';
  loadDocs();
}

// ── Vincular carpeta local ────────────────────────────────
async function scanFolder() {
  const path = document.getElementById('folder-path').value.trim();
  if (!path) return toast('Ingresa una ruta de carpeta', 'error');
  try {
    const res = await fetch(`${API}/api/folder/scan`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify({ folder_path: path })
    });
    const data = await res.json();
    if (!res.ok) { toast(data.detail || 'Ruta no válida', 'error'); return; }
    folderFiles = data.files;
    renderFolderFiles();
    toast(`${data.total} archivos encontrados`);
  } catch (e) { toast('Error al escanear la carpeta', 'error'); }
}

function renderFolderFiles() {
  const result = document.getElementById('folder-result');
  const list = document.getElementById('folder-files-list');

  if (!folderFiles.length) {
    list.innerHTML = '<p style="padding:16px;text-align:center;color:var(--text3);font-size:13px">No se encontraron archivos soportados</p>';
  } else {
    list.innerHTML = folderFiles.map((f, i) => `
      <div style="display:flex;align-items:center;gap:10px;padding:9px 13px;border-bottom:1px solid var(--border)">
        <input type="checkbox" class="folder-file-cb" id="ff-${i}" checked style="width:14px;height:14px;accent-color:var(--accent)">
        ${getFileIcon(f.ext)}
        <span style="flex:1;font-size:13px;font-weight:500;color:var(--text2)">${f.name}</span>
        <span style="font-size:11px;color:var(--text3);font-family:var(--mono)">${f.size} KB</span>
      </div>
    `).join('');
  }
  result.style.display = 'block';
}

function selectAllFolderFiles() {
  document.querySelectorAll('.folder-file-cb').forEach(cb => cb.checked = true);
}

async function importSelected() {
  const selected = [];
  document.querySelectorAll('.folder-file-cb').forEach((cb, i) => {
    if (cb.checked && folderFiles[i]) selected.push(folderFiles[i].path);
  });
  if (!selected.length) return toast('Selecciona al menos un archivo', 'error');

  const area = document.getElementById('folder-area').value;
  toast(`Importando ${selected.length} archivos...`);

  try {
    const res = await fetch(`${API}/api/folder/import`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify({ file_paths: selected, area })
    });
    const data = await res.json();
    const ok = data.results.filter(r => r.status === 'ok').length;
    const err = data.results.filter(r => r.status === 'error').length;
    if (ok > 0) toast(`✓ ${ok} archivos importados${err > 0 ? `, ${err} con error` : ''}`);
    else toast(`${err} archivos con error`, 'error');
    document.getElementById('folder-result').style.display = 'none';
    document.getElementById('folder-path').value = '';
    folderFiles = [];
    loadDocs();
  } catch (e) { toast('Error al importar', 'error'); }
}
