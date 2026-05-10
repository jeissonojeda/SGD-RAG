// ═══════════════════════════════════════════════
// chat.js — Asistente IA y mensajes
// ═══════════════════════════════════════════════

function handleChatKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function quickAsk(text) {
  document.getElementById('chat-input').value = text;
  sendMessage();
}

async function sendMessage() {
  const input = document.getElementById('chat-input');
  const question = input.value.trim();
  if (!question) return;

  const sendBtn = document.getElementById('send-btn');
  sendBtn.disabled = true;
  input.value = '';
  input.style.height = 'auto';

  // Ocultar pantalla vacía
  const empty = document.getElementById('chat-empty');
  if (empty) empty.style.display = 'none';

  appendMessage('user', question);

  const typingId = 'typing-' + Date.now();
  appendTyping(typingId);

  try {
    const res = await fetch(`${API}/api/chat`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      // doc_ids: null para buscar en TODOS los documentos automáticamente
      body: JSON.stringify({ question, doc_ids: null })
    });
    const data = await res.json();
    removeTyping(typingId);
    if (res.ok) {
      appendMessage('ai', data.answer, data.sources);
    } else {
      appendMessage('ai', '⚠️ Error: ' + (data.detail || 'No se pudo obtener respuesta'));
    }
  } catch (e) {
    removeTyping(typingId);
    appendMessage('ai', '⚠️ No se pudo conectar al servidor. Verifica que el servidor esté corriendo.');
  }

  sendBtn.disabled = false;
  input.focus();
}

function appendMessage(role, text, sources = []) {
  const container = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = `message ${role}`;

  const avatar = role === 'user'
    ? (currentUser?.username[0]?.toUpperCase() || 'U')
    : 'AI';

  const sourcesHtml = sources?.length ? `
    <div class="msg-sources">
      ${sources.map(s => `
        <span class="source-tag" title="${s.preview || ''}">
          📄 ${s.filename}${s.page ? ` · p.${s.page + 1}` : ''}
        </span>
      `).join('')}
    </div>
  ` : '';

  // Formato básico: negritas y saltos de línea
  const formatted = text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/\n/g, '<br>');

  div.innerHTML = `
    <div class="msg-avatar">${avatar}</div>
    <div class="msg-content">
      <div class="msg-bubble">${formatted}</div>
      ${sourcesHtml}
      <div class="msg-time">${now()}</div>
    </div>
  `;

  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function appendTyping(id) {
  const container = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = 'message ai';
  div.id = id;
  div.innerHTML = `
    <div class="msg-avatar">AI</div>
    <div class="msg-content">
      <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    </div>
  `;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function removeTyping(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}
