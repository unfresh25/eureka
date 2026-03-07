/**
 * Eureka Legal Agent — Chat Frontend
 * Talks to the FastAPI backend at /api/v1/
 */

const API_BASE = "http://localhost:8000/api/v1";

// ── State ─────────────────────────────────────────────────────────────────

let conversationId = null;
let isLoading = false;

// ── DOM refs ──────────────────────────────────────────────────────────────

const messagesContainer = document.getElementById("messagesContainer");
const welcomeScreen     = document.getElementById("welcomeScreen");
const messageInput      = document.getElementById("messageInput");
const sendBtn           = document.getElementById("sendBtn");
const charCount         = document.getElementById("charCount");
const thinkingBar       = document.getElementById("thinkingBar");
const thinkingText      = document.getElementById("thinkingText");
const conversationsList = document.getElementById("conversationsList");
const btnNewChat        = document.getElementById("btnNewChat");
const sidebarToggle     = document.getElementById("sidebarToggle");
const sidebar           = document.getElementById("sidebar");
const chatHeaderTitle   = document.getElementById("chatHeaderTitle");

// ── API Client ────────────────────────────────────────────────────────────

async function sendMessage(message) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, conversation_id: conversationId }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.message || `HTTP ${res.status}`);
  }
  return res.json();
}

async function fetchConversations() {
  const res = await fetch(`${API_BASE}/conversations?page_size=30`);
  if (!res.ok) return [];
  const data = await res.json();
  return data.items || [];
}

// ── Thinking indicator ────────────────────────────────────────────────────

const THINKING_STATES = {
  default:           "🔍 Analizando...",
  consulta:          "💬 Consultando base de conocimiento...",
  generar_documento: "📄 Generando documento legal...",
  clarificar:        "🤔 Elaborando respuesta...",
};

function showThinking(intent = "default") {
  thinkingText.textContent = THINKING_STATES[intent] || THINKING_STATES.default;
  thinkingBar.style.display = "flex";
  thinkingBar.style.animation = "fadeIn 0.2s ease";
}

function hideThinking() {
  thinkingBar.style.display = "none";
}

// ── Message rendering ─────────────────────────────────────────────────────

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderMarkdown(text) {
  // Minimal markdown: bold, italic, inline code, line breaks
  return escapeHtml(text)
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/`(.*?)`/g, "<code>$1</code>")
    .replace(/\n/g, "<br>");
}

function intentBadgeHTML(intent) {
  const labels = {
    consulta:          { label: "Consulta", icon: "❓" },
    generar_documento: { label: "Documento generado", icon: "📄" },
    clarificar:        { label: "Información requerida", icon: "💡" },
  };
  const info = labels[intent] || labels.consulta;
  return `<div class="intent-badge ${intent}">${info.icon} ${info.label}</div>`;
}

function sourcesHTML(sources) {
  if (!sources || sources.length === 0) return "";
  const chips = sources.map(s =>
    `<div class="source-chip">
      <span class="src-label">📋 ${escapeHtml(s.title)}</span>
      <span class="src-text">${escapeHtml(s.excerpt)}</span>
    </div>`
  ).join("");
  return `<div class="message-sources">${chips}</div>`;
}

function documentCardHTML(doc) {
  if (!doc) return "";
  return `
    <div class="doc-download-card">
      <div class="doc-icon">📄</div>
      <div class="doc-info">
        <div class="doc-type">${escapeHtml(doc.doc_type)}</div>
        <div class="doc-name">Documento listo para descargar</div>
      </div>
      <a class="doc-download-btn" href="${API_BASE}/documents/${doc.id}/download" download>
        ⬇ Descargar .docx
      </a>
    </div>`;
}

function appendMessage(role, content, { intent, sources, document } = {}) {
  // Hide welcome screen on first message
  if (welcomeScreen) welcomeScreen.style.display = "none";

  const msg = document.createElement("div");
  msg.className = `message ${role}`;

  const avatarIcon = role === "user" ? "👤" : "⚖️";

  let bodyContent = "";
  if (role === "assistant") {
    bodyContent += intent ? intentBadgeHTML(intent) : "";
    bodyContent += `<div class="message-bubble">${renderMarkdown(content)}</div>`;
    bodyContent += sourcesHTML(sources);
    bodyContent += documentCardHTML(document);
  } else {
    bodyContent = `<div class="message-bubble">${renderMarkdown(content)}</div>`;
  }

  msg.innerHTML = `
    <div class="message-avatar">${avatarIcon}</div>
    <div class="message-body">${bodyContent}</div>
  `;

  messagesContainer.appendChild(msg);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// ── Sending ───────────────────────────────────────────────────────────────

async function handleSend() {
  const text = messageInput.value.trim();
  if (!text || isLoading) return;

  isLoading = true;
  sendBtn.disabled = true;
  messageInput.value = "";
  charCount.textContent = "0 / 4000";
  autoResizeTextarea();

  appendMessage("user", text);
  showThinking();

  try {
    const data = await sendMessage(text);

    // Update conversation state
    conversationId = data.conversation_id;
    chatHeaderTitle.textContent = text.slice(0, 50) + (text.length > 50 ? "…" : "");

    // Update thinking to match detected intent
    showThinking(data.intent);

    appendMessage("assistant", data.response, {
      intent:   data.intent,
      sources:  data.sources,
      document: data.document,
    });

    // Refresh sidebar
    loadConversations();
  } catch (err) {
    appendMessage("assistant", `⚠️ Error: ${err.message}. Por favor inténtelo de nuevo.`);
  } finally {
    hideThinking();
    isLoading = false;
    sendBtn.disabled = false;
    messageInput.focus();
  }
}

// ── Conversations sidebar ─────────────────────────────────────────────────

async function loadConversations() {
  try {
    const convs = await fetchConversations();
    if (convs.length === 0) {
      conversationsList.innerHTML = `<div class="empty-conversations">No hay conversaciones aún</div>`;
      return;
    }
    conversationsList.innerHTML = convs.map(c => `
      <div class="conversation-item ${c.id === conversationId ? "active" : ""}"
           data-id="${c.id}">
        <span class="conv-icon">💬</span>
        <span class="conv-title">${escapeHtml(c.title || "Conversación")}</span>
      </div>
    `).join("");

    // Click to switch conversation
    conversationsList.querySelectorAll(".conversation-item").forEach(el => {
      el.addEventListener("click", () => switchConversation(el.dataset.id));
    });
  } catch (_) {
    // Sidebar failure is non-critical
  }
}

function switchConversation(id) {
  conversationId = id;
  // Clear messages and reload
  messagesContainer.innerHTML = `
    <div class="message assistant" style="animation:fadeIn 0.3s ease">
      <div class="message-avatar">⚖️</div>
      <div class="message-body">
        <div class="message-bubble">Conversación cargada. ¿En qué le puedo ayudar?</div>
      </div>
    </div>
  `;
  chatHeaderTitle.textContent = "Conversación";
  // Update active state
  conversationsList.querySelectorAll(".conversation-item").forEach(el => {
    el.classList.toggle("active", el.dataset.id === id);
  });
}

function startNewChat() {
  conversationId = null;
  messagesContainer.innerHTML = `
    <div class="welcome-screen" id="welcomeScreen">
      <div class="welcome-icon">⚖️</div>
      <h1 class="welcome-title">Asistente Legal Eureka</h1>
      <p class="welcome-subtitle">Consulte sus dudas legales o solicite la generación de documentos jurídicos personalizados.</p>
    </div>
  `;
  chatHeaderTitle.textContent = "Nueva conversación";
  messageInput.focus();
  conversationsList.querySelectorAll(".conversation-item").forEach(el => el.classList.remove("active"));
}

// ── Textarea auto-resize ──────────────────────────────────────────────────

function autoResizeTextarea() {
  messageInput.style.height = "auto";
  messageInput.style.height = Math.min(messageInput.scrollHeight, 140) + "px";
}

// ── Event listeners ───────────────────────────────────────────────────────

messageInput.addEventListener("input", () => {
  const len = messageInput.value.length;
  charCount.textContent = `${len} / 4000`;
  sendBtn.disabled = len === 0 || isLoading;
  autoResizeTextarea();
});

messageInput.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
});

sendBtn.addEventListener("click", handleSend);
btnNewChat.addEventListener("click", startNewChat);

sidebarToggle.addEventListener("click", () => {
  sidebar.classList.toggle("collapsed");
  sidebar.classList.toggle("mobile-open"); // mobile
});

// Quick action buttons
document.querySelectorAll(".quick-action-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const prompt = btn.dataset.prompt;
    messageInput.value = prompt;
    charCount.textContent = `${prompt.length} / 4000`;
    sendBtn.disabled = false;
    autoResizeTextarea();
    handleSend();
  });
});

// ── Init ──────────────────────────────────────────────────────────────────

loadConversations();
messageInput.focus();
