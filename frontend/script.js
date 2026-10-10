// ---- Agent portfolio data ----
// Only agents that are actually LIVE go here, one entry per agent, added the
// same day it deploys (per the incremental-linking approach). Anything not
// yet built shows as a placeholder card instead -- never a fake/broken link.
//
// Agent cards are data-driven: edit agents.json (same folder), not this file.
  // Each entry: { "title", "pattern", "desc", "url" (https GitHub link), "status" (optional) }.
  // Entries without a valid https url render as a non-clickable card.
  const AGENTS_URL = "agents.json";

  function normalizeAgents(raw) {
    if (!Array.isArray(raw)) return [];
    return raw
      .filter((a) => a && typeof a.title === "string" && a.title.trim())
      .map((a) => ({
        title: a.title.trim(),
        pattern: typeof a.pattern === "string" ? a.pattern : "",
        desc: typeof a.desc === "string" ? a.desc : "",
        status: typeof a.status === "string" ? a.status : "",
        // https only: blocks javascript: and other schemes from a bad edit
        url: typeof a.url === "string" && /^https:\/\//i.test(a.url) ? a.url : "",
      }));
  }

  async function loadAgents() {
    try {
      const res = await fetch(AGENTS_URL, { cache: "no-cache" });
      if (!res.ok) return [];
      return normalizeAgents(await res.json());
    } catch (err) {
      return []; // a missing/broken agents.json must never break the page
    }
  }

  const CHASSIS_TECH = [
    "Docker", "Terraform", "Kubernetes (EKS)", "GitHub Actions CI/CD", "Langfuse",
  ];
  
  function addText(parent, tag, className, text) {
    if (!text) return;
    const el = document.createElement(tag);
    el.className = className;
    el.textContent = text; // textContent, never innerHTML: JSON content can't inject markup
    parent.appendChild(el);
  }

  function renderAgents(agents) {
    const list = document.getElementById("agent-list");
    list.innerHTML = "";

    agents.forEach((agent) => {
      const card = document.createElement(agent.url ? "a" : "div");
      card.className = "agent-card" + (agent.url ? "" : " agent-card-static");
      if (agent.url) {
        card.href = agent.url;
        card.target = "_blank";
        card.rel = "noopener noreferrer";
      }
      addText(card, "p", "agent-card-title", agent.title);
      addText(card, "p", "agent-card-pattern", agent.pattern);
      addText(card, "p", "agent-card-desc", agent.desc);
      if (agent.status) addText(card, "p", "agent-card-status", agent.status);
      list.appendChild(card);
    });

    // Always show one honest placeholder card -- signals active, ongoing
    // work rather than leaving the section looking finished-and-thin.
    const placeholder = document.createElement("div");
    placeholder.className = "agent-card agent-card-placeholder";
    placeholder.textContent = "More agents added weekly";
    list.appendChild(placeholder);
  }
  
  function renderChassisChips() {
    const chipList = document.getElementById("chassis-chips");
    chipList.innerHTML = "";
    CHASSIS_TECH.forEach((tech) => {
      const li = document.createElement("li");
      li.textContent = tech;
      chipList.appendChild(li);
    });
  }
  
  // ---- Chat logic ----
  const chatMessages = document.getElementById("chat-messages");
  const chatForm = document.getElementById("chat-form");
  const chatInput = document.getElementById("chat-input");
  const sendButton = chatForm.querySelector(".chat-send");
  
  // Session ID persisted in localStorage -- the SERVER is now the source of
  // truth for conversation history (stored in S3, via the backend's
  // /chat and /history routes), not the browser. Reusing the same session_id
  // across page reloads means a returning visitor's history can be fetched
  // back from the server instead of starting over each time.
  let sessionId = localStorage.getItem("twin_session_id");
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem("twin_session_id", sessionId);
  }
  
  function appendMessage(text, className) {
    const el = document.createElement("div");
    el.className = `message ${className}`;
    el.textContent = text;
    chatMessages.appendChild(el);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return el;
  }
  
  // On page load, fetch any prior history for this session_id and replay it
  // into the chat UI -- a returning visitor sees their earlier conversation
  // instead of a blank chat, since the server (not the browser) now holds
  // the actual conversation state.
  async function loadHistory() {
    try {
      const response = await fetch(`${CONFIG.API_BASE_URL}/history?session_id=${sessionId}`);
      if (!response.ok) return;  // fail quietly -- worst case, chat just starts blank
      const data = await response.json();
      for (const turn of data.history || []) {
        appendMessage(turn.content, turn.role === "user" ? "message-user" : "message-bot");
      }
    } catch (err) {
      // Network hiccup on page load shouldn't block the chat from being usable --
      // just starts blank, same as a first-time visitor.
    }
  }
  
  function showError(el, message) {
    el.textContent = message;
    el.className = "message message-error";
  }

  // Rate limited -- by the backend's per-visitor/daily limits (JSON body with
  // a friendly message) or by API Gateway's own throttle (generic body).
  // Either way, tell the visitor plainly instead of showing "something went wrong".
  async function showRateLimited(response, el) {
    let message = "You're sending messages quickly -- give me a moment and try again.";
    try {
      const limited = await response.json();
      if (limited.error) message = limited.error;
    } catch (e) { /* keep default message */ }
    showError(el, message);
  }

  // Streaming path. Resolves true when the request was fully handled (a reply,
  // or a definitive error already shown) and false when the caller should fall
  // back to the plain /chat endpoint -- i.e. streaming itself looks broken or
  // unreachable and NOTHING has been shown to the visitor yet.
  async function streamReply(question, el) {
    let response;
    try {
      response = await fetch(`${CONFIG.STREAM_URL}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: question, session_id: sessionId }),
      });
    } catch (err) {
      return false; // network / CORS / DNS failure before any response
    }

    if (response.status === 429) { await showRateLimited(response, el); return true; }
    if (response.status === 400) {
      showError(el, "I couldn't process that message. Please try rephrasing it.");
      return true;
    }
    if (!response.ok || !response.body) return false; // 403/404/5xx: streaming is down

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let text = "";
    let finished = false;

    function handle(ev) {
      if (ev.type === "delta") {
        text += ev.text;
        el.className = "message message-bot";
        el.textContent = text;
      } else if (ev.type === "replace") {
        // Output guardrail tripped AFTER the text streamed: swap the whole message.
        text = ev.text;
        el.className = "message message-bot";
        el.textContent = text;
      } else if (ev.type === "done") {
        finished = true;
      } else if (ev.type === "error") {
        finished = "error";
      }
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let nl;
        while ((nl = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, nl).trim();
          buffer = buffer.slice(nl + 1);
          if (line) { try { handle(JSON.parse(line)); } catch (e) { /* skip a malformed line */ } }
        }
      }
    } catch (err) {
      // connection dropped mid-stream; handled below
    }

    if (finished === true) return true;
    if (!text) return false; // nothing shown yet: safe to retry through /chat
    // Some text was already shown, then the stream broke: keep it, say so.
    el.textContent = text + "\n\n[The reply was interrupted -- please ask again.]";
    return true;
  }

  // Plain (non-streaming) path -- the original behaviour, and the fallback.
  async function plainReply(question, el) {
    // No history sent here anymore -- the backend loads it from S3 using
    // session_id alone. Smaller request payloads, and the client and
    // server can never drift out of sync on what "the conversation" is.
    const response = await fetch(`${CONFIG.API_BASE_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question, session_id: sessionId }),
    });

    if (response.status === 429) { await showRateLimited(response, el); return; }
    if (!response.ok) throw new Error(`Server responded ${response.status}`);

    const data = await response.json();
    el.textContent = data.reply;
    el.className = "message message-bot";
  }

  async function sendMessage(question) {
    appendMessage(question, "message-user");
    const pendingEl = appendMessage("Thinking…", "message-bot message-pending");

    sendButton.disabled = true;
    chatInput.disabled = true;

    try {
      let handled = false;
      if (CONFIG.STREAM_URL) handled = await streamReply(question, pendingEl);
      if (!handled) await plainReply(question, pendingEl);
    } catch (err) {
      showError(pendingEl, "Something went wrong reaching my backend — please try again in a moment.");
    } finally {
      sendButton.disabled = false;
      chatInput.disabled = false;
      chatInput.focus();
    }
  }

  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const question = chatInput.value.trim();
    if (!question) return;
    chatInput.value = "";
    sendMessage(question);
  });
  
  // ---- Resume link ----
  const resumeLink = document.getElementById("resume-link");
  resumeLink.href = CONFIG.RESUME_URL;
  resumeLink.download = "Anmol_Bhargava_Resume.pdf";
  
  // ---- Init ----
  renderAgents([]); // placeholder immediately; real cards replace it when agents.json loads
  loadAgents().then(renderAgents);
  renderChassisChips();
  loadHistory();