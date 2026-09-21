// ---- Agent portfolio data ----
// Only agents that are actually LIVE go here, one entry per agent, added the
// same day it deploys (per the incremental-linking approach). Anything not
// yet built shows as a placeholder card instead -- never a fake/broken link.
const AGENTS = [
    // Populated as Agents 25, 26, 19 go live (Sep 5-6). Example shape:
    // { title: "GraphRAG Brand Knowledge", pattern: "GraphRAG (Neo4j-based knowledge graph)", desc: "Cross-campaign performance Q&A over a brand knowledge graph.", url: "https://..." },
  ];
  
  const CHASSIS_TECH = [
    "Docker", "Terraform", "Kubernetes (EKS)", "GitHub Actions CI/CD", "Langfuse",
  ];
  
  function renderAgents() {
    const list = document.getElementById("agent-list");
    list.innerHTML = "";
  
    AGENTS.forEach((agent) => {
      const card = document.createElement("a");
      card.className = "agent-card";
      card.href = agent.url;
      card.target = "_blank";
      card.rel = "noopener noreferrer";
      card.innerHTML = `
        <p class="agent-card-title">${agent.title}</p>
        <p class="agent-card-pattern">${agent.pattern}</p>
        <p class="agent-card-desc">${agent.desc}</p>
      `;
      list.appendChild(card);
    });
  
    // Always show one honest placeholder card if there's room for more --
    // signals active, ongoing work rather than leaving the section looking finished-and-thin.
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
  
  async function sendMessage(question) {
    appendMessage(question, "message-user");
    const pendingEl = appendMessage("Thinking…", "message-bot message-pending");
  
    sendButton.disabled = true;
    chatInput.disabled = true;
  
    try {
      // No history sent here anymore -- the backend loads it from S3 using
      // session_id alone. Smaller request payloads, and the client and
      // server can never drift out of sync on what "the conversation" is.
      const response = await fetch(`${CONFIG.API_BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: question, session_id: sessionId }),
      });
  
      if (!response.ok) {
        throw new Error(`Server responded ${response.status}`);
      }
  
      const data = await response.json();
      pendingEl.textContent = data.reply;
      pendingEl.className = "message message-bot";
    } catch (err) {
      pendingEl.textContent = "Something went wrong reaching my backend — please try again in a moment.";
      pendingEl.className = "message message-error";
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
  renderAgents();
  renderChassisChips();
  loadHistory();