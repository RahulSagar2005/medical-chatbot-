document.addEventListener("DOMContentLoaded", () => {
  setupMedicineStore();
  setupDoctorFilter();
  setupChat();
});

function setupMedicineStore() {
  const searchInput = document.getElementById("medicineSearch");
  const categoryFilter = document.getElementById("categoryFilter");
  const cards = Array.from(document.querySelectorAll(".medicine-card-wrap"));

  function applyFilters() {
    const query = (searchInput?.value || "").trim().toLowerCase();
    const category = categoryFilter?.value || "all";
    cards.forEach((card) => {
      const matchesName = card.dataset.name.includes(query);
      const matchesCategory = category === "all" || card.dataset.category === category;
      card.classList.toggle("d-none", !(matchesName && matchesCategory));
    });
  }

  searchInput?.addEventListener("input", applyFilters);
  categoryFilter?.addEventListener("change", applyFilters);

  document.querySelectorAll(".add-cart-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      button.disabled = true;
      try {
        const response = await fetch("/cart/add", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ medicine_id: button.dataset.id }),
        });
        renderCart(await response.json());
      } finally {
        button.disabled = false;
      }
    });
  });

  if (document.getElementById("cartItems")) {
    fetch("/cart").then((response) => response.json()).then(renderCart).catch(() => {});
  }
}

function renderCart(cart) {
  const itemsEl = document.getElementById("cartItems");
  const totalEl = document.getElementById("cartTotal");
  const countEl = document.getElementById("cartCount");
  if (!itemsEl || !totalEl || !countEl) return;

  countEl.textContent = cart.count || 0;
  totalEl.textContent = `$${Number(cart.total || 0).toFixed(2)}`;

  if (!cart.items || cart.items.length === 0) {
    itemsEl.innerHTML = '<p class="text-muted">Your cart is empty.</p>';
    return;
  }

  itemsEl.innerHTML = cart.items.map((item) => `
    <div class="cart-item">
      <div class="d-flex justify-content-between gap-3">
        <strong>${escapeHtml(item.name)}</strong>
        <span>$${Number(item.subtotal).toFixed(2)}</span>
      </div>
      <small class="text-muted">${escapeHtml(item.category)} · Qty ${item.quantity}</small>
    </div>
  `).join("");
}

function setupDoctorFilter() {
  const filter = document.getElementById("doctorFilter");
  const cards = Array.from(document.querySelectorAll(".doctor-card-wrap"));
  filter?.addEventListener("change", () => {
    cards.forEach((card) => {
      const show = filter.value === "all" || card.dataset.department === filter.value;
      card.classList.toggle("d-none", !show);
    });
  });
}

function setupChat() {
  const welcomeTime = document.getElementById("welcomeTime");
  if (welcomeTime) {
    welcomeTime.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
}

function autoResize(el) {
  el.style.height = "auto";
  el.style.height = `${Math.min(el.scrollHeight, 100)}px`;
}

function handleKey(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
}

function scrollToBottom() {
  const messagesEl = document.getElementById("chatMessages");
  if (messagesEl) messagesEl.scrollTop = messagesEl.scrollHeight;
}

function hideQuickReplies() {
  const quickReplies = document.getElementById("quickReplies");
  if (quickReplies) quickReplies.style.display = "none";
}

function sendChip(button) {
  const inputEl = document.getElementById("userInput");
  if (!inputEl) return;
  inputEl.value = button.textContent;
  sendMessage();
}

function addUserMessage(text) {
  hideQuickReplies();
  const messagesEl = document.getElementById("chatMessages");
  if (!messagesEl) return;
  const div = document.createElement("div");
  div.className = "message user-message";
  div.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  messagesEl.appendChild(div);
  scrollToBottom();
}

function addBotMessage(text) {
  const messagesEl = document.getElementById("chatMessages");
  if (!messagesEl) return;
  const div = document.createElement("div");
  div.className = "message bot-message";
  const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  div.innerHTML = `<div class="bubble">${escapeHtml(text)}<span class="msg-time">${time}</span></div>`;
  messagesEl.appendChild(div);
  scrollToBottom();
}

function addTyping() {
  const messagesEl = document.getElementById("chatMessages");
  if (!messagesEl) return;
  const div = document.createElement("div");
  div.className = "message bot-message";
  div.id = "typingMsg";
  div.innerHTML = '<div class="bubble typing-bubble"><span></span><span></span><span></span></div>';
  messagesEl.appendChild(div);
  scrollToBottom();
}

function removeTyping() {
  document.getElementById("typingMsg")?.remove();
}

async function sendMessage() {
  const inputEl = document.getElementById("userInput");
  const sendBtn = document.getElementById("sendBtn");
  const text = inputEl?.value.trim();
  if (!text) return;

  addUserMessage(text);
  inputEl.value = "";
  inputEl.style.height = "auto";
  if (sendBtn) sendBtn.disabled = true;
  addTyping();

  try {
    const response = await fetch("/get", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ msg: text }),
    });
    const data = await response.json();
    removeTyping();
    addBotMessage(data.response || data.answer || "Sorry, I could not get a response.");
  } catch (error) {
    removeTyping();
    addBotMessage("Sorry, something went wrong. Please try again.");
  }

  if (sendBtn) sendBtn.disabled = false;
  inputEl.focus();
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
