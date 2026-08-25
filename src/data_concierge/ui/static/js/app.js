// Data Concierge - Main Application JavaScript

const API_BASE = '/api/v1';
const VERIFIED_SIMILARITY_THRESHOLD = 0.5;  // Use verified notebook if >= 50% match
const VERIFIED_SUGGESTION_THRESHOLD = 0.4;  // Show "similar notebook exists" if >= 40% match

// =============================================================================
// Typed Standards independent verifier (https://typedstandards.org)
// -----------------------------------------------------------------------------
// The label reads "Typed Standards verified" with a check badge. It links out
// to typedstandards.org, which re-runs the checks in the reader's browser, so
// the claim stays independently confirmable rather than a blind assertion.
// =============================================================================
// Render the "Typed Standards verified" button. The href MUST
// be the served commitment URL (`evidence_verify_url`, the same link the admin
// panel uses) — the verifier fetches it in the reader's browser. A GitHub-raw
// URL does NOT work here: the verifier rejects raw.githubusercontent.com's
// text/plain content-type, which is why the chat button previously failed while
// the admin one (commitment URL) worked.
function typedStandardsVerifyButton(verifyHref, extraClass = '') {
    return `<a href="${escapeHtml(verifyHref)}" target="_blank" rel="noopener"
               class="btn btn-sm btn-outline-secondary ${extraClass}"
               title="Independently verify this evidence at typedstandards.org — checks run in your browser">
                <i class="bi bi-patch-check-fill me-1"></i>Typed Standards verified
            </a>`;
}

// State
let chats = {};
let currentChatId = null;
let currentNotebook = null;
let isAuthenticated = false;
let _pendingQuery = null;  // Query waiting for login to complete
let _auth0Enabled = false;

// =============================================================================
// Theme Toggle
// =============================================================================

function toggleTheme() {
    const html = document.documentElement;
    const current = html.getAttribute('data-bs-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-bs-theme', next);
    localStorage.setItem('dc-theme', next);
    updateThemeIcon(next);
    updateHljsTheme(next);
    updateLogos(next);
}

function updateThemeIcon(theme) {
    const btn = document.getElementById('themeToggle');
    const icon = btn ? btn.querySelector('i') : null;
    if (icon) icon.className = theme === 'dark' ? 'bi bi-sun-fill' : 'bi bi-moon-stars';
    if (btn) btn.setAttribute('aria-pressed', theme === 'dark' ? 'true' : 'false');
}

function updateHljsTheme(theme) {
    const dark = document.getElementById('hljs-theme-dark');
    const light = document.getElementById('hljs-theme-light');
    if (dark && light) {
        dark.disabled = theme === 'light';
        light.disabled = theme === 'dark';
    }
}

function updateLogos(theme) {
    const src = theme === 'light' ? '/static/images/logo-light.svg' : '/static/images/logo.png';
    document.querySelectorAll('img.dc-logo').forEach(img => { img.src = src; });
}

// Apply saved theme on load (light is the default)
(function initTheme() {
    const saved = localStorage.getItem('dc-theme') || 'light';
    document.documentElement.setAttribute('data-bs-theme', saved);
    // Defer icon/hljs update until DOM ready
    document.addEventListener('DOMContentLoaded', () => {
        updateThemeIcon(saved);
        updateHljsTheme(saved);
        updateLogos(saved);
    });
})();

// =============================================================================
// Sidebar (collapse / mobile drawer)
// =============================================================================

function isMobileViewport() {
    return window.matchMedia('(max-width: 768px)').matches;
}

function applySidebar(collapsed) {
    document.body.classList.toggle('sidebar-collapsed', collapsed);
    const btn = document.getElementById('sidebarToggle');
    if (btn) btn.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
    // Pull the off-screen sidebar out of the tab order + a11y tree when collapsed,
    // so keyboard / screen-reader users don't land on hidden controls.
    const sidebar = document.querySelector('.sidebar');
    if (sidebar) {
        sidebar.inert = collapsed;
        sidebar.setAttribute('aria-hidden', collapsed ? 'true' : 'false');
    }
}

function toggleSidebar() {
    const collapsed = !document.body.classList.contains('sidebar-collapsed');
    applySidebar(collapsed);
    // Persist desktop preference only; mobile defaults to closed each load.
    if (!isMobileViewport()) {
        localStorage.setItem('dc-sidebar', collapsed ? 'collapsed' : 'open');
    }
}

// Close the drawer after navigation on small screens.
function closeSidebarMobile() {
    if (isMobileViewport()) applySidebar(true);
}

function initSidebar() {
    const saved = localStorage.getItem('dc-sidebar');
    // Mobile: always start collapsed. Desktop: honor saved preference (default open).
    const collapsed = isMobileViewport() ? true : saved === 'collapsed';
    applySidebar(collapsed);
}

// =============================================================================
// Conversation search + de-duplication
// =============================================================================

let _chatSearchTerm = '';

// Signature of "the same conversation": title + its full (role, content)
// sequence. Only EXACT duplicates collapse, so two different conversations that
// merely open with the same question are never merged. Matches the server's
// notion (gateway/chats.py _chat_signature). Empty/in-progress chats (no user
// message yet) get no signature and are never merged.
function chatSignature(chat) {
    const messages = chat.messages || [];
    if (!messages.some(m => m.role === 'user' && m.content)) return '';
    const parts = messages.map(m => (m.role || '') + '\n' + (m.content || ''));
    return ((chat.title || '') + '\n--\n' + parts.join('\n--\n')).trim();
}

// Prune duplicate conversations locally, keeping the richest representative per
// signature (most messages, then most recent). The active / deep-linked chat is
// never removed (so a deep link can't delete its own target), and the longest
// conversation always wins (so an answered chat is never dropped for a thin
// retry). This is a local-only tidy-up; the server de-duplicates authoritatively
// on GET /chats (see gateway/chats.py dedupe_user_chats).
function dedupeChats({ protectedId = null } = {}) {
    const protect = protectedId || currentChatId || location.hash.slice(1) || null;

    // Group by signature (skip empty / in-progress chats with no signature).
    const groups = new Map();
    for (const chat of Object.values(chats)) {
        const sig = chatSignature(chat);
        if (!sig) continue;
        if (!groups.has(sig)) groups.set(sig, []);
        groups.get(sig).push(chat);
    }

    const removeIds = [];
    for (const group of groups.values()) {
        if (group.length < 2) continue;
        group.sort((a, b) => {
            const ma = (a.messages || []).length, mb = (b.messages || []).length;
            if (mb !== ma) return mb - ma;          // most messages wins
            return new Date(b.createdAt) - new Date(a.createdAt);
        });
        for (let i = 1; i < group.length; i++) {
            if (group[i].id === protect) continue;  // never remove the active/deep-linked chat
            removeIds.push(group[i].id);
        }
    }

    // Local-only prune (keeps the sidebar tidy immediately + handles the
    // signed-out case). The server collapses duplicates authoritatively on the
    // next GET /chats, so we never delete server-side from the client here.
    if (removeIds.length) {
        removeIds.forEach(id => { delete chats[id]; });
        saveChatsToStorage();
    }
    return removeIds.length;
}

// DOM Elements
const landingPage = document.getElementById('landingPage');
const chatInterface = document.getElementById('chatInterface');
const chatList = document.getElementById('chatList');
const chatMessages = document.getElementById('chatMessages');
const chatTitle = document.getElementById('chatTitle');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const newChatBtn = document.getElementById('newChatBtn');
const landingSearchInput = document.getElementById('landingSearchInput');
const landingSearchBtn = document.getElementById('landingSearchBtn');

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    loadChatsFromStorage();
    dedupeChats();
    initSidebar();
    setupEventListeners();
    // Surface popular verified questions on the landing page (fail-safe).
    loadVerifiedSuggestions();
    // checkAuthStatus also loads server chats when authenticated
    await checkAuthStatus();

    // Restore from URL hash after all chats (local + server) are loaded
    const initialChatId = location.hash.slice(1);
    if (initialChatId && chats[initialChatId]) {
        currentChatId = initialChatId;
        renderChatList();
        showChatInterface();
    } else {
        renderChatList();
    }

    // Deep link: /?ask=<question> — the Library "Ask this" and share links land
    // here to start (or, if verified, instantly answer) that question.
    const askParam = new URLSearchParams(location.search).get('ask');
    if (askParam && askParam.trim()) {
        history.replaceState(null, '', location.pathname);  // clean the URL
        createNewChat(askParam.trim());
    }

    // Handle browser back/forward and direct hash links
    window.addEventListener('hashchange', () => {
        const chatId = location.hash.slice(1);
        if (chatId && chats[chatId]) {
            currentChatId = chatId;
            renderChatList();
            showChatInterface();
            renderMessages();
        } else if (!chatId) {
            currentChatId = null;
            renderChatList();
            landingPage.classList.remove('d-none');
            chatInterface.classList.add('d-none');
            toggleFooter(true);
        }
    });
});

function setupEventListeners() {
    // New chat button — return to the landing screen (search box + example
    // questions + verified suggestions) rather than dropping into an empty chat.
    // The actual chat is created lazily once the user asks a question.
    newChatBtn.addEventListener('click', () => {
        showLandingPage();
        closeSidebarMobile();
    });

    // Send message — Enter sends, Shift+Enter inserts a newline.
    // Username/password sign-in (the only method when social login is off)
    const passwordLoginForm = document.getElementById('passwordLoginForm');
    if (passwordLoginForm) passwordLoginForm.addEventListener('submit', submitPasswordLogin);

    sendBtn.addEventListener('click', sendMessage);
    chatInput.addEventListener('keydown', (e) => {
        // Don't intercept the Enter that confirms an IME composition candidate
        // (CJK / accented input) — that would send half-composed text.
        if (e.isComposing || e.keyCode === 229) return;
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    chatInput.addEventListener('input', autoGrowInput);

    // Scroll-to-latest button visibility + click
    const scrollBtn = document.getElementById('scrollBottomBtn');
    if (scrollBtn) scrollBtn.addEventListener('click', () => scrollMessagesToBottom(true));
    chatMessages.addEventListener('scroll', updateScrollButton);

    // Delegated handler for message-embedded controls (confidence toggle,
    // feedback thumbs, "people also asked" chips). Delegation avoids per-render
    // rebinding and keeps user text out of inline onclick attributes.
    chatMessages.addEventListener('click', (e) => {
        const el = e.target.closest('[data-action]');
        if (!el || !chatMessages.contains(el)) return;
        const action = el.dataset.action;
        if (action === 'toggle-confidence') {
            toggleConfidence(parseInt(el.dataset.index, 10), el);
        } else if (action === 'feedback') {
            submitFeedback(parseInt(el.dataset.index, 10), el.dataset.rating, el);
        } else if (action === 'ask-verified') {
            if (el.dataset.query) createNewChat(el.dataset.query);
        } else if (action === 'share') {
            shareVerified(el.dataset.id, el);
        }
    });

    // Press "/" anywhere to jump into the message box (unless already typing,
    // editing rich text, or a modal is open).
    document.addEventListener('keydown', (e) => {
        if (e.key !== '/' || e.metaKey || e.ctrlKey || e.altKey) return;
        const el = document.activeElement;
        const tag = (el && el.tagName) || '';
        if (tag === 'INPUT' || tag === 'TEXTAREA' || (el && el.isContentEditable)) return;
        if (document.querySelector('.modal.show')) return;
        const target = chatInterface.classList.contains('d-none') ? landingSearchInput : chatInput;
        if (target) { e.preventDefault(); target.focus(); }
    });

    // Cmd/Ctrl+K opens the command palette (works even while typing).
    document.addEventListener('keydown', (e) => {
        if ((e.metaKey || e.ctrlKey) && (e.key === 'k' || e.key === 'K')) {
            e.preventDefault();
            openCommandPalette();
        }
    });
    const cmdkInput = document.getElementById('cmdkInput');
    if (cmdkInput) {
        cmdkInput.addEventListener('input', (e) => renderCommandResults(e.target.value));
        cmdkInput.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowDown') {
                e.preventDefault();
                _cmdkActiveIndex = Math.min(_cmdkActiveIndex + 1, _cmdkItems.length - 1);
                updateCmdkActive();
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                _cmdkActiveIndex = Math.max(_cmdkActiveIndex - 1, 0);
                updateCmdkActive();
            } else if (e.key === 'Enter') {
                e.preventDefault();
                runCmdk(_cmdkActiveIndex);
            }
        });
    }

    // Press "?" anywhere (outside inputs/modals) to open the shortcuts overlay.
    document.addEventListener('keydown', (e) => {
        if (e.key !== '?' || e.metaKey || e.ctrlKey || e.altKey) return;
        const el = document.activeElement;
        const tag = (el && el.tagName) || '';
        if (tag === 'INPUT' || tag === 'TEXTAREA' || (el && el.isContentEditable)) return;
        if (document.querySelector('.modal.show')) return;
        e.preventDefault();
        showShortcutsModal();
    });

    // Rename the current conversation by double-clicking its header title.
    if (chatTitle) {
        chatTitle.addEventListener('dblclick', startRenameChat);
        // Use the `.editing` class (not the contenteditable value) as the state
        // flag — it's set to "plaintext-only", not "true". commitRenameChat()
        // clears the class first, so the trailing blur after Enter/Escape is a
        // no-op (no double-commit).
        chatTitle.addEventListener('keydown', (e) => {
            if (!chatTitle.classList.contains('editing')) return;
            if (e.key === 'Enter') { e.preventDefault(); commitRenameChat(true); chatTitle.blur(); }
            else if (e.key === 'Escape') { e.preventDefault(); commitRenameChat(false); chatTitle.blur(); }
        });
        chatTitle.addEventListener('blur', () => {
            if (chatTitle.classList.contains('editing')) commitRenameChat(true);
        });
    }

    // Landing search
    landingSearchBtn.addEventListener('click', () => {
        const query = landingSearchInput.value.trim();
        if (query) {
            createNewChat(query);
        }
    });
    landingSearchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const query = landingSearchInput.value.trim();
            if (query) createNewChat(query);
        }
    });

    // Example questions
    document.querySelectorAll('.example-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            createNewChat(btn.textContent.trim());
        });
    });

    // Export the active conversation as a Markdown file.
    const exportBtn = document.getElementById('exportChatBtn');
    if (exportBtn) exportBtn.addEventListener('click', exportCurrentChat);

    // Visible rename affordance (double-clicking the title still works).
    const renameBtn = document.getElementById('renameChatBtn');
    if (renameBtn) renameBtn.addEventListener('click', startRenameChat);

    // Submit for review button
    document.getElementById('submitForReviewBtn').addEventListener('click', submitForReview);

    // Notebook collapse/expand buttons
    document.getElementById('collapseAllBtn').addEventListener('click', () => toggleAllCells(true));
    document.getElementById('expandAllBtn').addEventListener('click', () => toggleAllCells(false));

    // Conversation search
    const chatSearch = document.getElementById('chatSearch');
    if (chatSearch) {
        chatSearch.addEventListener('input', (e) => {
            _chatSearchTerm = e.target.value.trim().toLowerCase();
            renderChatList();
        });
    }

    // Re-evaluate sidebar layout on viewport changes (mobile <-> desktop)
    window.addEventListener('resize', () => {
        if (!isMobileViewport() && localStorage.getItem('dc-sidebar') !== 'collapsed') {
            applySidebar(false);
        }
    });
}

// =============================================================================
// Authentication
// =============================================================================

let _currentUsername = '';
let _isAdmin = false;

async function checkAuthStatus() {
    try {
        const response = await fetch(`${API_BASE}/auth/status`);
        if (response.ok) {
            const data = await response.json();
            isAuthenticated = data.authenticated;
            _auth0Enabled = !!data.auth0_enabled;
            _currentUsername = data.username || '';
            _isAdmin = !!data.is_admin;
        }
    } catch (error) {
        isAuthenticated = false;
    }
    updateAuthUI();
    updateAuth0Button();
    if (isAuthenticated) {
        await loadChatsFromServer();
    }
}

function updateAuth0Button() {
    // Show/hide the whole social-login button group (GitHub + LinkedIn).
    const group = document.getElementById('auth0Buttons');
    if (group) group.classList.toggle('d-none', !_auth0Enabled);
    // The "or" divider only makes sense when both methods are offered.
    const divider = document.getElementById('loginDivider');
    if (divider) divider.classList.toggle('d-none', !_auth0Enabled);
}

// Username/password sign-in. Always available: a self-hosted or local install
// has no social provider configured, and this is the only way in.
async function submitPasswordLogin(event) {
    if (event) event.preventDefault();
    const btn = document.getElementById('passwordLoginBtn');
    const errorEl = document.getElementById('loginError');
    const username = (document.getElementById('loginUsername') || {}).value || '';
    const password = (document.getElementById('loginPassword') || {}).value || '';
    if (errorEl) errorEl.classList.add('d-none');
    if (btn) { btn.disabled = true; btn.classList.add('disabled'); }
    try {
        const response = await fetch(`${API_BASE}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ username, password })
        });
        if (!response.ok) {
            let detail = 'Sign in failed. Check your username and password.';
            try { const d = await response.json(); if (d.detail) detail = d.detail; } catch (e) { /* keep default */ }
            throw new Error(detail);
        }
        // Cookie is set; reload so every view picks up the authenticated session.
        window.location.reload();
    } catch (error) {
        if (errorEl) {
            errorEl.textContent = error.message;
            errorEl.classList.remove('d-none');
        }
        if (btn) { btn.disabled = false; btn.classList.remove('disabled'); }
    }
}

// Start the Auth0 flow. Passing a connection ("github" / "linkedin") sends the
// user straight to that provider; omitting it shows Auth0's provider picker.
function loginWithAuth0(connection) {
    const next = new URLSearchParams(location.search).get('next') || '/';
    const conn = connection ? `&connection=${encodeURIComponent(connection)}` : '';
    window.location.href = `${API_BASE}/auth/auth0/login?next=${encodeURIComponent(next)}${conn}`;
}

function updateAuthUI() {
    const loginBtn = document.getElementById('loginBtn');
    const logoutBtn = document.getElementById('logoutBtn');
    const authStatus = document.getElementById('authStatus');
    const adminBtn = document.getElementById('adminNavBtn');

    if (isAuthenticated) {
        loginBtn.classList.add('d-none');
        logoutBtn.classList.remove('d-none');
        authStatus.classList.remove('d-none');
        if (_currentUsername) {
            authStatus.innerHTML = `<i class="bi bi-person-check-fill text-success me-1"></i>${escapeHtml(_currentUsername)}`;
        }
    } else {
        loginBtn.classList.remove('d-none');
        logoutBtn.classList.add('d-none');
        authStatus.classList.add('d-none');
    }
    // Admin shortcut in the navbar — only for signed-in admins.
    if (adminBtn) adminBtn.classList.toggle('d-none', !(isAuthenticated && _isAdmin));
}

function showLoginModal(pendingQuery = null) {
    _pendingQuery = pendingQuery;
    const errorEl = document.getElementById('loginError');
    if (errorEl) errorEl.classList.add('d-none');
    updateAuth0Button();
    const modalEl = document.getElementById('loginModal');
    const modal = new bootstrap.Modal(modalEl);
    modal.show();
    // Focus the social sign-in button once the modal is shown.
    modalEl.addEventListener('shown.bs.modal', () => {
        const btn = document.getElementById('auth0LoginBtn');
        if (btn && _auth0Enabled) { btn.focus(); return; }
        const userField = document.getElementById('loginUsername');
        if (userField) userField.focus();
    }, { once: true });
}


async function doLogout() {
    try {
        await fetch(`${API_BASE}/auth/logout`, { method: 'POST' });
    } catch (error) {
        // Ignore errors on logout
    }
    isAuthenticated = false;
    _currentUsername = '';
    _isAdmin = false;
    updateAuthUI();
    showToast('Logged out.', 'info');
}

// Chat Management
function createNewChat(initialQuery = null) {
    const chatId = generateId();
    chats[chatId] = {
        id: chatId,
        title: 'New Chat',
        messages: [],
        createdAt: new Date().toISOString()
    };

    currentChatId = chatId;
    location.hash = chatId;  // persistent URL
    saveChatsToStorage();
    renderChatList();
    showChatInterface();

    if (initialQuery) {
        addUserMessage(initialQuery);
        processQuery(initialQuery);
    }
}

function selectChat(chatId) {
    currentChatId = chatId;
    location.hash = chatId;  // persistent URL
    renderChatList();
    showChatInterface();
    renderMessages();
    closeSidebarMobile();
}

function deleteChat(chatId) {
    // Deletion is permanent (local + server) — always confirm. The delete icon
    // is also always visible on touch devices, where a stray tap could land.
    const title = (chats[chatId] && chats[chatId].title) || 'this conversation';
    if (!confirm(`Delete "${title}"? This cannot be undone.`)) return;
    delete chats[chatId];
    if (currentChatId === chatId) {
        currentChatId = null;
        showLandingPage();
    }
    saveChatsToStorage();
    renderChatList();
    if (isAuthenticated) {
        deleteChatFromServer(chatId).catch(() => {});
    }
}

function toggleFooter(show) {
    const footer = document.getElementById('siteFooter');
    if (footer) footer.classList.toggle('d-none', !show);
}

function showLandingPage() {
    landingPage.classList.remove('d-none');
    chatInterface.classList.add('d-none');
    toggleFooter(true);
    currentChatId = null;
    history.replaceState(null, '', location.pathname);  // clear hash
    renderChatList();
}

function showChatInterface() {
    landingPage.classList.add('d-none');
    chatInterface.classList.remove('d-none');
    toggleFooter(false);
    renderMessages();

    const chat = chats[currentChatId];
    if (chat) {
        chatTitle.textContent = chat.title || 'New Chat';
    }
}

// Keyboard shortcuts overlay
function showShortcutsModal() {
    const el = document.getElementById('shortcutsModal');
    if (el) new bootstrap.Modal(el).show();
}

// =============================================================================
// Command palette (Cmd/Ctrl+K)
// =============================================================================
let _cmdkActiveIndex = 0;
let _cmdkItems = [];

function cmdkActions() {
    const actions = [
        { icon: 'bi-plus-lg', label: 'New chat', run: () => showLandingPage() },
        { icon: 'bi-patch-check', label: 'Browse the Verified Library', run: () => { window.location.href = '/library'; } },
        { icon: 'bi-circle-half', label: 'Toggle light / dark theme', run: () => toggleTheme() },
        { icon: 'bi-question-circle', label: 'Open the User Guide', run: () => { window.location.href = '/docs'; } },
        { icon: 'bi-keyboard', label: 'Keyboard shortcuts', run: () => showShortcutsModal() },
    ];
    if (_isAdmin) actions.push({ icon: 'bi-speedometer2', label: 'Open the Admin dashboard', run: () => { window.location.href = '/admin'; } });
    return actions;
}

function openCommandPalette() {
    const modalEl = document.getElementById('commandPalette');
    if (!modalEl) return;
    const input = document.getElementById('cmdkInput');
    input.value = '';
    renderCommandResults('');
    bootstrap.Modal.getOrCreateInstance(modalEl).show();
    modalEl.addEventListener('shown.bs.modal', () => input.focus(), { once: true });
}

function renderCommandResults(term) {
    term = (term || '').trim().toLowerCase();
    const results = document.getElementById('cmdkResults');
    _cmdkItems = [];
    let html = '';

    const convos = Object.values(chats)
        .filter(c => !term
            || (c.title || '').toLowerCase().includes(term)
            || (c.messages || []).some(m => (m.content || '').toLowerCase().includes(term)))
        .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
        .slice(0, 6);
    const actions = cmdkActions().filter(a => !term || a.label.toLowerCase().includes(term));

    if (convos.length) {
        html += `<div class="cmdk-section">Conversations</div>`;
        convos.forEach(c => {
            const idx = _cmdkItems.length;
            _cmdkItems.push({ run: () => selectChat(c.id) });
            html += `<button type="button" class="cmdk-item" data-idx="${idx}" role="option">
                <i class="bi bi-chat-left-text" aria-hidden="true"></i>
                <span class="cmdk-label">${escapeHtml(c.title || 'Untitled')}</span></button>`;
        });
    }
    if (actions.length) {
        html += `<div class="cmdk-section">Actions</div>`;
        actions.forEach(a => {
            const idx = _cmdkItems.length;
            _cmdkItems.push({ run: a.run });
            html += `<button type="button" class="cmdk-item" data-idx="${idx}" role="option">
                <i class="bi ${a.icon}" aria-hidden="true"></i>
                <span class="cmdk-label">${escapeHtml(a.label)}</span></button>`;
        });
    }
    if (!_cmdkItems.length) html = `<div class="cmdk-empty">No matches</div>`;
    results.innerHTML = html;
    _cmdkActiveIndex = 0;
    updateCmdkActive();
    results.querySelectorAll('.cmdk-item').forEach(el => {
        el.addEventListener('click', () => runCmdk(parseInt(el.dataset.idx, 10)));
        el.addEventListener('mousemove', () => {
            _cmdkActiveIndex = parseInt(el.dataset.idx, 10);
            updateCmdkActive();
        });
    });
}

function updateCmdkActive() {
    const items = document.querySelectorAll('#cmdkResults .cmdk-item');
    items.forEach((el, i) => el.classList.toggle('active', i === _cmdkActiveIndex));
    const active = items[_cmdkActiveIndex];
    if (active) active.scrollIntoView({ block: 'nearest' });
}

function runCmdk(idx) {
    const item = _cmdkItems[idx];
    if (!item) return;
    bootstrap.Modal.getOrCreateInstance(document.getElementById('commandPalette')).hide();
    // Defer so the modal fully closes before navigation / re-render.
    setTimeout(() => item.run(), 150);
}

// Inline rename of the current conversation (double-click the header title).
function startRenameChat() {
    if (!chats[currentChatId]) return;
    chatTitle.setAttribute('contenteditable', 'plaintext-only');
    chatTitle.classList.add('editing');
    chatTitle.focus();
    const range = document.createRange();
    range.selectNodeContents(chatTitle);
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
}

function commitRenameChat(save) {
    const chat = chats[currentChatId];
    chatTitle.removeAttribute('contenteditable');
    chatTitle.classList.remove('editing');
    if (!chat) return;
    if (save) {
        const next = chatTitle.textContent.trim().slice(0, 80);
        chat.title = next || chat.title || 'New Chat';
        chatTitle.textContent = chat.title;
        saveChatsToStorage();
        renderChatList();
        if (isAuthenticated) syncChatToServer(currentChatId, chat).catch(() => {});
    } else {
        chatTitle.textContent = chat.title || 'New Chat';
    }
}

// Regenerate the latest answer by re-running the most recent user query. The new
// answer is appended below (not replaced) so the user can compare.
let _isRegenerating = false;
function regenerateAnswer() {
    if (_isRegenerating) return;
    const chat = chats[currentChatId];
    if (!chat) return;
    let userQuery = null;
    for (let i = chat.messages.length - 1; i >= 0; i--) {
        if (chat.messages[i].role === 'user' && chat.messages[i].content) {
            userQuery = chat.messages[i].content;
            break;
        }
    }
    if (!userQuery) return;
    _isRegenerating = true;
    Promise.resolve(processQuery(userQuery)).finally(() => { _isRegenerating = false; });
}

// Message Handling
function sendMessage() {
    const message = chatInput.value.trim();
    if (!message || !currentChatId) return;

    chatInput.value = '';
    autoGrowInput();
    addUserMessage(message);
    processQuery(message);
}

// Grow the message textarea with its content, up to a capped height. Also
// publishes the composer height so the jump-to-latest button stays above it.
function autoGrowInput() {
    if (!chatInput) return;
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + 'px';
    const composer = chatInput.closest('.chat-input-container');
    if (composer && chatInterface) {
        chatInterface.style.setProperty('--dc-composer-h', (composer.offsetHeight + 14) + 'px');
    }
}

// Scroll the message list to the newest message.
function scrollMessagesToBottom(smooth = false) {
    if (!chatMessages) return;
    chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: smooth ? 'smooth' : 'auto' });
    // Defer to the next frame so layout (clientHeight) is settled before we
    // decide whether the jump-to-latest button is needed.
    requestAnimationFrame(updateScrollButton);
}

// Show the "jump to latest" button only when the list is scrollable AND the
// user has scrolled away from the bottom.
function updateScrollButton() {
    const btn = document.getElementById('scrollBottomBtn');
    if (!btn || !chatMessages) return;
    const scrollable = chatMessages.scrollHeight - chatMessages.clientHeight;
    const distance = scrollable - chatMessages.scrollTop;
    btn.classList.toggle('d-none', scrollable < 40 || distance < 120);
}

// =============================================================================
// Copy to clipboard (code cells + assistant answers)
// =============================================================================

async function copyToClipboard(text, btn) {
    try {
        await navigator.clipboard.writeText(text);
    } catch {
        // Fallback for non-secure contexts
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); } catch {}
        ta.remove();
    }
    if (btn) {
        const icon = btn.querySelector('i');
        const prev = icon ? icon.className : '';
        if (icon) icon.className = 'bi bi-check2';
        btn.classList.add('copied');
        setTimeout(() => {
            if (icon) icon.className = prev;
            btn.classList.remove('copied');
        }, 1300);
    }
}

// Copy the source of a notebook code cell (called from the cell's copy button).
function copyCodeCell(btn) {
    const cell = btn.closest('.notebook-cell');
    const code = cell && cell.querySelector('.cell-source code');
    if (code) copyToClipboard(code.textContent, btn);
}

// Copy the raw markdown of an assistant answer by message index.
function copyAnswer(index, btn) {
    const chat = chats[currentChatId];
    const msg = chat && chat.messages[index];
    if (msg) copyToClipboard(msg.content || '', btn);
}

// Serialize the active conversation to a self-contained Markdown transcript and
// trigger a download. Pure client-side — no server round-trip. Lets a user keep
// or share an answer (with its sources) outside the app.
function chatToMarkdown(chat) {
    const title = (chat.title || 'Conversation').trim();
    const when = chat.createdAt ? new Date(chat.createdAt) : null;
    const lines = [`# ${title}`, ''];
    lines.push(`*Exported from Verikan — the verified Data Concierge${when && !isNaN(when) ? ' · ' + when.toLocaleString() : ''}*`, '');
    for (const msg of (chat.messages || [])) {
        const role = msg.role === 'user' ? 'You' : 'Verikan';
        const content = (msg.content || '').trim();
        if (!content && !(msg.sourceLinks && msg.sourceLinks.length)) continue;
        lines.push(`## ${role}`, '');
        if (content) lines.push(content, '');
        // Include quick-answer source links if present so citations survive export.
        const links = msg.sourceLinks || (msg.quickAnswer && msg.quickAnswer.source_links) || [];
        if (Array.isArray(links) && links.length) {
            lines.push('**Sources:**', '');
            for (const l of links) {
                const name = (l && (l.name || l.url)) || 'source';
                const url = (l && l.url) || '';
                lines.push(url ? `- [${name}](${url})` : `- ${name}`);
            }
            lines.push('');
        }
    }
    return lines.join('\n');
}

// Populate the landing page's "verified answers" section from the verified
// library (notebooks + quick answers). Fail-safe: any error or empty library
// leaves the section hidden, so the static sample questions remain the
// baseline. The list endpoints are public and read-only (no usage increment),
// so this is safe for anonymous and signed-in users alike.
async function loadVerifiedSuggestions() {
    const wrap = document.getElementById('verifiedSuggestions');
    const grid = document.getElementById('verifiedSuggestionsGrid');
    if (!wrap || !grid) return;

    const norm = (s) => (s || '').trim().toLowerCase().replace(/\s+/g, ' ').replace(/[.?!]+$/, '');
    // Don't repeat questions already shown as static sample buttons.
    const seen = new Set(
        Array.from(document.querySelectorAll('.example-grid .example-btn'))
            .map(b => norm(b.textContent))
    );

    try {
        const [nbRes, ansRes] = await Promise.all([
            fetch(`${API_BASE}/verified-notebooks`).catch(() => null),
            fetch(`${API_BASE}/verified-answers`).catch(() => null),
        ]);
        const items = [];
        if (nbRes && nbRes.ok) {
            const d = await nbRes.json();
            for (const nb of (d.notebooks || [])) {
                if (nb.query) items.push({ q: nb.query, usage: nb.usage_count || 0 });
            }
        }
        if (ansRes && ansRes.ok) {
            const d = await ansRes.json();
            for (const a of (d.answers || [])) {
                if (a.query) items.push({ q: a.query, usage: a.usage_count || 0 });
            }
        }
        // Most-used first; dedupe by normalized question; cap at 5.
        items.sort((a, b) => b.usage - a.usage);
        const picks = [];
        for (const it of items) {
            const key = norm(it.q);
            if (!key || seen.has(key)) continue;
            seen.add(key);
            picks.push(it.q);
            if (picks.length >= 5) break;
        }
        if (!picks.length) return;  // nothing new to show — keep section hidden

        grid.innerHTML = '';
        for (const q of picks) {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'example-btn';
            btn.setAttribute('data-verified', 'true');
            btn.innerHTML = `<i class="bi bi-patch-check text-success me-1" aria-hidden="true"></i>${escapeHtml(q)}`;
            btn.addEventListener('click', () => createNewChat(q));
            grid.appendChild(btn);
        }
        wrap.classList.remove('d-none');
    } catch {
        // Network/parse error — leave the section hidden; static samples remain.
    }
}

function exportCurrentChat() {
    const chat = chats[currentChatId];
    if (!chat || !(chat.messages || []).length) {
        showToast('Nothing to export yet — ask a question first.');
        return;
    }
    const md = chatToMarkdown(chat);
    const slug = (chat.title || 'conversation')
        .toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 60) || 'conversation';
    const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${slug}.md`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function addUserMessage(content) {
    const chat = chats[currentChatId];
    if (!chat) return;

    chat.messages.push({
        role: 'user',
        content: content,
        timestamp: new Date().toISOString()
    });

    // Update title from first message
    if (chat.messages.length === 1) {
        chat.title = content.length > 40 ? content.substring(0, 40) + '...' : content;
        chatTitle.textContent = chat.title;
        renderChatList();
    }

    saveChatsToStorage();
    renderMessages();
}

// Gather the chat context the backend needs to understand follow-ups:
// recent turns (so "what about Ohio?" can be resolved) and the query_id of
// the latest generated notebook (so "fix the chart" can edit it). Verified
// and quick answers carry no editable notebook, so they never become the
// revision target. The just-typed message is excluded — it travels as the
// query itself.
function buildFollowUpContext(query) {
    const chat = chats[currentChatId];
    if (!chat || !Array.isArray(chat.messages)) {
        return { conversation: null, previousQueryId: null };
    }
    let messages = chat.messages;
    const last = messages[messages.length - 1];
    if (last && last.role === 'user' && last.content === query) {
        messages = messages.slice(0, -1);
    }
    const turns = messages
        .filter(m => (m.role === 'user' || m.role === 'assistant') && m.content)
        .slice(-8)
        .map(m => ({ role: m.role, content: String(m.content).substring(0, 4000) }));
    const prev = [...messages].reverse().find(m =>
        m.role === 'assistant' && m.queryId && !m.isVerified && !m.isQuickAnswer
        && m.confidenceLevel !== 'verified' && m.hadNotebook !== false);
    return {
        conversation: turns.length ? turns : null,
        previousQueryId: prev ? prev.queryId : null
    };
}

async function processQuery(query, isFollowUp = false) {
    const followCtx = buildFollowUpContext(query);
    // In an ongoing authenticated chat the server understands the follow-up
    // (rewriting it or editing the notebook), so the client-side verified
    // shortcut would misfire on context-dependent wording — skip it and let
    // the server run its own verified-cache checks on the resolved query.
    const isChatFollowUp = isAuthenticated && followCtx.conversation !== null;

    showTypingIndicator();
    appendThinking(isChatFollowUp
        ? "Reading the conversation so far...\n"
        : "Searching verified notebooks...\n");

    let thinkingInterval = null;
    let lastThinkingLength = 0;

    try {
        // First check for verified notebooks (skipped for chat follow-ups)
        const verifiedResults = isChatFollowUp ? [] : await searchVerifiedNotebooks(query);

        // If we have a high-confidence verified match, use it instead of generating new
        if (verifiedResults.length > 0 && verifiedResults[0].similarity_score >= VERIFIED_SIMILARITY_THRESHOLD) {
            const verifiedMatch = verifiedResults[0];
            const score = Math.round(verifiedMatch.similarity_score * 100);
            appendThinking(`Found a verified answer (${score}% match)! Loading details...\n`);

            // Fetch notebook + log in parallel
            const [verifiedData] = await Promise.all([
                getVerifiedNotebook(verifiedMatch.notebook_id),
                fetch(`${API_BASE}/query-log`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: query,
                        source: 'verified_cache',
                        query_id: verifiedMatch.notebook_id,
                        similarity_score: verifiedMatch.similarity_score,
                        verified_query: verifiedMatch.query,
                        had_notebook: true,
                        notebook_url: `/api/v1/verified-notebooks/${verifiedMatch.notebook_id}`
                    })
                }).catch(e => console.warn('Failed to record verified-cache log', e)),
                // Minimum display time so the user can read the thinking steps
                new Promise(r => setTimeout(r, 1200)),
            ]);

            appendThinking("Done!\n");
            await new Promise(r => setTimeout(r, 400));

            hideTypingIndicator();
            addAssistantMessage({
                answer: verifiedData.answer || verifiedMatch.answer,
                confidence: verifiedMatch.similarity_score,
                notebook: verifiedData.notebook_json,
                query_id: verifiedMatch.notebook_id,
                isVerified: true,
                verifiedQuery: verifiedMatch.query,
                similarityScore: verifiedMatch.similarity_score,
                githubUrl: verifiedData.github_url || null,
                evidenceVerifyUrl: verifiedData.evidence_verify_url || null
            });
            return;
        }

        // No verified match — login required to run a new query
        if (!isAuthenticated) {
            hideTypingIndicator();
            addAssistantMessage({
                answer: 'You need to **log in** to run new queries. Verified notebooks are available without login.',
                confidence: 0
            });
            showLoginModal(query);
            return;
        }

        // Start polling for thinking updates (simulated for now - will show tool calls)
        appendThinking(isChatFollowUp
            ? "Working out whether this updates the previous notebook or needs a fresh analysis...\n\n"
            : "No verified match found. Running a new analysis...\n\n");
        let queryStartTime = Date.now();
        const maxPollTime = 60000; // 60 seconds max

        // Simulated thinking stages for better UX
        const thinkingStages = [
            "Searching for relevant datasets...",
            "Found relevant data, now loading it...",
            "Analyzing the data to answer your question...",
            "Preparing the final answer and notebook..."
        ];
        let currentStage = 0;

        // Show the first stage immediately
        appendThinking(thinkingStages[currentStage] + "\n");
        currentStage++;

        thinkingInterval = setInterval(() => {
            const elapsed = Date.now() - queryStartTime;
            // Show stages every 3 seconds
            if (elapsed > currentStage * 3000 && currentStage < thinkingStages.length) {
                appendThinking(thinkingStages[currentStage] + "\n");
                currentStage++;
            }

            // Stop after max time
            if (elapsed > maxPollTime && thinkingInterval) {
                clearInterval(thinkingInterval);
                thinkingInterval = null;
            }
        }, 500);

        // Always use analyze mode with WPRDC data source
        // The LLM agent handles everything: search, load, analyze, answer
        const queryPayload = {
            query: query,
            include_notebook: true,
            include_visualization: true,
            concierge_mode: 'analyze',
            data_source: 'wprdc'
        };
        if (followCtx.conversation && followCtx.conversation.length > 0) {
            queryPayload.conversation = followCtx.conversation;
            if (followCtx.previousQueryId) {
                queryPayload.previous_query_id = followCtx.previousQueryId;
            }
        }
        const response = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(queryPayload)
        });

        // Clear the thinking interval
        if (thinkingInterval) {
            clearInterval(thinkingInterval);
            thinkingInterval = null;
        }

        if (!response.ok) {
            const err = new Error('Query failed with HTTP ' + response.status);
            err.status = response.status;
            throw err;
        }

        const data = await response.json();

        hideTypingIndicator();

        // Surface server-side suggestions (returned when the analysis failed,
        // timed out, or found no answer) as clickable chips.
        if (Array.isArray(data.suggested_questions) && data.suggested_questions.length > 0) {
            data.suggestedQuestions = data.suggested_questions;
        } else if (data.confidence_level === 'error') {
            data.suggestedQuestions = extractSuggestedQuestions(query, data.answer);
        }

        // Check if the response indicates an escalation/error
        const isEscalation = data.answer && (
            data.answer.includes('human data specialist') ||
            data.answer.includes('specialist queue') ||
            data.answer.includes('escalate this')
        );

        // For recommend mode, add suggested follow-up questions
        // (always using analyze mode with WPRDC, so this is skipped)
        if (false) {
            data.isRecommendation = true;
            data.suggestedQuestions = extractSuggestedQuestions(query, data.answer);
        }

        // If escalation happened, add helpful context and alternative suggestions
        if (isEscalation) {
            data.answer = `I wasn't able to fetch live data for this query (some data sources may require API configuration). However, I can still help!\n\n` +
                `**Here's what you can do:**\n` +
                `- Try a different phrasing or ask about available data sources\n` +
                `- Download a verified notebook if one exists for a similar question\n` +
                `- Ask me to explain what data is available on this topic`;
            data.suggestedQuestions = [
                `What data sources cover this topic?`,
                `Show me what datasets are available`,
                `Explain how to access this data manually`
            ];
        }

        addAssistantMessage(data, verifiedResults);

    } catch (error) {
        // Clear the thinking interval on error
        if (thinkingInterval) {
            clearInterval(thinkingInterval);
            thinkingInterval = null;
        }

        hideTypingIndicator();

        // Fail gracefully: never show a raw error to the user. Pick a friendly
        // explanation for what happened and always recommend alternatives.
        console.error('Query failed:', error);
        const status = error && error.status;
        let friendly;
        if (status === 401 || status === 403) {
            friendly = 'Your session has expired, so I couldn\'t run that question. ' +
                'Please **log in** again and re-ask it.';
        } else if (status === 429) {
            friendly = 'I\'m getting a lot of questions right now and had to pause. ' +
                'Please wait a minute and try again.';
        } else if (status === 502 || status === 503 || status === 504) {
            friendly = 'That question took longer than the server allows, so the analysis was cut short. ' +
                'A more specific question usually works better — try narrowing it to one place, ' +
                'one time period, or one metric.';
        } else if (!status) {
            friendly = 'I couldn\'t reach the server — your connection may have dropped mid-analysis. ' +
                'Please check your internet connection and try again.';
        } else {
            friendly = 'Something went wrong on our side while processing your question, so I ' +
                'couldn\'t finish this one. Please try again in a moment, or try one of the ' +
                'suggestions below.';
        }

        addAssistantMessage({
            answer: friendly,
            confidence: 0,
            suggestedQuestions: extractSuggestedQuestions(query, '')
        });

        if (status === 401 || status === 403) {
            isAuthenticated = false;
            showLoginModal(query);
        }
    }
}

// Extract or generate suggested follow-up questions
function extractSuggestedQuestions(originalQuery, answer) {
    // Generate contextual follow-up questions based on the query
    // Keep questions simple to avoid tier_3 classification (avoid complex comparisons)
    const queryLower = originalQuery.toLowerCase();
    const suggestions = [];

    if (queryLower.includes('gdp')) {
        suggestions.push(
            `What is the current US GDP?`,
            `Show me GDP data for California`,
            `What was the GDP growth last year?`
        );
    } else if (queryLower.includes('unemployment') || queryLower.includes('jobs') || queryLower.includes('employment')) {
        suggestions.push(
            `What is the current national unemployment rate?`,
            `Show me unemployment data for Texas`,
            `What are the latest employment statistics?`
        );
    } else if (queryLower.includes('inflation') || queryLower.includes('cpi') || queryLower.includes('prices')) {
        suggestions.push(
            `What is the current inflation rate?`,
            `Show me CPI data for the past year`,
            `What are current food prices?`
        );
    } else if (queryLower.includes('population') || queryLower.includes('census') || queryLower.includes('demographics')) {
        suggestions.push(
            `What is the US population?`,
            `Show me population data for New York`,
            `What are the latest census statistics?`
        );
    } else if (queryLower.includes('income') || queryLower.includes('wages') || queryLower.includes('salary')) {
        suggestions.push(
            `What is the median household income?`,
            `Show me wage data for California`,
            `What are average salaries by occupation?`
        );
    } else {
        // Generic follow-ups
        suggestions.push(
            `Show me the latest statistics`,
            `Get me specific data on this topic`,
            `What are the current numbers?`
        );
    }

    return suggestions.slice(0, 3);
}

function addAssistantMessage(data, verifiedResults = null) {
    const chat = chats[currentChatId];
    if (!chat) return;

    chat.messages.push({
        role: 'assistant',
        content: data.answer || 'I could not generate a response.',
        timestamp: new Date().toISOString(),
        confidence: data.confidence,
        confidenceBreakdown: data.confidence_breakdown || null,
        confidenceUnavailable: data.confidence_unavailable || null,
        confidenceExplanation: data.confidence_explanation || null,
        notebook: data.notebook,
        queryId: data.query_id,
        verifiedMatches: verifiedResults,
        isVerified: data.isVerified || false,
        verifiedQuery: data.verifiedQuery,
        similarityScore: data.similarityScore,
        githubUrl: data.githubUrl || null,
        evidenceVerifyUrl: data.evidenceVerifyUrl || null,
        isRecommendation: data.isRecommendation || false,
        suggestedQuestions: data.suggestedQuestions || [],
        isQuickAnswer: data.is_quick_answer || false,
        quickAnswer: data.quick_answer || null,
        sourceLinks: data.source_links || [],
        isRevision: data.is_revision || false,
        revisedFromQueryId: data.revised_from_query_id || null,
        // 'verified' marks server-side verified-cache hits, whose queryId has
        // no notebook stored under /notebooks/{id}; hadNotebook=false marks
        // responses (e.g. a declined revision) that shipped no notebook at
        // all. Both must be skipped as revision targets and get no dead
        // "View Notebook" button.
        confidenceLevel: data.confidence_level || null,
        hadNotebook: !!(data.notebook || data.notebook_url)
    });

    saveChatsToStorage();
    renderMessages();

    // The notebook check runs after the answer is returned (#131). Poll for
    // it and refresh the confidence display in place when it lands.
    if (data.verification_pending && data.query_id) {
        pollNotebookVerification(chat.id, data.query_id);
    }
}

// Poll for a notebook verification verdict and fold the updated confidence
// into the stored message. Gives up rather than polling forever: on Cloud Run
// a background verification can stall or be lost when the instance scales
// down, so "pending" is a state that may never resolve.
const VERIFICATION_POLL_MS = 5000;
const VERIFICATION_MAX_POLLS = 48;  // ~4 minutes

async function pollNotebookVerification(chatId, queryId, attempt = 0) {
    if (attempt >= VERIFICATION_MAX_POLLS) return;

    let result;
    try {
        const res = await fetch(`/api/v1/notebooks/${encodeURIComponent(queryId)}/verification`);
        if (!res.ok) return;
        result = await res.json();
    } catch (e) {
        return;  // transient; a later query will schedule its own poll
    }

    if (result.status === 'pending') {
        setTimeout(() => pollNotebookVerification(chatId, queryId, attempt + 1),
                   VERIFICATION_POLL_MS);
        return;
    }
    if (result.status !== 'complete' || typeof result.confidence !== 'number') return;

    const chat = chats[chatId];
    if (!chat) return;
    const msg = [...chat.messages].reverse().find(m => m.queryId === queryId);
    if (!msg) return;

    msg.confidence = result.confidence;
    msg.confidenceBreakdown = result.confidence_breakdown || msg.confidenceBreakdown;
    msg.confidenceUnavailable = result.confidence_unavailable || null;
    msg.confidenceExplanation = result.confidence_explanation || null;
    // Keep the adversarial review verdict so the confidence panel can show
    // what the adversarial method review found (third signal).
    if (result.review && result.review.reviewed) {
        msg.notebookReview = {
            summary: result.review.summary || null,
            findings: (result.review.findings || []).map(f => ({
                severity: f.severity, title: f.title
            }))
        };
    }
    saveChatsToStorage();
    if (currentChatId === chatId) renderMessages();
}

// Rendering
// Bucket a chat by recency for the sidebar section headers.
function chatDateGroup(date) {
    const now = new Date();
    const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const diffDays = Math.floor((startOfToday - new Date(date.getFullYear(), date.getMonth(), date.getDate())) / 86400000);
    if (diffDays <= 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays <= 7) return 'Previous 7 Days';
    if (diffDays <= 30) return 'Previous 30 Days';
    return 'Older';
}

// Does a chat match the current search term (title or any message text)?
function chatMatchesSearch(chat) {
    if (!_chatSearchTerm) return true;
    if ((chat.title || '').toLowerCase().includes(_chatSearchTerm)) return true;
    return (chat.messages || []).some(m => (m.content || '').toLowerCase().includes(_chatSearchTerm));
}

function chatItemHtml(chat) {
    const isActive = chat.id === currentChatId;
    const pinned = !!chat.pinned;
    return `
        <div class="chat-item ${isActive ? 'active' : ''}" role="button" tabindex="0"
             onclick="selectChat('${chat.id}')"
             onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();selectChat('${chat.id}');}"
             title="${escapeHtml(chat.title)}">
            <i class="bi ${pinned ? 'bi-pin-angle-fill' : 'bi-chat-left-text'} chat-item-icon" aria-hidden="true"></i>
            <div class="chat-item-body">
                <div class="chat-title">${escapeHtml(chat.title)}</div>
            </div>
            <button class="chat-pin ${pinned ? 'pinned' : ''}" type="button"
                    aria-label="${pinned ? 'Unpin conversation' : 'Pin conversation'}"
                    title="${pinned ? 'Unpin' : 'Pin'}"
                    onclick="event.stopPropagation(); togglePinChat('${chat.id}')">
                <i class="bi ${pinned ? 'bi-pin-angle-fill' : 'bi-pin-angle'}" aria-hidden="true"></i>
            </button>
            <button class="chat-del" type="button" aria-label="Delete conversation"
                    onclick="event.stopPropagation(); deleteChat('${chat.id}')">
                <i class="bi bi-trash3" aria-hidden="true"></i>
            </button>
        </div>
    `;
}

function renderChatList() {
    const all = Object.values(chats).filter(chatMatchesSearch);

    if (all.length === 0) {
        chatList.innerHTML = _chatSearchTerm
            ? `<div class="sidebar-empty"><i class="bi bi-search d-block mb-2 fs-5"></i>No conversations match “${escapeHtml(_chatSearchTerm)}”.</div>`
            : `<div class="sidebar-empty"><i class="bi bi-chat-square-text d-block mb-2 fs-5"></i>No conversations yet.<br>Start by asking a question.</div>`;
        return;
    }

    const byRecency = (a, b) => new Date(b.createdAt) - new Date(a.createdAt);
    let html = '';

    // Pinned conversations float to the top in their own group.
    const pinned = all.filter(c => c.pinned).sort(byRecency);
    if (pinned.length) {
        html += `<div class="chat-group-label"><i class="bi bi-pin-angle-fill me-1" aria-hidden="true"></i>Pinned</div>`;
        html += pinned.map(chatItemHtml).join('');
    }

    // Everything else, grouped by recency.
    const GROUP_ORDER = ['Today', 'Yesterday', 'Previous 7 Days', 'Previous 30 Days', 'Older'];
    const groups = {};
    for (const chat of all.filter(c => !c.pinned).sort(byRecency)) {
        const g = chatDateGroup(new Date(chat.createdAt));
        (groups[g] = groups[g] || []).push(chat);
    }
    for (const groupName of GROUP_ORDER) {
        const items = groups[groupName];
        if (!items || !items.length) continue;
        html += `<div class="chat-group-label">${groupName}</div>`;
        html += items.map(chatItemHtml).join('');
    }
    chatList.innerHTML = html;
}

// Pin / unpin a conversation (persists locally + syncs to the server).
function togglePinChat(chatId) {
    const chat = chats[chatId];
    if (!chat) return;
    chat.pinned = !chat.pinned;
    saveChatsToStorage();
    renderChatList();
    if (isAuthenticated) syncChatToServer(chatId, chat).catch(() => {});
}

function renderMessages() {
    const chat = chats[currentChatId];
    if (!chat) return;

    // Preserve the typing indicator if it exists — renderMessages rebuilds
    // the chat history via innerHTML which would destroy it.
    const typingIndicator = document.getElementById('typingIndicator');
    const hadIndicator = !!typingIndicator;
    if (typingIndicator) typingIndicator.remove();

    chatMessages.innerHTML = chat.messages.map((msg, index) => {
        const time = new Date(msg.timestamp).toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit'
        });

        if (msg.role === 'user') {
            return `
                <div class="message user">
                    <div class="message-content">
                        <div class="message-bubble">${escapeHtml(msg.content)}</div>
                        <div class="message-meta">${time}</div>
                    </div>
                    <div class="message-avatar"><i class="bi bi-person"></i></div>
                </div>
            `;
        } else {
            // Verified answers carry a similarity score, not a pipeline
            // confidence — showing it as "% confidence" under a "Verified
            // Answer" badge was misleading. The banner already shows "% match".
            const confidence = (!msg.isVerified && msg.confidence)
                ? `${Math.round(msg.confidence * 100)}% confidence` : '';
            // hadNotebook === false means this response shipped no notebook
            // (declined revision); a server verified-cache hit's queryId has
            // no fetchable /notebooks/{id} either. Older stored messages
            // predate both fields and keep the legacy behaviour.
            const hasNotebook = !msg.isQuickAnswer
                && (msg.notebook || msg.queryId)
                && msg.hadNotebook !== false
                && !(msg.confidenceLevel === 'verified' && !msg.notebook);

            // "Why this confidence?" — the pipeline returns a 5-factor
            // breakdown; make the % clickable to reveal it (generated answers
            // only, where the score is a real pipeline confidence).
            // A factor the pipeline could not measure comes back as null with a
            // reason, so the panel is worth opening for those too — not only
            // when at least one factor has a number.
            const bd = msg.confidenceBreakdown;
            const confUnavailable = msg.confidenceUnavailable || {};
            const hasBreakdown = !!(confidence && (
                (bd && CONFIDENCE_FACTORS.some(([k]) => typeof bd[k] === 'number'))
                || Object.keys(confUnavailable).length > 0
            ));
            const confidenceMeta = confidence
                ? (hasBreakdown
                    ? `· <button type="button" class="confidence-chip" data-action="toggle-confidence" data-index="${index}" aria-expanded="false">${confidence} <i class="bi bi-chevron-down" aria-hidden="true"></i></button>`
                    : '· ' + confidence)
                : '';
            const confidencePanel = hasBreakdown
                ? `<div class="confidence-panel d-none" id="conf-panel-${index}">${renderConfidenceBars(bd || {}, confUnavailable, msg.confidenceExplanation, msg.notebookReview)}</div>`
                : '';

            // A revised notebook (chat follow-up edit) gets a small badge so
            // the user can see this answer updated an earlier notebook.
            const revisionBadge = msg.isRevision
                ? `<div class="mb-2"><span class="badge bg-primary"><i class="bi bi-pencil-square me-1"></i>Notebook updated</span></div>`
                : '';

            // Show verified banner if this is a verified response
            let verifiedBanner = '';
            if (msg.isVerified) {
                const score = Math.round(msg.similarityScore * 100);
                verifiedBanner = `
                    <div class="verified-response">
                        <span class="badge bg-success me-2"><i class="bi bi-patch-check-fill me-1"></i>Verified Answer</span>
                        <span class="small text-muted">From verified notebook (${score}% match to: "${escapeHtml(msg.verifiedQuery)}")</span>
                    </div>
                `;
            } else if (msg.verifiedMatches && msg.verifiedMatches.length > 0 &&
                       msg.verifiedMatches[0].similarity_score >= VERIFIED_SUGGESTION_THRESHOLD) {
                // Show that similar verified notebooks exist (only if >= 40% similarity)
                const match = msg.verifiedMatches[0];
                const score = Math.round(match.similarity_score * 100);
                verifiedBanner = `
                    <div class="verified-banner">
                        <h6><i class="bi bi-patch-check-fill me-2"></i>Similar Verified Notebook</h6>
                        <p class="mb-1 small">A related verified notebook exists: "${escapeHtml(match.query)}"</p>
                        <small class="text-muted">${score}% similarity</small>
                        <button class="btn btn-sm btn-outline-success ms-2" onclick="downloadVerifiedNotebook('${match.notebook_id}')">
                            <i class="bi bi-download me-1"></i>Download
                        </button>
                    </div>
                `;
            }

            // Suggested follow-up questions (for recommendation mode)
            let suggestedQuestionsHtml = '';
            if (msg.suggestedQuestions && msg.suggestedQuestions.length > 0) {
                suggestedQuestionsHtml = `
                    <div class="suggested-questions mt-3">
                        <div class="small text-muted mb-2"><i class="bi bi-lightbulb me-1"></i>Try asking:</div>
                        <div class="d-flex flex-wrap gap-2">
                            ${msg.suggestedQuestions.map(q => `
                                <button class="btn btn-sm btn-outline-primary suggested-q-btn" onclick="askFollowUp('${escapeHtml(q).replace(/'/g, "\\'")}')">
                                    ${escapeHtml(q)}
                                </button>
                            `).join('')}
                        </div>
                    </div>
                `;
            }

            // "People also asked" — the verified search already returned the
            // top-5 matches; the banner shows #1, so surface #2–5 (above the
            // suggestion threshold) as clickable verified follow-ups. Each
            // click starts a new chat and is served instantly from the library.
            let peopleAlsoAskedHtml = '';
            if (!msg.isVerified && Array.isArray(msg.verifiedMatches) && msg.verifiedMatches.length > 1) {
                const seen = new Set();
                const related = msg.verifiedMatches.slice(1)
                    .filter(m => m && m.query && m.similarity_score >= VERIFIED_SUGGESTION_THRESHOLD)
                    .filter(m => { const k = m.query.toLowerCase().trim(); if (seen.has(k)) return false; seen.add(k); return true; })
                    .slice(0, 4);
                if (related.length) {
                    peopleAlsoAskedHtml = `
                        <div class="people-also-asked mt-3">
                            <div class="paa-header"><i class="bi bi-patch-check me-1" aria-hidden="true"></i>People also asked</div>
                            <div class="d-flex flex-wrap gap-2">
                                ${related.map(m => `
                                    <button type="button" class="example-btn" data-verified="true"
                                            data-action="ask-verified" data-query="${escapeAttr(m.query)}">
                                        ${escapeHtml(m.query)}
                                    </button>
                                `).join('')}
                            </div>
                        </div>
                    `;
                }
            }

            // Quick answer source links
            let sourceLinksHtml = '';
            if (msg.isQuickAnswer && msg.sourceLinks && msg.sourceLinks.length > 0) {
                sourceLinksHtml = `
                    <div class="source-links mt-2">
                        <div class="source-links-header">
                            <i class="bi bi-link-45deg me-1"></i>Sources
                        </div>
                        <div class="source-links-list">
                            ${msg.sourceLinks.map(link => `
                                <a href="${escapeHtml(link.url)}" target="_blank" rel="noopener noreferrer" class="source-link-item" title="${escapeHtml(link.description || '')}">
                                    <i class="bi bi-box-arrow-up-right me-1"></i>
                                    <span>${escapeHtml(link.name)}</span>
                                </a>
                            `).join('')}
                        </div>
                    </div>
                `;
            }

            // Quick answer submit button — rendered inside the single shared
            // actions row below (two stacked button strips looked broken).
            const quickAnswerSubmitHtml = msg.isQuickAnswer ? `
                <button class="btn btn-sm btn-outline-secondary" onclick="submitQuickAnswerForReview(${index})">
                    <i class="bi bi-check-circle me-1"></i>Submit for Admin Review
                </button>` : '';

            return `
                <div class="message assistant">
                    <div class="message-avatar"><i class="bi bi-robot"></i></div>
                    <div class="message-content">
                        ${revisionBadge}
                        ${verifiedBanner}
                        <div class="message-bubble">${marked.parse(msg.content)}</div>
                        ${sourceLinksHtml}
                        <div class="message-meta">${time} ${confidenceMeta}${msg.isQuickAnswer ? ' · <span class="quick-answer-badge">Quick Answer</span>' : ''}</div>
                        ${confidencePanel}
                        ${suggestedQuestionsHtml}
                        ${peopleAlsoAskedHtml}
                        <div class="message-actions">
                            <button class="btn btn-sm btn-outline-secondary msg-copy-btn"
                                    title="Copy answer" aria-label="Copy answer"
                                    onclick="copyAnswer(${index}, this)">
                                <i class="bi bi-clipboard me-1"></i>Copy
                            </button>
                            ${quickAnswerSubmitHtml}
                            ${hasNotebook ? `
                            <button class="btn btn-sm btn-outline-primary" onclick="showNotebook(${index})">
                                <i class="bi bi-journal-code me-1"></i>View Notebook
                            </button>` : ''}
                            ${msg.isVerified && msg.evidenceVerifyUrl ? typedStandardsVerifyButton(msg.evidenceVerifyUrl) : ''}
                            ${msg.isVerified && msg.queryId ? `
                            <button type="button" class="btn btn-sm btn-outline-secondary" data-action="share" data-id="${escapeAttr(msg.queryId)}" title="Copy a public link to this verified answer">
                                <i class="bi bi-share me-1"></i>Share
                            </button>` : ''}
                            ${(index === chat.messages.length - 1 && msg.role === 'assistant' && !msg.isVerified && !msg.isQuickAnswer) ? `
                            <button class="btn btn-sm btn-outline-secondary" onclick="regenerateAnswer()" title="Re-run this question for a fresh answer">
                                <i class="bi bi-arrow-clockwise me-1"></i>Regenerate
                            </button>` : ''}
                            <span class="feedback-group" role="group" aria-label="Was this answer helpful?">
                                <button type="button" class="feedback-btn ${msg.feedback === 'up' ? 'active' : ''}"
                                        data-action="feedback" data-index="${index}" data-rating="up"
                                        title="Helpful" aria-label="Helpful"${msg.feedback ? ' disabled' : ''}>
                                    <i class="bi bi-hand-thumbs-up" aria-hidden="true"></i>
                                </button>
                                <button type="button" class="feedback-btn ${msg.feedback === 'down' ? 'active' : ''}"
                                        data-action="feedback" data-index="${index}" data-rating="down"
                                        title="Not helpful" aria-label="Not helpful"${msg.feedback ? ' disabled' : ''}>
                                    <i class="bi bi-hand-thumbs-down" aria-hidden="true"></i>
                                </button>
                                <span class="feedback-thanks text-muted small ${msg.feedback ? '' : 'd-none'}">Thanks for the feedback</span>
                            </span>
                        </div>
                    </div>
                </div>
            `;
        }
    }).join('');

    // Re-attach the typing indicator if it was present before re-render
    if (hadIndicator && typingIndicator) {
        chatMessages.appendChild(typingIndicator);
    }

    // Scroll to bottom (and refresh the jump-to-latest button state)
    scrollMessagesToBottom();
}

function showTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.id = 'typingIndicator';
    indicator.className = 'message assistant';
    indicator.innerHTML = `
        <div class="message-avatar"><i class="bi bi-robot"></i></div>
        <div class="message-content">
            <div class="message-bubble">
                <div class="thinking-label" style="font-size: 0.85em; color: var(--dc-text-muted); margin-bottom: 8px;">
                    <i class="bi bi-lightning-charge-fill"></i> Thinking...
                </div>
                <div id="thinkingContent" style="font-size: 0.9em; color: var(--dc-text-secondary); line-height: 1.6; max-height: 200px; overflow-y: auto; margin-bottom: 8px; white-space: pre-wrap;"></div>
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
        </div>
    `;
    chatMessages.appendChild(indicator);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Function to append thinking text to the indicator
function appendThinking(text) {
    const thinkingContent = document.getElementById('thinkingContent');
    if (thinkingContent) {
        thinkingContent.textContent += text;
        // Auto-scroll the thinking content
        thinkingContent.scrollTop = thinkingContent.scrollHeight;
        // Also scroll the chat
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

function hideTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) indicator.remove();
}

// Notebook Functions
async function showNotebook(messageIndex) {
    const chat = chats[currentChatId];
    if (!chat) return;

    const msg = chat.messages[messageIndex];
    if (!msg) return;

    const cellsContainer = document.getElementById('notebookCells');
    const cellCount = document.getElementById('notebookCellCount');
    const downloadBtn = document.getElementById('downloadNotebookBtn');
    const modalTitle = document.getElementById('notebookModalTitle');

    // Show loading state
    cellsContainer.innerHTML = `
        <div class="text-center text-muted p-5">
            <div class="spinner-border text-primary" role="status"></div>
            <p class="mt-3">Loading notebook...</p>
        </div>
    `;

    const modal = new bootstrap.Modal(document.getElementById('notebookModal'));
    modal.show();

    // Update title. Already-verified notebooks can't be re-submitted — hide
    // the Submit for Review action for them.
    const verifyBtn = document.getElementById('verifyTypedStandardsBtn');
    const submitBtn = document.getElementById('submitForReviewBtn');
    if (submitBtn) submitBtn.classList.toggle('d-none', !!msg.isVerified);
    if (msg.isVerified) {
        modalTitle.innerHTML = '<span class="badge bg-success me-2">Verified</span>Notebook';
        // Offer independent verification only when this notebook has a minted
        // evidence package (its served commitment URL). Without one there is
        // nothing for the verifier to fetch — matches the admin panel, which
        // also only links to verify when evidence_verify_url is present.
        if (verifyBtn && msg.evidenceVerifyUrl) {
            verifyBtn.href = msg.evidenceVerifyUrl;
            verifyBtn.classList.remove('d-none');
        } else if (verifyBtn) {
            verifyBtn.classList.add('d-none');
        }
    } else {
        modalTitle.textContent = 'Generated Notebook';
        if (verifyBtn) verifyBtn.classList.add('d-none');
    }

    // "Open in Colab" — only for verified notebooks published to GitHub
    // (generated notebooks aren't on GitHub, so there's nothing to open).
    const colabBtn = document.getElementById('openColabBtn');
    const colab = colabUrl(msg.githubUrl);
    if (colabBtn && colab) {
        colabBtn.href = colab;
        colabBtn.classList.remove('d-none');
    } else if (colabBtn) {
        colabBtn.classList.add('d-none');
    }

    // Try to get notebook - first from message, then from API
    let notebook = msg.notebook;

    // If notebook not in message but we have a queryId, fetch it from API
    if ((!notebook || !notebook.cells) && msg.queryId) {
        try {
            const response = await fetch(`${API_BASE}/notebooks/${msg.queryId}`);
            if (response.ok) {
                notebook = await response.json();
                // Cache it in the message for next time
                msg.notebook = notebook;
                saveChatsToStorage();
            }
        } catch (error) {
            console.error('Failed to fetch notebook:', error);
        }
    }

    currentNotebook = notebook;

    if (notebook && notebook.cells) {
        const cells = notebook.cells;
        cellCount.textContent = `${cells.length} cells`;

        // Render cells
        cellsContainer.innerHTML = cells.map((cell, i) => renderNotebookCell(cell, i)).join('');

        // Apply syntax highlighting
        cellsContainer.querySelectorAll('pre code').forEach(block => {
            hljs.highlightElement(block);
        });

        // Setup download - use API endpoint for reliable download
        if (msg.queryId) {
            downloadBtn.href = `${API_BASE}/notebooks/${msg.queryId}/download`;
            downloadBtn.download = `data_concierge_${msg.queryId}.ipynb`;
        } else {
            // Fallback to blob download
            const blob = new Blob([JSON.stringify(notebook, null, 2)], { type: 'application/json' });
            downloadBtn.href = URL.createObjectURL(blob);
            downloadBtn.download = 'data_concierge_notebook.ipynb';
        }
    } else {
        cellsContainer.innerHTML = `
            <div class="text-center text-muted p-5">
                <i class="bi bi-journal-x display-4"></i>
                <p class="mt-3">Notebook content not available</p>
                <small class="text-muted">The notebook may still be generating or an error occurred.</small>
            </div>
        `;
        cellCount.textContent = '0 cells';
    }
}

function renderNotebookCell(cell, index) {
    const isCode = cell.cell_type === 'code';
    const source = Array.isArray(cell.source) ? cell.source.join('') : (cell.source || '');

    // Get output if it's a code cell
    let outputHtml = '';
    if (isCode && cell.outputs && cell.outputs.length > 0) {
        const outputText = cell.outputs.map(out => {
            if (out.text) return Array.isArray(out.text) ? out.text.join('') : out.text;
            if (out.data && out.data['text/plain']) {
                const plain = out.data['text/plain'];
                return Array.isArray(plain) ? plain.join('') : plain;
            }
            return '';
        }).join('\n');

        if (outputText.trim()) {
            outputHtml = `
                <div class="cell-output">
                    <div class="cell-output-label">Output:</div>
                    ${escapeHtml(outputText)}
                </div>
            `;
        }
    }

    if (isCode) {
        return `
            <div class="notebook-cell" data-cell-index="${index}">
                <div class="cell-header" onclick="toggleCell(${index})">
                    <div>
                        <span class="cell-type-badge cell-type-code">Code</span>
                        <span class="cell-number ms-2">[${index + 1}]</span>
                    </div>
                    <div class="cell-header-actions">
                        <button class="cell-copy-btn" type="button" title="Copy code" aria-label="Copy code"
                                onclick="event.stopPropagation(); copyCodeCell(this)">
                            <i class="bi bi-clipboard"></i>
                        </button>
                        <i class="bi bi-chevron-down cell-chevron"></i>
                    </div>
                </div>
                <div class="cell-body" id="cell-body-${index}">
                    <pre class="cell-source"><code class="language-python">${escapeHtml(source)}</code></pre>
                    ${outputHtml}
                </div>
            </div>
        `;
    } else {
        // Markdown cell
        return `
            <div class="notebook-cell" data-cell-index="${index}">
                <div class="cell-header" onclick="toggleCell(${index})">
                    <div>
                        <span class="cell-type-badge cell-type-markdown">Markdown</span>
                        <span class="cell-number ms-2">[${index + 1}]</span>
                    </div>
                    <i class="bi bi-chevron-down cell-chevron"></i>
                </div>
                <div class="cell-body" id="cell-body-${index}">
                    <div class="cell-markdown-content">${marked.parse(source)}</div>
                </div>
            </div>
        `;
    }
}

function toggleCell(index) {
    const body = document.getElementById(`cell-body-${index}`);
    if (body) {
        body.classList.toggle('collapsed');
        const header = body.previousElementSibling;
        const icon = header.querySelector('.cell-chevron') || header.querySelector('i');
        if (icon) {
            icon.className = body.classList.contains('collapsed')
                ? 'bi bi-chevron-right cell-chevron'
                : 'bi bi-chevron-down cell-chevron';
        }
    }
}

function toggleAllCells(collapse) {
    document.querySelectorAll('.cell-body').forEach((body, i) => {
        if (collapse) {
            body.classList.add('collapsed');
        } else {
            body.classList.remove('collapsed');
        }
        const header = body.previousElementSibling;
        const icon = header.querySelector('.cell-chevron') || header.querySelector('i');
        if (icon) {
            icon.className = collapse
                ? 'bi bi-chevron-right cell-chevron'
                : 'bi bi-chevron-down cell-chevron';
        }
    });
}

async function submitForReview() {
    if (!currentNotebook || !currentChatId) {
        showToast('No notebook to submit');
        return;
    }

    const chat = chats[currentChatId];
    const lastUserMsg = chat.messages.filter(m => m.role === 'user').pop();
    const lastAssistantMsg = chat.messages.filter(m => m.role === 'assistant').pop();

    try {
        const response = await fetch(`${API_BASE}/notebooks/submit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: lastUserMsg?.content || '',
                answer: lastAssistantMsg?.content || '',
                notebook_json: currentNotebook,
                filename: `notebook_${Date.now()}.ipynb`,
                query_id: lastAssistantMsg?.queryId || null,
            })
        });

        if (response.ok) {
            showToast('Notebook submitted for review!', 'success');
            bootstrap.Modal.getInstance(document.getElementById('notebookModal')).hide();
        } else {
            throw new Error('Submission failed');
        }
    } catch (error) {
        showToast('Failed to submit notebook: ' + error.message, 'danger');
    }
}

// Verified Notebooks
async function searchVerifiedNotebooks(query) {
    try {
        const response = await fetch(`${API_BASE}/verified-notebooks/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, threshold: 0.2, max_results: 5 })
        });

        if (response.ok) {
            const data = await response.json();
            return data.results || [];
        }
    } catch (error) {
        console.error('Verified notebook search error:', error);
    }
    return [];
}

async function getVerifiedNotebook(notebookId) {
    try {
        const response = await fetch(`${API_BASE}/verified-notebooks/${notebookId}`);
        if (response.ok) {
            return await response.json();
        }
    } catch (error) {
        console.error('Failed to get verified notebook:', error);
    }
    return {};
}

async function downloadVerifiedNotebook(notebookId) {
    window.location.href = `${API_BASE}/verified-notebooks/${notebookId}/download`;
}

// =============================================================================
// Server-side chat sync
// =============================================================================

/**
 * Load chats from the server and merge with local storage.
 * Server is authoritative for chats that exist on both sides.
 * Local-only chats are uploaded to the server.
 */
async function loadChatsFromServer() {
    try {
        const response = await fetch(`${API_BASE}/chats`, { credentials: 'include' });
        if (!response.ok) return;
        const data = await response.json();
        const serverChats = data.chats || {};

        // Upload any chats that are local-only (created before login / on another device
        // that hasn't synced yet)
        const uploadPromises = [];
        for (const [id, chat] of Object.entries(chats)) {
            if (!serverChats[id]) {
                uploadPromises.push(syncChatToServer(id, chat).catch(() => {}));
            }
        }

        // Merge: server wins for any chat that exists on both sides
        chats = { ...chats, ...serverChats };

        await Promise.all(uploadPromises);

        // Drop duplicate conversations introduced by the merge.
        dedupeChats();

        // Persist merged result locally
        try {
            const persisted = {};
            for (const [id, chat] of Object.entries(chats)) {
                persisted[id] = {
                    ...chat,
                    messages: (chat.messages || []).map(msg => {
                        const { notebook, quickAnswer, verifiedMatches, ...rest } = msg;
                        return rest;
                    })
                };
            }
            localStorage.setItem('dc_chats', JSON.stringify(persisted));
        } catch {}

        renderChatList();

        // If a hash chat now exists after server sync, restore it
        const hashChatId = location.hash.slice(1);
        if (hashChatId && chats[hashChatId] && !currentChatId) {
            currentChatId = hashChatId;
            renderChatList();
            showChatInterface();
        }
    } catch (e) {
        console.warn('Failed to load chats from server', e);
    }
}

/**
 * Push a single chat to the server (best-effort, fire-and-forget).
 * Heavy fields like notebook JSON are stripped before sending.
 */
async function syncChatToServer(chatId, chatData) {
    const stripped = {
        id: chatData.id,
        title: chatData.title,
        createdAt: chatData.createdAt,
        pinned: !!chatData.pinned,
        messages: (chatData.messages || []).map(({ notebook, quickAnswer, ...rest }) => rest),
    };
    await fetch(`${API_BASE}/chats/${chatId}`, {
        method: 'PUT',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(stripped),
    });
}

/** Delete a chat from the server (best-effort). */
async function deleteChatFromServer(chatId) {
    await fetch(`${API_BASE}/chats/${chatId}`, {
        method: 'DELETE',
        credentials: 'include',
    });
}

// =============================================================================
// Storage
// =============================================================================

// Storage
function loadChatsFromStorage() {
    try {
        const stored = localStorage.getItem('dc_chats');
        if (stored) {
            const parsed = JSON.parse(stored);
            // Re-save immediately with stripped fields to recover from any bloated
            // existing entries that were preventing future saves.
            chats = parsed;
            try {
                const clean = {};
                for (const [id, chat] of Object.entries(parsed)) {
                    clean[id] = {
                        ...chat,
                        messages: (chat.messages || []).map(msg => {
                            const { notebook, quickAnswer, verifiedMatches, ...rest } = msg;
                            return rest;
                        })
                    };
                }
                localStorage.setItem('dc_chats', JSON.stringify(clean));
            } catch {}
        }
    } catch (error) {
        console.error('Failed to load chats:', error);
        chats = {};
    }
}

function saveChatsToStorage() {
    try {
        // Strip large in-memory-only fields before persisting
        const persisted = {};
        for (const [id, chat] of Object.entries(chats)) {
            persisted[id] = {
                ...chat,
                messages: (chat.messages || []).map(msg => {
                    const { notebook, quickAnswer, verifiedMatches, ...rest } = msg;
                    return rest;
                })
            };
        }
        localStorage.setItem('dc_chats', JSON.stringify(persisted));
    } catch (error) {
        console.error('Failed to save chats to localStorage:', error);
    }
    // Best-effort server sync for current chat
    if (currentChatId && chats[currentChatId] && isAuthenticated) {
        syncChatToServer(currentChatId, chats[currentChatId]).catch(() => {});
    }
}

// Utilities
function generateId() {
    return Math.random().toString(36).substring(2, 10);
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// escapeHtml handles &, <, > but not quotes — needed when interpolating into
// a double-quoted HTML attribute (e.g. data-query="...").
function escapeAttr(text) {
    return escapeHtml(text).replace(/"/g, '&quot;');
}

// github.com/<repo>/blob/<branch>/<path> -> Colab "open from GitHub" deep link.
function colabUrl(githubUrl) {
    if (!githubUrl || !githubUrl.includes('github.com/')) return null;
    return githubUrl.replace('https://github.com/', 'https://colab.research.google.com/github/');
}

// Copy a shareable public Library deep link for a verified answer/notebook.
async function shareVerified(id, btn) {
    if (!id) return;
    const url = `${location.origin}/library#lib-${id}`;
    try {
        await navigator.clipboard.writeText(url);
    } catch {
        const ta = document.createElement('textarea');
        ta.value = url; ta.style.position = 'fixed'; ta.style.opacity = '0';
        document.body.appendChild(ta); ta.select();
        try { document.execCommand('copy'); } catch {}
        ta.remove();
    }
    showToast('Share link copied to clipboard', 'success');
    if (btn) {
        const icon = btn.querySelector('i');
        const prev = icon ? icon.className : '';
        if (icon) icon.className = 'bi bi-check2 me-1';
        btn.classList.add('copied');
        setTimeout(() => { if (icon) icon.className = prev; btn.classList.remove('copied'); }, 1300);
    }
}

// =============================================================================
// Confidence breakdown ("Why this confidence?")
// =============================================================================
// The 5 primary pipeline factors (core/confidence.py), in display order.
const CONFIDENCE_FACTORS = [
    ['notebook_verification', 'Notebook check', 'The notebook was run and its output reproduces the numbers in this answer'],
    ['answer_grounding', 'Answer grounding', 'How well the answer is backed by the retrieved data'],
    ['data_retrieval_quality', 'Data retrieval', 'Quality and quantity of the data that was retrieved'],
    ['source_metadata_quality', 'Source quality', 'Authority and completeness of the data source'],
    ['query_answer_alignment', 'Query alignment', 'How directly the answer addresses the question'],
    ['computation_complexity', 'Computation', 'Reliability given the computation involved'],
];

// A factor with no value was not measured. Say so, and say why, instead of
// dropping the row — silently omitting it made a partial score look complete.
function renderConfidenceBars(bd, unavailable, explanation, review) {
    const reasons = unavailable || {};
    const rows = CONFIDENCE_FACTORS
        .map(([k, label, desc]) => {
            if (typeof bd[k] === 'number') {
                const pct = Math.max(0, Math.min(100, Math.round(bd[k] * 100)));
                const tone = pct >= 75 ? 'good' : pct >= 50 ? 'mid' : 'low';
                return `
                <div class="conf-row" title="${escapeAttr(desc)}">
                    <span class="conf-label">${label}</span>
                    <span class="conf-track"><span class="conf-fill conf-${tone}" style="width:${pct}%"></span></span>
                    <span class="conf-val">${pct}%</span>
                </div>`;
            }
            const why = reasons[k];
            if (!why) return '';
            return `
                <div class="conf-row conf-row-unmeasured" title="${escapeAttr(why)}">
                    <span class="conf-label">${label}</span>
                    <span class="conf-track conf-track-empty"></span>
                    <span class="conf-val conf-val-unmeasured">not measured</span>
                </div>
                <div class="conf-why">${escapeHtml(why)}</div>`;
        }).join('');

    // Reasons that are not tied to one factor (whole-query escalation or error).
    const factorKeys = new Set(CONFIDENCE_FACTORS.map(([k]) => k));
    const otherReasons = Object.entries(reasons)
        .filter(([k]) => !factorKeys.has(k))
        .map(([, why]) => `<div class="conf-why">${escapeHtml(why)}</div>`)
        .join('');

    const caption = explanation
        ? `<div class="conf-caption conf-caption-partial"><i class="bi bi-exclamation-circle me-1" aria-hidden="true"></i>${escapeHtml(explanation)}</div>`
        : '<div class="conf-caption"><i class="bi bi-info-circle me-1" aria-hidden="true"></i>How this confidence was scored</div>';

    // Adversarial review verdict (third signal) — what the method
    // method check found once it ran on the generated notebook.
    let reviewHtml = '';
    if (review) {
        const findings = review.findings || [];
        if (findings.length === 0) {
            reviewHtml = `<div class="conf-why"><i class="bi bi-shield-check me-1" aria-hidden="true"></i>Adversarial review found no issues with the notebook's method.</div>`;
        } else {
            const items = findings.slice(0, 5).map(f =>
                `<li><strong>${escapeHtml(f.severity)}</strong> — ${escapeHtml(f.title)}</li>`).join('');
            const more = findings.length > 5
                ? `<li class="text-muted">+${findings.length - 5} more in the admin Notebook Reviews pane</li>` : '';
            reviewHtml = `<div class="conf-why">
                <i class="bi bi-shield-exclamation me-1" aria-hidden="true"></i>Adversarial review flagged ${findings.length} issue${findings.length === 1 ? '' : 's'}:
                <ul class="mb-0 mt-1 ps-4">${items}${more}</ul>
            </div>`;
        }
    }

    return `<div class="confidence-breakdown-inner">
        ${caption}
        ${rows}
        ${otherReasons}
        ${reviewHtml}
    </div>`;
}

function toggleConfidence(index, btn) {
    const panel = document.getElementById(`conf-panel-${index}`);
    if (!panel) return;
    const open = !panel.classList.toggle('d-none');
    if (btn) {
        btn.setAttribute('aria-expanded', open ? 'true' : 'false');
        const icon = btn.querySelector('i');
        if (icon) icon.className = open ? 'bi bi-chevron-up' : 'bi bi-chevron-down';
    }
}

// =============================================================================
// Answer feedback (👍 / 👎)
// =============================================================================
async function submitFeedback(index, rating, btn) {
    const chat = chats[currentChatId];
    const msg = chat && chat.messages[index];
    if (!msg || (rating !== 'up' && rating !== 'down')) return;

    // Find the user question that prompted this answer (for admin context).
    let userQuery = '';
    for (let i = index - 1; i >= 0; i--) {
        if (chat.messages[i].role === 'user') { userQuery = chat.messages[i].content; break; }
    }

    // Reflect the choice immediately and lock the group (prevents double-submit;
    // persists across reloads since msg.feedback is saved with the chat).
    msg.feedback = rating;
    const group = btn && btn.closest('.feedback-group');
    if (group) {
        group.querySelectorAll('.feedback-btn').forEach(b => {
            b.disabled = true;
            b.classList.toggle('active', b.dataset.rating === rating);
        });
        const thanks = group.querySelector('.feedback-thanks');
        if (thanks) thanks.classList.remove('d-none');
    }
    saveChatsToStorage();

    const source = msg.isVerified || msg.isQuickAnswer ? 'verified_cache' : 'generated';
    try {
        await fetch(`${API_BASE}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                rating,
                query: userQuery,
                answer: msg.content || '',
                query_id: msg.queryId || null,
                source,
            }),
        });
    } catch {
        // Best-effort — the in-UI acknowledgement already showed.
    }
}

function showToast(message, type = 'info') {
    const toastEl = document.getElementById('toast');
    document.getElementById('toastBody').textContent = message;
    toastEl.className = `toast bg-${type === 'success' ? 'success' : type === 'danger' ? 'danger' : 'body'}`;

    // Header icon + title follow the toast type.
    const [iconClass, title] = {
        success: ['bi bi-check-circle-fill me-2', 'Success'],
        danger: ['bi bi-exclamation-triangle-fill me-2', 'Error'],
    }[type] || ['bi bi-info-circle me-2 text-primary', 'Notice'];
    const headerIcon = toastEl.querySelector('.toast-header i');
    const headerTitle = toastEl.querySelector('.toast-header strong');
    if (headerIcon) headerIcon.className = iconClass;
    if (headerTitle) headerTitle.textContent = title;

    new bootstrap.Toast(toastEl).show();
}

// Handle clicking on suggested follow-up questions
function askFollowUp(question) {
    if (!currentChatId) return;
    addUserMessage(question);
    processQuery(question, true);  // true = isFollowUp, use analyze mode
}

// Quick Answer Submission
async function submitQuickAnswerForReview(messageIndex) {
    const chat = chats[currentChatId];
    if (!chat) return;

    const msg = chat.messages[messageIndex];
    if (!msg || !msg.isQuickAnswer) return;

    // Find the user query that triggered this answer
    let userQuery = '';
    for (let i = messageIndex - 1; i >= 0; i--) {
        if (chat.messages[i].role === 'user') {
            userQuery = chat.messages[i].content;
            break;
        }
    }

    try {
        const response = await fetch(`${API_BASE}/answers/submit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: userQuery,
                answer: msg.quickAnswer || msg.content,
                source_links: msg.sourceLinks || [],
                confidence: msg.confidence || 0
            })
        });

        if (response.ok) {
            showToast('Quick answer submitted for admin review!', 'success');
        } else {
            throw new Error('Submission failed');
        }
    } catch (error) {
        showToast('Failed to submit: ' + error.message, 'danger');
    }
}

// Make functions globally accessible
window.selectChat = selectChat;
window.deleteChat = deleteChat;
window.showNotebook = showNotebook;
window.downloadVerifiedNotebook = downloadVerifiedNotebook;
window.toggleCell = toggleCell;
window.askFollowUp = askFollowUp;
window.submitQuickAnswerForReview = submitQuickAnswerForReview;
window.showLoginModal = showLoginModal;
window.submitPasswordLogin = submitPasswordLogin;
window.doLogout = doLogout;
window.toggleSidebar = toggleSidebar;
window.closeSidebarMobile = closeSidebarMobile;
window.copyCodeCell = copyCodeCell;
window.copyAnswer = copyAnswer;
window.regenerateAnswer = regenerateAnswer;
window.showShortcutsModal = showShortcutsModal;
window.togglePinChat = togglePinChat;
