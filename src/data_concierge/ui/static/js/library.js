// Public Verified Library catalog — browse/search/sort verified answers +
// notebooks. Consumes the public GET /verified-notebooks and /verified-answers.

const API = '/api/v1';
let _items = [];

function escapeHtml(t) {
    if (!t) return '';
    const d = document.createElement('div');
    d.textContent = t;
    return d.innerHTML;
}
function escapeAttr(t) { return escapeHtml(t).replace(/"/g, '&quot;'); }

// github.com/<repo>/blob/<branch>/<path> -> Colab "open from GitHub" deep link.
function colabUrl(githubUrl) {
    if (!githubUrl || !githubUrl.includes('github.com/')) return null;
    return githubUrl.replace('https://github.com/', 'https://colab.research.google.com/github/');
}

function relativeAge(iso) {
    if (!iso) return '';
    const then = new Date(iso);
    if (isNaN(then)) return '';
    const days = Math.floor((Date.now() - then.getTime()) / 86400000);
    if (days <= 0) return 'today';
    if (days === 1) return 'yesterday';
    if (days < 30) return `${days}d ago`;
    if (days < 365) return `${Math.floor(days / 30)}mo ago`;
    return `${Math.floor(days / 365)}y ago`;
}

async function loadLibrary() {
    const grid = document.getElementById('libGrid');
    try {
        const [nbRes, ansRes] = await Promise.all([
            fetch(`${API}/verified-notebooks`).catch(() => null),
            fetch(`${API}/verified-answers`).catch(() => null),
        ]);
        const items = [];
        if (nbRes && nbRes.ok) {
            const d = await nbRes.json();
            for (const nb of (d.notebooks || [])) {
                items.push({
                    kind: 'notebook', id: nb.notebook_id, query: nb.query || '', answer: nb.answer || '',
                    source: nb.data_source || '', confidence: nb.confidence, usage: nb.usage_count || 0,
                    verified_at: nb.verified_at, download_url: nb.download_url,
                    github_url: nb.github_url, colab_url: colabUrl(nb.github_url),
                    verify_url: nb.evidence_verify_url,
                });
            }
        }
        if (ansRes && ansRes.ok) {
            const d = await ansRes.json();
            for (const a of (d.answers || [])) {
                items.push({
                    kind: 'answer', id: a.answer_id, query: a.query || '', answer: a.answer || '',
                    source: a.data_source || '', confidence: a.confidence, usage: a.usage_count || 0,
                    verified_at: a.verified_at, github_url: a.github_url, colab_url: null, verify_url: null,
                });
            }
        }
        _items = items;
        render();
    } catch (e) {
        grid.innerHTML = `<div class="col-12 empty-state">
            <i class="bi bi-exclamation-triangle text-danger" aria-hidden="true"></i>
            <div class="empty-title">Couldn't load the library</div>
            <p class="small">${escapeHtml(e.message)}</p>
        </div>`;
    }
}

function currentView() {
    const term = (document.getElementById('libSearch').value || '').trim().toLowerCase();
    const type = document.getElementById('libType').value;
    const sort = document.getElementById('libSort').value;
    let list = _items.slice();
    if (type !== 'all') list = list.filter(i => i.kind === type);
    if (term) list = list.filter(i =>
        i.query.toLowerCase().includes(term) ||
        i.answer.toLowerCase().includes(term) ||
        (i.source || '').toLowerCase().includes(term));
    const cmp = {
        usage: (a, b) => (b.usage - a.usage),
        recent: (a, b) => new Date(b.verified_at || 0) - new Date(a.verified_at || 0),
        confidence: (a, b) => (b.confidence || 0) - (a.confidence || 0),
        alpha: (a, b) => a.query.localeCompare(b.query),
    }[sort] || (() => 0);
    return list.sort(cmp);
}

function card(i) {
    const conf = i.confidence != null ? `${Math.round(i.confidence * 100)}% confidence` : '';
    const age = relativeAge(i.verified_at);
    const kindBadge = i.kind === 'notebook'
        ? '<span class="badge bg-primary"><i class="bi bi-journal-code me-1"></i>Notebook</span>'
        : '<span class="badge bg-info"><i class="bi bi-chat-quote me-1"></i>Quick answer</span>';
    const askUrl = `/?ask=${encodeURIComponent(i.query)}`;
    return `
    <div class="col-12 col-lg-6">
        <div class="card lib-card p-3" id="lib-${escapeAttr(i.id)}">
            <div class="d-flex justify-content-between align-items-start gap-2 mb-1">
                <div class="lib-q">${escapeHtml(i.query)}</div>
                ${kindBadge}
            </div>
            <div class="lib-meta">
                ${i.source ? `<span><i class="bi bi-globe" aria-hidden="true"></i>${escapeHtml(i.source)}</span>` : ''}
                ${conf ? `<span><i class="bi bi-graph-up" aria-hidden="true"></i>${conf}</span>` : ''}
                <span><i class="bi bi-arrow-repeat" aria-hidden="true"></i>used ${i.usage}×</span>
                ${age ? `<span title="${escapeAttr(i.verified_at || '')}"><i class="bi bi-patch-check" aria-hidden="true"></i>verified ${age}</span>` : ''}
            </div>
            ${i.answer ? `<div class="lib-answer">${escapeHtml(i.answer.slice(0, 220))}${i.answer.length > 220 ? '…' : ''}</div>` : ''}
            <div class="lib-actions">
                <a href="${escapeAttr(askUrl)}" class="btn btn-sm btn-primary"><i class="bi bi-chat-dots me-1"></i>Ask this</a>
                ${i.download_url ? `<a href="${escapeAttr(i.download_url)}" class="btn btn-sm btn-outline-secondary" download><i class="bi bi-download me-1"></i>Notebook</a>` : ''}
                ${i.colab_url ? `<a href="${escapeAttr(i.colab_url)}" target="_blank" rel="noopener" class="btn btn-sm btn-outline-secondary"><i class="bi bi-box-arrow-up-right me-1"></i>Open in Colab</a>` : ''}
                ${i.verify_url ? `<a href="${escapeAttr(i.verify_url)}" target="_blank" rel="noopener" class="btn btn-sm btn-outline-secondary" title="Independently verify at typedstandards.org"><i class="bi bi-patch-check-fill me-1"></i>Verify</a>` : ''}
                <button type="button" class="btn btn-sm btn-outline-secondary" data-share="${escapeAttr(i.id)}"><i class="bi bi-share me-1"></i>Share</button>
            </div>
        </div>
    </div>`;
}

function render() {
    const grid = document.getElementById('libGrid');
    const list = currentView();
    document.getElementById('libCount').textContent =
        `${list.length} of ${_items.length} verified ${_items.length === 1 ? 'entry' : 'entries'}`;
    if (!list.length) {
        grid.innerHTML = `<div class="col-12 empty-state">
            <i class="bi bi-search" aria-hidden="true"></i>
            <div class="empty-title">Nothing matches</div>
            <p class="small">Try a different search or filter.</p>
        </div>`;
        return;
    }
    grid.innerHTML = list.map(card).join('');
    grid.querySelectorAll('[data-share]').forEach(btn => {
        btn.addEventListener('click', () => shareEntry(btn.dataset.share, btn));
    });
    // Deep-link highlight (#lib-<id>)
    const hash = location.hash.slice(1);
    if (hash) {
        const el = document.getElementById(hash.startsWith('lib-') ? hash : `lib-${hash}`);
        if (el) { el.classList.add('lib-highlight'); el.scrollIntoView({ block: 'center' }); }
    }
}

async function shareEntry(id, btn) {
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
    showToast('Link copied to clipboard', 'success');
    if (btn) {
        const icon = btn.querySelector('i');
        const prev = icon ? icon.className : '';
        if (icon) icon.className = 'bi bi-check2 me-1';
        setTimeout(() => { if (icon) icon.className = prev; }, 1300);
    }
}

function showToast(message, type = 'info') {
    const el = document.getElementById('toast');
    document.getElementById('toastBody').textContent = message;
    el.className = `toast bg-${type === 'success' ? 'success' : type === 'danger' ? 'danger' : 'body'}`;
    const [iconClass, title] = type === 'success'
        ? ['bi bi-check-circle-fill me-2', 'Success']
        : ['bi bi-info-circle me-2 text-primary', 'Notice'];
    const hi = el.querySelector('.toast-header i');
    const ht = el.querySelector('.toast-header strong');
    if (hi) hi.className = iconClass;
    if (ht) ht.textContent = title;
    new bootstrap.Toast(el).show();
}

document.addEventListener('DOMContentLoaded', () => {
    loadLibrary();
    ['libSearch', 'libType', 'libSort'].forEach(id => {
        const el = document.getElementById(id);
        el.addEventListener(id === 'libSearch' ? 'input' : 'change', render);
    });
    // Reveal the Admin link only for signed-in admins (the route is backend-protected).
    fetch(`${API}/auth/status`).then(r => r.ok ? r.json() : null).then(d => {
        if (d && d.authenticated && d.is_admin) {
            document.getElementById('libAdminLink')?.classList.remove('d-none');
        }
    }).catch(() => {});
});
