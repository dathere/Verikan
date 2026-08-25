// Fair Store data dictionary browser (#136).
// Consumes /api/v1/fairstore/search and /api/v1/fairstore/resource/{id} to let
// anyone explore a dataset's columns — types, stats, top values — before
// reading a row. Public, read-only: verified public metadata, not the data.

function escapeHtml(t) {
    if (t === null || t === undefined) return '';
    const d = document.createElement('div');
    d.textContent = String(t);
    return d.innerHTML;
}
function escapeAttr(t) { return escapeHtml(t).replace(/"/g, '&quot;'); }

function showToast(message) {
    const el = document.getElementById('toast');
    const body = document.getElementById('toastBody');
    if (!el || !body || !window.bootstrap) return;
    body.textContent = message;
    bootstrap.Toast.getOrCreateInstance(el).show();
}

// Compact one column's whitelisted stats into a readable line.
function renderStats(stats) {
    if (!stats || typeof stats !== 'object') return '';
    const order = [
        ['min', 'min'], ['max', 'max'], ['mean', 'mean'],
        ['q2_median', 'median'], ['median', 'median'],
        ['stddev', 'std'], ['cardinality', 'distinct'], ['nullcount', 'nulls'],
    ];
    const seen = new Set();
    const parts = [];
    for (const [key, label] of order) {
        if (seen.has(label)) continue;
        const v = stats[key];
        if (v === undefined || v === null || v === '') continue;
        seen.add(label);
        parts.push(`${label} <code>${escapeHtml(v)}</code>`);
    }
    return parts.join(' · ');
}

function renderTopValues(top) {
    if (!Array.isArray(top) || !top.length) return '';
    return top.slice(0, 5)
        .map(tv => `${escapeHtml(tv.value)}${tv.count != null ? ` (${escapeHtml(tv.count)})` : ''}`)
        .join(', ');
}

let _searchSeq = 0;

async function runSearch(query) {
    const grid = document.getElementById('dictGrid');
    const count = document.getElementById('dictCount');
    const q = (query || '').trim();
    if (!q) {
        grid.innerHTML = '<div class="col-12 empty-state"><i class="bi bi-search me-2"></i>Search above to explore the data dictionary.</div>';
        count.textContent = '';
        return;
    }

    const seq = ++_searchSeq;
    grid.innerHTML = '<div class="col-12 empty-state"><div class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></div>Searching…</div>';

    let data;
    try {
        const res = await fetch(`/api/v1/fairstore/search?q=${encodeURIComponent(q)}&limit=24`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        data = await res.json();
    } catch (e) {
        if (seq !== _searchSeq) return;
        grid.innerHTML = '<div class="col-12 empty-state text-danger"><i class="bi bi-exclamation-triangle me-2"></i>Could not reach the data dictionary.</div>';
        count.textContent = '';
        return;
    }
    if (seq !== _searchSeq) return;  // a newer search superseded this one

    if (data.available === false) {
        grid.innerHTML = '<div class="col-12 empty-state"><i class="bi bi-info-circle me-2"></i>No data dictionary has been onboarded for this site yet.</div>';
        count.textContent = '';
        return;
    }

    const results = data.results || [];
    count.textContent = results.length
        ? `${results.length} dataset${results.length === 1 ? '' : 's'} for “${q}”`
        : '';

    if (!results.length) {
        grid.innerHTML = `<div class="col-12 empty-state"><i class="bi bi-inbox me-2"></i>No datasets match “${escapeHtml(q)}”.</div>`;
        return;
    }

    grid.innerHTML = results.map(r => `
        <div class="col-md-6 col-lg-4">
            <div class="card dict-card p-3" role="button" tabindex="0"
                 data-resource-id="${escapeAttr(r.resource_id)}"
                 data-title="${escapeAttr(r.dataset_title || r.resource_name)}">
                <div class="dict-title">${escapeHtml(r.dataset_title || r.resource_name)}</div>
                <div class="dict-meta">
                    <span><i class="bi bi-list-columns-reverse"></i>${escapeHtml(r.column_count)} columns</span>
                    <span><i class="bi bi-database"></i>${escapeHtml(r.record_count)} rows</span>
                </div>
                ${r.description ? `<p class="lib-answer mt-2 mb-0">${escapeHtml(String(r.description).slice(0, 160))}</p>` : ''}
            </div>
        </div>`).join('');
}

async function openResource(resourceId, title) {
    const listView = document.getElementById('dictListView');
    const detailView = document.getElementById('dictDetailView');
    const detail = document.getElementById('dictDetail');

    listView.classList.add('d-none');
    detailView.classList.remove('d-none');
    detail.innerHTML = '<div class="empty-state"><div class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></div>Loading the schematic…</div>';
    window.scrollTo({ top: 0, behavior: 'smooth' });

    let d;
    try {
        const res = await fetch(`/api/v1/fairstore/resource/${encodeURIComponent(resourceId)}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        d = await res.json();
    } catch (e) {
        detail.innerHTML = '<div class="empty-state text-danger"><i class="bi bi-exclamation-triangle me-2"></i>Could not load this dataset.</div>';
        return;
    }

    const cols = d.columns_detail || [];
    const rows = cols.map(c => `
        <tr>
            <td class="col-name">${escapeHtml(c.name)}${c.label && c.label !== c.name ? `<div class="col-stats">${escapeHtml(c.label)}</div>` : ''}</td>
            <td><span class="col-type">${escapeHtml(c.type || '—')}</span></td>
            <td class="col-stats">${renderStats(c.stats) || '<span class="text-muted">—</span>'}</td>
            <td class="col-top">${renderTopValues(c.top_values) || '<span class="text-muted">—</span>'}</td>
        </tr>`).join('');

    detail.innerHTML = `
        <div class="pane-header mb-3">
            <div>
                <h2 class="admin-page-title h4 mb-1">${escapeHtml(d.dataset_title || title || 'Dataset')}</h2>
                <div class="dict-meta">
                    <span><i class="bi bi-list-columns-reverse"></i>${escapeHtml(d.column_count)} columns</span>
                    <span><i class="bi bi-database"></i>${escapeHtml(d.record_count)} rows</span>
                </div>
                ${d.description ? `<p class="lib-answer mt-2 mb-0">${escapeHtml(d.description)}</p>` : ''}
            </div>
        </div>
        <div class="card p-0 dict-detail-wrap">
            <table class="col-table">
                <thead>
                    <tr><th>Column</th><th>Type</th><th>Statistics</th><th>Top values</th></tr>
                </thead>
                <tbody>${rows || '<tr><td colspan="4" class="text-muted p-3">No column detail available.</td></tr>'}</tbody>
            </table>
        </div>`;
}

function backToList() {
    document.getElementById('dictDetailView').classList.add('d-none');
    document.getElementById('dictListView').classList.remove('d-none');
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function debounce(fn, ms) {
    let t;
    return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

document.addEventListener('DOMContentLoaded', () => {
    const search = document.getElementById('dictSearch');
    const debounced = debounce(() => runSearch(search.value), 300);
    search.addEventListener('input', debounced);
    search.addEventListener('keydown', (e) => { if (e.key === 'Enter') runSearch(search.value); });

    // Delegated: open a card's resource detail (click or keyboard).
    document.getElementById('dictGrid').addEventListener('click', (e) => {
        const card = e.target.closest('.dict-card');
        if (card) openResource(card.dataset.resourceId, card.dataset.title);
    });
    document.getElementById('dictGrid').addEventListener('keydown', (e) => {
        if (e.key !== 'Enter' && e.key !== ' ') return;
        const card = e.target.closest('.dict-card');
        if (card) { e.preventDefault(); openResource(card.dataset.resourceId, card.dataset.title); }
    });

    document.getElementById('dictBack').addEventListener('click', backToList);

    // Deep link: /dictionary?q=census runs the search on load.
    const params = new URLSearchParams(window.location.search);
    const q = params.get('q');
    if (q) { search.value = q; runSearch(q); }
});
