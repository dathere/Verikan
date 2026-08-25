// Data Concierge - Admin Panel JavaScript

const API_BASE = '/api/v1';
let currentSubmissionId = null;

// =============================================================================
// Typed Standards independent verifier (https://typedstandards.org)
// -----------------------------------------------------------------------------
// The label reads "Typed Standards verified" with a check badge. It links out
// to typedstandards.org, which re-runs the checks in the reader's browser, so
// the claim stays independently confirmable rather than a blind assertion.
// =============================================================================
const TYPED_STANDARDS_VERIFY_URL = 'https://typedstandards.org/verify';

// Convert a GitHub blob (HTML) URL to its raw content URL so the verifier can
// fetch the published artifact directly; pass anything else through unchanged.
function typedStandardsRawUrl(githubUrl) {
    if (!githubUrl) return null;
    const m = githubUrl.match(/^https:\/\/github\.com\/([^/]+\/[^/]+)\/blob\/(.+)$/);
    return m ? `https://raw.githubusercontent.com/${m[1]}/${m[2]}` : githubUrl;
}

// Build the verifier deep-link. A hosted URL goes through ?url= (mirrors the
// Typed Standards badge's buildVerifyHref); with no URL we link to the bare
// verifier page so the reader can paste a hash or upload a bundle.
function typedStandardsVerifyHref(githubUrl) {
    const raw = typedStandardsRawUrl(githubUrl);
    return raw
        ? `${TYPED_STANDARDS_VERIFY_URL}?url=${encodeURIComponent(raw)}`
        : TYPED_STANDARDS_VERIFY_URL;
}

// Render the "Typed Standards verified" button.
function typedStandardsVerifyButton(githubUrl, extraClass = '') {
    const href = typedStandardsVerifyHref(githubUrl);
    return `<a href="${escapeHtml(href)}" target="_blank" rel="noopener"
               class="btn btn-sm btn-outline-secondary ${extraClass}"
               title="Independently verify this evidence at typedstandards.org — checks run in your browser">
                <i class="bi bi-patch-check-fill me-1"></i>Typed Standards verified
            </a>`;
}

/**
 * Wrapper around fetch that always sends credentials (cookies) and surfaces
 * the API error detail on non-2xx responses instead of a generic message.
 */
async function adminFetch(url, options = {}) {
    const response = await window.fetch(url, { credentials: 'include', ...options });
    if (!response.ok) {
        let detail = `${response.status} ${response.statusText}`;
        try {
            const body = await response.json();
            if (body.detail) detail = body.detail;
        } catch {}
        throw new Error(detail);
    }
    return response;
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadAdminIdentity();
    loadStats();
    loadOverviewActivity();
    loadPendingSubmissions();
    // Populate the Approved Members nav badge immediately — pending access
    // requests are time-sensitive and must be visible without opening the tab.
    loadPendingRequests();
    setupEventListeners();
});

function setupEventListeners() {
    // Tab changes
    const overviewTab = document.getElementById('overview-tab');
    if (overviewTab) {
        overviewTab.addEventListener('shown.bs.tab', () => {
            loadStats();
            loadOverviewActivity();
        });
    }
    document.getElementById('pending-tab').addEventListener('shown.bs.tab', loadPendingSubmissions);
    document.getElementById('all-tab').addEventListener('shown.bs.tab', loadAllSubmissions);
    document.getElementById('verified-tab').addEventListener('shown.bs.tab', loadVerifiedNotebooks);
    document.getElementById('settings-tab').addEventListener('shown.bs.tab', loadGitHubSettings);

    // Status filter
    document.getElementById('statusFilter').addEventListener('change', loadAllSubmissions);

    // Search
    document.getElementById('searchBtn').addEventListener('click', performSearch);
    document.getElementById('searchInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') performSearch();
    });

    // Approve/Reject buttons
    document.getElementById('approveBtn').addEventListener('click', () => reviewSubmission('approve'));
    document.getElementById('rejectBtn').addEventListener('click', () => reviewSubmission('reject'));

    // MCP Servers tab
    document.getElementById('mcp-tab').addEventListener('shown.bs.tab', loadMcpServers);
    document.getElementById('mcpAddServerBtn').addEventListener('click', addMcpServer);
    document.getElementById('mcpExecuteToolBtn').addEventListener('click', executeMcpTool);
    document.querySelectorAll('input[name="mcpTransport"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            const isStdio = e.target.value === 'stdio';
            document.getElementById('mcpStdioFields').classList.toggle('d-none', !isStdio);
            document.getElementById('mcpSseFields').classList.toggle('d-none', isStdio);
        });
    });

    // CKAN Sites tab
    document.getElementById('ckansites-tab').addEventListener('shown.bs.tab', loadCkanSites);
    document.getElementById('ckanAddSiteBtn').addEventListener('click', addCkanSite);
    const ckanQualityInput = document.getElementById('ckanSiteQuality');
    if (ckanQualityInput) {
        ckanQualityInput.addEventListener('input', (e) => {
            document.getElementById('ckanSiteQualityValue').textContent =
                parseFloat(e.target.value).toFixed(2);
        });
    }

    // Approved members tab
    document.getElementById('approved-tab').addEventListener('shown.bs.tab', loadApprovedMembers);
    document.getElementById('addApprovedForm').addEventListener('submit', addApprovedMember);

    // Admins tab
    document.getElementById('admins-tab').addEventListener('shown.bs.tab', loadAdmins);
    document.getElementById('addAdminForm').addEventListener('submit', grantAdmin);

    // Query logs tab
    document.getElementById('querylogs-tab').addEventListener('shown.bs.tab', loadQueryLogs);
    document.getElementById('qlRefreshBtn').addEventListener('click', loadQueryLogs);

    // Notebook reviews tab
    document.getElementById('nbreviews-tab').addEventListener('shown.bs.tab', loadNotebookReviews);
    document.getElementById('nbrRefreshBtn').addEventListener('click', loadNotebookReviews);
    document.getElementById('nbrStatusFilter').addEventListener('change', renderNotebookReviews);
    document.getElementById('nbrFindingsFilter').addEventListener('change', renderNotebookReviews);

    // Feedback tab
    document.getElementById('feedback-tab').addEventListener('shown.bs.tab', loadFeedback);
    document.getElementById('fbRefreshBtn').addEventListener('click', loadFeedback);
    document.getElementById('fbFilter').addEventListener('change', loadFeedback);
    document.getElementById('qlUserFilter').addEventListener('change', renderQueryLogs);
    document.getElementById('qlSourceFilter').addEventListener('change', renderQueryLogs);

    // Landing page settings (#109)
    document.getElementById('landing-tab').addEventListener('shown.bs.tab', loadLandingSettings);
    document.getElementById('landingSettingsForm').addEventListener('submit', saveLandingSettings);

    // System prompt editor
    document.getElementById('systemprompt-tab').addEventListener('shown.bs.tab', loadSystemPrompt);
    document.getElementById('spTarget').addEventListener('change', onSystemPromptTargetChange);
    document.getElementById('spSaveBtn').addEventListener('click', saveSystemPrompt);
    document.getElementById('spResetBtn').addEventListener('click', resetSystemPrompt);

    // Query settings (runtime)
    document.getElementById('runtime-tab').addEventListener('shown.bs.tab', loadRuntimeSettings);
    document.getElementById('rtSaveBtn').addEventListener('click', saveRuntimeSettings);
    document.getElementById('rtResetBtn').addEventListener('click', resetRuntimeSettings);

    // GitHub settings
    document.getElementById('githubSettingsForm').addEventListener('submit', saveGitHubSettings);
    document.getElementById('ghTestBtn').addEventListener('click', testGitHubConnection);
    // Secret-reveal toggles: flip the eye icon and expose the state to AT.
    const wireRevealToggle = (btnId, inputId, what) => {
        const btn = document.getElementById(btnId);
        btn.addEventListener('click', () => {
            const input = document.getElementById(inputId);
            const reveal = input.type === 'password';
            input.type = reveal ? 'text' : 'password';
            const icon = btn.querySelector('i');
            if (icon) icon.className = reveal ? 'bi bi-eye-slash' : 'bi bi-eye';
            btn.setAttribute('aria-pressed', reveal ? 'true' : 'false');
            btn.setAttribute('aria-label', `${reveal ? 'Hide' : 'Show'} ${what}`);
        });
    };
    wireRevealToggle('ghTokenToggle', 'ghToken', 'token');
    wireRevealToggle('ghWebhookSecretToggle', 'ghWebhookSecret', 'webhook secret');
    document.getElementById('ghPauseBtn').addEventListener('click', openPauseModal);
    document.getElementById('ghPauseConfirmBtn').addEventListener('click', confirmPauseToggle);
}

// Load Stats
async function loadStats() {
    try {
        const response = await adminFetch(`${API_BASE}/notebooks/admin/stats`);
        if (response.ok) {
            const stats = await response.json();
            document.getElementById('statTotal').textContent = stats.total_submissions || 0;
            document.getElementById('statPending').textContent = stats.pending || 0;
            document.getElementById('statApproved').textContent = stats.approved || 0;
            document.getElementById('statRejected').textContent = stats.rejected || 0;
            document.getElementById('statVerified').textContent = stats.total_verified || 0;
            document.getElementById('statUsage').textContent = stats.total_usage || 0;
            document.getElementById('pendingBadge').textContent = stats.pending || 0;
            renderStatusDonut(stats);
        }
    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

// =============================================================================
// Overview charts (dependency-free SVG; strokes use CSS vars so they adapt to
// the active theme automatically).
// =============================================================================

// Donut of submission status (pending / approved / rejected).
function renderStatusDonut(stats) {
    const host = document.getElementById('statusDonut');
    if (!host) return;

    const segments = [
        { label: 'Approved', value: stats.approved || 0, color: 'var(--dc-success)' },
        { label: 'Pending',  value: stats.pending || 0,  color: 'var(--dc-warning)' },
        { label: 'Rejected', value: stats.rejected || 0, color: 'var(--dc-danger)' },
    ];
    const total = segments.reduce((s, x) => s + x.value, 0);

    if (total === 0) {
        host.innerHTML = `<div class="activity-empty"><i class="bi bi-inbox me-1"></i>No submissions yet</div>`;
        return;
    }

    const R = 46, C = 2 * Math.PI * R, SW = 14;
    let acc = 0;
    const arcs = segments.filter(s => s.value > 0).map(s => {
        const frac = s.value / total;
        const len = frac * C;
        const rot = acc * 360 - 90;       // start at 12 o'clock
        acc += frac;
        return `<circle cx="60" cy="60" r="${R}" fill="none" stroke="${s.color}"
                    stroke-width="${SW}" stroke-dasharray="${len.toFixed(2)} ${(C - len).toFixed(2)}"
                    transform="rotate(${rot.toFixed(2)} 60 60)" stroke-linecap="butt"></circle>`;
    }).join('');

    const legend = segments.map(s => `
        <div class="lg-row">
            <span class="lg-dot" style="background:${s.color}"></span>
            <span>${s.label}</span>
            <span class="lg-val">${s.value}</span>
        </div>`).join('');

    host.innerHTML = `
        <div class="donut-wrap">
            <svg width="124" height="124" viewBox="0 0 120 120" role="img" aria-label="Submission status breakdown">
                <circle cx="60" cy="60" r="${R}" fill="none" stroke="var(--dc-surface-hover)" stroke-width="${SW}"></circle>
                ${arcs}
                <text x="60" y="58" text-anchor="middle" class="donut-center-num">${total}</text>
                <text x="60" y="74" text-anchor="middle" class="donut-center-lbl">Total</text>
            </svg>
            <div class="donut-legend">${legend}</div>
        </div>`;
}

// 14-day query-activity bar chart, sourced from the query logs.
async function loadOverviewActivity() {
    const host = document.getElementById('activityChart');
    if (!host) return;
    try {
        const response = await adminFetch(`${API_BASE}/admin/query-logs?limit=500`);
        if (!response.ok) throw new Error('Failed to load activity');
        const data = await response.json();
        renderActivityChart(data.logs || []);
    } catch (error) {
        host.innerHTML = `<div class="activity-empty">Activity unavailable</div>`;
    }
}

function renderActivityChart(logs) {
    const host = document.getElementById('activityChart');
    if (!host) return;

    const DAYS = 14;
    const buckets = [];
    const now = new Date();
    const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    for (let i = DAYS - 1; i >= 0; i--) {
        const d = new Date(startOfToday.getTime() - i * 86400000);
        buckets.push({ date: d, count: 0 });
    }
    const indexByKey = {};
    buckets.forEach((b, i) => { indexByKey[b.date.toDateString()] = i; });

    (logs || []).forEach(l => {
        if (!l.timestamp) return;
        const d = new Date(l.timestamp);
        if (isNaN(d)) return;
        const key = new Date(d.getFullYear(), d.getMonth(), d.getDate()).toDateString();
        if (key in indexByKey) buckets[indexByKey[key]].count += 1;
    });

    const max = Math.max(1, ...buckets.map(b => b.count));
    const totalCount = buckets.reduce((s, b) => s + b.count, 0);

    if (totalCount === 0) {
        host.innerHTML = `<div class="activity-empty"><i class="bi bi-graph-up me-1"></i>No queries in the last ${DAYS} days</div>`;
        return;
    }

    host.innerHTML = `<div class="activity-chart">` + buckets.map(b => {
        const pct = b.count === 0 ? 0 : Math.max(6, Math.round((b.count / max) * 100));
        const dayNum = b.date.getDate();
        const label = b.date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        return `
            <div class="activity-bar-col" title="${label}: ${b.count} quer${b.count === 1 ? 'y' : 'ies'}">
                <div class="activity-bar ${b.count === 0 ? 'empty' : ''}" style="height:${pct}%"></div>
                <div class="activity-bar-label">${dayNum}</div>
            </div>`;
    }).join('') + `</div>`;
}

// Load Pending Submissions
async function loadPendingSubmissions() {
    const container = document.getElementById('pendingList');

    try {
        const response = await adminFetch(`${API_BASE}/notebooks/submissions?status_filter=pending`);
        if (!response.ok) throw new Error('Failed to load');

        const data = await response.json();

        if (data.submissions.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted py-5">
                    <i class="bi bi-check-circle display-4 text-success"></i>
                    <p class="mt-3">No pending submissions to review</p>
                </div>
            `;
            return;
        }

        container.innerHTML = data.submissions.map(sub => renderSubmissionCard(sub, true)).join('');
        bindClick(container, '.js-review-submission', (el) => openReviewModal(el.dataset.submissionId));
    } catch (error) {
        container.innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle me-2"></i>
                Failed to load submissions: ${error.message}
            </div>
        `;
    }
}

// Load All Submissions
async function loadAllSubmissions() {
    const container = document.getElementById('allList');
    const filter = document.getElementById('statusFilter').value;

    try {
        const url = filter === 'all'
            ? `${API_BASE}/notebooks/submissions`
            : `${API_BASE}/notebooks/submissions?status_filter=${filter}`;

        const response = await adminFetch(url);
        if (!response.ok) throw new Error('Failed to load');

        const data = await response.json();

        if (data.submissions.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted py-5">
                    <i class="bi bi-inbox display-4"></i>
                    <p class="mt-3">No submissions found</p>
                </div>
            `;
            return;
        }

        container.innerHTML = data.submissions.map(sub => renderSubmissionCard(sub, sub.status === 'pending')).join('');
        bindClick(container, '.js-review-submission', (el) => openReviewModal(el.dataset.submissionId));
    } catch (error) {
        container.innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle me-2"></i>
                Failed to load: ${error.message}
            </div>
        `;
    }
}

// Load Verified Notebooks
async function syncVerifiedFromGitHub() {
    const btn = document.getElementById('syncFromGithubBtn');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Syncing...';
    }
    try {
        const response = await adminFetch(`${API_BASE}/verified-notebooks/sync-from-github`, {
            method: 'POST',
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || 'Sync failed');
        }
        const updated = (data.updated || []).length;
        const failed = (data.failed || []).length;
        const missing = (data.missing_path || []).length;
        showToast(
            `Synced ${updated} notebook(s) from GitHub` +
            (failed ? ` · ${failed} failed` : '') +
            (missing ? ` · ${missing} not on GitHub` : ''),
            failed ? 'warning' : 'success',
        );
        loadVerifiedNotebooks();
    } catch (error) {
        showToast('Sync failed: ' + error.message, 'danger');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="bi bi-cloud-download me-1"></i>Sync from GitHub';
        }
    }
}

// Build + store a Typed Standards evidence package for every verified notebook,
// so each gets a resolvable verify link. Mirrors the sync button's UX.
async function backfillEvidence() {
    const btn = document.getElementById('backfillEvidenceBtn');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Backfilling...';
    }
    try {
        // adminFetch throws on any non-2xx, so a returned response is always ok.
        const response = await adminFetch(`${API_BASE}/verified-notebooks/backfill-evidence`, {
            method: 'POST',
        });
        const data = await response.json();
        const stored = data.stored || 0;
        const total = data.count || 0;
        const failed = Math.max(0, total - stored);
        if (stored === 0) {
            showToast(
                `No packages stored (${total} notebook(s)). Is evidence signing enabled on the server?`,
                'warning',
            );
        } else {
            // Surface partial failures the way the Sync button does.
            showToast(
                `Minted ${stored} of ${total} evidence package(s)` + (failed ? ` · ${failed} failed` : ''),
                failed ? 'warning' : 'success',
            );
        }
        loadVerifiedNotebooks();
    } catch (error) {
        showToast('Backfill failed: ' + error.message, 'danger');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="bi bi-shield-check me-1"></i>Backfill evidence';
        }
    }
}

// Copy a notebook's Typed Standards verify link to the clipboard.
function copyVerifyLink(url) {
    if (!url) return;
    const done = () => showToast('Verify link copied to clipboard', 'success');
    const fail = () => showToast('Could not copy — link: ' + url, 'warning');
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(url).then(done, fail);
    } else {
        fail();
    }
}

async function loadVerifiedNotebooks() {
    const container = document.getElementById('verifiedList');

    try {
        const response = await adminFetch(`${API_BASE}/verified-notebooks`);
        if (!response.ok) throw new Error('Failed to load');

        const data = await response.json();

        if (data.notebooks.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted py-5">
                    <i class="bi bi-collection display-4"></i>
                    <p class="mt-3">No verified notebooks yet</p>
                    <small>Approve pending submissions to build the library</small>
                </div>
            `;
            return;
        }

        container.innerHTML = data.notebooks.map(nb => {
            const ghPathHtml = nb.github_url
                ? `<a href="${escapeHtml(nb.github_url)}" target="_blank" rel="noopener"><code>${escapeHtml(nb.github_path)}</code> <i class="bi bi-box-arrow-up-right"></i></a>`
                : `<code>${escapeHtml(nb.github_path)}</code>`;
            const ghLine = nb.github_path
                ? `<div class="meta small"><i class="bi bi-github me-1"></i>Source of truth: ${ghPathHtml}${nb.github_synced_at ? ` · synced ${formatDate(nb.github_synced_at)}` : ''}</div>`
                : '';
            const nbInTok = nb.input_tokens != null ? nb.input_tokens.toLocaleString() : '';
            const nbOutTok = nb.output_tokens != null ? nb.output_tokens.toLocaleString() : '';
            const nbTotalTok = (nb.input_tokens || 0) + (nb.output_tokens || 0);
            const nbTokensHtml = nbTotalTok > 0
                ? ` · <i class="bi bi-cpu me-1"></i><span title="In: ${nbInTok} / Out: ${nbOutTok}">${nbTotalTok.toLocaleString()} tokens</span>`
                : '';
            const nbConfHtml = nb.confidence ? ` · ${Math.round(nb.confidence * 100)}% confidence` : '';
            return `
            <div class="submission-card">
                <div class="d-flex justify-content-between align-items-start">
                    <div>
                        <div class="query">${escapeHtml(nb.query)}</div>
                        <div class="meta">
                            ${nb.submitted_by ? `<i class="bi bi-person-circle me-1"></i>Submitted by <strong>${escapeHtml(nb.submitted_by)}</strong> · ` : ''}
                            Verified by ${escapeHtml(nb.verified_by)} ·
                            ${formatDate(nb.verified_at)} ·
                            Used ${nb.usage_count} time(s)${nbConfHtml}${nbTokensHtml}
                        </div>
                        ${ghLine}
                        ${nb.tags && nb.tags.length > 0 ? `
                            <div class="mt-2">
                                ${nb.tags.slice(0, 5).map(tag => `<span class="badge bg-secondary me-1">${escapeHtml(tag)}</span>`).join('')}
                            </div>
                        ` : ''}
                    </div>
                    <div>
                        <span class="status-badge status-approved">VERIFIED</span>
                    </div>
                </div>
                <div class="mt-3 d-flex flex-wrap gap-2">
                    <button type="button" class="btn btn-sm btn-primary js-open-notebook"
                            data-notebook-id="${escapeHtml(nb.notebook_id)}">
                        <i class="bi bi-eye me-1"></i>View
                    </button>
                    <a href="${API_BASE}/verified-notebooks/${nb.notebook_id}/download"
                       class="btn btn-sm btn-outline-secondary">
                        <i class="bi bi-download me-1"></i>Download
                    </a>
                    ${nb.github_url ? `
                        <a href="${escapeHtml(nb.github_url)}" target="_blank" rel="noopener"
                           class="btn btn-sm btn-outline-secondary">
                            <i class="bi bi-github me-1"></i>View on GitHub
                        </a>
                    ` : ''}
                    ${nb.evidence_verify_url ? `
                        <a href="${escapeHtml(nb.evidence_verify_url)}" target="_blank" rel="noopener"
                           class="btn btn-sm btn-outline-secondary"
                           title="Independently verify this signed package at typedstandards.org">
                            <i class="bi bi-patch-check-fill me-1"></i>Typed Standards verified
                        </a>
                        <button type="button" class="btn btn-sm btn-outline-secondary js-copy-verify"
                                data-verify-url="${escapeHtml(nb.evidence_verify_url)}"
                                title="Copy this notebook's verify link">
                            <i class="bi bi-clipboard me-1"></i>Copy verify link
                        </button>
                    ` : typedStandardsVerifyButton(nb.github_url, '')}
                </div>
            </div>
        `;
        }).join('');

        // Wire actions via data attributes + listeners rather than inline
        // onclick handlers, so the values never enter a JS execution context.
        bindClick(container, '.js-open-notebook', (el) => openVerifiedNotebook(el.dataset.notebookId));
        bindClick(container, '.js-copy-verify', (el) => copyVerifyLink(el.dataset.verifyUrl));
    } catch (error) {
        container.innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle me-2"></i>
                Failed to load: ${error.message}
            </div>
        `;
    }
}

// Perform Search
async function performSearch() {
    const query = document.getElementById('searchInput').value.trim();
    const container = document.getElementById('searchResults');

    if (!query) {
        container.innerHTML = '<p class="text-muted">Enter a search query</p>';
        return;
    }

    try {
        const response = await adminFetch(`${API_BASE}/verified-notebooks/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, threshold: 0.1, max_results: 10 })
        });

        if (!response.ok) throw new Error('Search failed');

        const data = await response.json();

        if (data.results.length === 0) {
            container.innerHTML = `
                <div class="alert alert-info">
                    <i class="bi bi-info-circle me-2"></i>
                    No matching verified notebooks found
                </div>
            `;
            return;
        }

        container.innerHTML = `
            <p class="text-success mb-3"><i class="bi bi-check-circle me-2"></i>Found ${data.results.length} match(es)</p>
            ${data.results.map(r => `
                <div class="search-result">
                    <div class="d-flex justify-content-between align-items-start">
                        <div>
                            <div class="fw-medium">${escapeHtml(r.query)}</div>
                            <small class="text-muted">Used ${r.usage_count} time(s)</small>
                        </div>
                        <div class="score">${Math.round(r.similarity_score * 100)}%</div>
                    </div>
                    ${r.answer ? `<p class="mt-2 mb-0 small text-muted">${escapeHtml(r.answer.substring(0, 150))}...</p>` : ''}
                </div>
            `).join('')}
        `;
    } catch (error) {
        container.innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle me-2"></i>
                Search failed: ${error.message}
            </div>
        `;
    }
}

// Render Submission Card
function renderSubmissionCard(sub, showActions = false) {
    const statusClass = {
        pending: 'status-pending',
        approved: 'status-approved',
        rejected: 'status-rejected'
    }[sub.status] || '';

    const submitterLabel = escapeHtml(sub.submitted_by || 'anonymous');
    const submitterEmail = sub.submitter_email ? escapeHtml(sub.submitter_email) : '';
    const submitterAuth = sub.submitter_auth_type ? escapeHtml(sub.submitter_auth_type) : '';
    const submitterDetails = [
        submitterEmail && submitterEmail !== submitterLabel
            ? `<a href="mailto:${submitterEmail}" class="text-decoration-none">${submitterEmail}</a>`
            : '',
        submitterAuth ? `via ${submitterAuth}` : '',
    ].filter(Boolean).join(' · ');

    const subInTok = sub.input_tokens != null ? sub.input_tokens.toLocaleString() : '';
    const subOutTok = sub.output_tokens != null ? sub.output_tokens.toLocaleString() : '';
    const subTotalTok = (sub.input_tokens || 0) + (sub.output_tokens || 0);
    const subTokensHtml = subTotalTok > 0
        ? ` · <i class="bi bi-cpu me-1"></i><span title="In: ${subInTok} / Out: ${subOutTok}">${subTotalTok.toLocaleString()} tokens</span>`
        : '';

    return `
        <div class="submission-card">
            <div class="d-flex justify-content-between align-items-start">
                <div style="flex: 1;">
                    <div class="query">${escapeHtml(sub.query)}</div>
                    <div class="meta">
                        <i class="bi bi-person-circle me-1"></i>
                        Asked by <strong>${submitterLabel}</strong>${submitterDetails ? ` (${submitterDetails})` : ''}
                        · ${formatDate(sub.submitted_at)}
                        ${sub.confidence ? ` · ${Math.round(sub.confidence * 100)}% confidence` : ''}${subTokensHtml}
                    </div>
                    ${sub.tags && sub.tags.length > 0 ? `
                        <div class="mt-2">
                            ${sub.tags.slice(0, 5).map(tag => `<span class="badge bg-secondary me-1">${escapeHtml(tag)}</span>`).join('')}
                        </div>
                    ` : ''}
                </div>
                <div>
                    <span class="status-badge ${statusClass}">${sub.status.toUpperCase()}</span>
                </div>
            </div>
            ${sub.answer ? `
                <div class="mt-3 p-3 rounded-code-block rounded small">
                    ${escapeHtml(sub.answer.substring(0, 200))}${sub.answer.length > 200 ? '...' : ''}
                </div>
            ` : ''}
            ${sub.admin_notes ? `
                <div class="mt-2 small text-muted">
                    <i class="bi bi-chat-left-text me-1"></i>
                    ${escapeHtml(sub.admin_notes)}
                </div>
            ` : ''}
            ${showActions ? `
                <div class="mt-3">
                    <button class="btn btn-sm btn-primary js-review-submission" data-submission-id="${escapeHtml(sub.submission_id)}">
                        <i class="bi bi-eye me-1"></i>Review
                    </button>
                </div>
            ` : ''}
        </div>
    `;
}

// Open Review Modal
async function openReviewModal(submissionId) {
    currentSubmissionId = submissionId;

    // Start loading logs immediately (don't wait for notebook rendering)
    const logsContainer = document.getElementById('agentLogsContainer');
    logsContainer.innerHTML = '<p class="text-muted p-3">Loading logs...</p>';
    document.getElementById('logBadge').classList.add('d-none');
    document.getElementById('reviewFindingsBadge').classList.add('d-none');
    document.getElementById('methodReviewContainer').innerHTML =
        '<p class="text-muted p-3">Loading review…</p>';
    const logsPromise = loadAgentLogs(submissionId);

    try {
        const response = await adminFetch(`${API_BASE}/notebooks/submissions/${submissionId}`);
        if (!response.ok) throw new Error('Failed to load submission');

        const sub = await response.json();

        document.getElementById('reviewQuery').textContent = sub.query;

        // Verification + adversarial review verdict for the originating
        // query (#131). New submissions carry query_id; older auto-submits
        // embed it in the filename.
        let vQueryId = sub.query_id || null;
        if (!vQueryId && sub.filename) {
            const m = sub.filename.match(/^auto_([0-9a-f-]{36})\.ipynb$/);
            if (m) vQueryId = m[1];
        }
        loadMethodReview(vQueryId);

        // Show a link to the published GitHub artifact when this submission has
        // already been approved into the verified library (#111).
        const ghLink = document.getElementById('reviewGithubLink');
        if (sub.github_url) {
            ghLink.href = sub.github_url;
            ghLink.classList.remove('d-none');
        } else {
            ghLink.removeAttribute('href');
            ghLink.classList.add('d-none');
        }

        document.getElementById('reviewAnswer').innerHTML = sub.answer
            ? marked.parse(sub.answer)
            : '<em class="text-muted">No answer available</em>';

        const notebookInfo = document.getElementById('reviewNotebookInfo');
        const downloadBtn = document.getElementById('reviewDownloadBtn');
        const notebookCells = document.getElementById('reviewNotebookCells');

        if (sub.notebook_json) {
            const cells = sub.notebook_json.cells || [];
            notebookInfo.textContent = `${cells.length} cells`;

            const blob = new Blob([JSON.stringify(sub.notebook_json, null, 2)], { type: 'application/json' });
            downloadBtn.href = URL.createObjectURL(blob);
            downloadBtn.download = sub.filename || `submission_${submissionId.substring(0, 8)}.ipynb`;
            downloadBtn.classList.remove('d-none');

            if (cells.length > 0) {
                try {
                    notebookCells.innerHTML = cells.map((cell, i) => renderReviewCell(cell, i)).join('');
                    notebookCells.querySelectorAll('pre code').forEach(block => {
                        if (window.hljs) hljs.highlightElement(block);
                    });
                } catch (renderErr) {
                    console.error('[admin] Error rendering notebook cells:', renderErr);
                    notebookCells.innerHTML = '<p class="text-muted p-3 mb-0 small">Error rendering notebook cells</p>';
                }
            } else {
                notebookCells.innerHTML = '<p class="text-muted p-3 mb-0 small">Empty notebook</p>';
            }
        } else {
            notebookInfo.textContent = 'No notebook';
            downloadBtn.classList.add('d-none');
            notebookCells.innerHTML = '<p class="text-muted p-3 mb-0 small">No notebook attached</p>';
        }

        const notesEl = document.getElementById('adminNotes');
        notesEl.value = '';
        notesEl.classList.remove('is-invalid');

        const modal = new bootstrap.Modal(document.getElementById('reviewModal'));
        modal.show();
    } catch (error) {
        console.error('[admin] Failed to load submission:', error);
        showToast('Failed to load submission: ' + error.message, 'danger');
    }
}

// Fetch and render the execution + adversarial-review verdict (#131) into
// the review modal's Method Review tab.
async function loadMethodReview(queryId) {
    const container = document.getElementById('methodReviewContainer');
    const badge = document.getElementById('reviewFindingsBadge');
    if (!queryId) {
        container.innerHTML = `<div class="empty-state p-4">
            <i class="bi bi-clipboard2-x"></i>
            <p class="mb-0">No verification record is linked to this submission.</p>
        </div>`;
        return;
    }
    let record;
    try {
        const res = await window.fetch(`${API_BASE}/notebooks/${encodeURIComponent(queryId)}/verification`,
            { credentials: 'include' });
        if (!res.ok) throw new Error(`${res.status}`);
        record = await res.json();
    } catch (e) {
        container.innerHTML = '<p class="text-muted p-3">Could not load the verification record.</p>';
        return;
    }
    container.innerHTML = renderVerificationReport(record);

    const findings = ((record.review || {}).findings) || [];
    if (findings.length > 0) {
        badge.textContent = findings.length;
        const serious = findings.some(f => f.severity === 'critical' || f.severity === 'high');
        badge.className = `badge ms-1 ${serious ? 'bg-danger' : 'bg-warning text-dark'}`;
    } else if ((record.review || {}).reviewed) {
        badge.textContent = '✓';
        badge.className = 'badge ms-1 bg-success';
    }
}

// Shared, properly formatted report for one verification/review record.
// Used by the Method Review tab and the Notebook Reviews pane details.
function renderVerificationReport(record) {
    if (!record || record.status === 'not_found') {
        return '<p class="text-muted p-3 mb-0">No verification has run for this notebook yet.</p>';
    }
    if (record.status === 'disabled') {
        return '<p class="text-muted p-3 mb-0">Notebook verification is disabled on this deployment.</p>';
    }
    if (record.status === 'pending') {
        return `<p class="text-muted p-3 mb-0"><i class="bi bi-hourglass-split me-2"></i>
            Verification is still running — the notebook is being executed and reviewed.</p>`;
    }

    const v = record.verdict || {};
    const r = record.review || {};
    const chips = [];

    if (record.status === 'error') {
        chips.push('<span class="badge bg-danger">verifier error</span>');
    }
    if (v.executed === true) {
        chips.push(`<span class="badge bg-success"><i class="bi bi-play-circle me-1"></i>Executed ${escapeHtml(String(v.cells_executed ?? ''))}/${escapeHtml(String(v.cells_total ?? ''))} cells</span>`);
    } else if (v.executed === false && v.score === 0) {
        chips.push(`<span class="badge bg-danger"><i class="bi bi-x-circle me-1"></i>Execution failed</span>`);
    } else if (v.reason) {
        chips.push(`<span class="badge bg-secondary" title="${escapeAttrSafe(v.reason)}">Execution not measured</span>`);
    }
    const claims = (v.claimed_values || []).length;
    if (v.executed && claims > 0) {
        const rec = (v.reconciled_values || []).length;
        const cls = rec === claims ? 'bg-success' : (rec === 0 ? 'bg-danger' : 'bg-warning text-dark');
        chips.push(`<span class="badge ${cls}" title="Numeric claims in the answer re-derived by the notebook's own output">Reconciled ${rec}/${claims} claims</span>`);
    }
    if (typeof r.score === 'number') {
        const pct = Math.round(r.score * 100);
        const cls = pct >= 80 ? 'bg-success' : (pct >= 50 ? 'bg-warning text-dark' : 'bg-danger');
        chips.push(`<span class="badge ${cls}" title="Adversarial method-review score">Review ${pct}%</span>`);
    }
    if (typeof record.combined_score === 'number') {
        chips.push(`<span class="badge bg-secondary" title="Combined verification factor (execution 70% / review 30%)">Combined ${Math.round(record.combined_score * 100)}%</span>`);
    }
    if (typeof record.confidence_before === 'number' && typeof record.confidence === 'number') {
        const delta = record.confidence - record.confidence_before;
        const cls = delta > 0.005 ? 'text-success' : (delta < -0.005 ? 'text-danger' : 'text-muted');
        chips.push(`<span class="badge bg-secondary">Confidence ${Math.round(record.confidence_before * 100)}% → <span class="${cls}">${Math.round(record.confidence * 100)}%</span></span>`);
    }

    let body = `<div class="d-flex flex-wrap gap-2 mb-3">${chips.join('')}</div>`;

    if (r.summary) {
        body += `<p class="mb-3" style="white-space:pre-wrap;">${escapeHtml(r.summary)}</p>`;
    }

    const findings = r.findings || [];
    if (findings.length > 0) {
        const order = { critical: 0, high: 1, medium: 2, low: 3 };
        const sorted = [...findings].sort((a, b) => (order[a.severity] ?? 9) - (order[b.severity] ?? 9));
        body += sorted.map(f => `
            <div class="mb-3 pb-2" style="border-bottom: 1px solid var(--dc-border);">
                <div class="d-flex align-items-center gap-2 mb-1 flex-wrap">
                    ${severityBadge(f.severity, '')}
                    <strong>${escapeHtml(f.title || '')}</strong>
                    ${f.cell_index != null ? `<span class="text-muted small">cell ${escapeHtml(String(f.cell_index))}</span>` : ''}
                </div>
                <div class="small" style="white-space:pre-wrap; color: var(--dc-text-secondary);">${escapeHtml(f.detail || '')}</div>
            </div>`).join('');
    } else if (r.reviewed) {
        body += `<p class="text-success mb-2"><i class="bi bi-shield-check me-1"></i>The adversarial review found no issues with the notebook's method.</p>`;
    } else if (r.reason) {
        body += `<p class="text-muted mb-2">${escapeHtml(r.reason)}</p>`;
    }

    if (v.execution_error) {
        body += `
            <details class="mt-2">
                <summary class="text-muted small" style="cursor:pointer;">Execution error</summary>
                <pre class="rounded-code-block rounded p-2 mt-1 mb-0" style="white-space:pre-wrap;max-height:240px;overflow-y:auto;font-size:0.78rem;">${escapeHtml(v.execution_error)}</pre>
            </details>`;
    }
    if (record.error) {
        body += `<p class="text-danger small mb-0">${escapeHtml(record.error)}</p>`;
    }

    return `<div class="p-3">${body}</div>`;
}

// escapeHtml already covers quotes; alias kept for title-attribute call sites
// so intent is explicit at the usage.
function escapeAttrSafe(s) { return escapeHtml(String(s ?? '')); }

// Read-only viewer for an already-verified notebook (#72), including its
// verifier provenance (#73) and agent logs (#71). Unlike openReviewModal,
// this has no approve/reject controls — the notebook is already in the
// library.
async function openVerifiedNotebook(notebookId) {
    const cellsContainer = document.getElementById('viewNotebookCells');
    const logsContainer = document.getElementById('viewAgentLogsContainer');
    cellsContainer.innerHTML = '<p class="text-muted p-3">Loading notebook...</p>';
    logsContainer.innerHTML = '<p class="text-muted p-3">Loading logs...</p>';
    document.getElementById('viewLogBadge').classList.add('d-none');

    const modal = new bootstrap.Modal(document.getElementById('viewModal'));
    modal.show();

    try {
        const response = await adminFetch(`${API_BASE}/verified-notebooks/${notebookId}`);
        if (!response.ok) throw new Error('Failed to load verified notebook');

        const nb = await response.json();

        document.getElementById('viewQuery').textContent = nb.query || '';

        // Verifier provenance (#73): who verified it and when, plus any notes.
        const meta = [];
        if (nb.verified_by) meta.push(`Verified by <strong>${escapeHtml(nb.verified_by)}</strong>`);
        if (nb.verified_at) meta.push(formatDate(nb.verified_at));
        if (nb.submitted_by) meta.push(`Submitted by ${escapeHtml(nb.submitted_by)}`);
        document.getElementById('viewMeta').innerHTML = meta.join(' · ');

        const notesEl = document.getElementById('viewAdminNotes');
        if (nb.admin_notes) {
            notesEl.innerHTML = `<i class="bi bi-chat-left-text me-1"></i><strong>Reviewer note:</strong> ${escapeHtml(nb.admin_notes)}`;
            notesEl.classList.remove('d-none');
        } else {
            notesEl.classList.add('d-none');
        }

        // GitHub link, mirroring the review modal.
        const ghLink = document.getElementById('viewGithubLink');
        if (nb.github_url) {
            ghLink.href = nb.github_url;
            ghLink.classList.remove('d-none');
        } else {
            ghLink.removeAttribute('href');
            ghLink.classList.add('d-none');
        }

        // Typed Standards verify call-to-action (always available — falls back
        // to the bare verifier page when there's no published GitHub URL).
        const verifyLink = document.getElementById('viewVerifyLink');
        if (verifyLink) verifyLink.href = typedStandardsVerifyHref(nb.github_url);

        const cells = (nb.notebook_json && nb.notebook_json.cells) || [];
        document.getElementById('viewNotebookInfo').textContent = `${cells.length} cells`;
        if (cells.length > 0) {
            try {
                cellsContainer.innerHTML = cells.map((cell, i) => renderReviewCell(cell, i)).join('');
                cellsContainer.querySelectorAll('pre code').forEach(block => {
                    if (window.hljs) hljs.highlightElement(block);
                });
            } catch (renderErr) {
                console.error('[admin] Error rendering verified notebook cells:', renderErr);
                cellsContainer.innerHTML = '<p class="text-muted p-3 mb-0 small">Error rendering notebook cells</p>';
            }
        } else {
            cellsContainer.innerHTML = '<p class="text-muted p-3 mb-0 small">No notebook cells</p>';
        }

        // Agent logs (#71): keyed by the originating submission_id, retained
        // after approval. Older verified notebooks may predate log capture.
        if (nb.submission_id) {
            loadAgentLogs(nb.submission_id, 'viewAgentLogsContainer', 'viewLogBadge');
        } else {
            logsContainer.innerHTML = '<p class="text-muted p-3 small">No agent logs available for this notebook.</p>';
        }
    } catch (error) {
        console.error('[admin] Failed to load verified notebook:', error);
        cellsContainer.innerHTML = `<p class="text-danger p-3 small">Failed to load: ${escapeHtml(error.message)}</p>`;
        logsContainer.innerHTML = '<p class="text-muted p-3 small">Failed to load logs.</p>';
    }
}

function renderReviewCell(cell, index) {
    const isCode = cell.cell_type === 'code';
    const source = Array.isArray(cell.source) ? cell.source.join('') : (cell.source || '');
    const labelHtml = isCode
        ? `<span class="cell-type-badge cell-type-code me-2">Code [${index + 1}]</span>`
        : `<span class="cell-type-badge cell-type-markdown me-2">Markdown [${index + 1}]</span>`;

    if (isCode) {
        let outputHtml = '';
        const outputs = cell.outputs || [];
        if (outputs.length > 0) {
            const outputText = outputs.map(out => {
                if (out.text) return Array.isArray(out.text) ? out.text.join('') : out.text;
                if (out.data && out.data['text/plain']) {
                    const p = out.data['text/plain'];
                    return Array.isArray(p) ? p.join('') : p;
                }
                return '';
            }).join('\n').trim();
            if (outputText) {
                outputHtml = `<div style="background-color:var(--dc-code-bg);padding:0.5rem 1rem;font-family:monospace;font-size:0.8rem;white-space:pre-wrap;color:var(--dc-text-secondary);border-top:1px solid var(--dc-border);">${escapeHtml(outputText)}</div>`;
            }
        }
        return `
            <div style="border-bottom:1px solid var(--dc-border);">
                <div style="padding:0.35rem 0.75rem;background-color:var(--dc-bg-alt);display:flex;align-items:center;">${labelHtml}</div>
                <pre style="margin:0;padding:0.75rem 1rem;background-color:var(--dc-code-bg);font-size:0.82rem;overflow-x:auto;"><code class="language-python">${escapeHtml(source)}</code></pre>
                ${outputHtml}
            </div>`;
    } else {
        return `
            <div style="border-bottom:1px solid var(--dc-border);">
                <div style="padding:0.35rem 0.75rem;background-color:var(--dc-bg-alt);display:flex;align-items:center;">${labelHtml}</div>
                <div style="padding:0.75rem 1rem;background-color:var(--dc-surface);font-size:0.88rem;">${marked.parse(source)}</div>
            </div>`;
    }
}

async function loadAgentLogs(submissionId, containerId = 'agentLogsContainer', badgeId = 'logBadge') {
    const container = document.getElementById(containerId);
    const badge = badgeId ? document.getElementById(badgeId) : null;

    try {
        const url = `${API_BASE}/notebooks/submissions/${submissionId}/logs`;
        console.log('[admin] Loading agent logs:', url);
        const response = await window.fetch(url, { credentials: 'include' });

        if (!response.ok) {
            console.warn('[admin] Logs endpoint returned', response.status);
            container.innerHTML = '<p class="text-muted p-3 small">No logs available for this submission.</p>';
            return;
        }

        const data = await response.json();
        const log = data.log || [];
        console.log('[admin] Agent log entries:', log.length);

        if (log.length === 0) {
            container.innerHTML = '<p class="text-muted p-3 small">No logs available for this submission.</p>';
            return;
        }

        if (badge) {
            badge.textContent = log.length;
            badge.classList.remove('d-none');
        }

        container.innerHTML = log.map((entry, i) => renderLogEntry(entry, i)).join('');
    } catch (err) {
        console.error('[admin] Failed to load agent logs:', err);
        container.innerHTML = '<p class="text-muted p-3 small">Failed to load logs.</p>';
    }
}

function renderLogEntry(entry, index) {
    const ts = entry.timestamp
        ? `<span class="text-muted small me-2">${escapeHtml(entry.timestamp.replace('T', ' ').substring(0, 19))}</span>`
        : '';

    if (entry.type === 'session_start') {
        const tools = (entry.tools_available || []).map(t => `<code>${escapeHtml(t)}</code>`).join(', ');
        return `
            <div class="border-bottom p-3">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span class="badge bg-secondary"><i class="bi bi-play-circle me-1"></i>Session Start</span>
                    <span class="text-muted small">${ts}model: ${escapeHtml(entry.model || '?')} · source: ${escapeHtml(entry.data_source || '?')}</span>
                </div>
                <div class="small mb-1"><strong class="text-muted">Query:</strong> ${escapeHtml(entry.query || '')}</div>
                ${tools ? `<div class="small mb-1"><strong class="text-muted">Tools:</strong> ${tools}</div>` : ''}
                ${entry.system_prompt ? `
                <details class="small">
                    <summary class="text-muted" style="cursor:pointer;">System prompt (${entry.system_prompt.length} chars · sha256 ${escapeHtml((entry.system_prompt_sha256 || '').substring(0, 12))}…)</summary>
                    <pre class="rounded-code-block rounded p-2 mt-1 mb-0" style="white-space:pre-wrap;max-height:300px;overflow-y:auto;font-size:0.78rem;">${escapeHtml(entry.system_prompt)}</pre>
                </details>` : ''}
            </div>`;
    }

    if (entry.type === 'llm_response') {
        const texts = (entry.texts || []).map(t => escapeHtml(t)).join('\n\n');
        const toolCalls = (entry.tool_calls || []).map(tc =>
            `<div class="ms-3 mt-1"><code class="text-info">${escapeHtml(tc.tool)}</code>(<span class="text-warning">${escapeHtml(JSON.stringify(tc.input).substring(0, 200))}</span>)</div>`
        ).join('');
        const tokens = entry.tokens || {};
        const cacheTokens = (tokens.cache_creation_input || 0) + (tokens.cache_read_input || 0);
        const tokenStr = `${tokens.input || 0}+${tokens.output || 0}${cacheTokens ? `+${cacheTokens} cache` : ''} tokens`;
        const modelStr = entry.model ? ` · ${escapeHtml(entry.model)}` : '';

        return `
            <div class="border-bottom p-3">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span class="badge bg-primary">LLM Response · Iteration ${entry.iteration || '?'}</span>
                    <span class="text-muted small">${ts}${entry.duration_ms || 0}ms · ${tokenStr} · stop: ${escapeHtml(entry.stop_reason || '?')}${modelStr}</span>
                </div>
                ${texts ? `<div class="rounded-code-block rounded p-2 mb-2 small" style="white-space:pre-wrap;max-height:200px;overflow-y:auto;">${texts}</div>` : ''}
                ${toolCalls ? `<div class="small"><strong class="text-muted">Tool calls:</strong>${toolCalls}</div>` : ''}
            </div>`;
    }

    if (entry.type === 'tool_execution') {
        const result = escapeHtml((entry.result || '').substring(0, 1500));
        const input = escapeHtml(JSON.stringify(entry.input || {}).substring(0, 300));
        const isError = entry.status === 'error';
        const badgeClass = isError ? 'bg-danger' : 'bg-warning text-dark';
        const meta = [
            entry.source ? escapeHtml(entry.source) : null,
            entry.operation_type ? escapeHtml(entry.operation_type) : null,
            `${entry.duration_ms || 0}ms`,
        ].filter(Boolean).join(' · ');
        const resultChars = entry.result_chars || (entry.result || '').length;
        const truncNote = entry.result_truncated ? ' · truncated' : '';

        return `
            <div class="border-bottom p-3" style="background-color:var(--dc-code-bg);">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span class="badge ${badgeClass}"><i class="bi bi-gear me-1"></i>${escapeHtml(entry.tool || '?')}${isError ? ' · error' : ''}</span>
                    <span class="text-muted small">${ts}${meta}</span>
                </div>
                <div class="small mb-1"><strong class="text-muted">Input:</strong> <code>${input}</code></div>
                <details class="small">
                    <summary class="text-muted" style="cursor:pointer;">Result (${resultChars} chars${truncNote})</summary>
                    <pre class="rounded-code-block rounded p-2 mt-1 mb-0" style="white-space:pre-wrap;max-height:300px;overflow-y:auto;font-size:0.78rem;">${result}</pre>
                </details>
            </div>`;
    }

    if (entry.type === 'llm_retry' || entry.type === 'model_fallback') {
        const label = entry.type === 'llm_retry'
            ? `Retry ${entry.attempt || '?'} · ${escapeHtml(entry.model || '?')}`
            : `Fallback: ${escapeHtml(entry.from_model || '?')} → ${escapeHtml(entry.to_model || '?')}`;
        return `
            <div class="border-bottom p-2 px-3">
                <span class="badge bg-warning text-dark me-2"><i class="bi bi-arrow-repeat me-1"></i>${label}</span>
                <span class="small text-muted">${ts}${escapeHtml(entry.error || '')}</span>
            </div>`;
    }

    if (entry.type === 'error') {
        return `
            <div class="border-bottom p-3">
                <span class="badge bg-danger me-2"><i class="bi bi-x-circle me-1"></i>Error · Iteration ${entry.iteration || '?'}</span>
                <span class="small text-muted">${ts}${escapeHtml(entry.exception_type || '')}</span>
                <pre class="rounded-code-block rounded p-2 mt-2 mb-0 small" style="white-space:pre-wrap;">${escapeHtml(entry.error || '')}</pre>
            </div>`;
    }

    if (entry.type === 'summary') {
        const totals = entry.token_totals || {};
        const cacheTotal = (totals.cache_creation_input || 0) + (totals.cache_read_input || 0);
        const tokenStr = totals.input !== undefined
            ? ` · ${totals.input}+${totals.output || 0}${cacheTotal ? `+${cacheTotal} cache` : ''} tokens`
            : '';
        return `
            <div class="border-top p-3" style="background-color:var(--dc-surface-hover);">
                <span class="badge bg-success me-2">Summary</span>
                <span class="small text-muted">
                    ${entry.total_iterations || 0} iterations ·
                    ${entry.total_tool_calls || 0} tool calls ·
                    ${entry.total_elapsed_ms || 0}ms total ·
                    model: ${escapeHtml(entry.model || '?')} ·
                    source: ${escapeHtml(entry.data_source || '?')}${tokenStr}
                </span>
            </div>`;
    }

    return `<div class="border-bottom p-3 small text-muted">${escapeHtml(JSON.stringify(entry))}</div>`;
}

// Review Submission (Approve/Reject)
async function reviewSubmission(action) {
    if (!currentSubmissionId) return;

    const notesEl = document.getElementById('adminNotes');
    const adminNotes = notesEl.value.trim();
    // #112: a reason is mandatory for both approve and reject. It is stored
    // on the submission and written into the GitHub commit message.
    if (!adminNotes) {
        notesEl.classList.add('is-invalid');
        notesEl.focus();
        showToast('A reason is required to approve or reject.', 'danger');
        return;
    }
    notesEl.classList.remove('is-invalid');
    const endpoint = action === 'approve' ? 'approve' : 'reject';

    try {
        const response = await adminFetch(`${API_BASE}/notebooks/submissions/${currentSubmissionId}/${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                reviewed_by: 'admin',
                admin_notes: adminNotes
            })
        });

        if (!response.ok) throw new Error('Review failed');

        bootstrap.Modal.getInstance(document.getElementById('reviewModal')).hide();
        showToast(`Submission ${action === 'approve' ? 'approved' : 'rejected'} successfully!`,
                  action === 'approve' ? 'success' : 'warning');

        // Reload data
        loadStats();
        loadPendingSubmissions();

    } catch (error) {
        showToast('Failed to review submission: ' + error.message, 'danger');
    }
}

// Utilities
function formatDate(dateStr) {
    if (!dateStr) return '';
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// Escape a value for use as HTML text OR inside a double-quoted HTML attribute.
// Unlike the old textContent/innerHTML trick, this also escapes both quote
// characters, so an interpolated value can never break out of a quoted
// attribute (e.g. data-* attributes read back via element.dataset).
//
// NOTE: this is NOT sufficient for values placed inside a JS string literal in
// an inline event handler (e.g. onclick="fn('...')") — HTML-entity escaping is
// undone before the JS parser runs, so a "'" would still terminate the string.
// Such values must instead be carried in a data-* attribute and wired up with
// bindClick() below, so they never enter an executable context.
function escapeHtml(text) {
    if (!text) return '';
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

// Attach a click handler to every element matching `selector` within `root`.
// The handler receives the matched element, so per-item data travels via
// data-* attributes (element.dataset) instead of being concatenated into an
// inline onclick — the value is never parsed as executable attribute text.
function bindClick(root, selector, handler) {
    root.querySelectorAll(selector).forEach((el) => {
        el.addEventListener('click', () => handler(el));
    });
}

function showToast(message, type = 'info') {
    const toastEl = document.getElementById('toast');
    document.getElementById('toastBody').textContent = message;
    toastEl.className = `toast bg-${type === 'success' ? 'success' : type === 'danger' ? 'danger' : type === 'warning' ? 'warning' : 'body'}`;

    // Header icon + title follow the toast type (a green toast titled
    // "Notification" with a blue info icon read as unfinished).
    const [iconClass, title] = {
        success: ['bi bi-check-circle-fill me-2', 'Success'],
        danger: ['bi bi-exclamation-triangle-fill me-2', 'Error'],
        warning: ['bi bi-exclamation-circle-fill me-2', 'Warning'],
    }[type] || ['bi bi-info-circle me-2 text-primary', 'Notice'];
    const headerIcon = toastEl.querySelector('.toast-header i');
    const headerTitle = toastEl.querySelector('.toast-header strong');
    if (headerIcon) headerIcon.className = iconClass;
    if (headerTitle) headerTitle.textContent = title;

    new bootstrap.Toast(toastEl).show();
}

// =============================================================================
// GitHub Settings
// =============================================================================

async function loadGitHubSettings() {
    try {
        const response = await adminFetch(`${API_BASE}/settings/github`);
        if (!response.ok) throw new Error('Failed to load');

        const settings = await response.json();
        renderGitHubStatus(settings);
        document.getElementById('ghRepo').value = settings.repo || '';
        document.getElementById('ghBranch').value = settings.branch || 'main';
        document.getElementById('ghDraftsFolder').value = settings.drafts_folder || 'drafts';
        document.getElementById('ghVerifiedFolder').value = settings.verified_folder || 'verified';
        document.getElementById('ghVerifiedAnswersFolder').value =
            settings.verified_answers_folder || 'verified-answers';

        // Show token hint
        const hint = document.getElementById('ghTokenHint');
        if (settings.token_set) {
            hint.textContent = `Current token: ${settings.token_masked}. Leave blank to keep it.`;
        } else {
            hint.textContent = 'No token configured. Enter a GitHub personal access token.';
        }
        // Always clear the password field so it acts as "enter new"
        document.getElementById('ghToken').value = '';

        // Webhook secret hint — server only tells us whether one is set, never echoes the value
        const whHint = document.getElementById('ghWebhookSecretHint');
        if (settings.webhook_secret_set) {
            whHint.textContent =
                'Webhook secret configured. Leave blank to keep it, or enter a new value to rotate.';
        } else {
            whHint.textContent =
                'No webhook secret configured. Enter a shared secret to verify inbound GitHub webhooks.';
        }
        document.getElementById('ghWebhookSecret').value = '';
    } catch (error) {
        console.error('Failed to load GitHub settings:', error);
    }
}

async function saveGitHubSettings(e) {
    e.preventDefault();
    const alertEl = document.getElementById('githubSettingsAlert');

    const body = {
        token: document.getElementById('ghToken').value,
        repo: document.getElementById('ghRepo').value,
        branch: document.getElementById('ghBranch').value,
        drafts_folder: document.getElementById('ghDraftsFolder').value,
        verified_folder: document.getElementById('ghVerifiedFolder').value,
        verified_answers_folder: document.getElementById('ghVerifiedAnswersFolder').value,
        webhook_secret: document.getElementById('ghWebhookSecret').value,
    };

    try {
        const response = await adminFetch(`${API_BASE}/settings/github`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        if (!response.ok) throw new Error('Failed to save');

        alertEl.innerHTML = `
            <div class="alert alert-success alert-dismissible fade show">
                <i class="bi bi-check-circle me-2"></i>GitHub settings saved successfully.
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        loadGitHubSettings();
    } catch (error) {
        alertEl.innerHTML = `
            <div class="alert alert-danger alert-dismissible fade show">
                <i class="bi bi-exclamation-triangle me-2"></i>Failed to save settings: ${escapeHtml(error.message)}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
    }
}

// ---- Landing page settings (#109) ----
async function loadLandingSettings() {
    try {
        const response = await adminFetch(`${API_BASE}/settings/landing`);
        if (!response.ok) throw new Error('Failed to load');

        const s = await response.json();
        document.getElementById('lpTitle').value = s.title || '';
        document.getElementById('lpTagline').value = s.tagline || '';
        document.getElementById('lpShowBeta').checked = !!s.show_beta_badge;
        document.getElementById('lpLogoUrl').value = s.logo_url || '';
        document.getElementById('lpCallToAction').value = s.call_to_action || '';
        document.getElementById('lpSearchPlaceholder').value = s.search_placeholder || '';
        document.getElementById('lpTryAsking').value = s.try_asking_label || '';
        document.getElementById('lpPoweredByLabel').value = s.powered_by_label || '';
        document.getElementById('lpPoweredByUrl').value = s.powered_by_url || '';
        document.getElementById('lpSampleQuestions').value =
            (s.sample_questions || []).join('\n');
    } catch (error) {
        console.error('Failed to load landing settings:', error);
    }
}

async function saveLandingSettings(e) {
    e.preventDefault();
    const alertEl = document.getElementById('landingSettingsAlert');

    const sampleQuestions = document.getElementById('lpSampleQuestions').value
        .split('\n')
        .map(q => q.trim())
        .filter(q => q.length > 0);

    const body = {
        title: document.getElementById('lpTitle').value,
        tagline: document.getElementById('lpTagline').value,
        show_beta_badge: document.getElementById('lpShowBeta').checked,
        logo_url: document.getElementById('lpLogoUrl').value,
        call_to_action: document.getElementById('lpCallToAction').value,
        search_placeholder: document.getElementById('lpSearchPlaceholder').value,
        try_asking_label: document.getElementById('lpTryAsking').value,
        powered_by_label: document.getElementById('lpPoweredByLabel').value,
        powered_by_url: document.getElementById('lpPoweredByUrl').value,
        sample_questions: sampleQuestions,
    };

    try {
        const response = await adminFetch(`${API_BASE}/settings/landing`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!response.ok) throw new Error('Failed to save');

        alertEl.innerHTML = `
            <div class="alert alert-success alert-dismissible fade show">
                <i class="bi bi-check-circle me-2"></i>Landing page saved. Changes are live on the home page.
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        loadLandingSettings();
    } catch (error) {
        alertEl.innerHTML = `
            <div class="alert alert-danger alert-dismissible fade show">
                <i class="bi bi-exclamation-triangle me-2"></i>Failed to save: ${escapeHtml(error.message)}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
    }
}

// =============================================================================
// System Prompt editor
// =============================================================================

let _spSettings = null;
let _spTarget = 'ckan';

// Valid editor targets — each maps to `{base}_template` / `{base}_placeholders`
// / `{base}_is_custom` keys in the /settings/system-prompt payload.
const _SP_TARGETS = ['ckan', 'mcp', 'notebook_header', 'notebook_results', 'notebook_review'];

function _spBase() { return _SP_TARGETS.includes(_spTarget) ? _spTarget : 'ckan'; }
function _spKey() { return _spBase() + '_template'; }

function _spAlert(type, msg) {
    const icon = type === 'success' ? 'bi-check-circle' : 'bi-exclamation-triangle';
    document.getElementById('systemPromptAlert').innerHTML = `
        <div class="alert alert-${type} alert-dismissible fade show">
            <i class="bi ${icon} me-2"></i>${escapeHtml(msg)}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>`;
}

async function loadSystemPrompt() {
    try {
        const response = await adminFetch(`${API_BASE}/settings/system-prompt`);
        _spSettings = await response.json();
        _spTarget = document.getElementById('spTarget').value || 'ckan';
        renderSystemPrompt();
    } catch (error) {
        _spAlert('danger', 'Failed to load system prompt: ' + error.message);
    }
}

function renderSystemPrompt() {
    if (!_spSettings) return;
    const base = _spBase();
    document.getElementById('spTemplate').value = _spSettings[base + '_template'] || '';

    const placeholders = _spSettings[base + '_placeholders'];
    document.getElementById('spPlaceholders').innerHTML =
        (placeholders || []).map(p => `<code>{${escapeHtml(p)}}</code>`).join(' ');

    const isCustom = !!_spSettings[base + '_is_custom'];
    const badge = document.getElementById('spStatus');
    badge.textContent = isCustom ? 'Customized' : 'Default';
    badge.className = 'badge align-self-center ' + (isCustom ? 'bg-primary' : 'bg-secondary');
}

// Stash the current edit before switching targets so it isn't lost.
function onSystemPromptTargetChange() {
    if (_spSettings) _spSettings[_spKey()] = document.getElementById('spTemplate').value;
    _spTarget = document.getElementById('spTarget').value;
    renderSystemPrompt();
}

async function saveSystemPrompt() {
    const body = {};
    body[_spKey()] = document.getElementById('spTemplate').value;
    try {
        const response = await adminFetch(`${API_BASE}/settings/system-prompt`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await response.json();
        _spSettings = data.settings;
        renderSystemPrompt();
        _spAlert('success', 'System prompt saved. New chat queries use it immediately.');
        showToast('System prompt saved', 'success');
    } catch (error) {
        // The API returns a helpful placeholder-error detail on 400.
        _spAlert('danger', 'Could not save: ' + error.message);
    }
}

async function resetSystemPrompt() {
    if (!confirm('Reset this prompt to the shipped default? Your custom text will be discarded.')) return;
    const body = {};
    body[_spKey()] = '';  // blank => reset to default server-side
    try {
        const response = await adminFetch(`${API_BASE}/settings/system-prompt`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await response.json();
        _spSettings = data.settings;
        renderSystemPrompt();
        showToast('Reset to default', 'success');
    } catch (error) {
        showToast('Failed to reset: ' + error.message, 'danger');
    }
}

// =============================================================================
// Query settings (runtime) — analysis timeout
// =============================================================================

let _rtSettings = null;

function _rtAlert(type, msg) {
    const icon = type === 'success' ? 'bi-check-circle' : 'bi-exclamation-triangle';
    document.getElementById('runtimeAlert').innerHTML = `
        <div class="alert alert-${type} alert-dismissible fade show">
            <i class="bi ${icon} me-2"></i>${escapeHtml(msg)}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>`;
}

function renderRuntimeSettings() {
    if (!_rtSettings) return;
    const input = document.getElementById('rtQueryTimeout');
    input.value = _rtSettings.query_timeout_seconds;
    input.min = _rtSettings.min_query_timeout_seconds || 30;
    input.max = _rtSettings.max_query_timeout_seconds || 3600;

    const badge = document.getElementById('rtStatus');
    const isCustom = !!_rtSettings.query_timeout_is_custom;
    badge.textContent = isCustom ? 'Customized' : `Default (${_rtSettings.default_query_timeout_seconds}s)`;
    badge.className = 'badge align-self-center ' + (isCustom ? 'bg-primary' : 'bg-secondary');
}

async function loadRuntimeSettings() {
    try {
        const response = await adminFetch(`${API_BASE}/settings/runtime`);
        _rtSettings = await response.json();
        renderRuntimeSettings();
    } catch (error) {
        _rtAlert('danger', 'Failed to load query settings: ' + error.message);
    }
}

async function _postRuntimeSettings(body, successMsg) {
    try {
        const response = await adminFetch(`${API_BASE}/settings/runtime`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await response.json();
        _rtSettings = data.settings;
        renderRuntimeSettings();
        _rtAlert('success', successMsg);
        showToast(successMsg, 'success');
    } catch (error) {
        // The API returns a helpful range-error detail on 400.
        _rtAlert('danger', 'Could not save: ' + error.message);
    }
}

async function saveRuntimeSettings() {
    const raw = document.getElementById('rtQueryTimeout').value;
    const seconds = parseInt(raw, 10);
    if (!raw || isNaN(seconds)) {
        _rtAlert('danger', 'Enter a timeout in seconds (or use Reset to Default).');
        return;
    }
    await _postRuntimeSettings(
        { query_timeout_seconds: seconds },
        'Query settings saved. New queries use the updated timeout immediately.'
    );
}

async function resetRuntimeSettings() {
    if (!confirm('Reset the analysis timeout to the shipped default?')) return;
    await _postRuntimeSettings({ query_timeout_seconds: 0 }, 'Reset to default');
}

async function testGitHubConnection() {
    const resultEl = document.getElementById('ghTestResult');
    resultEl.innerHTML = '<span class="text-muted"><i class="bi bi-hourglass-split me-2"></i>Testing connection...</span>';

    try {
        const response = await adminFetch(`${API_BASE}/settings/github/test`, { method: 'POST' });
        if (!response.ok) throw new Error('Test request failed');

        const result = await response.json();
        if (result.ok) {
            resultEl.innerHTML = `
                <div class="alert alert-success">
                    <i class="bi bi-check-circle me-2"></i>${escapeHtml(result.message)}
                    ${result.private ? ' <span class="badge bg-secondary">Private</span>' : ' <span class="badge bg-info">Public</span>'}
                </div>
            `;
        } else {
            resultEl.innerHTML = `
                <div class="alert alert-danger">
                    <i class="bi bi-x-circle me-2"></i>${escapeHtml(result.message)}
                </div>
            `;
        }
    } catch (error) {
        resultEl.innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-x-circle me-2"></i>Connection test failed: ${escapeHtml(error.message)}
            </div>
        `;
    }
}

// Render the status badge + Pause/Resume button state from a GET /settings/github payload.
// Three derived states based on backend's {configured, paused, active} fields:
//   - Not configured:  missing repo or token  -> grey badge, Pause disabled
//   - Active:          configured && !paused  -> green badge, button says "Pause"
//   - Paused:          configured && paused   -> yellow badge, button says "Resume"
function renderGitHubStatus(settings) {
    const badge = document.getElementById('ghStatusBadge');
    const btn = document.getElementById('ghPauseBtn');
    const btnLabel = document.getElementById('ghPauseBtnLabel');
    const btnIcon = btn.querySelector('i');

    if (!settings.configured) {
        badge.className = 'badge bg-secondary';
        badge.textContent = 'Not configured';
        btn.disabled = true;
        btn.dataset.action = '';
        btnLabel.textContent = 'Pause';
        btnIcon.className = 'bi bi-pause-circle me-1';
        // Reset to the neutral Pause styling so we don't leave the
        // disabled button stuck in the previous state's color (e.g.
        // green outline from Resume) when the user clears creds.
        btn.className = 'btn btn-outline-warning';
    } else if (settings.paused) {
        badge.className = 'badge bg-warning text-dark';
        badge.textContent = 'Paused';
        btn.disabled = false;
        btn.dataset.action = 'resume';
        btnLabel.textContent = 'Resume';
        btnIcon.className = 'bi bi-play-circle me-1';
        btn.className = 'btn btn-outline-success';
    } else {
        badge.className = 'badge bg-success';
        badge.textContent = 'Active';
        btn.disabled = false;
        btn.dataset.action = 'pause';
        btnLabel.textContent = 'Pause';
        btnIcon.className = 'bi bi-pause-circle me-1';
        btn.className = 'btn btn-outline-warning';
    }
}

// Show the Pause/Resume confirmation modal. Body text spells out the
// consequences so the action stays deliberate — Option 2 of the SoT redesign.
function openPauseModal() {
    const btn = document.getElementById('ghPauseBtn');
    const action = btn.dataset.action;  // 'pause' or 'resume'
    const title = document.getElementById('ghPauseModalTitle');
    const body = document.getElementById('ghPauseModalBody');
    const confirmBtn = document.getElementById('ghPauseConfirmBtn');
    const confirmLabel = document.getElementById('ghPauseConfirmLabel');
    const confirmIcon = confirmBtn.querySelector('i');

    if (action === 'pause') {
        title.textContent = 'Pause GitHub publishing?';
        body.innerHTML = `
            While paused, the app will <strong>halt all GitHub I/O</strong>:
            new submissions and approvals will stay local only, and the verified
            library will <strong>drift from the GitHub source of truth</strong>
            until you resume. The token is preserved — Resume to bring everything
            back online.
        `;
        confirmLabel.textContent = 'Pause publishing';
        confirmIcon.className = 'bi bi-pause-circle me-1';
        confirmBtn.className = 'btn btn-warning';
        confirmBtn.dataset.targetPaused = 'true';
    } else {
        title.textContent = 'Resume GitHub publishing?';
        body.innerHTML = `
            Resuming will re-enable all GitHub I/O. Local submissions/approvals
            made while paused are NOT auto-published on resume — push them
            manually if needed.
        `;
        confirmLabel.textContent = 'Resume publishing';
        confirmIcon.className = 'bi bi-play-circle me-1';
        confirmBtn.className = 'btn btn-success';
        confirmBtn.dataset.targetPaused = 'false';
    }
    bootstrap.Modal.getOrCreateInstance(document.getElementById('ghPauseModal')).show();
}

async function confirmPauseToggle() {
    const confirmBtn = document.getElementById('ghPauseConfirmBtn');
    const targetPaused = confirmBtn.dataset.targetPaused === 'true';
    const alertEl = document.getElementById('githubSettingsAlert');
    const modal = bootstrap.Modal.getOrCreateInstance(document.getElementById('ghPauseModal'));

    try {
        const response = await adminFetch(`${API_BASE}/settings/github/pause`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ paused: targetPaused }),
        });
        if (!response.ok) {
            const errJson = await response.json().catch(() => ({}));
            throw new Error(errJson.detail || 'Request failed');
        }
        const result = await response.json();
        modal.hide();
        alertEl.innerHTML = `
            <div class="alert alert-success alert-dismissible fade show">
                <i class="bi bi-check-circle me-2"></i>${escapeHtml(result.message)}.
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        loadGitHubSettings();
    } catch (error) {
        alertEl.innerHTML = `
            <div class="alert alert-danger alert-dismissible fade show">
                <i class="bi bi-exclamation-triangle me-2"></i>${escapeHtml(error.message)}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
    }
}

// =============================================================================
// Pending Access Requests
// =============================================================================

async function loadPendingRequests() {
    const section = document.getElementById('pendingRequestsSection');
    const container = document.getElementById('pendingRequestsList');
    const badge = document.getElementById('pendingRequestsBadge');
    const tabBadge = document.getElementById('approvedTabBadge');

    try {
        const response = await adminFetch(`${API_BASE}/admin/pending-requests`);
        if (!response.ok) throw new Error('Failed to load');
        const data = await response.json();

        if (!data.requests || data.requests.length === 0) {
            section.classList.add('d-none');
            tabBadge.classList.add('d-none');
            return;
        }

        section.classList.remove('d-none');
        badge.textContent = data.count;
        tabBadge.textContent = data.count;
        tabBadge.classList.remove('d-none');

        container.innerHTML = `
            <div class="list-group" style="max-width: 600px;">
                ${data.requests.map(r => `
                    <div class="list-group-item border-warning d-flex justify-content-between align-items-center">
                        <div>
                            <i class="bi bi-person-exclamation me-2 text-warning"></i>
                            <strong>${escapeHtml(r.email)}</strong>
                            ${r.name ? ` <span class="text-muted">(${escapeHtml(r.name)})</span>` : ''}
                            <br>
                            <small class="text-muted">
                                Requested ${r.requested_at ? new Date(r.requested_at).toLocaleString() : 'recently'}
                            </small>
                        </div>
                        <div class="d-flex gap-2">
                            <button class="btn btn-sm btn-success js-approve-request" data-email="${escapeHtml(r.email)}" title="Approve">
                                <i class="bi bi-check-lg me-1"></i>Approve
                            </button>
                            <button class="btn btn-sm btn-outline-secondary js-dismiss-request" data-email="${escapeHtml(r.email)}" title="Dismiss">
                                <i class="bi bi-x-lg"></i>
                            </button>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
        bindClick(container, '.js-approve-request', (el) => approvePendingRequest(el.dataset.email));
        bindClick(container, '.js-dismiss-request', (el) => dismissPendingRequest(el.dataset.email));
    } catch (error) {
        section.classList.add('d-none');
        tabBadge.classList.add('d-none');
    }
}

async function approvePendingRequest(email) {
    try {
        const response = await adminFetch(`${API_BASE}/admin/pending-requests/${encodeURIComponent(email)}/approve`, {
            method: 'POST'
        });
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.detail || 'Failed to approve');
        }
        showToast(`${email} approved`, 'success');
        loadApprovedMembers();
    } catch (error) {
        showToast('Failed: ' + error.message, 'danger');
    }
}

async function dismissPendingRequest(email) {
    if (!confirm(`Dismiss the access request from "${email}"? They can request again by signing in.`)) return;
    try {
        const response = await adminFetch(`${API_BASE}/admin/pending-requests/${encodeURIComponent(email)}`, {
            method: 'DELETE'
        });
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.detail || 'Failed to dismiss');
        }
        showToast(`Request from ${email} dismissed`, 'success');
        loadPendingRequests();
    } catch (error) {
        showToast('Failed: ' + error.message, 'danger');
    }
}

// =============================================================================
// Approved Members
// =============================================================================

async function loadApprovedMembers() {
    loadPendingRequests();
    const container = document.getElementById('approvedList');
    try {
        const response = await adminFetch(`${API_BASE}/admin/approved-members`);
        if (!response.ok) throw new Error('Failed to load approved members');
        const data = await response.json();

        if (!data.members || data.members.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted py-4">
                    <i class="bi bi-person-x display-4"></i>
                    <p class="mt-3">No approved members yet. Add one above to enable social login.</p>
                </div>
            `;
            return;
        }

        container.innerHTML = `
            <div class="list-group" style="max-width: 600px;">
                ${data.members.map(m => {
                    const displayLabel = m.display_name
                        ? `<strong>${escapeHtml(m.display_name)}</strong> <span class="text-muted">(${escapeHtml(m.email)})</span>`
                        : `<strong>${escapeHtml(m.email)}</strong>`;
                    return `
                    <div class="list-group-item list-group-item-action d-flex justify-content-between align-items-center">
                        <div>
                            <i class="bi bi-person-check me-2 text-success"></i>
                            ${displayLabel}
                            <br>
                            <small class="text-muted">
                                Added by ${escapeHtml(m.added_by || 'admin')}
                                ${m.added_at ? ' · ' + new Date(m.added_at).toLocaleString() : ''}
                            </small>
                        </div>
                        <button class="btn btn-sm btn-outline-danger js-remove-member" data-email="${escapeHtml(m.email)}">
                            <i class="bi bi-trash"></i>
                        </button>
                    </div>
                `}).join('')}
            </div>
            <p class="text-muted small mt-2">${data.count} approved member(s)</p>
        `;
        bindClick(container, '.js-remove-member', (el) => removeApprovedMember(el.dataset.email));
    } catch (error) {
        container.innerHTML = `<div class="alert alert-danger">${escapeHtml(error.message)}</div>`;
    }
}

async function addApprovedMember(e) {
    e.preventDefault();
    const alertEl = document.getElementById('addApprovedAlert');
    const email = document.getElementById('approvedEmail').value.trim();
    if (!email) return;

    try {
        const response = await adminFetch(`${API_BASE}/admin/approved-members`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email })
        });
        const data = await response.json();
        if (!response.ok) {
            alertEl.innerHTML = `<div class="alert alert-danger alert-dismissible fade show">
                ${escapeHtml(data.detail || 'Failed to add member')}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button></div>`;
            return;
        }
        alertEl.innerHTML = `<div class="alert alert-success alert-dismissible fade show">
            <i class="bi bi-check-circle me-2"></i>${escapeHtml(email)} added.
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button></div>`;
        document.getElementById('approvedEmail').value = '';
        loadApprovedMembers();
    } catch (error) {
        alertEl.innerHTML = `<div class="alert alert-danger">${escapeHtml(error.message)}</div>`;
    }
}

async function removeApprovedMember(email) {
    if (!confirm(`Remove '${email}' from the approved list?`)) return;
    try {
        const response = await adminFetch(`${API_BASE}/admin/approved-members/${encodeURIComponent(email)}`, {
            method: 'DELETE'
        });
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.detail || 'Failed to remove');
        }
        showToast(`Removed ${email}`, 'success');
        loadApprovedMembers();
    } catch (error) {
        showToast('Failed: ' + error.message, 'danger');
    }
}

// =============================================================================
// Query Logs
// =============================================================================

let _queryLogsCache = [];

async function loadQueryLogs() {
    const tbody = document.getElementById('queryLogsBody');
    tbody.innerHTML = `<tr><td colspan="9" class="text-center text-muted py-4">Loading logs...</td></tr>`;
    try {
        const response = await adminFetch(`${API_BASE}/admin/query-logs?limit=500`);
        if (!response.ok) throw new Error('Failed to load query logs');
        const data = await response.json();
        _queryLogsCache = data.logs || [];

        // Update summary cards
        const summary = data.summary || {};
        const bySource = summary.by_source || {};
        document.getElementById('qlTotal').textContent = summary.total || 0;
        document.getElementById('qlGenerated').textContent = bySource.generated || 0;
        document.getElementById('qlCached').textContent = bySource.verified_cache || 0;
        document.getElementById('qlUsers').textContent = summary.unique_users || 0;
        document.getElementById('qlHitRate').textContent =
            summary.cache_hit_rate != null ? `${summary.cache_hit_rate}%` : '—';

        // Populate user filter
        const users = Object.keys(summary.by_user || {}).sort();
        const sel = document.getElementById('qlUserFilter');
        const current = sel.value;
        sel.innerHTML = '<option value="">All users</option>' +
            users.map(u => `<option value="${escapeHtml(u)}">${escapeHtml(u)}</option>`).join('');
        if (users.includes(current)) sel.value = current;

        renderQueryLogs();
    } catch (error) {
        tbody.innerHTML = `<tr><td colspan="9" class="text-danger">${escapeHtml(error.message)}</td></tr>`;
    }
}

function renderQueryLogs() {
    const tbody = document.getElementById('queryLogsBody');
    const userFilter = document.getElementById('qlUserFilter').value;
    const sourceFilter = document.getElementById('qlSourceFilter').value;

    let logs = _queryLogsCache;
    if (userFilter) logs = logs.filter(l => l.user === userFilter);
    if (sourceFilter) logs = logs.filter(l => l.source === sourceFilter);

    if (logs.length === 0) {
        tbody.innerHTML = `<tr><td colspan="9" class="text-center text-muted py-4">
            <i class="bi bi-inbox me-2"></i>No query logs match the current filters</td></tr>`;
        return;
    }

    tbody.innerHTML = logs.map(l => {
        // Compact timestamp — full precision stays in the title attribute.
        const tsDate = l.timestamp ? new Date(l.timestamp) : null;
        const ts = tsDate && !isNaN(tsDate)
            ? tsDate.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
            : '';
        const tsFull = tsDate && !isNaN(tsDate) ? tsDate.toLocaleString() : '';
        const srcBadge = sourceBadge(l.source);
        const conf = l.confidence != null ? `${Math.round(l.confidence * 100)}%` : '—';
        const tier = l.tier || '—';
        const ms = l.processing_time_ms != null ? l.processing_time_ms : '—';
        const inTok = l.input_tokens != null ? l.input_tokens.toLocaleString() : '';
        const outTok = l.output_tokens != null ? l.output_tokens.toLocaleString() : '';
        const totalTok = (l.input_tokens || 0) + (l.output_tokens || 0);
        const tokensHtml = totalTok > 0
            ? `<span title="In: ${inTok} / Out: ${outTok}">${totalTok.toLocaleString()}</span>`
            : '<span class="text-muted">—</span>';
        const nb = l.had_notebook
            ? (l.notebook_url
                ? `<a href="${escapeHtml(l.notebook_url)}/download">download</a>`
                : '<span class="text-success">yes</span>')
            : '<span class="text-muted">—</span>';
        const queryText = l.query || '';
        const queryDisplay = queryText.length > 80 ? queryText.substring(0, 80) + '…' : queryText;
        return `
            <tr>
                <td class="text-muted" style="white-space:nowrap;" title="${escapeHtml(tsFull)}">${escapeHtml(ts)}</td>
                <td><strong>${escapeHtml(l.user || 'anonymous')}</strong>
                    ${l.auth_type ? `<br><small class="text-muted">${escapeHtml(l.auth_type)}</small>` : ''}</td>
                <td class="ql-query" title="${escapeHtml(queryText)}">${escapeHtml(queryDisplay)}
                    ${l.error ? `<br><small class="text-danger">${escapeHtml(l.error)}</small>` : ''}
                    ${l.verified_query ? `<br><small class="text-info">match: ${escapeHtml(l.verified_query)}</small>` : ''}</td>
                <td>${srcBadge}</td>
                <td>${escapeHtml(tier)}</td>
                <td>${conf}${l.similarity_score != null ? `<br><small class="text-muted">sim ${Math.round(l.similarity_score * 100)}%</small>` : ''}</td>
                <td>${tokensHtml}</td>
                <td>${ms}</td>
                <td>${nb}</td>
            </tr>
        `;
    }).join('');
}

function sourceBadge(src) {
    if (src === 'generated') return '<span class="badge bg-success">generated</span>';
    if (src === 'verified_cache') return '<span class="badge bg-info">cached</span>';
    if (src === 'revision') return '<span class="badge bg-primary">revision</span>';
    if (src === 'error') return '<span class="badge bg-danger">error</span>';
    return `<span class="badge bg-secondary">${escapeHtml(src || 'unknown')}</span>`;
}

// =============================================================================
// Notebook Reviews (execution + adversarial method review)
// =============================================================================

let _nbReviewsCache = [];

async function loadNotebookReviews() {
    const tbody = document.getElementById('nbReviewsBody');
    tbody.innerHTML = `<tr><td colspan="8" class="text-center text-muted py-4">Loading reviews...</td></tr>`;
    try {
        const response = await adminFetch(`${API_BASE}/admin/notebook-reviews?limit=200`);
        const data = await response.json();
        _nbReviewsCache = data.reviews || [];

        const summary = data.summary || {};
        const sev = summary.findings_by_severity || {};
        const totalFindings = Object.values(sev).reduce((a, b) => a + b, 0);
        document.getElementById('nbrTotal').textContent = summary.total || 0;
        document.getElementById('nbrExecuted').textContent = summary.executed_ok || 0;
        document.getElementById('nbrReviewed').textContent = summary.reviewed || 0;
        document.getElementById('nbrFindings').textContent = totalFindings;
        document.getElementById('nbrAvgScore').textContent =
            summary.avg_combined_score != null ? `${Math.round(summary.avg_combined_score * 100)}%` : '—';

        renderNotebookReviews();
    } catch (error) {
        tbody.innerHTML = `<tr><td colspan="8" class="text-danger">${escapeHtml(error.message)}</td></tr>`;
    }
}

function severityBadge(severity, count) {
    const cls = { critical: 'bg-danger', high: 'bg-danger', medium: 'bg-warning text-dark', low: 'bg-secondary' }[severity]
        || 'bg-secondary';
    const label = count !== '' && count != null ? `${count} ${severity}` : severity;
    return `<span class="badge ${cls}" title="${escapeHtml(severity)}">${escapeHtml(label)}</span>`;
}

function reviewCellHtml(record) {
    const review = (record || {}).review;
    if (!review || !review.reviewed) {
        const reason = review && review.reason ? review.reason : 'Review did not run';
        return `<span class="text-muted" title="${escapeHtml(reason)}">not reviewed</span>`;
    }
    const findings = review.findings || [];
    let summaryLabel;
    if (findings.length === 0) {
        summaryLabel = `<span class="text-success"><i class="bi bi-check-circle me-1"></i>clean</span>`;
    } else {
        const counts = {};
        findings.forEach(f => { counts[f.severity] = (counts[f.severity] || 0) + 1; });
        summaryLabel = ['critical', 'high', 'medium', 'low']
            .filter(s => counts[s])
            .map(s => severityBadge(s, counts[s]))
            .join(' ');
    }
    // Full, properly formatted report on expand — same renderer as the
    // review modal's Method Review tab, in a readable full-width panel.
    return `
        <details>
            <summary style="cursor:pointer;">${summaryLabel}</summary>
            <div class="mt-2 rounded" style="background:var(--dc-bg-alt);min-width:340px;max-width:720px;max-height:420px;overflow-y:auto;">
                ${renderVerificationReport(record)}
            </div>
        </details>`;
}

function renderNotebookReviews() {
    const tbody = document.getElementById('nbReviewsBody');
    const statusFilter = document.getElementById('nbrStatusFilter').value;
    const findingsFilter = document.getElementById('nbrFindingsFilter').value;

    let reviews = _nbReviewsCache;
    if (statusFilter) reviews = reviews.filter(r => r.status === statusFilter);
    if (findingsFilter === 'with') {
        reviews = reviews.filter(r => ((r.review || {}).findings || []).length > 0);
    } else if (findingsFilter === 'clean') {
        reviews = reviews.filter(r =>
            (r.review || {}).reviewed && ((r.review || {}).findings || []).length === 0);
    }

    if (reviews.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8" class="text-center text-muted py-4">
            <i class="bi bi-inbox me-2"></i>No notebook reviews match the current filters</td></tr>`;
        return;
    }

    tbody.innerHTML = reviews.map(r => {
        const tsDate = r.completed_at ? new Date(r.completed_at) : null;
        const ts = tsDate && !isNaN(tsDate)
            ? tsDate.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
            : '';
        const tsFull = tsDate && !isNaN(tsDate) ? tsDate.toLocaleString() : '';

        const statusMap = {
            complete: '<span class="badge bg-success">complete</span>',
            pending: '<span class="badge bg-warning text-dark">pending</span>',
            error: '<span class="badge bg-danger">error</span>',
        };
        const statusHtml = statusMap[r.status] || `<span class="badge bg-secondary">${escapeHtml(r.status || '?')}</span>`;

        const verdict = r.verdict || {};
        let executedHtml = '<span class="text-muted">—</span>';
        if (r.status === 'complete') {
            executedHtml = verdict.executed
                ? `<span class="text-success"><i class="bi bi-check-circle me-1"></i>${escapeHtml(verdict.cells_executed != null ? String(verdict.cells_executed) : '')}/${escapeHtml(String(verdict.cells_total || 0))}</span>`
                : `<span class="text-danger" title="${escapeHtml(verdict.execution_error || (verdict.timed_out ? 'timed out' : ''))}">
                    <i class="bi bi-x-circle me-1"></i>failed</span>`;
        }

        let reconciledHtml = '<span class="text-muted">—</span>';
        const claims = (verdict.claimed_values || []).length;
        if (r.status === 'complete' && verdict.executed) {
            reconciledHtml = claims > 0
                ? `${(verdict.reconciled_values || []).length}/${claims}`
                : '<span class="text-muted" title="Answer makes no numeric claims">n/a</span>';
        }

        const score = r.combined_score;
        const scoreHtml = score != null ? `${Math.round(score * 100)}%` : '<span class="text-muted">—</span>';

        const before = r.confidence_before;
        const after = r.status === 'complete' ? r.confidence : null;
        let confHtml = '<span class="text-muted">—</span>';
        if (before != null && after != null) {
            const delta = after - before;
            const cls = delta > 0.005 ? 'text-success' : (delta < -0.005 ? 'text-danger' : 'text-muted');
            confHtml = `${Math.round(before * 100)}% → <span class="${cls}">${Math.round(after * 100)}%</span>`;
        } else if (before != null) {
            confHtml = `${Math.round(before * 100)}%`;
        }

        const queryText = r.query || r.query_id || '';
        const queryDisplay = queryText.length > 70 ? queryText.substring(0, 70) + '…' : queryText;
        const nbLink = r.query_id
            ? `<br><small><a href="${API_BASE}/notebooks/${encodeURIComponent(r.query_id)}/download">notebook</a></small>`
            : '';

        return `
            <tr>
                <td class="text-muted" style="white-space:nowrap;" title="${escapeHtml(tsFull)}">${escapeHtml(ts)}</td>
                <td class="ql-query" title="${escapeHtml(queryText)}">${escapeHtml(queryDisplay)}${nbLink}
                    ${r.error ? `<br><small class="text-danger">${escapeHtml(r.error)}</small>` : ''}</td>
                <td>${statusHtml}</td>
                <td>${executedHtml}</td>
                <td>${reconciledHtml}</td>
                <td>${reviewCellHtml(r)}</td>
                <td>${scoreHtml}</td>
                <td style="white-space:nowrap;">${confHtml}</td>
            </tr>
        `;
    }).join('');
}

// =============================================================================
// Answer Feedback (admin view)
// =============================================================================
async function loadFeedback() {
    const container = document.getElementById('feedbackList');
    const rating = document.getElementById('fbFilter').value;
    try {
        const url = `${API_BASE}/admin/feedback?limit=200${rating ? `&rating=${rating}` : ''}`;
        const response = await adminFetch(url);
        const data = await response.json();
        const s = data.summary || {};
        document.getElementById('fbTotal').textContent = s.total || 0;
        document.getElementById('fbUp').textContent = s.up || 0;
        document.getElementById('fbDown').textContent = s.down || 0;
        document.getElementById('fbSatisfaction').textContent =
            s.satisfaction_pct != null ? `${s.satisfaction_pct}%` : '—';

        const items = data.feedback || [];
        if (items.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <i class="bi bi-hand-thumbs-up" aria-hidden="true"></i>
                    <div class="empty-title">No feedback yet</div>
                    <p class="small">Ratings appear here as users react to answers.</p>
                </div>`;
            return;
        }

        container.innerHTML = items.map(f => {
            const up = f.rating === 'up';
            const ts = f.timestamp ? new Date(f.timestamp).toLocaleString() : '';
            const srcBadge = f.source ? sourceBadge(f.source) : '';
            return `
                <div class="submission-card">
                    <div class="d-flex justify-content-between align-items-start gap-3">
                        <div style="min-width:0;">
                            <div class="query">${escapeHtml(f.query || '(no question captured)')}</div>
                            <div class="meta">
                                <i class="bi bi-person-circle me-1"></i>${escapeHtml(f.user || 'anonymous')}
                                · ${escapeHtml(ts)} ${srcBadge}
                            </div>
                            ${f.answer_preview ? `<div class="mt-2 p-2 rounded-code-block rounded small">${escapeHtml(f.answer_preview.substring(0, 240))}${f.answer_preview.length > 240 ? '…' : ''}</div>` : ''}
                            ${f.note ? `<div class="mt-2 small text-muted"><i class="bi bi-chat-left-text me-1"></i>${escapeHtml(f.note)}</div>` : ''}
                        </div>
                        <span class="status-badge ${up ? 'status-approved' : 'status-rejected'} flex-shrink-0">
                            <i class="bi bi-hand-thumbs-${up ? 'up' : 'down'}-fill me-1"></i>${up ? 'Helpful' : 'Not helpful'}
                        </span>
                    </div>
                </div>`;
        }).join('');
    } catch (error) {
        container.innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle me-2"></i>Failed to load feedback: ${escapeHtml(error.message)}
            </div>`;
    }
}

// =============================================================================
// Admin Identity
// =============================================================================

async function loadAdminIdentity() {
    try {
        const response = await adminFetch(`${API_BASE}/auth/status`);
        if (!response.ok) return;
        const data = await response.json();
        if (data.authenticated) {
            const el = document.getElementById('adminIdentity');
            el.textContent = `Signed in as ${data.username}`;
            el.classList.remove('d-none');
        }
    } catch (error) {
        console.error('Failed to load admin identity:', error);
    }
}

// =============================================================================
// Admin Role Management
// =============================================================================

async function loadAdmins() {
    const container = document.getElementById('adminList');
    try {
        const response = await adminFetch(`${API_BASE}/admin/roles`);
        if (!response.ok) throw new Error('Failed to load admins');
        const data = await response.json();

        if (!data.admins || data.admins.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted py-4">
                    <i class="bi bi-shield-x display-4"></i>
                    <p class="mt-3">No admins configured. Add one above.</p>
                </div>
            `;
            return;
        }

        container.innerHTML = `
            <div class="list-group" style="max-width: 600px;">
                ${data.admins.map(a => {
                    const adminLabel = a.display_name
                        ? `<strong>${escapeHtml(a.display_name)}</strong> <span class="text-muted">(${escapeHtml(a.user_id)})</span>`
                        : `<strong>${escapeHtml(a.user_id)}</strong>`;
                    return `
                    <div class="list-group-item list-group-item-action d-flex justify-content-between align-items-center">
                        <div>
                            <i class="bi bi-shield-check me-2 text-warning"></i>
                            ${adminLabel}
                            <br>
                            <small class="text-muted">
                                Granted by ${escapeHtml(a.granted_by || 'unknown')}
                                ${a.granted_at ? ' \u00b7 ' + new Date(a.granted_at).toLocaleString() : ''}
                            </small>
                        </div>
                        <button class="btn btn-sm btn-outline-danger js-revoke-admin" data-user-id="${escapeHtml(a.user_id)}" title="Revoke admin access">
                            <i class="bi bi-shield-minus"></i>
                        </button>
                    </div>
                `}).join('')}
            </div>
            <p class="text-muted small mt-2">${data.count} admin(s) total</p>
        `;
        bindClick(container, '.js-revoke-admin', (el) => revokeAdmin(el.dataset.userId));
    } catch (error) {
        container.innerHTML = `<div class="alert alert-danger">${escapeHtml(error.message)}</div>`;
    }
}

async function grantAdmin(e) {
    e.preventDefault();
    const alertEl = document.getElementById('addAdminAlert');
    const userId = document.getElementById('adminUserId').value.trim();
    if (!userId) return;

    try {
        const response = await adminFetch(`${API_BASE}/admin/roles`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: userId })
        });
        const data = await response.json();

        if (!response.ok) {
            alertEl.innerHTML = `
                <div class="alert alert-danger alert-dismissible fade show">
                    <i class="bi bi-exclamation-triangle me-2"></i>${escapeHtml(data.detail || 'Failed to grant admin')}
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            `;
            return;
        }

        alertEl.innerHTML = `
            <div class="alert alert-success alert-dismissible fade show">
                <i class="bi bi-check-circle me-2"></i>Admin access granted to '${escapeHtml(userId)}'.
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        document.getElementById('adminUserId').value = '';
        loadAdmins();
    } catch (error) {
        alertEl.innerHTML = `
            <div class="alert alert-danger alert-dismissible fade show">
                <i class="bi bi-exclamation-triangle me-2"></i>Error: ${escapeHtml(error.message)}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
    }
}

async function revokeAdmin(userId) {
    if (!confirm(`Revoke admin access from "${userId}"?`)) return;
    try {
        const response = await adminFetch(`${API_BASE}/admin/roles/${encodeURIComponent(userId)}`, {
            method: 'DELETE'
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to revoke admin');
        }
        showToast(`Admin access revoked from '${userId}'`, 'success');
        loadAdmins();
    } catch (error) {
        showToast('Failed: ' + error.message, 'danger');
    }
}

// =============================================================================
// MCP Server Management
// =============================================================================

const MCP_API = '/api/v1/mcp';
let _currentMcpToolCall = { serverId: null, toolName: null };

async function loadMcpServers() {
    const container = document.getElementById('mcpServerList');
    try {
        const response = await adminFetch(`${MCP_API}/servers`);
        const data = await response.json();

        if (data.servers.length === 0) {
            container.innerHTML = `
                <div class="col-12 text-center text-muted py-5">
                    <i class="bi bi-plug display-4 d-block mb-3"></i>
                    <h5>No MCP servers configured</h5>
                    <p class="mb-3">Add an MCP server to extend Data Concierge with additional data sources.</p>
                    <button class="btn btn-primary" data-bs-toggle="modal" data-bs-target="#addMcpServerModal">
                        <i class="bi bi-plus-lg me-2"></i>Add Your First Server
                    </button>
                </div>
            `;
            return;
        }

        container.innerHTML = data.servers.map(s => renderMcpServerCard(s)).join('');
        bindClick(container, '.js-mcp-tool', (el) => openMcpToolCall(el.dataset.serverId, el.dataset.toolName));
        bindClick(container, '.js-mcp-connect', (el) => connectMcpServer(el.dataset.serverId));
        bindClick(container, '.js-mcp-disconnect', (el) => disconnectMcpServer(el.dataset.serverId));
        bindClick(container, '.js-mcp-remove', (el) => removeMcpServer(el.dataset.serverId, el.dataset.serverName));
    } catch (error) {
        container.innerHTML = `
            <div class="col-12">
                <div class="alert alert-danger">
                    <i class="bi bi-exclamation-triangle me-2"></i>
                    Failed to load MCP servers: ${escapeHtml(error.message)}
                </div>
            </div>
        `;
    }
}

function renderMcpServerCard(server) {
    const statusClass = `mcp-status-${server.status}`;
    const statusIcon = {
        connected: 'bi-check-circle-fill',
        disconnected: 'bi-circle',
        connecting: 'bi-arrow-repeat',
        error: 'bi-exclamation-circle-fill',
    }[server.status] || 'bi-circle';

    const transportLabel = { stdio: 'stdio', sse: 'SSE', streamable_http: 'HTTP' }[server.transport] || server.transport;
    const transportIcon = { stdio: 'bi-terminal', sse: 'bi-broadcast' }[server.transport] || 'bi-globe';

    const toolsHtml = server.tools.length > 0
        ? server.tools.map(t => `
            <span class="tool-chip js-mcp-tool" title="${escapeHtml(t.description)}"
                  data-server-id="${escapeHtml(server.id)}" data-tool-name="${escapeHtml(t.name)}">
                <i class="bi bi-wrench me-1"></i>${escapeHtml(t.name)}
            </span>
        `).join('')
        : '<span class="text-muted small">No tools discovered yet — connect to discover tools</span>';

    const connectBtn = server.status === 'connected'
        ? `<button class="btn btn-sm btn-outline-warning js-mcp-disconnect" data-server-id="${escapeHtml(server.id)}">
               <i class="bi bi-plug me-1"></i>Disconnect
           </button>`
        : `<button class="btn btn-sm btn-outline-success js-mcp-connect" data-server-id="${escapeHtml(server.id)}">
               <i class="bi bi-plug-fill me-1"></i>Connect
           </button>`;

    return `
        <div class="col-12">
            <div class="mcp-server-card">
                <div class="d-flex justify-content-between align-items-start">
                    <div class="flex-grow-1">
                        <div class="d-flex align-items-center gap-2 mb-1 flex-wrap">
                            <h6 class="mb-0">${escapeHtml(server.name)}</h6>
                            <span class="badge bg-secondary"><i class="bi ${transportIcon} me-1"></i>${transportLabel}</span>
                            <span class="status-badge ${statusClass}">
                                <i class="bi ${statusIcon} me-1"></i>${server.status}
                            </span>
                        </div>
                        <p class="text-muted mb-2 small">${escapeHtml(server.description || 'No description')}</p>
                        ${server.server_name ? `<p class="small mb-1 text-muted"><strong>Server:</strong> ${escapeHtml(server.server_name)}${server.server_version ? ' v' + escapeHtml(server.server_version) : ''}</p>` : ''}
                        ${server.command ? `<p class="small mb-1 font-monospace text-muted"><i class="bi bi-terminal me-1"></i>${escapeHtml(server.command)} ${server.args.map(a => escapeHtml(a)).join(' ')}</p>` : ''}
                        ${server.working_dir ? `<p class="small mb-1 text-muted"><i class="bi bi-folder me-1"></i>${escapeHtml(server.working_dir)}</p>` : ''}
                        ${server.url ? `<p class="small mb-1 text-muted"><i class="bi bi-globe me-1"></i>${escapeHtml(server.url)}</p>` : ''}
                        <div class="mt-2">
                            <strong class="small">Tools:</strong>
                            <div class="mt-1">${toolsHtml}</div>
                        </div>
                        ${server.last_error ? `<div class="text-danger small mt-2"><i class="bi bi-exclamation-triangle me-1"></i>${escapeHtml(server.last_error)}</div>` : ''}
                    </div>
                    <div class="d-flex gap-2 ms-3 flex-shrink-0">
                        ${connectBtn}
                        <button class="btn btn-sm btn-outline-danger js-mcp-remove" data-server-id="${escapeHtml(server.id)}"
                                data-server-name="${escapeHtml(server.name)}"
                                title="Remove server" aria-label="Remove MCP server ${escapeHtml(server.name)}">
                            <i class="bi bi-trash" aria-hidden="true"></i>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
}

async function addMcpServer() {
    const transport = document.querySelector('input[name="mcpTransport"]:checked').value;
    const payload = {
        name: document.getElementById('mcpServerName').value.trim(),
        description: document.getElementById('mcpServerDescription').value.trim(),
        transport,
        command: document.getElementById('mcpServerCommand').value.trim(),
        args: document.getElementById('mcpServerArgs').value.trim()
            .split('\n').map(a => a.trim()).filter(Boolean),
        working_dir: document.getElementById('mcpServerWorkingDir').value.trim(),
        url: document.getElementById('mcpServerUrl').value.trim(),
        categories: document.getElementById('mcpServerCategories').value.trim()
            .split(',').map(c => c.trim()).filter(Boolean),
        keywords: document.getElementById('mcpServerKeywords').value.trim()
            .split(',').map(k => k.trim()).filter(Boolean),
        auto_connect: document.getElementById('mcpServerAutoConnect').checked,
    };

    // Parse env vars
    const envText = document.getElementById('mcpServerEnv').value.trim();
    const env = {};
    if (envText) {
        envText.split('\n').forEach(line => {
            const [key, ...rest] = line.split('=');
            if (key && rest.length > 0) env[key.trim()] = rest.join('=').trim();
        });
    }
    payload.env = env;

    if (!payload.name) { showToast('Server name is required', 'danger'); return; }

    try {
        await adminFetch(`${MCP_API}/servers`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        showToast(`Server "${payload.name}" added`, 'success');
        bootstrap.Modal.getInstance(document.getElementById('addMcpServerModal')).hide();
        document.getElementById('addMcpServerForm').reset();
        // Reset transport UI state after form reset
        document.getElementById('mcpStdioFields').classList.remove('d-none');
        document.getElementById('mcpSseFields').classList.add('d-none');
        loadMcpServers();
    } catch (error) {
        showToast(error.message, 'danger');
    }
}

async function connectMcpServer(serverId) {
    showToast(`Connecting to ${serverId}…`, 'info');
    try {
        const response = await adminFetch(`${MCP_API}/servers/${serverId}/connect`, { method: 'POST' });
        const result = await response.json();
        showToast(`Connected! Found ${result.tools.length} tool(s)`, 'success');
        loadMcpServers();
    } catch (error) {
        showToast(`Connection failed: ${error.message}`, 'danger');
        loadMcpServers();
    }
}

async function disconnectMcpServer(serverId) {
    try {
        await adminFetch(`${MCP_API}/servers/${serverId}/disconnect`, { method: 'POST' });
        showToast('Disconnected', 'info');
        loadMcpServers();
    } catch (error) {
        showToast(`Error: ${error.message}`, 'danger');
    }
}

async function removeMcpServer(serverId, serverName) {
    if (!confirm(`Remove MCP server "${serverName || serverId}"?`)) return;
    try {
        await adminFetch(`${MCP_API}/servers/${serverId}`, { method: 'DELETE' });
        showToast('Server removed', 'success');
        loadMcpServers();
    } catch (error) {
        showToast(`Error: ${error.message}`, 'danger');
    }
}

function openMcpToolCall(serverId, toolName) {
    _currentMcpToolCall = { serverId, toolName };
    document.getElementById('mcpToolCallTitle').textContent = `Call: ${toolName}`;
    document.getElementById('mcpToolCallArgs').value = '{}';
    document.getElementById('mcpToolCallResult').classList.add('d-none');
    new bootstrap.Modal(document.getElementById('mcpToolCallModal')).show();
}

async function executeMcpTool() {
    const { serverId, toolName } = _currentMcpToolCall;
    if (!serverId || !toolName) return;

    let args;
    try {
        args = JSON.parse(document.getElementById('mcpToolCallArgs').value);
    } catch (e) {
        showToast('Invalid JSON arguments', 'danger');
        return;
    }

    const resultDiv = document.getElementById('mcpToolCallResult');
    const resultText = document.getElementById('mcpToolCallResultText');
    resultDiv.classList.remove('d-none');
    resultText.textContent = 'Executing…';

    try {
        const response = await adminFetch(`${MCP_API}/servers/${serverId}/tools/call`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tool_name: toolName, arguments: args }),
        });
        const result = await response.json();
        resultText.textContent = result.is_error
            ? `Error: ${result.text}`
            : result.text || JSON.stringify(result.content, null, 2);
    } catch (error) {
        resultText.textContent = `Error: ${error.message}`;
    }
}

// =============================================================================
// CKAN Sites
// =============================================================================

const CKAN_SITES_API = `${API_BASE}/admin/ckan-sites`;

async function loadCkanSites() {
    const container = document.getElementById('ckanSiteList');
    try {
        const response = await adminFetch(CKAN_SITES_API);
        const data = await response.json();
        if (!data.sites || data.sites.length === 0) {
            container.innerHTML = `
                <div class="col-12 text-center text-muted py-5">
                    <i class="bi bi-globe display-4"></i>
                    <p class="mt-3 mb-1">No CKAN sites registered yet</p>
                    <p class="small">Add a CKAN portal so the agent can search it when answering questions.</p>
                </div>
            `;
            return;
        }
        container.innerHTML = data.sites.map(s => renderCkanSiteCard(s)).join('');
        bindClick(container, '.js-ckan-remove', (el) => removeCkanSite(el.dataset.siteId));
    } catch (error) {
        container.innerHTML = `
            <div class="col-12">
                <div class="alert alert-danger">
                    <i class="bi bi-exclamation-triangle me-2"></i>
                    Failed to load CKAN sites: ${escapeHtml(error.message)}
                </div>
            </div>
        `;
    }
}

function renderCkanSiteCard(site) {
    const idSafe = escapeHtml(site.id || '');
    const keywords = Array.isArray(site.keywords) && site.keywords.length
        ? site.keywords.map(k =>
            `<span class="badge bg-secondary me-1">${escapeHtml(k)}</span>`
        ).join('')
        : '<span class="text-muted small">No keywords</span>';
    const orgBadge = site.organization
        ? `<span class="badge bg-info text-dark"><i class="bi bi-building me-1"></i>${escapeHtml(site.organization)}</span>`
        : '';
    const isDefault = site.added_by === 'default';
    return `
        <div class="col-12">
            <div class="mcp-server-card">
                <div class="d-flex justify-content-between align-items-start">
                    <div class="flex-grow-1">
                        <div class="d-flex align-items-center gap-2 mb-1 flex-wrap">
                            <h6 class="mb-0">${escapeHtml(site.name)}</h6>
                            <span class="badge bg-secondary"><i class="bi bi-hash me-1"></i>${idSafe}</span>
                            ${orgBadge}
                            ${isDefault ? '<span class="badge bg-secondary border">built-in</span>' : ''}
                        </div>
                        <p class="small mb-1">
                            <a href="${escapeHtml(site.url)}" target="_blank" rel="noopener"
                               class="text-decoration-none">
                                <i class="bi bi-globe me-1"></i>${escapeHtml(site.url)}
                            </a>
                        </p>
                        ${site.description
                            ? `<p class="text-muted mb-2 small">${escapeHtml(site.description)}</p>`
                            : ''}
                        <div class="mb-1">${keywords}</div>
                        <p class="small text-muted mb-0">
                            Quality: ${Number(site.quality_score ?? 0.85).toFixed(2)}
                            ${site.added_by ? `&middot; added by ${escapeHtml(site.added_by)}` : ''}
                        </p>
                    </div>
                    <div class="d-flex gap-2 ms-3 flex-shrink-0">
                        <button class="btn btn-sm btn-outline-danger js-ckan-remove"
                                data-site-id="${idSafe}" title="Remove site">
                            <i class="bi bi-trash"></i>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
}

async function addCkanSite() {
    const name = document.getElementById('ckanSiteName').value.trim();
    const url = document.getElementById('ckanSiteUrl').value.trim();
    if (!name || !url) {
        showToast('Name and URL are required', 'warning');
        return;
    }
    const siteId = document.getElementById('ckanSiteId').value.trim();
    const organization = document.getElementById('ckanSiteOrg').value.trim();
    const description = document.getElementById('ckanSiteDescription').value.trim();
    const keywordsRaw = document.getElementById('ckanSiteKeywords').value;
    const keywords = keywordsRaw
        .split(',')
        .map(k => k.trim())
        .filter(Boolean);
    const quality = parseFloat(document.getElementById('ckanSiteQuality').value);

    const payload = {
        name,
        url,
        organization: organization || null,
        description,
        keywords,
        quality_score: Number.isFinite(quality) ? quality : 0.85,
    };
    if (siteId) payload.site_id = siteId;

    try {
        const response = await adminFetch(CKAN_SITES_API, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await response.json();
        showToast(data.message || 'CKAN site added', 'success');
        document.getElementById('addCkanSiteForm').reset();
        document.getElementById('ckanSiteQualityValue').textContent = '0.85';
        bootstrap.Modal.getInstance(document.getElementById('addCkanSiteModal'))?.hide();
        loadCkanSites();
    } catch (error) {
        showToast(`Failed to add CKAN site: ${error.message}`, 'danger');
    }
}

async function removeCkanSite(siteId) {
    if (!confirm(`Remove CKAN site "${siteId}"?\n\nThe agent will stop searching this portal for new queries.`)) {
        return;
    }
    try {
        await adminFetch(`${CKAN_SITES_API}/${encodeURIComponent(siteId)}`, { method: 'DELETE' });
        showToast(`CKAN site "${siteId}" removed`, 'success');
        loadCkanSites();
    } catch (error) {
        showToast(`Failed to remove CKAN site: ${error.message}`, 'danger');
    }
}

// These handlers are wired up via bindClick() (data-* attributes +
// addEventListener) rather than inline onclick, so they no longer need to be
// exposed on the global window object.
