// MCP Server Management JavaScript

const MCP_API = '/api/v1/mcp';

// State
let currentToolCall = { serverId: null, toolName: null };

// DOM Ready
document.addEventListener('DOMContentLoaded', () => {
    loadServers();
    setupEventListeners();
});

function setupEventListeners() {
    // Transport type toggle
    document.querySelectorAll('input[name="transport"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            const isStdio = e.target.value === 'stdio';
            document.getElementById('stdioFields').classList.toggle('d-none', !isStdio);
            document.getElementById('sseFields').classList.toggle('d-none', isStdio);
        });
    });

    // Add server button
    document.getElementById('addServerBtn').addEventListener('click', addServer);

    // Execute tool button
    document.getElementById('executeToolBtn').addEventListener('click', executeTool);
}

// Bind a click handler to every element matching selector inside root.
function bindClick(root, selector, handler) {
    root.querySelectorAll(selector).forEach((el) => {
        el.addEventListener('click', () => handler(el));
    });
}

// Load all servers
async function loadServers() {
    const container = document.getElementById('serverList');

    try {
        const response = await fetch(`${MCP_API}/servers`);
        const data = await response.json();

        if (data.servers.length === 0) {
            container.innerHTML = `
                <div class="col-12 empty-state">
                    <i class="bi bi-plug" aria-hidden="true"></i>
                    <div class="empty-title">No MCP servers configured</div>
                    <p>Add an MCP server to extend Verikan with additional data sources.</p>
                    <button class="btn btn-primary" data-bs-toggle="modal" data-bs-target="#addServerModal">
                        <i class="bi bi-plus-lg me-2" aria-hidden="true"></i>Add Your First Server
                    </button>
                </div>
            `;
            return;
        }

        container.innerHTML = data.servers.map(server => renderServerCard(server)).join('');

        // Wire actions via data attributes rather than inline onclick, so
        // names/descriptions never enter a JS execution context.
        bindClick(container, '.js-mcp-tool', (el) => openToolCall(el.dataset.serverId, el.dataset.toolName));
        bindClick(container, '.js-mcp-connect', (el) => connectServer(el.dataset.serverId, el.dataset.serverName, el));
        bindClick(container, '.js-mcp-disconnect', (el) => disconnectServer(el.dataset.serverId));
        bindClick(container, '.js-mcp-remove', (el) => removeServer(el.dataset.serverId, el.dataset.serverName));

    } catch (error) {
        container.innerHTML = `
            <div class="col-12 empty-state">
                <i class="bi bi-exclamation-triangle text-danger" aria-hidden="true"></i>
                <div class="empty-title">Couldn't load MCP servers</div>
                <p class="small">${escapeHtml(error.message)}</p>
                <button class="btn btn-outline-secondary js-mcp-retry" type="button">
                    <i class="bi bi-arrow-clockwise me-1" aria-hidden="true"></i>Retry
                </button>
            </div>
        `;
        bindClick(container, '.js-mcp-retry', () => loadServers());
    }
}

function renderServerCard(server) {
    const statusClass = `status-${server.status}`;
    const statusIcon = {
        'connected': 'bi-check-circle-fill',
        'disconnected': 'bi-circle',
        'connecting': 'bi-arrow-repeat',
        'error': 'bi-exclamation-circle-fill',
    }[server.status] || 'bi-circle';

    const transportIcon = { stdio: 'bi-terminal', sse: 'bi-broadcast' }[server.transport] || 'bi-globe';
    const transportLabel = { stdio: 'stdio', sse: 'SSE', streamable_http: 'HTTP' }[server.transport] || server.transport;

    const toolsHtml = server.tools.length > 0
        ? server.tools.map(t => `
            <button type="button" class="tool-chip js-mcp-tool" title="${escapeHtml(t.description)}"
                  data-server-id="${escapeHtml(server.id)}" data-tool-name="${escapeHtml(t.name)}">
                <i class="bi bi-wrench me-1" aria-hidden="true"></i>${escapeHtml(t.name)}
            </button>
        `).join('')
        : '<span class="text-muted small">No tools discovered yet — connect to discover tools</span>';

    const connectBtn = server.status === 'connected'
        ? `<button class="btn btn-sm btn-outline-warning text-nowrap js-mcp-disconnect" data-server-id="${escapeHtml(server.id)}">
               <i class="bi bi-plug me-1" aria-hidden="true"></i>Disconnect
           </button>`
        : `<button class="btn btn-sm btn-outline-success text-nowrap js-mcp-connect"
                   data-server-id="${escapeHtml(server.id)}" data-server-name="${escapeHtml(server.name)}">
               <i class="bi bi-plug-fill me-1" aria-hidden="true"></i>Connect
           </button>`;

    const errorHtml = server.last_error
        ? `<div class="text-danger small mt-2"><i class="bi bi-exclamation-triangle me-1" aria-hidden="true"></i>${escapeHtml(server.last_error)}</div>`
        : '';

    return `
        <div class="col-12">
            <div class="server-card p-3">
                <div class="d-flex justify-content-between align-items-start flex-column flex-sm-row gap-3">
                    <div class="flex-grow-1" style="min-width: 0;">
                        <div class="d-flex align-items-center gap-2 mb-1 flex-wrap">
                            <h5 class="mb-0 h6">${escapeHtml(server.name)}</h5>
                            <span class="badge bg-secondary"><i class="bi ${transportIcon} me-1" aria-hidden="true"></i>${escapeHtml(transportLabel)}</span>
                            <span class="status-badge ${statusClass}">
                                <i class="bi ${statusIcon} me-1" aria-hidden="true"></i>${escapeHtml(server.status)}
                            </span>
                        </div>
                        <p class="text-muted mb-2 small">${escapeHtml(server.description || 'No description')}</p>

                        ${server.server_name ? `<p class="small mb-1 text-muted"><strong>Server:</strong> ${escapeHtml(server.server_name)} ${server.server_version ? 'v' + escapeHtml(server.server_version) : ''}</p>` : ''}
                        ${server.command ? `<p class="small mb-1 font-monospace text-muted"><i class="bi bi-terminal me-1" aria-hidden="true"></i>${escapeHtml(server.command)} ${server.args.map(a => escapeHtml(a)).join(' ')}</p>` : ''}
                        ${server.working_dir ? `<p class="small mb-1 text-muted"><i class="bi bi-folder me-1" aria-hidden="true"></i>${escapeHtml(server.working_dir)}</p>` : ''}
                        ${server.url ? `<p class="small mb-1 text-muted"><i class="bi bi-globe me-1" aria-hidden="true"></i>${escapeHtml(server.url)}</p>` : ''}

                        <div class="mt-2">
                            <strong class="small">Tools:</strong>
                            <div class="mt-1">${toolsHtml}</div>
                        </div>

                        ${server.categories.length > 0 ? `
                            <div class="mt-2">
                                <strong class="small">Categories:</strong>
                                ${server.categories.map(c => `<span class="badge bg-info me-1">${escapeHtml(c)}</span>`).join('')}
                            </div>
                        ` : ''}

                        ${errorHtml}
                    </div>
                    <div class="d-flex gap-2 flex-shrink-0">
                        ${connectBtn}
                        <button class="btn btn-sm btn-outline-danger js-mcp-remove"
                                data-server-id="${escapeHtml(server.id)}" data-server-name="${escapeHtml(server.name)}"
                                title="Remove server" aria-label="Remove MCP server ${escapeHtml(server.name)}">
                            <i class="bi bi-trash" aria-hidden="true"></i>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
}

// Add a new server
async function addServer() {
    const transport = document.querySelector('input[name="transport"]:checked').value;

    const payload = {
        name: document.getElementById('serverName').value.trim(),
        description: document.getElementById('serverDescription').value.trim(),
        transport: transport,
        command: document.getElementById('serverCommand').value.trim(),
        args: document.getElementById('serverArgs').value.trim()
            .split('\n').map(a => a.trim()).filter(a => a),
        working_dir: document.getElementById('serverWorkingDir').value.trim(),
        url: document.getElementById('serverUrl').value.trim(),
        categories: document.getElementById('serverCategories').value.trim()
            .split(',').map(c => c.trim()).filter(c => c),
        keywords: document.getElementById('serverKeywords').value.trim()
            .split(',').map(k => k.trim()).filter(k => k),
        auto_connect: document.getElementById('serverAutoConnect').checked,
    };

    // Parse environment variables
    const envText = document.getElementById('serverEnv').value.trim();
    const env = {};
    if (envText) {
        envText.split('\n').forEach(line => {
            const [key, ...rest] = line.split('=');
            if (key && rest.length > 0) {
                env[key.trim()] = rest.join('=').trim();
            }
        });
    }
    payload.env = env;

    if (!payload.name) {
        showToast('Server name is required', 'danger');
        return;
    }

    try {
        const response = await fetch(`${MCP_API}/servers`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Failed to add server');
        }

        const result = await response.json();
        showToast(`Server "${payload.name}" added successfully`, 'success');

        // Close modal and refresh
        bootstrap.Modal.getInstance(document.getElementById('addServerModal')).hide();
        document.getElementById('addServerForm').reset();
        loadServers();

    } catch (error) {
        showToast(error.message, 'danger');
    }
}

// Connect to a server. Shows in-button progress (a cold connect can take
// seconds) and uses the human-readable server name in feedback.
async function connectServer(serverId, serverName, btn) {
    const label = serverName || serverId;
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<span class="spinner-border spinner-border-sm me-1" role="status" aria-hidden="true"></span>Connecting…`;
    }
    showToast(`Connecting to ${label}…`, 'info');

    try {
        const response = await fetch(`${MCP_API}/servers/${serverId}/connect`, {
            method: 'POST',
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Connection failed');
        }

        const result = await response.json();
        showToast(`Connected to ${label} — found ${result.tools.length} tool(s)`, 'success');
        loadServers();

    } catch (error) {
        showToast(`Connection failed: ${error.message}`, 'danger');
        loadServers();
    }
}

// Disconnect from a server
async function disconnectServer(serverId) {
    try {
        await fetch(`${MCP_API}/servers/${serverId}/disconnect`, { method: 'POST' });
        showToast('Disconnected', 'info');
        loadServers();
    } catch (error) {
        showToast(`Error: ${error.message}`, 'danger');
    }
}

// Remove a server
async function removeServer(serverId, serverName) {
    if (!confirm(`Remove MCP server "${serverName || serverId}"?`)) return;

    try {
        await fetch(`${MCP_API}/servers/${serverId}`, { method: 'DELETE' });
        showToast('Server removed', 'success');
        loadServers();
    } catch (error) {
        showToast(`Error: ${error.message}`, 'danger');
    }
}

// Open tool call modal
function openToolCall(serverId, toolName) {
    currentToolCall = { serverId, toolName };
    document.getElementById('toolCallTitle').textContent = `Call: ${toolName}`;
    document.getElementById('toolCallArgs').value = '{}';
    document.getElementById('toolCallResult').classList.add('d-none');

    const modal = new bootstrap.Modal(document.getElementById('toolCallModal'));
    modal.show();
}

// Execute tool call
async function executeTool() {
    const { serverId, toolName } = currentToolCall;
    if (!serverId || !toolName) return;

    let args;
    try {
        args = JSON.parse(document.getElementById('toolCallArgs').value);
    } catch (e) {
        showToast('Invalid JSON arguments', 'danger');
        return;
    }

    const resultDiv = document.getElementById('toolCallResult');
    const resultText = document.getElementById('toolCallResultText');

    resultDiv.classList.remove('d-none');
    resultText.textContent = 'Executing...';

    try {
        const response = await fetch(`${MCP_API}/servers/${serverId}/tools/call`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tool_name: toolName, arguments: args }),
        });

        const result = await response.json();

        if (result.is_error) {
            resultText.textContent = `Error: ${result.text}`;
        } else {
            resultText.textContent = result.text || JSON.stringify(result.content, null, 2);
        }

    } catch (error) {
        resultText.textContent = `Error: ${error.message}`;
    }
}

// Utility functions
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
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

// Make functions globally accessible
window.connectServer = connectServer;
window.disconnectServer = disconnectServer;
window.removeServer = removeServer;
window.openToolCall = openToolCall;
