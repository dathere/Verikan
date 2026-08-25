// Shared theme handling for admin/docs/mcp_settings templates.
// Index page has its own equivalent inside app.js. Light is the default.

function updateLogos(theme) {
    var src = theme === 'light' ? '/static/images/logo-light.svg' : '/static/images/logo.png';
    document.querySelectorAll('img.dc-logo').forEach(function(img) { img.src = src; });
}

function updateThemeIcon(theme) {
    var btn = document.querySelector('#themeToggle') || document.querySelector('.theme-toggle');
    var icon = btn ? btn.querySelector('i') : null;
    if (icon) icon.className = theme === 'dark' ? 'bi bi-sun-fill' : 'bi bi-moon-stars';
    if (btn) btn.setAttribute('aria-pressed', theme === 'dark' ? 'true' : 'false');
}

function toggleTheme() {
    var html = document.documentElement;
    var cur = html.getAttribute('data-bs-theme');
    var next = cur === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-bs-theme', next);
    localStorage.setItem('dc-theme', next);
    updateThemeIcon(next);
    var dark = document.getElementById('hljs-theme-dark');
    var light = document.getElementById('hljs-theme-light');
    if (dark && light) { dark.disabled = next === 'light'; light.disabled = next === 'dark'; }
    updateLogos(next);
}

// Init icon/logo on load
document.addEventListener('DOMContentLoaded', function() {
    var theme = localStorage.getItem('dc-theme') || 'light';
    updateThemeIcon(theme);
    var dark = document.getElementById('hljs-theme-dark');
    var light = document.getElementById('hljs-theme-light');
    if (dark && light) { dark.disabled = theme === 'light'; light.disabled = theme === 'dark'; }
    updateLogos(theme);
});
