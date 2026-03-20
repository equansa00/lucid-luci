/**
 * LUCI Universal Navigation
 * Inject into any page with: <script src="/static/luci-nav.js"></script>
 */
const LuciNav = {
  pages: [
    { icon: "💬", label: "Chat",        url: "/",          desc: "Main AI interface",        key: "chat" },
    { icon: "📚", label: "Learn",       url: "/learn",     desc: "Curriculum & lessons",     key: "learn" },
    { icon: "⚡", label: "Agent",       url: "/agent",     desc: "Autonomous task execution", key: "agent" },
    { icon: "🔬", label: "Diagnostics", url: "/diagnose",  desc: "Error analysis & health",  key: "diagnose" },
    { icon: "📋", label: "Audit",       url: "/audit/ui",  desc: "Workspace health report",  key: "audit" },
  ],

  currentPath: window.location.pathname,

  getStyles() {
    return `
      #luci-nav-btn {
        background: none;
        border: 1px solid #2a2a3a;
        color: #888;
        padding: 5px 14px;
        border-radius: 4px;
        cursor: pointer;
        font-family: 'JetBrains Mono', 'SF Mono', monospace;
        font-size: 11px;
        letter-spacing: 1px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        transition: all .15s;
        margin-left: 8px;
        white-space: nowrap;
      }
      #luci-nav-btn:hover { border-color: #D4AF37; color: #D4AF37; }
      #luci-nav-btn.open  { border-color: #D4AF37; color: #D4AF37; background: rgba(212,175,55,.06); }

      #luci-nav-dropdown {
        display: none;
        position: fixed;
        top: 57px;
        left: 0;
        width: 260px;
        background: #0a0a0e;
        border: 1px solid #2a2a3a;
        border-top: 2px solid #D4AF37;
        border-radius: 0 0 10px 0;
        z-index: 9999;
        box-shadow: 8px 8px 40px rgba(0,0,0,.9);
        overflow: hidden;
        animation: navSlideIn .12s ease;
      }
      #luci-nav-dropdown.open { display: block; }
      @keyframes navSlideIn {
        from { opacity: 0; transform: translateY(-8px); }
        to   { opacity: 1; transform: translateY(0); }
      }

      .luci-nav-header {
        padding: 10px 16px 8px;
        font-size: 9px;
        color: #D4AF37;
        letter-spacing: 3px;
        text-transform: uppercase;
        border-bottom: 1px solid #1a1a1a;
        font-family: 'JetBrains Mono', monospace;
      }

      .luci-nav-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 11px 16px;
        cursor: pointer;
        border-left: 3px solid transparent;
        border-bottom: 1px solid #0f0f13;
        transition: all .1s;
        text-decoration: none;
      }
      .luci-nav-item:hover {
        background: rgba(212,175,55,.05);
        border-left-color: #D4AF37;
      }
      .luci-nav-item.active {
        background: rgba(212,175,55,.08);
        border-left-color: #D4AF37;
      }
      .luci-nav-icon {
        font-size: 15px;
        width: 22px;
        text-align: center;
        flex-shrink: 0;
      }
      .luci-nav-text { flex: 1; min-width: 0; }
      .luci-nav-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #ccc;
        letter-spacing: .5px;
        font-weight: 500;
      }
      .luci-nav-item.active .luci-nav-label { color: #D4AF37; font-weight: 700; }
      .luci-nav-desc {
        font-size: 10px;
        color: #444;
        margin-top: 1px;
        font-family: 'JetBrains Mono', monospace;
      }
      .luci-nav-active-dot {
        width: 5px; height: 5px;
        border-radius: 50%;
        background: #D4AF37;
        box-shadow: 0 0 6px #D4AF37;
        flex-shrink: 0;
      }
      .luci-nav-footer {
        padding: 8px 16px;
        font-size: 9px;
        color: #2a2a2a;
        text-align: center;
        letter-spacing: 2px;
        font-family: 'JetBrains Mono', monospace;
        border-top: 1px solid #111;
      }
    `;
  },

  inject() {
    if (document.getElementById('luci-nav-btn')) return;

    // Inject styles
    const style = document.createElement('style');
    style.textContent = this.getStyles();
    document.head.appendChild(style);

    // Find header
    const header = document.querySelector('.header');
    if (!header) return;

    // Build menu button
    const btn = document.createElement('button');
    btn.id = 'luci-nav-btn';
    btn.innerHTML = '⊞ MENU';
    btn.onclick = () => this.toggle();

    // Build dropdown
    const dropdown = document.createElement('div');
    dropdown.id = 'luci-nav-dropdown';

    const items = this.pages.map(p => {
      const isActive = this.currentPath === p.url ||
        (p.url !== '/' && p.url !== '' && this.currentPath.startsWith(p.url));
      return `
        <a class="luci-nav-item${isActive ? ' active' : ''}" href="${p.url}">
          <span class="luci-nav-icon">${p.icon}</span>
          <div class="luci-nav-text">
            <div class="luci-nav-label">${p.label}</div>
            <div class="luci-nav-desc">${p.desc}</div>
          </div>
          ${isActive ? '<div class="luci-nav-active-dot"></div>' : ''}
        </a>
      `;
    }).join('');

    dropdown.innerHTML = `
      <div class="luci-nav-header">⚡ LUCI SYSTEM</div>
      ${items}
      <div class="luci-nav-footer">ESC TO CLOSE</div>
    `;

    // Insert button after back link
    const back = header.querySelector('a.back, a[href="/"]');
    if (back) {
      back.insertAdjacentElement('afterend', btn);
    } else {
      header.prepend(btn);
    }

    document.body.appendChild(dropdown);

    // Close handlers
    document.addEventListener('click', e => {
      if (!btn.contains(e.target) && !dropdown.contains(e.target)) {
        this.close();
      }
    });
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') this.close();
    });
  },

  toggle() {
    const btn      = document.getElementById('luci-nav-btn');
    const dropdown = document.getElementById('luci-nav-dropdown');
    if (!btn || !dropdown) return;
    const isOpen = dropdown.classList.contains('open');
    if (isOpen) {
      this.close();
    } else {
      dropdown.classList.add('open');
      btn.classList.add('open');
    }
  },

  close() {
    const btn      = document.getElementById('luci-nav-btn');
    const dropdown = document.getElementById('luci-nav-dropdown');
    if (btn)      btn.classList.remove('open');
    if (dropdown) dropdown.classList.remove('open');
  }
};

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => LuciNav.inject());
} else {
  LuciNav.inject();
}
