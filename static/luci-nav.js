/**
 * LUCI Universal Navigation
 * Inject into any page with: <script src="/static/luci-nav.js"></script>
 * Then call: LuciNav.inject()
 */

const LuciNav = {
  pages: [
    { icon: "💬", label: "Chat",        url: "/",           desc: "Main AI interface" },
    { icon: "📚", label: "Learn",       url: "/learn",      desc: "Curriculum & lessons" },
    { icon: "⚡", label: "Agent",       url: "/agent",      desc: "Autonomous task execution" },
    { icon: "🔬", label: "Diagnostics", url: "/diagnose",   desc: "Error analysis & health" },
    { icon: "📋", label: "Audit",       url: "/audit/ui",   desc: "Workspace health report" },
    { icon: "🧠", label: "Memory",      url: "/#memory",    desc: "Active memory store", onclick: "LuciNav.openMemory()" },
    { icon: "📊", label: "Progress",    url: "/learn#progress", desc: "Learning progress" },
  ],

  currentPath: window.location.pathname,

  inject() {
    // Find existing header back button and add menu next to it
    const header = document.querySelector('.header');
    if (!header) return;

    // Don't inject twice
    if (document.getElementById('luci-nav-menu')) return;

    // Create menu button
    const menuBtn = document.createElement('div');
    menuBtn.id = 'luci-nav-trigger';
    menuBtn.style.cssText = `
      position: relative; display: inline-flex; align-items: center;
      margin-left: 8px;
    `;
    menuBtn.innerHTML = `
      <button onclick="LuciNav.toggle()" style="
        background: none; border: 1px solid #2a2a3a; color: #888;
        padding: 5px 12px; border-radius: 4px; cursor: pointer;
        font-family: inherit; font-size: 11px; letter-spacing: 1px;
        display: flex; align-items: center; gap: 6px; transition: all .15s;
      " onmouseover="this.style.borderColor='#D4AF37';this.style.color='#D4AF37'"
         onmouseout="this.style.borderColor='#2a2a3a';this.style.color='#888'">
        ⊞ MENU
      </button>
    `;

    // Create dropdown
    const dropdown = document.createElement('div');
    dropdown.id = 'luci-nav-menu';
    dropdown.style.cssText = `
      display: none; position: fixed; top: 56px; left: 0;
      width: 280px; background: #0c0c10;
      border: 1px solid #2a2a3a; border-top: none;
      border-radius: 0 0 10px 0; z-index: 9999;
      box-shadow: 4px 4px 30px rgba(0,0,0,.8);
      overflow: hidden;
    `;

    const menuItems = LuciNav.pages.map(p => {
      const isActive = LuciNav.currentPath === p.url ||
                       (p.url !== '/' && LuciNav.currentPath.startsWith(p.url.split('#')[0]));
      const clickHandler = p.onclick
        ? `onclick="${p.onclick}; LuciNav.close()"`
        : `onclick="window.location='${p.url}'"`;
      return `
        <div ${clickHandler} style="
          display: flex; align-items: center; gap: 12px;
          padding: 12px 16px; cursor: pointer;
          background: ${isActive ? 'rgba(212,175,55,.08)' : 'transparent'};
          border-left: 3px solid ${isActive ? '#D4AF37' : 'transparent'};
          border-bottom: 1px solid #111; transition: all .1s;
        "
        onmouseover="this.style.background='rgba(212,175,55,.05)'"
        onmouseout="this.style.background='${isActive ? 'rgba(212,175,55,.08)' : 'transparent'}'">
          <span style="font-size: 16px; width: 24px; text-align: center">${p.icon}</span>
          <div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px;
              color: ${isActive ? '#D4AF37' : '#ccc'}; font-weight: ${isActive ? '700' : '400'};
              letter-spacing: 1px">${p.label}</div>
            <div style="font-size: 10px; color: #555; margin-top: 2px">${p.desc}</div>
          </div>
          ${isActive ? '<span style="margin-left:auto;font-size:10px;color:#D4AF37">●</span>' : ''}
        </div>
      `;
    }).join('');

    dropdown.innerHTML = `
      <div style="padding: 10px 16px 8px; border-bottom: 1px solid #1a1a1a;
        font-size: 9px; color: #444; letter-spacing: 3px; text-transform: uppercase">
        LUCI Navigation
      </div>
      ${menuItems}
      <div style="padding: 10px 16px; border-top: 1px solid #111;
        font-size: 10px; color: #333; text-align: center; letter-spacing: 1px">
        Press ESC to close
      </div>
    `;

    // Insert after back button
    const backBtn = header.querySelector('.back, a[href="/"]');
    if (backBtn) {
      backBtn.parentNode.insertBefore(menuBtn, backBtn.nextSibling);
    } else {
      header.insertBefore(menuBtn, header.firstChild);
    }

    // Add dropdown to body
    document.body.appendChild(dropdown);

    // Close on outside click
    document.addEventListener('click', function(e) {
      if (!document.getElementById('luci-nav-trigger')?.contains(e.target) &&
          !document.getElementById('luci-nav-menu')?.contains(e.target)) {
        LuciNav.close();
      }
    });

    // Close on ESC
    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape') LuciNav.close();
    });
  },

  toggle() {
    const menu = document.getElementById('luci-nav-menu');
    if (!menu) return;
    menu.style.display = menu.style.display === 'none' ? 'block' : 'none';
  },

  close() {
    const menu = document.getElementById('luci-nav-menu');
    if (menu) menu.style.display = 'none';
  },

  openMemory() {
    // Trigger memory overlay if on main page, otherwise go to main page
    if (window.location.pathname === '/') {
      const btn = document.querySelector('[onclick*="memory"], [onclick*="Memory"]');
      if (btn) btn.click();
    } else {
      window.location = '/#memory';
    }
  }
};

// Auto-inject when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => LuciNav.inject());
} else {
  LuciNav.inject();
}
