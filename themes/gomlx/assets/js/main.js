/* GoMLX site JS — theme, tabs, copy, TOC scroll spy */
(function() {
  'use strict';

  /* ── Theme toggle ── */
  const root = document.documentElement;
  const toggleBtn = document.getElementById('theme-toggle');
  const saved = localStorage.getItem('gomlx-theme');
  const preferred = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  const theme = saved || preferred;
  root.setAttribute('data-theme', theme);

  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      const current = root.getAttribute('data-theme');
      const next = current === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      localStorage.setItem('gomlx-theme', next);
    });
  }

  /* ── Code tabs ── */
  document.querySelectorAll('.code-tabs').forEach(tabBar => {
    const window_ = tabBar.closest('.code-window');
    if (!window_) return;
    const panels = window_.querySelectorAll('.code-panel');
    const tabs = tabBar.querySelectorAll('.code-tab');

    tabs.forEach((tab, i) => {
      tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('is-active'));
        panels.forEach(p => p.classList.remove('is-active'));
        tab.classList.add('is-active');
        if (panels[i]) panels[i].classList.add('is-active');
      });
    });
  });

  /* ── Copy buttons (homepage code window) ── */
  document.querySelectorAll('.copy-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const panel = btn.closest('.code-window').querySelector('.code-panel.is-active pre');
      if (!panel) return;
      navigator.clipboard.writeText(panel.innerText.trim()).then(() => {
        btn.classList.add('copied');
        btn.querySelector('span') && (btn.textContent = 'Copied!');
        setTimeout(() => {
          btn.classList.remove('copied');
          btn.innerHTML = `<img src="${window.GomlxBaseUrl}img/copy.svg" class="svg-icon" alt="icon"> Copy`;
        }, 2000);
      });
    });
  });

  /* ── Docs: copy buttons on code blocks ── */
  document.querySelectorAll('.docs-body pre').forEach(pre => {
    const wrap = document.createElement('div');
    wrap.className = 'code-copy-wrap';
    pre.parentNode.insertBefore(wrap, pre);
    wrap.appendChild(pre);

    const btn = document.createElement('button');
    btn.className = 'code-copy-btn';
    btn.innerHTML = `<img src="${window.GomlxBaseUrl}img/copy-small.svg" class="svg-icon" alt="icon"> Copy`;
    wrap.appendChild(btn);

    btn.addEventListener('click', () => {
      navigator.clipboard.writeText(pre.innerText.trim()).then(() => {
        btn.textContent = 'Copied!';
        btn.classList.add('copied');
        setTimeout(() => {
          btn.innerHTML = `<img src="${window.GomlxBaseUrl}img/copy-small.svg" class="svg-icon" alt="icon"> Copy`;
          btn.classList.remove('copied');
        }, 2000);
      });
    });
  });

  /* ── TOC scroll spy ── */
  const tocLinks = document.querySelectorAll('.docs-toc a[href^="#"]');
  if (tocLinks.length) {
    const headings = Array.from(tocLinks).map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);

    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const id = entry.target.getAttribute('id');
          tocLinks.forEach(a => a.classList.remove('is-active'));
          const active = document.querySelector(`.docs-toc a[href="#${id}"]`);
          if (active) active.classList.add('is-active');
        }
      });
    }, { rootMargin: '-60px 0px -60% 0px', threshold: 0 });

    headings.forEach(h => observer.observe(h));
  }

  /* ── Mobile sidebar toggle ── */
  const hamburger = document.getElementById('nav-hamburger');
  const sidebar = document.querySelector('.docs-sidebar');
  if (hamburger && sidebar) {
    hamburger.addEventListener('click', () => {
      const open = sidebar.classList.toggle('is-open');
      hamburger.setAttribute('aria-expanded', String(open));
    });
    document.addEventListener('click', e => {
      if (sidebar.classList.contains('is-open') && !sidebar.contains(e.target) && !hamburger.contains(e.target)) {
        sidebar.classList.remove('is-open');
        hamburger.setAttribute('aria-expanded', 'false');
      }
    });
  }

  /* ── Keyboard search shortcut ── */
  document.addEventListener('keydown', e => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault();
      document.getElementById('search-trigger')?.focus();
    }
  });

  /* ── Smooth nav active state on scroll ── */
  const nav = document.getElementById('site-nav');
  if (nav) {
    window.addEventListener('scroll', () => {
      nav.classList.toggle('is-scrolled', window.scrollY > 8);
    }, { passive: true });
  }

})();
