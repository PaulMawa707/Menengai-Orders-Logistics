// Notifications: trip events (stops visited, late or missed; trips completed, ended with missed stops, not started),
// shown under the bell and as browser pop-ups while the dashboard is open. Uses helpers from trips.js.

const ALERT_KINDS = {
  visit: { label: 'Visited', icon: '✅', cls: 'completed' },
  late: { label: 'Late', icon: '⏰', cls: 'late' },
  missed: { label: 'Missed', icon: '❌', cls: 'ended' },
  completed: { label: 'Completed', icon: '🏁', cls: 'completed' },
  ended: { label: 'Ended with missed stops', icon: '⚠️', cls: 'ended' },
  not_started: { label: 'Not started', icon: '🕒', cls: 'late' },
};
const ALERTS_POLL_MS = 120000;
const SEEN_KEY = 'menengai.alerts.seen';
const READ_KEY = 'menengai.alerts.read';
const KEEP_KEYS = 3000;
const MAX_POPUPS = 4;

function initAlerts({ openTrip }) {
  const el = {
    bell: document.getElementById('alerts-bell'),
    count: document.getElementById('alerts-count'),
    panel: document.getElementById('alerts-panel'),
    list: document.getElementById('alerts-list'),
    filters: document.getElementById('alerts-filters'),
    foot: document.getElementById('alerts-foot'),
    read: document.getElementById('alerts-read'),
    enable: document.getElementById('alerts-enable'),
  };
  if (!el.bell) return;
  const stored = (k) => {
    try {
      return new Set(JSON.parse(localStorage.getItem(k) || '[]'));
    } catch {
      return new Set();
    }
  };
  const keep = (set, keys) => {
    const merged = new Set([...set, ...keys]);
    return new Set([...merged].slice(-KEEP_KEYS));
  };
  let firstRun = localStorage.getItem(SEEN_KEY) === null;
  // Popped up already (seen) and opened under the bell (read).
  let seen = stored(SEEN_KEY);
  let read = stored(READ_KEY);
  let alerts = [];
  let now = 0;
  let kind = null;
  let loaded = false;

  const canNotify = () => 'Notification' in window && Notification.permission === 'granted';

  function updateEnable() {
    if (!('Notification' in window)) {
      el.enable.classList.add('hidden');
      return;
    }
    el.enable.classList.toggle('hidden', Notification.permission === 'granted');
    el.enable.textContent = Notification.permission === 'denied' ? 'Pop-ups blocked in the browser' : 'Enable pop-ups';
    el.enable.disabled = Notification.permission === 'denied';
  }

  function popUp(a) {
    try {
      const n = new Notification(a.title, { body: a.detail, tag: a.key, icon: '/static/img/login-hero.jpg' });
      n.onclick = () => {
        window.focus();
        openTrip(a.tripId, a.day);
        n.close();
      };
    } catch {
      // Some browsers only allow notifications from a service worker; the bell still shows them.
    }
  }

  function renderBadge() {
    const unread = alerts.filter((a) => !read.has(a.key)).length;
    el.count.textContent = unread > 99 ? '99+' : String(unread);
    el.count.classList.toggle('hidden', !unread);
    el.bell.title = unread ? `${unread} new notification${unread === 1 ? '' : 's'}` : 'Notifications';
  }

  function renderList() {
    const counts = {};
    alerts.forEach((a) => (counts[a.kind] = (counts[a.kind] || 0) + 1));
    el.filters.innerHTML = [
      `<button type="button" class="chip ${kind === null ? 'active' : ''}" data-kind="">All ${alerts.length}</button>`,
      ...Object.entries(ALERT_KINDS)
        .filter(([k]) => counts[k])
        .map(([k, m]) => `<button type="button" class="chip ${kind === k ? 'active' : ''}" data-kind="${k}">${m.icon} ${m.label} ${counts[k]}</button>`),
    ].join('');
    el.filters.querySelectorAll('.chip').forEach((b) =>
      b.addEventListener('click', () => {
        kind = b.dataset.kind || null;
        renderList();
      }),
    );
    const rows = alerts.filter((a) => !kind || a.kind === kind).slice(0, 150);
    el.list.innerHTML = rows.length
      ? rows
          .map((a) => {
            const m = ALERT_KINDS[a.kind] || ALERT_KINDS.visit;
            return `<li class="alert-item ${read.has(a.key) ? '' : 'unread'}" data-trip="${esc(a.tripId)}" data-day="${esc(a.day)}">
              <span class="alert-icon ${m.cls}">${m.icon}</span>
              <span class="alert-text"><b>${esc(a.title)}</b><small>${esc(a.detail)}</small></span>
              <span class="alert-time">${fmtTime(a.at)}<small>${esc(ago(a.at, now))}</small></span>
            </li>`;
          })
          .join('')
      : `<li class="alerts-empty">${loaded ? 'No notifications in the last 24 hours.' : 'Loading…'}</li>`;
    el.list.querySelectorAll('.alert-item').forEach((li) =>
      li.addEventListener('click', () => {
        close();
        openTrip(li.dataset.trip, li.dataset.day);
      }),
    );
  }

  async function poll() {
    try {
      const d = await apiJson('/api/logistics/alerts');
      loaded = true;
      now = d.now;
      alerts = d.alerts || [];
      const fresh = alerts.filter((a) => !seen.has(a.key));
      if (firstRun) {
        // Earlier events are listed under the bell, but neither popped up nor counted as new.
        read = keep(read, alerts.map((a) => a.key));
        localStorage.setItem(READ_KEY, JSON.stringify([...read]));
        firstRun = false;
      } else if (fresh.length && canNotify()) {
        fresh.slice(0, MAX_POPUPS).forEach(popUp);
        if (fresh.length > MAX_POPUPS) {
          try {
            new Notification(`${fresh.length - MAX_POPUPS} more trip notifications`, { body: 'Open the bell on the dashboard to see them all.', tag: 'menengai-more' });
          } catch {
            // Shown under the bell anyway.
          }
        }
      }
      seen = keep(seen, alerts.map((a) => a.key));
      localStorage.setItem(SEEN_KEY, JSON.stringify([...seen]));
      el.foot.textContent = `Last 24 hours · checked ${fmtTime(now)} EAT · every 2 minutes`;
    } catch (err) {
      el.foot.textContent = `Could not check notifications: ${err.message}`;
    }
    renderBadge();
    if (!el.panel.classList.contains('hidden')) renderList();
  }

  function markRead() {
    read = keep(read, alerts.map((a) => a.key));
    localStorage.setItem(READ_KEY, JSON.stringify([...read]));
    renderBadge();
    renderList();
  }

  function open() {
    el.panel.classList.remove('hidden');
    el.bell.setAttribute('aria-expanded', 'true');
    updateEnable();
    renderList();
  }

  function close() {
    if (el.panel.classList.contains('hidden')) return;
    el.panel.classList.add('hidden');
    el.bell.setAttribute('aria-expanded', 'false');
    markRead();
  }

  el.bell.addEventListener('click', (e) => {
    e.stopPropagation();
    if (el.panel.classList.contains('hidden')) open();
    else close();
  });
  el.panel.addEventListener('click', (e) => e.stopPropagation());
  document.addEventListener('click', close);
  document.addEventListener('keydown', (e) => e.key === 'Escape' && close());
  el.read.addEventListener('click', markRead);
  el.enable.addEventListener('click', async () => {
    if ('Notification' in window && Notification.permission === 'default') await Notification.requestPermission();
    updateEnable();
  });

  updateEnable();
  poll();
  setInterval(poll, ALERTS_POLL_MS);
}
