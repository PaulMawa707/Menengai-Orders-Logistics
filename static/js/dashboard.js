function pad(n) {
  return String(n).padStart(2, '0');
}

function tickClock() {
  const now = new Date();
  const utc = now.getTime() + now.getTimezoneOffset() * 60000;
  const eat = new Date(utc + 3 * 3600000);
  const el = document.getElementById('clock-time');
  if (!el) return;
  el.textContent =
    `${eat.getFullYear()}-${pad(eat.getMonth() + 1)}-${pad(eat.getDate())} ` +
    `${pad(eat.getHours())}:${pad(eat.getMinutes())}:${pad(eat.getSeconds())}`;
}

function toDatetimeLocal(d) {
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function initDatetimeDefaults() {
  const now = new Date();
  const from = new Date(now);
  from.setHours(6, 0, 0, 0);
  const end = new Date(now);
  end.setHours(18, 0, 0, 0);
  document.querySelectorAll('.route-from').forEach((el) => {
    if (!el.value) el.value = toDatetimeLocal(from);
  });
  document.querySelectorAll('.route-end').forEach((el) => {
    if (!el.value) el.value = toDatetimeLocal(end);
  });
}

function showToast(message, ok) {
  let toast = document.getElementById('app-toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'app-toast';
    toast.className = 'toast';
    document.body.appendChild(toast);
  }
  toast.textContent = message;
  toast.className = 'toast show ' + (ok ? 'ok' : 'err');
  setTimeout(() => toast.classList.remove('show'), 5000);
}

function setAlert(el, message, type) {
  if (!el) return;
  el.className = 'result-alert ' + type;
  el.innerHTML = message;
  el.classList.remove('hidden');
}

async function submitDispatchForm(form, endpoint) {
  const alertEl = form.querySelector('.result-alert');
  const btn = form.querySelector('[type="submit"]');
  const fd = new FormData(form);
  btn.disabled = true;
  if (alertEl) {
    alertEl.className = 'result-alert info';
    alertEl.textContent = 'Processing…';
    alertEl.classList.remove('hidden');
  }
  try {
    const res = await fetch(endpoint, { method: 'POST', body: fd, credentials: 'same-origin' });
    const data = await res.json();
    if (data.error === 0) {
      let html = `✅ ${data.message || 'Route created.'}`;
      if (data.summary) {
        html += `<br>Delivery points: ${data.summary.delivery_points} · Tonnage: ${data.summary.tonnage} · Amount: ${data.summary.amount} · Warehouse: ${data.summary.warehouse}`;
      }
      if (data.planning_url) {
        html += `<br><a href="${data.planning_url}" target="_blank" rel="noopener">Open Wialon Logistics</a>`;
      }
      setAlert(alertEl, html, 'ok');
      showToast('Route created successfully', true);
    } else {
      setAlert(alertEl, `❌ ${data.message || 'Unknown error'}`, 'err');
      showToast(data.message || 'Dispatch failed', false);
    }
  } catch (err) {
    setAlert(alertEl, `❌ ${err.message}`, 'err');
    showToast(err.message, false);
  } finally {
    btn.disabled = false;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  tickClock();
  setInterval(tickClock, 1000);
  initDatetimeDefaults();

  document.getElementById('optimized-form')?.addEventListener('submit', (e) => {
    e.preventDefault();
    submitDispatchForm(e.target, '/api/optimized/dispatch');
  });
});
