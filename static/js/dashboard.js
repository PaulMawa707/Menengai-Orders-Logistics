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
  const assetId = document.getElementById('asset-item-id');
  if (!assetId?.value) {
    showToast('Please select an asset', false);
    return;
  }

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

function initAssetPicker() {
  const searchInput = document.getElementById('asset-search');
  const hiddenInput = document.getElementById('asset-item-id');
  const listEl = document.getElementById('asset-suggestions');
  const selectedLabel = document.getElementById('asset-selected-label');
  const ordersInput = document.querySelector('input[name="orders"]');
  if (!searchInput || !hiddenInput || !listEl) return;

  let assets = [];
  let activeIndex = -1;
  let debounceTimer = null;

  function setSelected(asset) {
    if (!asset) {
      hiddenInput.value = '';
      searchInput.value = '';
      selectedLabel.textContent = '';
      selectedLabel.classList.add('hidden');
      return;
    }
    hiddenInput.value = String(asset.item_id);
    searchInput.value = asset.name;
    selectedLabel.textContent = `Selected: ${asset.name}`;
    selectedLabel.classList.remove('hidden');
    hideSuggestions();
  }

  function hideSuggestions() {
    listEl.classList.add('hidden');
    searchInput.setAttribute('aria-expanded', 'false');
    activeIndex = -1;
  }

  function renderSuggestions(items) {
    listEl.innerHTML = '';
    if (!items.length) {
      const li = document.createElement('li');
      li.className = 'empty';
      li.textContent = 'No matching assets';
      listEl.appendChild(li);
    } else {
      items.forEach((asset, index) => {
        const li = document.createElement('li');
        li.role = 'option';
        li.dataset.itemId = asset.item_id;
        li.dataset.name = asset.name;
        li.textContent = asset.name;
        li.addEventListener('mousedown', (e) => {
          e.preventDefault();
          setSelected(asset);
        });
        li.addEventListener('mouseenter', () => {
          activeIndex = index;
          highlightActive();
        });
        listEl.appendChild(li);
      });
    }
    listEl.classList.remove('hidden');
    searchInput.setAttribute('aria-expanded', 'true');
    activeIndex = 0;
    highlightActive();
  }

  function highlightActive() {
    [...listEl.querySelectorAll('li:not(.empty)')].forEach((li, idx) => {
      li.classList.toggle('active', idx === activeIndex);
    });
  }

  async function fetchAssets(query) {
    const params = new URLSearchParams({ q: query, limit: '50' });
    const res = await fetch(`/api/assets?${params}`, { credentials: 'same-origin' });
    const data = await res.json();
    if (data.error !== 0) throw new Error(data.message || 'Could not load assets');
    assets = data.assets || [];
    renderSuggestions(assets);
  }

  searchInput.addEventListener('input', () => {
    hiddenInput.value = '';
    selectedLabel.classList.add('hidden');
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      fetchAssets(searchInput.value.trim()).catch((err) => showToast(err.message, false));
    }, 180);
  });

  searchInput.addEventListener('focus', () => {
    fetchAssets(searchInput.value.trim()).catch((err) => showToast(err.message, false));
  });

  searchInput.addEventListener('keydown', (e) => {
    const options = [...listEl.querySelectorAll('li:not(.empty)')];
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (!options.length) return;
      activeIndex = Math.min(activeIndex + 1, options.length - 1);
      highlightActive();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (!options.length) return;
      activeIndex = Math.max(activeIndex - 1, 0);
      highlightActive();
    } else if (e.key === 'Enter') {
      if (!listEl.classList.contains('hidden') && options[activeIndex]) {
        e.preventDefault();
        setSelected({
          item_id: options[activeIndex].dataset.itemId,
          name: options[activeIndex].dataset.name,
        });
      }
    } else if (e.key === 'Escape') {
      hideSuggestions();
    }
  });

  document.addEventListener('click', (e) => {
    if (!e.target.closest('.asset-picker-wrap')) hideSuggestions();
  });

  async function suggestAssetFromOrders() {
    if (!ordersInput?.files?.length) return;
    const fd = new FormData();
    [...ordersInput.files].forEach((file) => fd.append('orders', file));
    try {
      const res = await fetch('/api/match-asset', {
        method: 'POST',
        body: fd,
        credentials: 'same-origin',
      });
      const data = await res.json();
      if (data.error === 0 && data.asset) {
        setSelected(data.asset);
        showToast(`Matched truck to ${data.asset.name}`, true);
      }
    } catch (_) {
      // Optional auto-match; ignore failures.
    }
  }

  ordersInput?.addEventListener('change', suggestAssetFromOrders);
}

document.addEventListener('DOMContentLoaded', () => {
  tickClock();
  setInterval(tickClock, 1000);
  initDatetimeDefaults();
  initAssetPicker();

  document.getElementById('optimized-form')?.addEventListener('submit', (e) => {
    e.preventDefault();
    submitDispatchForm(e.target, '/api/optimized/dispatch');
  });
});
