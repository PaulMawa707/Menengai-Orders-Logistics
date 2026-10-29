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

function initOrdersFileList(prefix = '') {
  const input = document.getElementById(`${prefix}orders-input`);
  const chooseBtn = document.getElementById(`${prefix}orders-choose-btn`);
  const listEl = document.getElementById(`${prefix}orders-file-list`);
  const hintEl = document.getElementById(`${prefix}orders-hint`);
  if (!input || !chooseBtn || !listEl) return;

  let managedFiles = [];
  const listeners = [];

  function notifyFilesChanged() {
    if (hintEl) {
      hintEl.textContent = managedFiles.length
        ? `${managedFiles.length} file(s) selected`
        : 'No files chosen';
    }
    listEl.classList.toggle('hidden', managedFiles.length === 0);
    listeners.forEach((callback) => callback());
  }

  function renderList() {
    listEl.innerHTML = '';
    managedFiles.forEach((file, index) => {
      const li = document.createElement('li');
      li.className = 'file-chip';
      const name = document.createElement('span');
      name.className = 'file-chip-name';
      name.title = file.name;
      name.textContent = file.name;
      const removeButton = document.createElement('button');
      removeButton.type = 'button';
      removeButton.className = 'file-chip-remove';
      removeButton.setAttribute('aria-label', `Remove ${file.name}`);
      removeButton.textContent = '×';
      removeButton.addEventListener('click', () => {
        managedFiles.splice(index, 1);
        renderList();
        notifyFilesChanged();
      });
      li.append(name, removeButton);
      listEl.appendChild(li);
    });
  }

  function addFiles(fileList) {
    const existing = new Set(managedFiles.map((f) => `${f.name}:${f.size}:${f.lastModified}`));
    [...fileList].forEach((file) => {
      const key = `${file.name}:${file.size}:${file.lastModified}`;
      if (!existing.has(key)) {
        managedFiles.push(file);
        existing.add(key);
      }
    });
    renderList();
    notifyFilesChanged();
  }

  chooseBtn.addEventListener('click', () => input.click());
  input.addEventListener('change', () => {
    const selectedFiles = [...(input.files || [])];
    input.value = '';
    if (selectedFiles.length) addFiles(selectedFiles);
  });

  return {
    getFiles: () => managedFiles,
    onChange(callback) {
      listeners.push(callback);
    },
    clear() {
      managedFiles = [];
      renderList();
      notifyFilesChanged();
    },
  };
}

function initAssetPicker(ordersFileList, prefix = '') {
  const searchInput = document.getElementById(`${prefix}asset-search`);
  const hiddenInput = document.getElementById(`${prefix}asset-item-id`);
  const listEl = document.getElementById(`${prefix}asset-suggestions`);
  const selectedLabel = document.getElementById(`${prefix}asset-selected-label`);
  const ordersSummary = document.getElementById(`${prefix}orders-summary`);
  if (!searchInput || !hiddenInput || !listEl) return;
  let onSelected = null;

  let assets = [];
  let activeIndex = -1;
  let debounceTimer = null;

  function setSelected(asset) {
    if (!asset) {
      hiddenInput.value = '';
      searchInput.value = '';
      selectedLabel.textContent = '';
      selectedLabel.classList.add('hidden');
      onSelected?.();
      return;
    }
    hiddenInput.value = String(asset.item_id);
    searchInput.value = asset.name;
    selectedLabel.textContent = `Selected: ${asset.name}`;
    selectedLabel.classList.remove('hidden');
    hideSuggestions();
    onSelected?.();
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
    const files = ordersFileList?.getFiles?.() || [];
    if (!files?.length) {
      setSelected(null);
      if (ordersSummary) {
        ordersSummary.textContent = '';
        ordersSummary.classList.add('hidden');
      }
      return;
    }
    const fd = new FormData();
    [...files].forEach((file) => fd.append('orders', file));
    try {
      const res = await fetch('/api/match-asset', {
        method: 'POST',
        body: fd,
        credentials: 'same-origin',
      });
      const data = await res.json();
      if (data.error !== 0) {
        throw new Error(data.message || 'Could not process the selected files');
      }
      if (ordersSummary && data.summary) {
        ordersSummary.textContent =
          `${data.summary.files} file(s) combined · ` +
          `${data.summary.delivery_points} delivery point(s) · ` +
          `${data.summary.tonnage} tonnes`;
        ordersSummary.classList.remove('hidden');
      }
      if (data.error === 0 && data.asset) {
        setSelected(data.asset);
        showToast(`Matched truck to ${data.asset.name}`, true);
      }
    } catch (err) {
      if (ordersSummary) {
        ordersSummary.textContent = '';
        ordersSummary.classList.add('hidden');
      }
      showToast(err.message, false);
    }
  }

  ordersFileList?.onChange(suggestAssetFromOrders);

  return {
    onChange(callback) {
      onSelected = callback;
    },
    clear: () => setSelected(null),
  };
}

document.addEventListener('DOMContentLoaded', () => {
  tickClock();
  setInterval(tickClock, 1000);
});
