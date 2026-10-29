const TRIP_META = {
  in_progress: { label: 'On the road', cls: 'in_progress' },
  scheduled: { label: 'Scheduled', cls: 'scheduled' },
  completed: { label: 'Completed', cls: 'completed' },
  ended: { label: 'Ended with missed stops', cls: 'ended' },
};
const VISIT_META = {
  on_time: { label: 'On time', cls: 'completed', hex: '#3fb950' },
  late: { label: 'Late', cls: 'late', hex: '#d29922' },
  early: { label: 'Early', cls: 'in_progress', hex: '#58a6ff' },
  missed: { label: 'Missed', cls: 'ended', hex: '#f85149' },
  pending: { label: 'Not yet', cls: 'pending', hex: '#8b949e' },
  planned: { label: 'Planned', cls: 'pending', hex: '#8b949e' },
};
const PAGE_SIZE = 15;
const MAX_DAYS = 31;
const EAT_S = 3 * 3600;

const esc = (s) =>
  String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const fmtNum = (n, d = 0) =>
  Number(n || 0).toLocaleString('en-KE', { minimumFractionDigits: d, maximumFractionDigits: d });
const fmtTime = (t, tz = EAT_S) => (t ? new Date((t + tz) * 1000).toISOString().slice(11, 16) : '—');
const fmtDay = (day) => {
  const d = new Date(`${day}T00:00:00Z`);
  return d.toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short', timeZone: 'UTC' });
};
const todayEat = () => new Date(Date.now() + EAT_S * 1000).toISOString().slice(0, 10);
const ago = (t, now) => {
  const m = Math.max(0, Math.round((now - t) / 60));
  return m < 1 ? 'just now' : m < 60 ? `${m} min ago` : `${Math.round(m / 6) / 10} h ago`;
};

function badge(meta) {
  return `<span class="badge ${meta.cls}"><span class="dot ${meta.cls}"></span>${esc(meta.label)}</span>`;
}

function progressBar(p) {
  if (!p || !p.total) return '';
  const seg = (n, cls) => (n ? `<span class="${cls}" style="width:${(n / p.total) * 100}%"></span>` : '');
  return `<div class="progress">${seg(p.onTime, 'completed')}${seg(p.early, 'in_progress')}${seg(p.late, 'late')}${seg(p.missed, 'ended')}</div>`;
}

async function apiJson(url, opts) {
  const res = await fetch(url, { credentials: 'same-origin', ...opts });
  let data;
  try {
    data = await res.json();
  } catch {
    throw new Error(`Request failed (${res.status})`);
  }
  if (!res.ok || data.error) throw new Error(data.message || `Request failed (${res.status})`);
  return data;
}

// ---- Tabs ----

function initTabs(trips, reports) {
  const tabs = [...document.querySelectorAll('.tab')];
  const panels = [...document.querySelectorAll('[data-panel]')];
  const names = tabs.map((t) => t.dataset.tab);
  function show(name) {
    tabs.forEach((t) => {
      const on = t.dataset.tab === name;
      t.classList.toggle('active', on);
      t.setAttribute('aria-selected', String(on));
    });
    panels.forEach((p) => {
      const on = p.dataset.panel === name;
      p.hidden = !on;
      p.classList.toggle('active', on);
    });
    if (name === 'trips') trips.activate();
    else trips.deactivate();
    if (name === 'reports') reports.activate();
    history.replaceState(null, '', name === 'trips' ? location.pathname : `#${name}`);
  }
  tabs.forEach((t) => t.addEventListener('click', () => show(t.dataset.tab)));
  const fromHash = location.hash.slice(1);
  show(names.includes(fromHash) ? fromHash : 'trips');
  return { show };
}

// ---- Plan a trip ----

const nextDay = (day) => new Date(Date.parse(`${day}T00:00:00Z`) + 86400000).toISOString().slice(0, 10);

function drawPlanMap(host, plan) {
  if (!window.L || !host) {
    if (host) host.innerHTML = '<div class="empty"><p>The map could not be loaded.</p></div>';
    return null;
  }
  const map = L.map(host, { scrollWheelZoom: false });
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '&copy; OpenStreetMap' }).addTo(map);
  const bounds = [];
  if (plan.geometry?.length) {
    L.polyline(plan.geometry, { color: '#6366f1', weight: 3, opacity: 0.7, dashArray: '6 6' }).addTo(map);
    bounds.push(...plan.geometry);
  }
  if (plan.depot) {
    L.marker([plan.depot.lat, plan.depot.lon], { icon: L.divIcon({ className: 'pin depot', html: 'W', iconSize: [26, 26] }) })
      .bindTooltip(esc(plan.depot.name))
      .addTo(map);
    bounds.push([plan.depot.lat, plan.depot.lon]);
  }
  plan.stops.forEach((s) => {
    L.marker([s.lat, s.lon], { icon: L.divIcon({ className: 'pin', html: `<span style="background:${VISIT_META.planned.hex}">${s.seq}</span>`, iconSize: [24, 24] }) })
      .bindTooltip(`${s.seq}. ${esc(s.name)} · ETA ${fmtTime(s.eta)}`)
      .addTo(map);
    bounds.push([s.lat, s.lon]);
  });
  if (bounds.length) map.fitBounds(bounds, { padding: [24, 24] });
  else map.setView([-0.29, 36.04], 8);
  setTimeout(() => map.invalidateSize(), 50);
  return map;
}

function initPlanTrip({ onDispatched }) {
  const drawer = document.getElementById('plan-drawer');
  const backdrop = document.getElementById('plan-drawer-backdrop');
  const el = {
    day: document.getElementById('plan-day'),
    from: document.getElementById('plan-from'),
    to: document.getElementById('plan-to'),
    warehouse: document.getElementById('plan-warehouse'),
    order: document.getElementById('plan-order'),
    asset: document.getElementById('plan-asset-item-id'),
    preview: document.getElementById('plan-preview'),
    dispatch: document.getElementById('plan-dispatch'),
    alert: document.getElementById('plan-alert'),
    result: document.getElementById('plan-result'),
  };
  const files = initOrdersFileList('plan-');
  const picker = initAssetPicker(files, 'plan-');
  let map = null;
  let busy = false;

  function setAlert(html, type) {
    el.alert.className = `result-alert ${type}`;
    el.alert.innerHTML = html;
  }

  function invalidate() {
    el.dispatch.disabled = true;
    el.result.innerHTML = '';
    el.alert.className = 'result-alert hidden';
    if (map) {
      map.remove();
      map = null;
    }
  }
  files.onChange(invalidate);
  picker.onChange(invalidate);
  [el.day, el.from, el.to, el.warehouse, el.order].forEach((i) => i.addEventListener('change', invalidate));

  function formData() {
    const list = files.getFiles();
    if (!list.length) throw new Error('Choose the orders Excel file(s)');
    if (!el.asset.value) throw new Error('Choose the vehicle');
    if (!el.day.value || !el.from.value || !el.to.value) throw new Error('Choose the day and the window');
    const endDay = el.to.value <= el.from.value ? nextDay(el.day.value) : el.day.value;
    const fd = new FormData();
    list.forEach((f) => fd.append('orders', f, f.name));
    fd.set('asset_item_id', el.asset.value);
    fd.set('warehouse', el.warehouse.value);
    fd.set('stop_order', el.order.value);
    fd.set('route_from', `${el.day.value}T${el.from.value}`);
    fd.set('route_end', `${endDay}T${el.to.value}`);
    return fd;
  }

  function render(d) {
    const p = d.plan;
    const tz = d.tz;
    const stops = p.stops
      .map(
        (s) => `<tr><td class="num">${s.seq}</td><td><div class="strong">${esc(s.name)}</div><div class="sub">${esc(
          [s.location, s.weightKg ? `${fmtNum(s.weightKg)} kg` : ''].filter(Boolean).join(' · '),
        )}</div></td><td class="right nowrap">${fmtNum(s.legKm, 1)} km</td><td class="right nowrap strong">${fmtTime(s.eta, tz)}</td></tr>`,
      )
      .join('');
    const sum = d.summary;
    el.result.innerHTML = `
      <div class="facts">
        <div class="fact"><span>Stops</span><b>${fmtNum(p.stops.length)}</b><small>${fmtNum(sum.tonnage, 2)} t${sum.amount ? ` · ${fmtNum(sum.amount)}` : ''}</small></div>
        <div class="fact"><span>Distance</span><b>${fmtNum(p.km, 1)} km</b><small>${p.routing === 'osrm' ? 'by road' : 'straight line'}</small></div>
        <div class="fact"><span>Driving</span><b>${fmtNum(p.driveMin)} min</b><small>${fmtNum(p.serviceMin)} min at stops</small></div>
        <div class="fact"><span>Back</span><b class="${p.overrun ? 'bad' : ''}">${fmtTime(p.returnAt, tz)}</b><small class="${p.overrun ? 'bad' : ''}">${p.overrun ? 'after the window ends' : 'within the window'}</small></div>
      </div>
      <div class="trip-map plan-map" id="plan-map"></div>
      <div class="card"><div class="card-head"><b>Schedule</b><span class="sub">${esc(d.asset.name)} · from ${esc(p.depot?.name || '')} at ${fmtTime(p.start, tz)}</span></div>
        <div class="table-wrap plan-stops"><table class="stops-table"><thead><tr><th class="num">#</th><th>Stop</th><th class="right">Leg</th><th class="right">ETA</th></tr></thead><tbody>${stops}</tbody></table></div>
      </div>`;
    map = drawPlanMap(document.getElementById('plan-map'), p);
  }

  async function preview() {
    if (busy) return;
    let fd;
    try {
      fd = formData();
    } catch (err) {
      setAlert(`⚠️ ${esc(err.message)}`, 'err');
      return;
    }
    busy = true;
    invalidate();
    el.preview.disabled = true;
    setAlert('Reading the orders and planning the schedule…', 'info');
    try {
      const d = await apiJson('/api/logistics/plan-preview', { method: 'POST', body: fd });
      el.alert.className = 'result-alert hidden';
      render(d);
      el.dispatch.disabled = false;
    } catch (err) {
      setAlert(`❌ ${esc(err.message)}`, 'err');
    } finally {
      busy = false;
      el.preview.disabled = false;
    }
  }

  async function dispatch() {
    if (busy) return;
    let fd;
    try {
      fd = formData();
    } catch (err) {
      setAlert(`⚠️ ${esc(err.message)}`, 'err');
      return;
    }
    busy = true;
    el.dispatch.disabled = true;
    el.preview.disabled = true;
    setAlert('Sending the route to Wialon…', 'info');
    try {
      const d = await apiJson('/api/optimized/dispatch', { method: 'POST', body: fd });
      showToast('Route created successfully', true);
      if (d.trip_id) {
        close();
        files.clear();
        picker.clear();
        onDispatched(d.trip_id, d.trip_day);
      } else {
        setAlert(`✅ ${esc(d.message || 'Route created.')}${d.trip_warning ? `<br>⚠️ ${esc(d.trip_warning)}` : ''}`, 'ok');
      }
    } catch (err) {
      setAlert(`❌ ${esc(err.message)}`, 'err');
      el.dispatch.disabled = false;
    } finally {
      busy = false;
      el.preview.disabled = false;
    }
  }

  function close() {
    drawer.classList.add('hidden');
    backdrop.classList.add('hidden');
    drawer.setAttribute('aria-hidden', 'true');
  }

  el.preview.addEventListener('click', preview);
  el.dispatch.addEventListener('click', dispatch);
  document.getElementById('plan-close').addEventListener('click', close);
  backdrop.addEventListener('click', close);
  document.addEventListener('keydown', (e) => e.key === 'Escape' && !drawer.classList.contains('hidden') && close());

  return {
    open(day) {
      if (!el.day.value) el.day.value = day || todayEat();
      drawer.classList.remove('hidden');
      backdrop.classList.remove('hidden');
      drawer.setAttribute('aria-hidden', 'false');
      if (map) setTimeout(() => map.invalidateSize(), 50);
    },
  };
}

// ---- Trips list ----

function initTrips(drawer) {
  const el = {
    from: document.getElementById('trips-from'),
    to: document.getElementById('trips-to'),
    run: document.getElementById('trips-run'),
    rangeError: document.getElementById('trips-range-error'),
    search: document.getElementById('trips-search'),
    exportBtn: document.getElementById('trips-export'),
    stats: [...document.querySelectorAll('#trips-stats .stat')],
    body: document.getElementById('trips-body'),
    pager: document.getElementById('trips-pager'),
    meta: document.getElementById('trips-meta'),
    storage: document.getElementById('trips-storage-alert'),
  };
  // Today and tomorrow, so trips planned for tomorrow show as scheduled.
  const today = todayEat();
  const tomorrow = nextDay(today);
  const st = { from: today, to: tomorrow, q: '', state: null, page: 1, sort: null, data: null, timer: null, active: false, seq: 0 };
  el.from.value = today;
  el.to.value = tomorrow;

  function rangeInvalid() {
    const f = el.from.value;
    const t = el.to.value;
    if (!f || !t || t < f) return 'The end day is before the start day';
    if ((Date.parse(t) - Date.parse(f)) / 86400000 + 1 > MAX_DAYS) return `Choose at most ${MAX_DAYS} days`;
    return '';
  }

  function updateExport() {
    el.exportBtn.href = `/api/logistics/report?${new URLSearchParams({ from: st.from, to: st.to })}`;
  }

  async function load(quiet) {
    const seq = ++st.seq;
    const params = new URLSearchParams({ from: st.from, to: st.to, page: String(st.page), pageSize: String(PAGE_SIZE) });
    if (st.q) params.set('q', st.q);
    if (st.state) params.set('state', st.state);
    if (st.sort) params.set('sort', st.sort);
    if (!quiet || !st.data) el.body.innerHTML = '<div class="skeleton-rows"><div></div><div></div><div></div><div></div></div>';
    el.run.disabled = true;
    const t0 = performance.now();
    try {
      const data = await apiJson(`/api/logistics/trips?${params}`);
      if (seq !== st.seq) return;
      st.data = data;
      el.storage.classList.toggle('hidden', data.storage !== false);
      render(data);
      el.meta.textContent = `Updated ${fmtTime(data.now, data.tz)} EAT · ${Math.round(performance.now() - t0)} ms · refreshes every minute`;
    } catch (err) {
      if (seq !== st.seq) return;
      el.body.innerHTML = `<div class="empty"><b>Could not load trips</b><p>${esc(err.message)}</p><button type="button" class="btn-ghost btn-sm" data-retry>Try again</button></div>`;
      el.body.querySelector('[data-retry]')?.addEventListener('click', () => load());
    } finally {
      if (seq === st.seq) el.run.disabled = false;
    }
  }

  function render(data) {
    el.stats.forEach((b) => {
      const k = b.dataset.state;
      b.querySelector('.stat-value').textContent = fmtNum(data.counts?.[k] ?? 0);
      b.classList.toggle('active', st.state === k);
      b.classList.toggle('dim', st.state !== null && st.state !== k);
    });
    const oneDay = st.from === st.to;
    const period = oneDay ? fmtDay(st.from) : `${fmtDay(st.from)} to ${fmtDay(st.to)}`;
    if (!data.rows.length) {
      const filtered = st.q || st.state;
      el.body.innerHTML = `<div class="empty"><b>${filtered ? 'No trips match' : `No trips for ${esc(period)}`}</b><p>${
        filtered ? 'Try clearing the filters.' : 'Use Plan a trip to upload orders; each dispatched route appears here on its day.'
      }</p></div>`;
      el.pager.innerHTML = '';
      return;
    }
    const offset = (data.page - 1) * data.pageSize;
    const th = (col, label, cls = '') => {
      const [c, dir] = (st.sort || '').split(':');
      const arrow = c === col ? (dir === 'desc' ? ' ▾' : ' ▴') : '';
      return `<th class="sortable ${cls}" data-sort="${col}">${label}${arrow}</th>`;
    };
    const rows = data.rows
      .map((r, i) => {
        const p = r.progress;
        const visited = p ? p.onTime + p.early + p.late : 0;
        const stops = p
          ? `${progressBar(p)}<div class="sub">${visited} of ${p.total} visited${p.late ? ` · <span class="warn">${p.late} late</span>` : ''}${
              p.missed ? ` · <span class="bad">${p.missed} missed</span>` : ''
            }</div>`
          : `<span class="sub">${fmtNum(r.stops)} stops planned</span>`;
        const source = r.source === 'upload' ? 'Uploaded' : 'Wialon Logistics';
        return `<tr data-id="${esc(r.id)}" tabindex="0">
          <td class="num">${offset + i + 1}</td>
          <td class="route-cell"><div class="strong">${esc(r.routeName)}</div><div class="sub">${esc(source)}${r.warehouse ? ` · ${esc(r.warehouse)}` : ''}</div></td>
          <td class="hide-md">${esc(r.unitName)}</td>
          <td class="nowrap">${oneDay ? '' : `<div class="strong">${esc(fmtDay(r.day))}</div>`}${fmtTime(r.windowFrom, data.tz)}–${fmtTime(r.windowTo, data.tz)}<div class="sub">back ${fmtTime(r.returnAt, data.tz)}</div></td>
          <td class="stops-cell">${stops}</td>
          <td class="right nowrap">${fmtNum(r.km, 1)} km</td>
          <td>${badge(TRIP_META[r.state])}</td>
        </tr>`;
      })
      .join('');
    el.body.innerHTML = `<div class="table-wrap"><table class="trips-table">
      <thead><tr><th class="num">#</th>${th('route', 'Route')}${th('vehicle', 'Vehicle', 'hide-md')}${th('window', oneDay ? 'Window' : 'Day and window')}${th('stops', 'Stops')}${th('km', 'Distance', 'right')}${th('state', 'State')}</tr></thead>
      <tbody>${rows}</tbody></table></div>`;
    el.body.querySelectorAll('tbody tr').forEach((tr) => {
      tr.addEventListener('click', () => drawer.open(tr.dataset.id));
      tr.addEventListener('keydown', (e) => e.key === 'Enter' && drawer.open(tr.dataset.id));
    });
    el.body.querySelectorAll('th[data-sort]').forEach((h) =>
      h.addEventListener('click', () => {
        const [c, dir] = (st.sort || '').split(':');
        st.sort = c !== h.dataset.sort ? `${h.dataset.sort}:asc` : dir === 'asc' ? `${c}:desc` : null;
        st.page = 1;
        load();
      }),
    );
    const pages = Math.max(1, Math.ceil(data.total / data.pageSize));
    el.pager.innerHTML =
      pages > 1
        ? `<span>${offset + 1}–${Math.min(offset + data.pageSize, data.total)} of ${fmtNum(data.total)}</span>
           <button type="button" class="btn-ghost btn-sm" data-page="${data.page - 1}" ${data.page <= 1 ? 'disabled' : ''}>‹ Prev</button>
           <button type="button" class="btn-ghost btn-sm" data-page="${data.page + 1}" ${data.page >= pages ? 'disabled' : ''}>Next ›</button>`
        : `<span>${fmtNum(data.total)} trip${data.total === 1 ? '' : 's'}</span>`;
    el.pager.querySelectorAll('[data-page]').forEach((b) =>
      b.addEventListener('click', () => {
        st.page = Number(b.dataset.page);
        load();
      }),
    );
  }

  el.run.addEventListener('click', () => {
    const invalid = rangeInvalid();
    el.rangeError.textContent = invalid;
    if (invalid) return;
    st.from = el.from.value;
    st.to = el.to.value;
    st.page = 1;
    updateExport();
    load();
  });
  [el.from, el.to].forEach((i) => i.addEventListener('change', () => (el.rangeError.textContent = rangeInvalid())));
  let debounce;
  el.search.addEventListener('input', () => {
    clearTimeout(debounce);
    debounce = setTimeout(() => {
      st.q = el.search.value.trim();
      st.page = 1;
      load();
    }, 250);
  });
  el.stats.forEach((b) =>
    b.addEventListener('click', () => {
      st.state = st.state === b.dataset.state ? null : b.dataset.state;
      st.page = 1;
      load();
    }),
  );
  updateExport();

  return {
    activate() {
      if (st.active) return;
      st.active = true;
      load(true);
      st.timer = setInterval(() => !document.hidden && load(true), 60000);
    },
    deactivate() {
      st.active = false;
      clearInterval(st.timer);
    },
    reload: () => load(true),
    day: () => st.from,
    showDay(day) {
      st.from = day;
      st.to = day;
      el.from.value = day;
      el.to.value = day;
      st.page = 1;
      updateExport();
      if (st.active) load();
    },
  };
}

// ---- Trip drawer ----

function initDrawer(onChanged) {
  const drawer = document.getElementById('trip-drawer');
  const backdrop = document.getElementById('trip-drawer-backdrop');
  const title = document.getElementById('trip-title');
  const subtitle = document.getElementById('trip-subtitle');
  const body = document.getElementById('trip-body');
  const st = { id: null, timer: null, map: null, layers: null, markers: new Map(), selected: null, data: null, fitted: false };

  function close() {
    st.id = null;
    clearInterval(st.timer);
    drawer.classList.add('hidden');
    backdrop.classList.add('hidden');
    drawer.setAttribute('aria-hidden', 'true');
    if (st.map) {
      st.map.remove();
      st.map = null;
    }
  }

  async function load() {
    const id = st.id;
    try {
      const data = await apiJson(`/api/logistics/trips/${encodeURIComponent(id)}`);
      if (id !== st.id) return;
      st.data = data;
      render(data);
    } catch (err) {
      if (id !== st.id) return;
      body.innerHTML = `<div class="empty"><b>Could not load the trip</b><p>${esc(err.message)}</p></div>`;
    }
  }

  function fact(label, value, hint, warn) {
    return `<div class="fact"><span>${esc(label)}</span><b class="${warn ? 'bad' : ''}">${esc(value)}</b>${hint ? `<small class="${warn ? 'bad' : ''}">${esc(hint)}</small>` : ''}</div>`;
  }

  function render(d) {
    const tz = d.tz;
    const t = d.tracking;
    const p = d.progress;
    const plan = d.plan;
    title.textContent = d.routeName;
    subtitle.textContent = `${d.unitName}${d.warehouse ? ` · ${d.warehouse}` : ''} · ${d.source === 'upload' ? 'Uploaded' : 'Wialon Logistics'}`;
    const visits = new Map((t?.visits || []).map((v) => [v.seq, v]));
    const legend = p
      ? `<div class="card progress-card">${progressBar(p)}<div class="legend">${[
          ['onTime', 'on_time'],
          ['late', 'late'],
          ['early', 'early'],
          ['missed', 'missed'],
          ['pending', 'pending'],
        ]
          .map(([k, s]) => `<span><span class="dot ${VISIT_META[s].cls}"></span>${VISIT_META[s].label} <b>${p[k]}</b></span>`)
          .join('')}${p.outOfOrder ? `<span class="warn">${p.outOfOrder} out of order</span>` : ''}</div></div>`
      : '';
    const routing =
      plan.routing === 'osrm' || plan.routing === 'wialon_osrm'
        ? 'Road routing'
        : plan.routing === 'wialon_road'
          ? 'Wialon road plan'
          : plan.routing === 'wialon'
            ? 'Wialon plan'
            : 'Straight line';
    const stopRows = plan.stops
      .map((s) => {
        const v = visits.get(s.seq);
        const late = v?.arrivedAt ? Math.round((v.arrivedAt - s.eta) / 60) : null;
        let arrived = '<span class="sub">—</span>';
        if (v?.arrivedAt) {
          arrived = `${fmtTime(v.arrivedAt, tz)}${late !== null && Math.abs(late) >= 1 ? `<div class="sub ${late > 0 ? 'warn' : 'info'}">${late > 0 ? '+' : ''}${late} min</div>` : ''}`;
        } else if (v?.closestM !== null && v?.closestM !== undefined) {
          arrived = `<span class="sub" title="Closest the vehicle came">${v.closestM >= 1000 ? `${fmtNum(v.closestM / 1000, 1)} km` : `${v.closestM} m`} away</span>`;
        }
        const status = v
          ? `${badge(VISIT_META[v.status])}${v.outOfOrder ? ' <span class="sub warn">out of order</span>' : ''}${v.leftAt ? ` <span class="sub">left ${fmtTime(v.leftAt, tz)}</span>` : ''}`
          : '<span class="sub">Planned</span>';
        const detail = [s.location || `${s.lat.toFixed(5)}, ${s.lon.toFixed(5)}`, s.legKm > 0 ? `${fmtNum(s.legKm, 1)} km` : '', s.weightKg ? `${fmtNum(s.weightKg)} kg` : '']
          .filter(Boolean)
          .join(' · ');
        return `<tr data-seq="${s.seq}"><td class="num">${s.seq}</td><td><div class="strong">${esc(s.name)}</div><div class="sub">${esc(detail)}</div></td>
          <td class="right nowrap">${fmtTime(s.eta, tz)}</td><td class="right nowrap">${arrived}</td><td>${status}</td></tr>`;
      })
      .join('');
    body.innerHTML = `
      <div class="drawer-actions">
        ${badge(TRIP_META[d.state])}
        <span class="sub">${esc(fmtDay(d.day))} · window ${fmtTime(d.windowFrom, tz)}–${fmtTime(d.windowTo, tz)}</span>
        <span class="spacer"></span>
        <a class="btn-ghost btn-sm" href="/api/logistics/trips/${encodeURIComponent(d.id)}/export">⬇ Export CSV</a>
        <button type="button" class="btn-ghost btn-sm danger" id="trip-remove">Remove</button>
      </div>
      <div class="facts">
        ${fact('Stops', fmtNum(d.stops), p ? `${p.onTime + p.late + p.early} visited` : 'Not started yet')}
        ${fact('Planned distance', `${fmtNum(plan.km, 1)} km`, `${routing} · ${fmtNum(plan.driveMin)} min driving`)}
        ${fact('Planned return', fmtTime(plan.returnAt, tz), plan.overrun ? 'After the window ends' : `${fmtNum(plan.serviceMin)} min at stops`, plan.overrun)}
        ${fact('Driven', t ? `${fmtNum(t.km, 1)} km` : '—', t?.startedAt ? `Left ${fmtTime(t.startedAt, tz)}${t.returnedAt ? ` · back ${fmtTime(t.returnedAt, tz)}` : ''}` : 'No movement yet')}
      </div>
      ${legend}
      <div class="trip-map" id="trip-map"></div>
      <div class="map-legend">
        <span><span class="line planned"></span> Planned route</span>
        ${t ? '<span><span class="line driven"></span> Driven (from Wialon positions)</span>' : ''}
        <span>W is the warehouse</span>
      </div>
      <div class="card">
        <div class="card-head"><b>Stops</b><span class="sub">Visited within ${fmtNum(plan.rules.stopRadiusM)} m · late after ETA + ${plan.rules.lateMin} min</span></div>
        <div class="table-wrap"><table class="stops-table"><thead><tr><th class="num">#</th><th>Stop</th><th class="right">ETA</th><th class="right">Arrived</th><th>Status</th></tr></thead><tbody>${stopRows}</tbody></table></div>
      </div>
      <p class="data-meta">${d.source === 'upload' ? `Uploaded by ${esc(d.createdBy || '—')}${d.files ? ` from ${esc(d.files)}` : ''}. ` : 'Read from Wialon Logistics. '}Progress is measured from the vehicle's positions in Wialon. Updated ${fmtTime(d.now, tz)} EAT.</p>`;
    body.querySelector('#trip-remove').addEventListener('click', remove);
    body.querySelectorAll('.stops-table tbody tr').forEach((tr) => tr.addEventListener('click', () => select(Number(tr.dataset.seq))));
    drawMap(d, visits);
  }

  function drawMap(d, visits) {
    const host = document.getElementById('trip-map');
    if (!window.L || !host) {
      if (host) host.innerHTML = '<div class="empty"><p>The map could not be loaded.</p></div>';
      return;
    }
    const center = st.map?.getCenter();
    const zoom = st.map?.getZoom();
    if (st.map) st.map.remove();
    const map = L.map(host, { scrollWheelZoom: false });
    st.map = map;
    st.markers = new Map();
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '&copy; OpenStreetMap' }).addTo(map);
    const bounds = [];
    if (d.plan.geometry?.length) {
      L.polyline(d.plan.geometry, { color: '#6366f1', weight: 3, opacity: 0.6, dashArray: '6 6' }).addTo(map);
      bounds.push(...d.plan.geometry);
    }
    if (d.tracking?.path?.length) {
      L.polyline(d.tracking.path, { color: '#0ea5e9', weight: 4, opacity: 0.85 }).addTo(map);
      bounds.push(...d.tracking.path);
    }
    if (d.plan.depot) {
      const dep = d.plan.depot;
      L.marker([dep.lat, dep.lon], { icon: L.divIcon({ className: 'pin depot', html: 'W', iconSize: [26, 26] }) })
        .bindTooltip(esc(dep.name))
        .addTo(map);
      bounds.push([dep.lat, dep.lon]);
    }
    d.plan.stops.forEach((s) => {
      const status = visits.get(s.seq)?.status || 'planned';
      const m = L.marker([s.lat, s.lon], {
        icon: L.divIcon({ className: 'pin', html: `<span style="background:${VISIT_META[status].hex}">${s.seq}</span>`, iconSize: [24, 24] }),
      })
        .bindTooltip(`${s.seq}. ${esc(s.name)} · ${VISIT_META[status].label}`)
        .on('click', () => select(s.seq))
        .addTo(map);
      st.markers.set(s.seq, m);
      bounds.push([s.lat, s.lon]);
    });
    const pos = d.tracking?.lastPos;
    if (pos && d.state === 'in_progress') {
      L.marker([pos.lat, pos.lon], { icon: L.divIcon({ className: 'pin vehicle', html: '🚚', iconSize: [28, 28] }) })
        .bindTooltip(`${esc(d.unitName)} · ${ago(pos.t, d.now)}`)
        .addTo(map);
      bounds.push([pos.lat, pos.lon]);
    }
    if (st.fitted && center) map.setView(center, zoom);
    else if (bounds.length) map.fitBounds(bounds, { padding: [24, 24] });
    else map.setView([-0.29, 36.04], 8);
    st.fitted = true;
    setTimeout(() => map.invalidateSize(), 50);
    if (st.selected) highlight(st.selected);
  }

  function highlight(seq) {
    body.querySelectorAll('.stops-table tbody tr').forEach((tr) => tr.classList.toggle('selected', Number(tr.dataset.seq) === seq));
  }

  function select(seq) {
    st.selected = seq;
    highlight(seq);
    const m = st.markers.get(seq);
    if (m && st.map) {
      st.map.setView(m.getLatLng(), Math.max(st.map.getZoom(), 14));
      m.openTooltip();
    }
    body.querySelector(`.stops-table tr[data-seq="${seq}"]`)?.scrollIntoView({ block: 'nearest' });
  }

  async function remove() {
    const question =
      st.data?.source === 'wialon'
        ? 'Remove this trip from the dashboard? The route stays in Wialon Logistics and is not read again.'
        : 'Remove this trip from the dashboard? The route already sent to Wialon Logistics is not changed.';
    if (!st.id || !confirm(question)) return;
    const btn = body.querySelector('#trip-remove');
    btn.disabled = true;
    try {
      await apiJson(`/api/logistics/trips/${encodeURIComponent(st.id)}`, { method: 'DELETE' });
      close();
      onChanged();
    } catch (err) {
      alert(err.message);
      btn.disabled = false;
    }
  }

  document.getElementById('trip-close').addEventListener('click', close);
  backdrop.addEventListener('click', close);
  document.addEventListener('keydown', (e) => e.key === 'Escape' && st.id && close());

  return {
    open(id) {
      st.id = id;
      st.selected = null;
      st.fitted = false;
      if (st.map) {
        st.map.remove();
        st.map = null;
      }
      title.textContent = 'Trip';
      subtitle.textContent = '';
      body.innerHTML = '<div class="skeleton-rows"><div></div><div></div><div></div><div></div><div></div></div>';
      drawer.classList.remove('hidden');
      backdrop.classList.remove('hidden');
      drawer.setAttribute('aria-hidden', 'false');
      clearInterval(st.timer);
      load();
      st.timer = setInterval(() => !document.hidden && st.id && load(), 60000);
    },
  };
}

document.addEventListener('DOMContentLoaded', () => {
  let trips;
  const drawer = initDrawer(() => trips.reload());
  trips = initTrips(drawer);
  const reports = initReports({ openTrip: (id) => drawer.open(id) });
  const tabs = initTabs(trips, reports);
  const openTrip = (id, day) => {
    if (day) trips.showDay(day);
    tabs.show('trips');
    if (id) drawer.open(id);
  };
  const plan = initPlanTrip({ onDispatched: openTrip });
  document.getElementById('plan-open').addEventListener('click', () => plan.open(trips.day()));
  window.menengaiTrips = { openTrip };
});
