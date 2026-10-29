// Reports tab: historical trips with every order planned against actual, as in Wialon Logistics.
// Shares helpers (esc, fmtNum, fmtTime, badge, apiJson, TRIP_META, VISIT_META, todayEat) with trips.js.

const REPORT_ORDER_FILTERS = {
  all: { label: 'All orders', match: () => true },
  visited: { label: 'Visited', match: (s) => isVisited(s) },
  not_visited: { label: 'Not visited', match: (s) => !isVisited(s) },
  in_time: { label: 'Visited in time', match: (s) => s.status === 'on_time' || s.status === 'early' },
  late: { label: 'Visited late', match: (s) => s.status === 'late' },
};
const isVisited = (s) => s.status === 'on_time' || s.status === 'early' || s.status === 'late';
const shiftDay = (day, n) => new Date(Date.parse(`${day}T00:00:00Z`) + n * 86400000).toISOString().slice(0, 10);
const fmtMin = (m) => {
  if (m === null || m === undefined) return '—';
  const h = Math.floor(Math.abs(m) / 60);
  const min = Math.abs(m) % 60;
  return h ? `${h} h ${String(min).padStart(2, '0')} min` : `${min} min`;
};
const devText = (m) => (m === null || m === undefined ? '—' : m === 0 ? 'On plan' : `${m > 0 ? '+' : '−'}${fmtMin(Math.abs(m))}`);
const stopLabel = (s) =>
  s.status === 'early' ? 'In time (early)' : s.status === 'on_time' ? 'In time' : s.status === 'pending' ? 'Not visited yet' : s.status === 'missed' ? 'Not visited' : 'Late';
const serviceMin = (s) => (s.arrivedAt && s.leftAt ? Math.round((s.leftAt - s.arrivedAt) / 60) : null);
const plannedServiceMin = (s) => Math.round((s.etd - s.eta) / 60);
const pct = (a, b) => (b ? `${Math.round((a / b) * 1000) / 10}%` : '0%');

function quickPeriods(today) {
  const d = new Date(`${today}T00:00:00Z`);
  const dow = (d.getUTCDay() + 6) % 7;
  const lastMon = shiftDay(today, -dow - 7);
  const lastMonthEnd = shiftDay(`${today.slice(0, 7)}-01`, -1);
  return [
    { label: 'Today', from: today, to: today },
    { label: 'Yesterday', from: shiftDay(today, -1), to: shiftDay(today, -1) },
    { label: 'Last week', from: lastMon, to: shiftDay(lastMon, 6) },
    { label: 'Last month', from: `${lastMonthEnd.slice(0, 7)}-01`, to: lastMonthEnd },
    { label: 'Last 7 days', from: shiftDay(today, -6), to: today },
    { label: 'Last 30 days', from: shiftDay(today, -29), to: today },
  ];
}

function tripFacts(t, orders = t.stops) {
  let visited = 0;
  let inTime = 0;
  let late = 0;
  let pending = 0;
  for (const s of orders) {
    if (isVisited(s)) visited++;
    if (s.status === 'on_time' || s.status === 'early') inTime++;
    if (s.status === 'late') late++;
    if (s.status === 'pending') pending++;
  }
  const a = t.actual;
  const lastVisit = Math.max(0, ...t.stops.map((s) => s.leftAt ?? s.arrivedAt ?? 0));
  const end = a?.returnedAt ?? (lastVisit || null);
  return {
    orders: orders.length,
    visited,
    inTime,
    late,
    notVisited: orders.length - visited,
    pending,
    plannedMin: Math.round((t.planned.returnAt - t.planned.start) / 60),
    actualMin: a?.startedAt && end && end > a.startedAt ? Math.round((end - a.startedAt) / 60) : null,
  };
}

function reportStats(rows) {
  const s = { routes: rows.length, orders: 0, visited: 0, inTime: 0, late: 0, notVisited: 0, pending: 0, plannedKm: 0, actualKm: 0, weightKg: 0, amount: 0 };
  let dev = 0;
  let devN = 0;
  let actualMin = 0;
  let actualOrders = 0;
  for (const { trip, orders } of rows) {
    const f = tripFacts(trip, orders);
    s.orders += f.orders;
    s.visited += f.visited;
    s.inTime += f.inTime;
    s.late += f.late;
    s.notVisited += f.notVisited;
    s.pending += f.pending;
    s.plannedKm += trip.planned.km;
    s.actualKm += trip.actual?.km ?? 0;
    const all = tripFacts(trip);
    if (all.actualMin !== null && trip.stops.length) {
      actualMin += all.actualMin;
      actualOrders += trip.stops.length;
    }
    for (const o of orders) {
      s.weightKg += o.weightKg || 0;
      s.amount += o.amount || 0;
      if (o.deviationMin !== null) {
        dev += o.deviationMin;
        devN++;
      }
    }
  }
  s.avgDeviationMin = devN ? Math.round(dev / devN) : null;
  s.avgOrderMin = actualOrders ? Math.round(actualMin / actualOrders) : null;
  return s;
}

function statTiles(s) {
  return [
    { label: 'Routes', value: fmtNum(s.routes) },
    { label: 'Number of orders', value: fmtNum(s.orders) },
    { label: 'Visited', value: fmtNum(s.visited), tone: 'info', note: pct(s.visited, s.orders) },
    { label: 'Visited in time', value: fmtNum(s.inTime), tone: 'good', note: pct(s.inTime, s.orders) },
    { label: 'Visited late', value: fmtNum(s.late), tone: 'warn', note: pct(s.late, s.orders) },
    { label: 'Not visited', value: fmtNum(s.notVisited), tone: 'bad', note: s.pending ? `${pct(s.notVisited, s.orders)} · ${fmtNum(s.pending)} still on the road` : pct(s.notVisited, s.orders) },
    { label: 'Avg time deviation', value: devText(s.avgDeviationMin), note: 'Actual minus planned arrival' },
    { label: 'Avg order time (actual)', value: fmtMin(s.avgOrderMin), note: 'Trip time / orders' },
    { label: 'Estimated mileage', value: `${fmtNum(s.plannedKm, 1)} km` },
    { label: 'Actual mileage', value: `${fmtNum(s.actualKm, 1)} km` },
    { label: 'Total weight', value: s.weightKg ? `${fmtNum(s.weightKg)} kg` : '—' },
    { label: 'Total cost', value: s.amount ? fmtNum(s.amount, 2) : '—' },
  ];
}

const REPORT_SORTS = {
  route: (r) => r.trip.routeName.toLowerCase(),
  vehicle: (r) => r.trip.unitName.toLowerCase(),
  day: (r) => r.trip.day,
  orders: (r) => tripFacts(r.trip, r.orders).orders,
  visited: (r) => tripFacts(r.trip, r.orders).inTime,
  start: (r) => r.trip.planned.start,
  return: (r) => r.trip.planned.returnAt,
  duration: (r) => tripFacts(r.trip).plannedMin,
  km: (r) => r.trip.planned.km,
  state: (r) => r.trip.state,
};

function initReports({ openTrip }) {
  const el = {
    kind: document.getElementById('rep-kind'),
    from: document.getElementById('rep-from'),
    to: document.getElementById('rep-to'),
    unit: document.getElementById('rep-unit'),
    orders: document.getElementById('rep-orders'),
    run: document.getElementById('rep-run'),
    periods: document.getElementById('rep-periods'),
    note: document.getElementById('rep-note'),
    results: document.getElementById('rep-results'),
  };
  const NOTE = el.note.textContent;
  const today = todayEat();
  el.from.value = shiftDay(today, -6);
  el.to.value = today;
  const st = { data: null, query: null, q: '', sort: null, open: new Set(), collapsed: new Set(), seq: 0 };
  let vehiclesLoaded = false;

  function renderPeriods() {
    el.periods.innerHTML = quickPeriods(today)
      .map((p) => `<button type="button" class="chip ${el.from.value === p.from && el.to.value === p.to ? 'active' : ''}" data-from="${p.from}" data-to="${p.to}">${p.label}</button>`)
      .join('');
    el.periods.querySelectorAll('.chip').forEach((b) =>
      b.addEventListener('click', () => {
        el.from.value = b.dataset.from;
        el.to.value = b.dataset.to;
        validate();
      }),
    );
  }

  function validate() {
    renderPeriods();
    const f = el.from.value;
    const t = el.to.value;
    const bad = !f || !t || t < f ? 'The end day is before the start day.' : (Date.parse(t) - Date.parse(f)) / 86400000 + 1 > 31 ? 'Choose at most 31 days.' : '';
    el.note.textContent = bad || NOTE;
    el.note.classList.toggle('bad', !!bad);
    el.run.disabled = !!bad;
    return !bad;
  }

  async function loadVehicles() {
    if (vehiclesLoaded) return;
    vehiclesLoaded = true;
    try {
      const d = await apiJson('/api/assets?limit=500');
      const list = [...(d.assets || [])].sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
      el.unit.innerHTML = `<option value="">All vehicles (${fmtNum(list.length)})</option>${list.map((a) => `<option value="${a.item_id}">${esc(a.name)}</option>`).join('')}`;
    } catch {
      vehiclesLoaded = false;
    }
  }

  async function execute() {
    if (!validate()) return;
    const query = { from: el.from.value, to: el.to.value, unit: el.unit.value };
    const seq = ++st.seq;
    st.query = query;
    st.open = new Set();
    st.collapsed = new Set();
    el.run.disabled = true;
    el.run.textContent = 'Running…';
    if (!st.data) {
      el.results.innerHTML = '<div class="card"><p class="loading-note">Loading trips and vehicle positions from Wialon…</p><div class="skeleton-rows"><div></div><div></div><div></div><div></div><div></div></div></div>';
    }
    const t0 = performance.now();
    try {
      const params = new URLSearchParams({ from: query.from, to: query.to });
      if (query.unit) params.set('unit', query.unit);
      const data = await apiJson(`/api/logistics/report-data?${params}`);
      if (seq !== st.seq) return;
      st.data = { ...data, ms: Math.round(performance.now() - t0) };
      render();
    } catch (err) {
      if (seq !== st.seq) return;
      el.results.innerHTML = `<div class="card"><div class="empty"><b>Could not run the report</b><p>${esc(err.message)}</p><button type="button" class="btn-ghost btn-sm" data-retry>Try again</button></div></div>`;
      el.results.querySelector('[data-retry]')?.addEventListener('click', execute);
    } finally {
      if (seq === st.seq) {
        el.run.disabled = false;
        el.run.textContent = '▶ Execute';
      }
    }
  }

  function filteredRows() {
    const filter = REPORT_ORDER_FILTERS[el.orders.value];
    const q = st.q.trim().toLowerCase();
    let rows = st.data.trips
      .filter((t) => !q || t.routeName.toLowerCase().includes(q) || t.unitName.toLowerCase().includes(q))
      .map((trip) => ({ trip, orders: trip.stops.filter(filter.match) }))
      .filter((r) => el.orders.value === 'all' || r.orders.length);
    if (st.sort) {
      const [col, dir] = st.sort.split(':');
      const key = REPORT_SORTS[col];
      rows = [...rows].sort((a, b) => {
        const x = key(a);
        const y = key(b);
        return (x < y ? -1 : x > y ? 1 : 0) * (dir === 'desc' ? -1 : 1);
      });
    }
    return rows;
  }

  function expanded(id) {
    return el.kind.value === 'orders' ? !st.collapsed.has(id) : st.open.has(id);
  }

  function exportHref() {
    const p = new URLSearchParams({ from: st.query.from, to: st.query.to, orders: el.orders.value, kind: el.kind.value });
    if (st.query.unit) p.set('unit', st.query.unit);
    return `/api/logistics/report?${p}`;
  }

  function render() {
    const d = st.data;
    const tz = d.tz;
    const rows = filteredRows();
    const stats = reportStats(rows);
    const period = d.from === d.to ? fmtDay(d.from) : `${fmtDay(d.from)} to ${fmtDay(d.to)}`;
    const unitName = st.query.unit ? el.unit.querySelector(`option[value="${st.query.unit}"]`)?.textContent : null;
    const filterText = [unitName ? `Vehicle: ${unitName}` : 'All vehicles', `Orders: ${REPORT_ORDER_FILTERS[el.orders.value].label}`, st.q.trim() && `Search: “${st.q.trim()}”`]
      .filter(Boolean)
      .join(' · ');
    const multiDay = d.from !== d.to;
    const th = (col, label, cls = '') => {
      const [c, dir] = (st.sort || '').split(':');
      return `<th class="sortable ${cls}" data-sort="${col}">${label}${c === col ? (dir === 'desc' ? ' ▾' : ' ▴') : ''}</th>`;
    };
    const t = (ts) => (ts ? fmtTime(ts, tz) : '—');
    const body = rows
      .map(({ trip, orders }) => {
        const f = tripFacts(trip);
        const open = expanded(trip.id);
        const main = `<tr class="report-row ${open ? 'open' : ''}" data-id="${esc(trip.id)}">
          <td class="chev">${open ? '▾' : '▸'}</td>
          <td class="route-cell"><div class="strong">${esc(trip.routeName)}</div><div class="sub">${trip.source === 'upload' ? 'Uploaded' : 'Wialon Logistics'}${trip.warehouse ? ` · ${esc(trip.warehouse)}` : ''}</div></td>
          <td>${esc(trip.unitName)}</td>
          <td class="nowrap">${esc(fmtDay(trip.day))}</td>
          <td class="right strong">${f.orders}</td>
          <td class="right nowrap"><span class="good">${f.inTime}</span> · <span class="${f.late ? 'warn' : 'sub'}">${f.late}</span> · <span class="${f.notVisited ? 'bad' : 'sub'}">${f.notVisited}</span></td>
          <td class="center nowrap">${t(trip.planned.start)} <span class="sub">/</span> ${t(trip.actual?.startedAt)}</td>
          <td class="center nowrap">${t(trip.planned.returnAt)} <span class="sub">/</span> ${t(trip.actual?.returnedAt)}</td>
          <td class="right nowrap">${fmtMin(f.plannedMin)} <span class="sub">/</span> ${fmtMin(f.actualMin)}</td>
          <td class="right nowrap">${fmtNum(trip.planned.km, 1)} <span class="sub">/</span> ${trip.actual ? fmtNum(trip.actual.km, 1) : '—'}</td>
          <td>${badge(TRIP_META[trip.state])}</td>
          <td><button type="button" class="icon-btn" data-map="${esc(trip.id)}" title="Planned and actual route on the map" aria-label="Planned and actual route on the map">🗺</button></td>
        </tr>`;
        return open ? main + `<tr class="orders-row"><td colspan="12">${ordersTable(orders, tz, trip)}</td></tr>` : main;
      })
      .join('');

    el.results.innerHTML = `
      <div class="report-head">
        <div><h3>${el.kind.value === 'orders' ? 'Routes and orders' : 'Routes'}</h3>
          <p class="sub">${esc(period)} · ${esc(filterText)}${d.failed ? ` · <span class="bad">positions of ${d.failed} trips could not be loaded</span>` : ''}</p></div>
        <div class="report-actions">
          <input type="search" class="trips-search" id="rep-search" placeholder="Search route or vehicle" value="${esc(st.q)}" />
          <a class="btn-ghost btn-sm" href="${exportHref()}">⬇ Excel</a>
        </div>
      </div>
      <div class="card"><div class="card-head"><b>Statistics</b></div>
        <div class="stat-tiles">${statTiles(stats)
          .map((s) => `<div class="tile"><span>${esc(s.label)}</span><b class="${s.tone || ''}">${esc(s.value)}</b>${s.note ? `<small>${esc(s.note)}</small>` : ''}</div>`)
          .join('')}</div>
      </div>
      <div class="card">${
        rows.length
          ? `<div class="table-wrap"><table class="trips-table report-table"><thead><tr><th></th>${th('route', 'Route')}${th('vehicle', 'Vehicle')}${th('day', multiDay ? 'Day' : 'Day')}${th('orders', 'Orders', 'right')}${th(
              'visited',
              'In time · late · not',
              'right',
            )}${th('start', 'Start est. / actual', 'center')}${th('return', 'Return est. / actual', 'center')}${th('duration', 'Duration est. / actual', 'right')}${th('km', 'km est. / actual', 'right')}${th('state', 'State')}<th></th></tr></thead><tbody>${body}</tbody></table></div>`
          : `<div class="empty"><b>No trips match</b><p>${d.trips.length ? 'Try another orders filter or clear the search.' : 'No trips for these days. Dispatched uploads and Wialon Logistics routes appear here.'}</p></div>`
      }</div>
      <p class="data-meta">Visited in time: the vehicle reached the order no later than its ETA plus the allowed lateness. A visit needs the vehicle to stop near the order (driving past does not count). Not visited: it did not stop there during the trip window. Actual times and mileage come from Wialon positions. Loaded in ${fmtNum(d.ms)} ms.</p>`;

    const search = el.results.querySelector('#rep-search');
    search.addEventListener('input', () => {
      st.q = search.value;
      const pos = search.selectionStart;
      render();
      const again = el.results.querySelector('#rep-search');
      again.focus();
      again.setSelectionRange(pos, pos);
    });
    el.results.querySelectorAll('tr.report-row').forEach((tr) =>
      tr.addEventListener('click', () => {
        const id = tr.dataset.id;
        const set = el.kind.value === 'orders' ? st.collapsed : st.open;
        if (set.has(id)) set.delete(id);
        else set.add(id);
        render();
      }),
    );
    el.results.querySelectorAll('[data-map]').forEach((b) =>
      b.addEventListener('click', (e) => {
        e.stopPropagation();
        openTrip(b.dataset.map);
      }),
    );
    el.results.querySelectorAll('th[data-sort]').forEach((h) =>
      h.addEventListener('click', () => {
        const [c, dir] = (st.sort || '').split(':');
        st.sort = c !== h.dataset.sort ? `${h.dataset.sort}:asc` : dir === 'asc' ? `${c}:desc` : null;
        render();
      }),
    );
  }

  function ordersTable(orders, tz, trip) {
    if (!orders.length) return '<div class="sub orders-empty">No orders match the orders filter.</div>';
    const t = (ts) => (ts ? fmtTime(ts, tz) : '—');
    const dev = (actual, planned) => (actual ? Math.round((actual - planned) / 60) : null);
    const depot = (leaving) => {
      if (el.orders.value !== 'all' || !trip.warehouse) return '';
      const planned = leaving ? trip.planned.start : trip.planned.returnAt;
      const actual = leaving ? trip.actual?.startedAt : trip.actual?.returnedAt;
      const d = dev(actual, planned);
      const label = actual ? badge({ label: leaving ? 'Left' : 'Back', cls: 'completed' }) : `<span class="sub">${leaving ? 'Departure not seen' : 'Return not seen'}</span>`;
      const times = `${t(planned)} <span class="sub">/</span> ${t(actual)}`;
      return `<tr class="depot-row">
          <td class="num"><span class="depot-chip">W</span></td>
          <td><div class="strong">${leaving ? 'Leave' : 'Return to'} ${esc(trip.warehouse)}</div><div class="sub">${leaving ? 'Departure from the warehouse' : 'Arrival back at the warehouse'}</div></td>
          <td>${label}</td>
          <td class="center nowrap">${leaving ? '' : times}</td>
          <td class="right nowrap ${d > 0 ? 'warn' : 'sub'}">${devText(d)}</td>
          <td class="center nowrap">${leaving ? times : ''}</td>
          <td></td><td></td><td></td>
        </tr>`;
    };
    const rows = orders
      .map((s) => {
        const svc = serviceMin(s);
        const tone = s.status === 'on_time' || s.status === 'early' ? 'completed' : s.status === 'late' ? 'late' : s.status === 'missed' ? 'ended' : 'pending';
        const detail = [s.location, s.invoice && `Invoice ${s.invoice}`].filter(Boolean).join(' · ') || '—';
        const away = !s.arrivedAt && s.closestM !== null && s.closestM !== undefined ? `<div class="sub">closest ${s.closestM >= 1000 ? `${fmtNum(s.closestM / 1000, 1)} km` : `${s.closestM} m`}</div>` : '';
        return `<tr>
          <td class="num">${s.seq}</td>
          <td><div class="strong">${esc(s.name)}</div><div class="sub">${esc(detail)}</div></td>
          <td>${badge({ label: stopLabel(s), cls: tone })}${s.outOfOrder ? '<div class="sub warn">out of sequence</div>' : ''}${away}</td>
          <td class="center nowrap">${t(s.eta)} <span class="sub">/</span> ${t(s.arrivedAt)}</td>
          <td class="right nowrap ${s.status === 'late' ? 'warn' : 'sub'}">${devText(s.deviationMin)}</td>
          <td class="center nowrap">${t(s.etd)} <span class="sub">/</span> ${t(s.leftAt)}</td>
          <td class="right nowrap">${fmtMin(plannedServiceMin(s))} <span class="sub">/</span> ${fmtMin(svc)}</td>
          <td class="right">${fmtNum(s.plannedKm, 1)}</td>
          <td class="right nowrap sub">${s.weightKg ? `${fmtNum(s.weightKg)} kg` : ''}${s.amount ? `<div>${fmtNum(s.amount, 2)}</div>` : s.weightKg ? '' : '—'}</td>
        </tr>`;
      })
      .join('');
    return `<table class="orders-table"><thead><tr><th class="num">#</th><th>Order</th><th>Status</th><th class="center">Arrival est. / actual</th><th class="right">Deviation</th><th class="center">Departure est. / actual</th><th class="right">Service est. / actual</th><th class="right">Est. km</th><th class="right">Load</th></tr></thead><tbody>${depot(true)}${rows}${depot(false)}</tbody></table>`;
  }

  el.run.addEventListener('click', execute);
  [el.from, el.to].forEach((i) => i.addEventListener('change', validate));
  [el.kind, el.orders].forEach((i) => i.addEventListener('change', () => st.data && render()));
  validate();

  return {
    activate: loadVehicles,
  };
}
