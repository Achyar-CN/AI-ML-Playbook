export class UIController {
  constructor({ controlsPanel, simulationManager, stateManager }) {
    this.controlsPanel    = controlsPanel;
    this.simulationManager = simulationManager;
    this.stateManager     = stateManager;

    this.statusBadge      = document.getElementById('status-badge');
    this.epochDisplay     = document.getElementById('epoch-display');
    this.metricsContainer = document.getElementById('metrics');
    this.metricKeys       = [];

    this._bindTopbarButtons();
    this._bindSpeedSlider();
  }

  // ── Topbar ─────────────────────────────────────────────────
  _bindTopbarButtons() {
    document.getElementById('start-btn')?.addEventListener('click', () => {
      this.setStatus('running');
      this.simulationManager.start();
    });
    document.getElementById('pause-btn')?.addEventListener('click', () => {
      this.setStatus('paused');
      this.simulationManager.stop();
    });
    document.getElementById('reset-btn')?.addEventListener('click', () => {
      this.setStatus('ready');
      this.simulationManager.stop();
      this.simulationManager.reset();
      this.renderMetrics([]);
      this._updateEpoch([]);
    });
  }

  _bindSpeedSlider() {
    const slider = document.getElementById('speed-slider');
    const label  = document.getElementById('speed-val');
    if (!slider) return;
    const update = () => {
      const v = Number(slider.value);
      if (label) label.innerHTML = `${v}<small>/s</small>`;
      this.simulationManager.setSpeed(v);
    };
    slider.addEventListener('input', update);
    update();
  }

  setStatus(state) {
    if (!this.statusBadge) return;
    const labels = { running: 'Running', paused: 'Paused', ready: 'Ready' };
    this.statusBadge.textContent = labels[state] || state;
    this.statusBadge.className   = `status-badge ${state === 'ready' ? '' : state}`.trim();
  }

  _updateEpoch(history) {
    if (!this.epochDisplay) return;
    if (!history || history.length === 0) {
      this.epochDisplay.textContent = '—';
      return;
    }
    const last     = history[history.length - 1];
    const meta     = this.simulationManager.currentMeta;
    const dp       = meta && meta.defaultParams;
    const maxEpoch = dp ? (dp.epochs ?? dp.maxDepth ?? '?') : '?';
    this.epochDisplay.textContent = `${last.epoch} / ${maxEpoch}`;
  }

  // ── Simulation menu ────────────────────────────────────────
  renderMenu(simulations) {
    const box = document.getElementById('algo-selector-box');
    if (!box) return;

    const select = document.createElement('select');
    select.className = 'algo-select';
    select.setAttribute('data-sim-select', '');
    simulations.forEach((sim) => {
      const opt = document.createElement('option');
      opt.value = sim.id;
      opt.textContent = sim.title;
      select.appendChild(opt);
    });
    box.appendChild(select);

    select.addEventListener('change', () => {
      const id = select.value;
      this.stateManager.setState({ sim: id });
      this.simulationManager.stop();
      this.simulationManager.selectSimulation(id);
      this.setSelectedSim(id);
      this.setStatus('ready');
      this.renderMetrics([]);
      this._updateEpoch([]);
    });

    const modelId = this.stateManager.get('sim', simulations[0]?.id);
    this.setSelectedSim(modelId);
  }

  setSelectedSim(id) {
    const select = document.querySelector('select[data-sim-select]');
    if (select) select.value = id;
    const sim = this.simulationManager.simulations?.get(id);
    if (sim) {
      this.renderParams(sim);
      this.setupMetrics(sim);
    }
  }

  // ── Parameters (sliders + toggles) ────────────────────────
  renderParams(sim) {
    const box = document.getElementById('param-box');
    if (!box) return;
    box.innerHTML = '';

    const defaults = sim.defaultParams || {};
    const meta     = sim.paramControls || [];

    meta.forEach((field) => {
      const value   = this.stateManager.get(field.name, defaults[field.name]);
      const wrapper = document.createElement('div');
      wrapper.className = 'param-field';

      if (field.type === 'boolean') {
        const row = document.createElement('div');
        row.className = 'toggle-row';
        const lbl = document.createElement('label');
        lbl.textContent = field.label;
        lbl.setAttribute('for', `param-${field.name}`);
        const inp = document.createElement('input');
        inp.id      = `param-${field.name}`;
        inp.type    = 'checkbox';
        inp.className = 'toggle';
        inp.checked = Boolean(value);
        inp.setAttribute('data-param-name', field.name);
        row.appendChild(lbl);
        row.appendChild(inp);
        wrapper.appendChild(row);
      } else {
        const header = document.createElement('div');
        header.className = 'param-header';
        const lbl = document.createElement('label');
        lbl.textContent = field.label;
        lbl.setAttribute('for', `param-${field.name}`);
        const valDisplay = document.createElement('span');
        valDisplay.className = 'param-value-display';
        valDisplay.textContent = value;
        header.appendChild(lbl);
        header.appendChild(valDisplay);
        wrapper.appendChild(header);

        const inp = document.createElement('input');
        inp.id = `param-${field.name}`;
        inp.type = 'range';
        inp.className = 'param-slider';
        inp.min   = field.min  != null ? field.min  : 0;
        inp.max   = field.max  != null ? field.max  : 100;
        inp.step  = field.step != null ? field.step : 1;
        inp.value = value;
        inp.setAttribute('data-param-name', field.name);
        inp.addEventListener('input', () => { valDisplay.textContent = inp.value; });
        wrapper.appendChild(inp);
      }

      if (field.description) {
        const desc = document.createElement('p');
        desc.className = 'param-desc';
        desc.textContent = field.description;
        wrapper.appendChild(desc);
      }

      box.appendChild(wrapper);
    });

    // Delegate change events (fires after slider released or checkbox toggled)
    box.addEventListener('change', (e) => {
      const el = e.target.closest('[data-param-name]');
      if (!el || !this.simulationManager.current) return;
      const pName = el.getAttribute('data-param-name');
      const val   = el.type === 'checkbox' ? el.checked : Number(el.value);
      this.simulationManager.updateCurrentParams({ [pName]: val });
      this.stateManager.setState({ [pName]: val });
      this.renderMetrics([]);
      this._updateEpoch([]);
    });
  }

  // ── Metrics panel ──────────────────────────────────────────
  setupMetrics(sim) {
    this.metricKeys = sim.metricKeys || ['loss'];
    if (!this.metricsContainer) return;
    this.metricsContainer.innerHTML = '';

    this.metricKeys.forEach((key) => {
      const card = document.createElement('div');
      card.className = 'metric-card';

      const header = document.createElement('div');
      header.className = 'metric-card-header';

      const name = document.createElement('span');
      name.className = 'metric-name';
      name.textContent = key.toUpperCase();

      const lastVal = document.createElement('span');
      lastVal.className = 'metric-last-value';
      lastVal.setAttribute('data-metric-val', key);
      lastVal.textContent = '—';

      header.appendChild(name);
      header.appendChild(lastVal);

      const canvas = document.createElement('canvas');
      canvas.setAttribute('data-metric', key);
      canvas.width  = 500;
      canvas.height = 120;

      card.appendChild(header);
      card.appendChild(canvas);
      this.metricsContainer.appendChild(card);
    });
  }

  renderMetrics(history) {
    this._updateEpoch(history);
    if (!this.metricsContainer) return;

    this.metricKeys.forEach((key) => {
      const canvas  = this.metricsContainer.querySelector(`canvas[data-metric="${key}"]`);
      const valSpan = this.metricsContainer.querySelector(`[data-metric-val="${key}"]`);
      if (!canvas) return;

      const ctx = canvas.getContext('2d');
      const W = canvas.width, H = canvas.height;
      const pad = { t: 8, r: 8, b: 20, l: 36 };

      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#f8fafc';
      ctx.fillRect(0, 0, W, H);

      if (!history || history.length === 0) {
        if (valSpan) valSpan.textContent = '—';
        ctx.fillStyle = '#94a3b8';
        ctx.font = '13px system-ui';
        ctx.textAlign = 'center';
        ctx.fillText('No data — press Run', W / 2, H / 2 + 5);
        return;
      }

      const values = history.map((item) => (item[key] !== undefined ? item[key] : 0));
      const latest = values[values.length - 1];
      const isAcc  = ['accuracy', 'f1', 'recall', 'precision'].includes(key);
      const isMape = key === 'mape';

      if (valSpan) {
        const disp = isAcc
          ? `${(latest * 100).toFixed(1)}%`
          : isMape ? `${latest.toFixed(1)}%`
          : latest.toFixed(4);
        valSpan.textContent = disp;
        valSpan.style.color = key === 'loss' ? '#dc2626' : '#1d4ed8';
      }

      const maxY   = Math.max(...values, 1e-6);
      const minY   = Math.min(...values, 0);
      const yRange = maxY - minY || 1;
      const chartW = W - pad.l - pad.r;
      const chartH = H - pad.t - pad.b;

      // Grid
      for (let i = 0; i <= 3; i++) {
        const yFrac = i / 3;
        const cy    = pad.t + yFrac * chartH;
        ctx.strokeStyle = '#e2e8f0'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(pad.l, cy); ctx.lineTo(W - pad.r, cy); ctx.stroke();
        ctx.fillStyle = '#94a3b8'; ctx.font = '9px system-ui'; ctx.textAlign = 'right';
        ctx.fillText((maxY - yFrac * yRange).toFixed(2), pad.l - 4, cy + 3);
      }
      ctx.fillStyle = '#94a3b8'; ctx.font = '9px system-ui'; ctx.textAlign = 'center';
      ctx.fillText('Epoch', W / 2, H - 4);

      const n = values.length;
      const lineColor = key === 'loss' ? '#dc2626' : '#1d4ed8';
      const areaColor = key === 'loss' ? 'rgba(220,38,38,.08)' : 'rgba(29,78,216,.08)';

      // Area fill
      ctx.beginPath();
      ctx.moveTo(pad.l, pad.t + chartH);
      for (let i = 0; i < n; i++) {
        const x = pad.l + (i / Math.max(n - 1, 1)) * chartW;
        const y = pad.t + ((maxY - values[i]) / yRange) * chartH;
        ctx.lineTo(x, y);
      }
      ctx.lineTo(pad.l + chartW, pad.t + chartH);
      ctx.closePath();
      ctx.fillStyle = areaColor;
      ctx.fill();

      // Line
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const x = pad.l + (i / Math.max(n - 1, 1)) * chartW;
        const y = pad.t + ((maxY - values[i]) / yRange) * chartH;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.strokeStyle = lineColor; ctx.lineWidth = 2; ctx.lineJoin = 'round';
      ctx.stroke();

      // Latest dot
      const lx = pad.l + chartW;
      const ly = pad.t + ((maxY - latest) / yRange) * chartH;
      ctx.beginPath(); ctx.arc(lx, ly, 3, 0, Math.PI * 2);
      ctx.fillStyle = lineColor; ctx.fill();
    });
  }
}

