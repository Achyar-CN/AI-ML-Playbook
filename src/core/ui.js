import { dataStore } from './dataStore.js';

// ── Preset scenarios ──────────────────────────────────────────────
const CLASS_PRESETS = [
  { label: 'Separable', params: { datasetType: 'linear',  noiseLevel: 0.02, nPoints: 100 } },
  { label: 'Noisy',     params: { datasetType: 'moons',   noiseLevel: 0.38, nPoints: 150 } },
  { label: 'Outliers',  params: { datasetType: 'linear',  noiseLevel: 0.32, nPoints: 120 } },
];
const REG_PRESETS = [
  { label: 'Separable', params: { datasetType: 'linear',    noiseLevel: 0.04, nPoints: 80  } },
  { label: 'Noisy',     params: { datasetType: 'sine',      noiseLevel: 0.4,  nPoints: 120 } },
  { label: 'Outliers',  params: { datasetType: 'noisy',     noiseLevel: 0.75, nPoints: 100 } },
];

export class UIController {
  constructor({ controlsPanel, simulationManager, stateManager, simulations }) {
    this.controlsPanel     = controlsPanel;
    this.simulationManager = simulationManager;
    this.stateManager      = stateManager;
    this.allSimulations    = simulations; // full metadata array

    this.statusBadge      = document.getElementById('status-badge');
    this.epochDisplay     = document.getElementById('epoch-display');
    this.metricsContainer = document.getElementById('metrics');
    this.metricKeys       = [];

    this._bindTopbarButtons();
    this._bindSpeedSlider();
    this._bindDarkToggle();
    this._bindAlgoInfoToggle();
    this._bindSectionCollapses();
  }

  // ── Topbar ──────────────────────────────────────────────────
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

  _bindDarkToggle() {
    const btn  = document.getElementById('dark-toggle');
    const icon = document.getElementById('dark-icon');
    if (!btn) return;
    const SUN  = '<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>';
    const MOON = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>';
    const apply = (dark) => {
      if (dark) {
        document.documentElement.dataset.theme = 'dark';
        if (icon) icon.innerHTML = SUN;
      } else {
        delete document.documentElement.dataset.theme;
        if (icon) icon.innerHTML = MOON;
      }
    };
    apply(localStorage.getItem('mlp-theme') === 'dark');
    btn.addEventListener('click', () => {
      const next = document.documentElement.dataset.theme !== 'dark';
      localStorage.setItem('mlp-theme', next ? 'dark' : 'light');
      apply(next);
    });
  }

  _bindAlgoInfoToggle() {
    const header = document.getElementById('algo-info-toggle');
    const box    = document.getElementById('algo-info-box');
    if (!header || !box) return;
    header.addEventListener('click', () => {
      // Pure class toggle — CSS handles max-height via .algo-info-body / .collapsed rules
      const collapsed = box.classList.toggle('collapsed');
      header.classList.toggle('collapsed', collapsed);
    });
  }

  _bindSectionCollapses() {
    [
      { headerId: 'data-toggle',   bodyId: 'data-param-box' },
      { headerId: 'params-toggle', bodyId: 'param-box'      },
    ].forEach(({ headerId, bodyId }) => {
      const header = document.getElementById(headerId);
      const body   = document.getElementById(bodyId);
      if (!header || !body) return;
      header.addEventListener('click', () => {
        const collapsed = body.classList.toggle('collapsed');
        header.classList.toggle('collapsed', collapsed);
      });
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
    // Always read max from the live params (reflects slider changes instantly)
    const p        = this.simulationManager.current?.params;
    const maxEpoch = p ? (p.epochs ?? p.nTrees ?? p.maxDepth ?? '?') : '?';
    if (!history || history.length === 0) {
      this.epochDisplay.textContent = `— / ${maxEpoch}`;
      return;
    }
    const last = history[history.length - 1];
    this.epochDisplay.textContent = `${last.epoch} / ${maxEpoch}`;
  }

  // ── Main menu: task tabs + algo selector ─────────────────────
  renderMenu(simulations) {
    const box = document.getElementById('algo-selector-box');
    if (!box) return;
    box.innerHTML = '';

    // Group by taskType
    const groups = {};
    simulations.forEach(sim => {
      if (!groups[sim.taskType]) groups[sim.taskType] = [];
      groups[sim.taskType].push(sim);
    });
    const taskTypes = Object.keys(groups);

    // Task tabs
    const tabBar = document.createElement('div');
    tabBar.className = 'task-tabs';
    taskTypes.forEach((type, i) => {
      const btn = document.createElement('button');
      btn.className = `task-tab${i === 0 ? ' active' : ''}`;
      btn.dataset.task = type;
      btn.textContent = type.charAt(0).toUpperCase() + type.slice(1);
      tabBar.appendChild(btn);
    });
    box.appendChild(tabBar);

    // Algo select
    const select = document.createElement('select');
    select.className = 'algo-select';
    select.setAttribute('data-sim-select', '');
    const populateAlgos = (taskType) => {
      select.innerHTML = '';
      groups[taskType].forEach(sim => {
        const opt = document.createElement('option');
        opt.value = sim.id;
        opt.textContent = sim.title;
        select.appendChild(opt);
      });
    };
    populateAlgos(taskTypes[0]);
    box.appendChild(select);

    // Tab switching
    let activeTask = taskTypes[0];
    tabBar.addEventListener('click', (e) => {
      const btn = e.target.closest('.task-tab');
      if (!btn) return;
      tabBar.querySelectorAll('.task-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeTask = btn.dataset.task;
      populateAlgos(activeTask);
      const firstId = groups[activeTask][0].id;
      this._switchSim(firstId);
    });

    select.addEventListener('change', () => this._switchSim(select.value));

    const initId = this.stateManager.get('sim', simulations[0]?.id);
    // Activate correct tab
    const initSim = simulations.find(s => s.id === initId);
    if (initSim) {
      activeTask = initSim.taskType;
      populateAlgos(activeTask);
      tabBar.querySelectorAll('.task-tab').forEach(b => {
        b.classList.toggle('active', b.dataset.task === activeTask);
      });
    }
    this.setSelectedSim(initId);
  }

  _switchSim(id) {
    this.simulationManager.stop();
    this.simulationManager.selectSimulation(id);
    this.setSelectedSim(id);
    this.setStatus('ready');
    this.renderMetrics([]);
    this._updateEpoch([]);
    this.stateManager.setState({ sim: id });
  }

  setSelectedSim(id) {
    const select = document.querySelector('select[data-sim-select]');
    if (select) select.value = id;
    const sim = this.simulationManager.simulations?.get(id);
    if (sim) {
      this.renderDataParams(sim);
      this.renderHyperParams(sim);
      this.renderAlgoInfo(sim);
      this.setupMetrics(sim);
    }
  }

  renderAlgoInfo(sim) {
    const box = document.getElementById('algo-info-box');
    if (!box) return;
    box.innerHTML = '';
    const info = sim.info;
    if (!info) return;

    if (info.tagline) {
      const tag = document.createElement('div');
      tag.className = 'algo-info-tagline';
      tag.textContent = info.tagline;
      box.appendChild(tag);
    }
    if (info.description) {
      const p = document.createElement('p');
      p.className = 'algo-info-desc';
      p.textContent = info.description;
      box.appendChild(p);
    }
    if (info.insights && info.insights.length) {
      const ul = document.createElement('ul');
      ul.className = 'algo-info-insights';
      info.insights.forEach(text => {
        const li = document.createElement('li');
        li.textContent = text;
        ul.appendChild(li);
      });
      box.appendChild(ul);
    }

    // max-height is controlled entirely by CSS classes (.algo-info-body / .collapsed)
    // Do NOT set inline max-height here — it would override the CSS toggle.
  }

  // ── Data section (dataset chips + data sliders) ──────────────
  renderDataParams(sim) {
    const box = document.getElementById('data-param-box');
    if (!box) return;
    box.innerHTML = '';

    // ── Preset scenario buttons ─────────────────────────────────
    const presets = sim.taskType === 'regression' ? REG_PRESETS : CLASS_PRESETS;
    const presetRow = document.createElement('div');
    presetRow.className = 'preset-row';
    presets.forEach(({ label, params }) => {
      const btn = document.createElement('button');
      btn.className = 'preset-btn';
      btn.textContent = label;
      btn.title = Object.entries(params).map(([k, v]) => `${k}: ${v}`).join(', ');
      btn.addEventListener('click', () => this._applyPreset(params, sim));
      presetRow.appendChild(btn);
    });
    box.appendChild(presetRow);

    const defaults = sim.defaultParams || {};
    const controls = sim.dataParamControls || [];

    controls.forEach(field => {
      const value = this.stateManager.get(field.name, defaults[field.name]);

      if (field.type === 'dataset') {
        // Chip grid for dataset selection
        const chips = document.createElement('div');
        chips.className = 'dataset-chips';
        field.options.forEach(opt => {
          const chip = document.createElement('button');
          chip.className = `dataset-chip${opt.id === (value || field.options[0].id) ? ' active' : ''}`;
          chip.dataset.value = opt.id;
          chip.textContent = opt.label;
          chip.addEventListener('click', () => {
            chips.querySelectorAll('.dataset-chip').forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            this._applyParam(field.name, opt.id);
          });
          chips.appendChild(chip);
        });
        box.appendChild(chips);
        return;
      }

      box.appendChild(this._buildSliderField(field, value));
    });

    // ── CSV import ──────────────────────────────────────────────
    box.appendChild(this._buildCSVImport(sim));
  }

  _applyPreset(params, sim) {
    // Update state manager so chip/slider UI reflects new values
    Object.entries(params).forEach(([k, v]) => this.stateManager.setState({ [k]: v }));
    this.simulationManager.updateCurrentParams(params);
    this.renderDataParams(sim); // re-render chips + sliders with new values
    this.setStatus('ready');
  }

  _buildCSVImport(sim) {
    const wrap = document.createElement('div');
    wrap.className = 'csv-import-section';

    const header = document.createElement('div');
    header.className = 'csv-import-header';

    const label = document.createElement('span');
    label.className = 'csv-import-label';
    label.textContent = 'Import CSV';

    const sampleLink = document.createElement('a');
    sampleLink.className = 'csv-sample-link';
    sampleLink.href = sim.taskType === 'regression'
      ? 'samples/regression_sample.csv'
      : 'samples/classification_sample.csv';
    sampleLink.download = '';
    sampleLink.textContent = 'sample';

    header.appendChild(label);
    header.appendChild(sampleLink);
    wrap.appendChild(header);

    // Status / filename row
    const status = document.createElement('div');
    status.className = 'csv-status';
    if (dataStore.points && dataStore.type === sim.taskType) {
      status.textContent = `\u2713 ${dataStore.filename || 'custom data'} (${dataStore.points.length} pts)`;
      status.classList.add('loaded');
    } else {
      status.textContent = 'No file loaded';
    }
    wrap.appendChild(status);

    const fileRow = document.createElement('div');
    fileRow.className = 'csv-file-row';

    const fileInput = document.createElement('input');
    fileInput.type   = 'file';
    fileInput.accept = '.csv';
    fileInput.className = 'csv-file-input';
    fileInput.id = 'csv-file-input';

    const fileLabel = document.createElement('label');
    fileLabel.htmlFor   = 'csv-file-input';
    fileLabel.className = 'btn csv-choose-btn';
    fileLabel.textContent = 'Choose file';

    const clearBtn = document.createElement('button');
    clearBtn.className   = 'btn csv-clear-btn';
    clearBtn.textContent = 'Clear';
    clearBtn.disabled    = !(dataStore.points && dataStore.type === sim.taskType);

    fileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (!file) return;
      this._handleCSVUpload(file, sim);
    });

    clearBtn.addEventListener('click', () => {
      dataStore.points   = null;
      dataStore.type     = null;
      dataStore.filename = null;
      this.simulationManager.updateCurrentParams({});
      this.renderDataParams(sim);
    });

    fileRow.appendChild(fileInput);
    fileRow.appendChild(fileLabel);
    fileRow.appendChild(clearBtn);
    wrap.appendChild(fileRow);

    const fmt = document.createElement('p');
    fmt.className = 'csv-format-hint';
    fmt.textContent = sim.taskType === 'regression'
      ? 'Format: x,y  (header row optional)'
      : 'Format: x1,x2,label  (label = 0 or 1)';
    wrap.appendChild(fmt);

    return wrap;
  }

  _handleCSVUpload(file, sim) {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const rows = e.target.result.trim().split('\n')
          .map(r => r.trim()).filter(r => r.length > 0);

        // Skip header row if first cell is non-numeric
        const startIdx = isNaN(parseFloat(rows[0].split(',')[0])) ? 1 : 0;
        const data = rows.slice(startIdx).map(row => row.split(',').map(v => parseFloat(v.trim())));

        if (sim.taskType === 'regression') {
          const raw = data.filter(r => r.length >= 2 && !isNaN(r[0]) && !isNaN(r[1]));
          if (!raw.length) throw new Error('No valid rows');
          const xs = raw.map(r => r[0]), ys = raw.map(r => r[1]);
          const normalize = (vals) => {
            const mn = Math.min(...vals), mx = Math.max(...vals);
            return mx === mn ? vals.map(() => 0) : vals.map(v => 2 * (v - mn) / (mx - mn) - 1);
          };
          const nxs = normalize(xs), nys = normalize(ys);
          dataStore.points   = nxs.map((x, i) => ({ x, y: nys[i] }));
          dataStore.type     = 'regression';
          dataStore.filename = file.name;
        } else {
          const raw = data.filter(r => r.length >= 3 && !isNaN(r[0]) && !isNaN(r[1]) && !isNaN(r[2]));
          if (!raw.length) throw new Error('No valid rows');
          const x1s = raw.map(r => r[0]), x2s = raw.map(r => r[1]);
          const normalize = (vals) => {
            const mn = Math.min(...vals), mx = Math.max(...vals);
            return mx === mn ? vals.map(() => 0) : vals.map(v => 2 * (v - mn) / (mx - mn) - 1);
          };
          const nx1 = normalize(x1s), nx2 = normalize(x2s);
          dataStore.points   = nx1.map((x, i) => ({ x, y: nx2[i], label: raw[i][2] === 1 ? 1 : 0 }));
          dataStore.type     = 'classification';
          dataStore.filename = file.name;
        }

        this.simulationManager.updateCurrentParams({});
        this.renderDataParams(sim);
        this.setStatus('ready');
      } catch {
        alert('CSV parse error. Check the format hint below the import button.');
      }
    };
    reader.readAsText(file);
  }

  // ── Hyperparameter section ────────────────────────────────────
  renderHyperParams(sim) {
    const box = document.getElementById('param-box');
    if (!box) return;
    box.innerHTML = '';

    const defaults = sim.defaultParams || {};
    const controls = sim.paramControls  || [];

    controls.forEach(field => {
      const value = this.stateManager.get(field.name, defaults[field.name]);

      if (field.type === 'boolean') {
        const wrapper = document.createElement('div');
        wrapper.className = 'param-field';
        const row = document.createElement('div');
        row.className = 'toggle-row';
        const lbl = document.createElement('label');
        lbl.textContent = field.label;
        lbl.setAttribute('for', `param-${field.name}`);
        const inp = document.createElement('input');
        inp.id = `param-${field.name}`;
        inp.type = 'checkbox';
        inp.className = 'toggle';
        inp.checked = Boolean(value);
        inp.setAttribute('data-param-name', field.name);
        inp.addEventListener('change', () => this._applyParam(field.name, inp.checked));
        row.appendChild(lbl);
        row.appendChild(inp);
        wrapper.appendChild(row);
        if (field.description) {
          const d = document.createElement('p');
          d.className = 'param-desc';
          d.textContent = field.description;
          wrapper.appendChild(d);
        }
        box.appendChild(wrapper);
        return;
      }

      if (field.type === 'select') {
        const wrapper = document.createElement('div');
        wrapper.className = 'param-field';
        const lbl = document.createElement('label');
        lbl.textContent = field.label;
        lbl.setAttribute('for', `param-${field.name}`);
        wrapper.appendChild(lbl);
        const sel = document.createElement('select');
        sel.id = `param-${field.name}`;
        sel.className = 'param-select';
        sel.setAttribute('data-param-name', field.name);
        field.options.forEach(opt => {
          const o = document.createElement('option');
          o.value = opt.value;
          o.textContent = opt.label;
          if (opt.value === value) o.selected = true;
          sel.appendChild(o);
        });
        sel.addEventListener('change', () => this._applyParam(field.name, sel.value));
        wrapper.appendChild(sel);
        if (field.description) {
          const d = document.createElement('p');
          d.className = 'param-desc';
          d.textContent = field.description;
          wrapper.appendChild(d);
        }
        box.appendChild(wrapper);
        return;
      }

      box.appendChild(this._buildSliderField(field, value));
    });
  }

  _buildSliderField(field, value) {
    const wrapper = document.createElement('div');
    wrapper.className = 'param-field';

    const header = document.createElement('div');
    header.className = 'param-header';
    const lbl = document.createElement('label');
    lbl.textContent = field.label;
    lbl.setAttribute('for', `param-${field.name}`);
    const valDisp = document.createElement('span');
    valDisp.className = 'param-value-display';
    valDisp.textContent = value;
    header.appendChild(lbl);
    header.appendChild(valDisp);
    wrapper.appendChild(header);

    const inp = document.createElement('input');
    inp.id        = `param-${field.name}`;
    inp.type      = 'range';
    inp.className = 'param-slider';
    inp.min   = field.min  != null ? field.min  : 0;
    inp.max   = field.max  != null ? field.max  : 100;
    inp.step  = field.step != null ? field.step : 1;
    inp.value = value;
    inp.setAttribute('data-param-name', field.name);

    inp.addEventListener('input', () => {
      valDisp.textContent = inp.value;
      // Keep epoch counter in sync while scrubbing epochs/maxDepth sliders
      if (field.name === 'epochs' || field.name === 'maxDepth') {
        if (this.simulationManager.current) {
          this.simulationManager.current.params[field.name] = Number(inp.value);
        }
        this._updateEpoch(this.simulationManager.current?.history || []);
      }
    });
    inp.addEventListener('change', () => this._applyParam(field.name, Number(inp.value)));

    wrapper.appendChild(inp);

    if (field.description) {
      const d = document.createElement('p');
      d.className = 'param-desc';
      d.textContent = field.description;
      wrapper.appendChild(d);
    }
    return wrapper;
  }

  _applyParam(name, value) {
    if (!this.simulationManager.current) return;
    this.simulationManager.updateCurrentParams({ [name]: value });
    this.stateManager.setState({ [name]: value });
    this.renderMetrics([]);
    this._updateEpoch([]);
  }

  // ── Metrics ──────────────────────────────────────────────────
  // Concise descriptions + business impact for each metric
  static METRIC_INFO = {
    loss:      { label: 'Loss',      desc: 'Error the model minimizes during training.',       impact: 'Lower = better fit. Plateau = converged.' },
    accuracy:  { label: 'Accuracy',  desc: 'Fraction of correctly classified samples.',        impact: 'Best with balanced classes. 90%+ is typically production-ready.' },
    recall:    { label: 'Recall',    desc: 'True positives / (True pos + False neg).',          impact: 'Critical when missing a positive is costly (e.g. disease detection).' },
    precision: { label: 'Precision', desc: 'True positives / (True pos + False pos).',          impact: 'Critical when false alarms are costly (e.g. spam filters, fraud).' },
    f1:        { label: 'F1',        desc: 'Harmonic mean of Precision and Recall.',            impact: 'Best single metric for imbalanced datasets.' },
    mae:       { label: 'MAE',       desc: 'Mean Absolute Error — avg magnitude of errors.',   impact: 'Interpretable in same units as target. Robust to outliers.' },
    rmse:      { label: 'RMSE',      desc: 'Root Mean Squared Error — penalizes large errors.', impact: 'Sensitive to outliers. Use when large errors are especially bad.' },
    mape:      { label: 'MAPE',      desc: 'Mean Absolute Percentage Error.',                   impact: 'Scale-independent. Useful for comparing across different datasets.' },
    nmae:      { label: 'NMAE',      desc: 'Normalized MAE relative to mean target value.',    impact: 'Allows comparison when target scale varies across runs.' },
  };

  setupMetrics(sim) {
    this.metricKeys = sim.metricKeys || ['loss'];
    if (!this.metricsContainer) return;
    this.metricsContainer.innerHTML = '';

    this.metricKeys.forEach(key => {
      const info = UIController.METRIC_INFO[key] || { label: key.toUpperCase(), desc: '', impact: '' };

      const card = document.createElement('div');
      card.className = 'metric-card';

      // Header: name + last value
      const header = document.createElement('div');
      header.className = 'metric-card-header';

      const nameWrap = document.createElement('div');
      nameWrap.className = 'metric-name-wrap';

      const name = document.createElement('span');
      name.className = 'metric-name';
      name.textContent = info.label.toUpperCase();
      nameWrap.appendChild(name);

      const lastVal = document.createElement('span');
      lastVal.className = 'metric-last-value';
      lastVal.setAttribute('data-metric-val', key);
      lastVal.textContent = '—';

      header.appendChild(nameWrap);
      header.appendChild(lastVal);
      card.appendChild(header);

      // Description + business impact
      if (info.desc) {
        const descWrap = document.createElement('div');
        descWrap.className = 'metric-desc';
        descWrap.innerHTML = `<span class="metric-desc-text">${info.desc}</span>`
          + (info.impact ? ` <span class="metric-impact">${info.impact}</span>` : '');
        card.appendChild(descWrap);
      }

      const canvas = document.createElement('canvas');
      canvas.setAttribute('data-metric', key);
      canvas.width  = 500;
      canvas.height = 120;
      card.appendChild(canvas);

      this.metricsContainer.appendChild(card);
    });
  }

  renderMetrics(history) {
    this._updateEpoch(history);
    if (!this.metricsContainer) return;

    this.metricKeys.forEach(key => {
      const canvas  = this.metricsContainer.querySelector(`canvas[data-metric="${key}"]`);
      const valSpan = this.metricsContainer.querySelector(`[data-metric-val="${key}"]`);
      if (!canvas) return;

      const ctx = canvas.getContext('2d');
      const W = canvas.width, H = canvas.height;
      const pad = { t: 8, r: 8, b: 20, l: 36 };
      const dark  = document.documentElement.dataset.theme === 'dark';
      const cBg   = dark ? '#1e293b' : '#f8fafc';
      const cGrid = dark ? '#334155' : '#e2e8f0';
      const cMute = dark ? '#64748b' : '#94a3b8';

      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = cBg;
      ctx.fillRect(0, 0, W, H);

      if (!history || history.length === 0) {
        if (valSpan) valSpan.textContent = '—';
        ctx.fillStyle = cMute;
        ctx.font = '13px system-ui';
        ctx.textAlign = 'center';
        ctx.fillText('No data — press Run', W / 2, H / 2 + 5);
        return;
      }

      const values = history.map(item => item[key] !== undefined ? item[key] : 0);
      const latest = values[values.length - 1];
      const isAcc  = ['accuracy', 'f1', 'recall', 'precision'].includes(key);
      const isMape = key === 'mape';

      if (valSpan) {
        const disp = isAcc ? `${(latest * 100).toFixed(1)}%`
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

      for (let i = 0; i <= 3; i++) {
        const yFrac = i / 3, cy = pad.t + yFrac * chartH;
        ctx.strokeStyle = cGrid; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(pad.l, cy); ctx.lineTo(W - pad.r, cy); ctx.stroke();
        ctx.fillStyle = cMute; ctx.font = '9px system-ui'; ctx.textAlign = 'right';
        ctx.fillText((maxY - yFrac * yRange).toFixed(2), pad.l - 4, cy + 3);
      }
      ctx.fillStyle = cMute; ctx.font = '9px system-ui'; ctx.textAlign = 'center';
      ctx.fillText('Epoch', W / 2, H - 4);

      const n = values.length;
      const lineColor = key === 'loss' ? '#dc2626' : '#1d4ed8';
      const areaColor = key === 'loss' ? 'rgba(220,38,38,.08)' : 'rgba(29,78,216,.08)';

      ctx.beginPath();
      ctx.moveTo(pad.l, pad.t + chartH);
      for (let i = 0; i < n; i++) {
        const x = pad.l + (i / Math.max(n - 1, 1)) * chartW;
        const y = pad.t + ((maxY - values[i]) / yRange) * chartH;
        ctx.lineTo(x, y);
      }
      ctx.lineTo(pad.l + chartW, pad.t + chartH);
      ctx.closePath();
      ctx.fillStyle = areaColor; ctx.fill();

      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const x = pad.l + (i / Math.max(n - 1, 1)) * chartW;
        const y = pad.t + ((maxY - values[i]) / yRange) * chartH;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.strokeStyle = lineColor; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.stroke();

      const lx = pad.l + chartW;
      const ly = pad.t + ((maxY - latest) / yRange) * chartH;
      ctx.beginPath(); ctx.arc(lx, ly, 3, 0, Math.PI * 2);
      ctx.fillStyle = lineColor; ctx.fill();
    });
  }
}
