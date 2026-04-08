import { dataStore }   from './dataStore.js';
import { computePCA }  from '../utils/pca.js';

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
    this._initTooltip();
  }

  // ── Shared floating tooltip ──────────────────────────────────
  _initTooltip() {
    const el = document.createElement('div');
    el.id = 'tip-bubble';
    document.body.appendChild(el);
    this._tipEl = el;
  }

  _buildTooltipIcon(text) {
    if (!text) return null;
    const icon = document.createElement('span');
    icon.className  = 'tip-icon';
    icon.textContent = 'i';
    icon.tabIndex   = 0;

    const show = (e) => {
      const el   = this._tipEl;
      el.textContent = text;
      el.classList.add('visible');
      this._positionTip(e.currentTarget);
    };
    const hide = () => this._tipEl.classList.remove('visible');

    icon.addEventListener('mouseenter', show);
    icon.addEventListener('focus',      show);
    icon.addEventListener('mouseleave', hide);
    icon.addEventListener('blur',       hide);
    return icon;
  }

  _positionTip(anchor) {
    const el   = this._tipEl;
    const rect = anchor.getBoundingClientRect();
    const gap  = 6;
    // prefer right of icon, shift left if overflows viewport
    let left = rect.right + gap;
    let top  = rect.top - 4;
    el.style.left = '0'; el.style.top = '0'; // reset so offsetWidth is measurable
    const bw = el.offsetWidth || 220;
    if (left + bw > window.innerWidth - 8) left = rect.left - bw - gap;
    if (top + el.offsetHeight > window.innerHeight - 8) top = window.innerHeight - el.offsetHeight - 8;
    el.style.left = `${Math.max(8, left)}px`;
    el.style.top  = `${Math.max(8, top)}px`;
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

  // ── Data section ─────────────────────────────────────────────
  renderDataParams(sim) {
    const box = document.getElementById('data-param-box');
    if (!box) return;
    box.innerHTML = '';

    const defaults  = sim.defaultParams || {};
    const controls  = sim.dataParamControls || [];

    // Separate testSplit from scenario controls
    const testSplitField    = controls.find(f => f.name === 'testSplit');
    const scenarioControls  = controls.filter(f => f.name !== 'testSplit');

    // Default mode: 'scenario'. Switch to 'csv' if CSV data is loaded for this task.
    if (!this._dataSourceMode) this._dataSourceMode = 'scenario';
    if (dataStore.points && dataStore.type === sim.taskType) this._dataSourceMode = 'csv';

    // ── Source tabs: Scenario | CSV ─────────────────────────────
    const sourceTabRow = document.createElement('div');
    sourceTabRow.className = 'data-source-tabs';

    const showSection = (mode) => {
      this._dataSourceMode = mode;
      scenarioSection.style.display = mode === 'scenario' ? '' : 'none';
      csvSection.style.display      = mode === 'csv'      ? '' : 'none';
      sourceTabRow.querySelectorAll('.data-source-tab').forEach(b =>
        b.classList.toggle('active', b.dataset.mode === mode)
      );
    };

    ['scenario', 'csv'].forEach(mode => {
      const btn = document.createElement('button');
      btn.className  = `data-source-tab${mode === this._dataSourceMode ? ' active' : ''}`;
      btn.dataset.mode = mode;
      btn.textContent = mode === 'scenario' ? 'Scenario' : 'CSV';
      btn.addEventListener('click', () => showSection(mode));
      sourceTabRow.appendChild(btn);
    });
    box.appendChild(sourceTabRow);

    // ── Scenario section ─────────────────────────────────────────
    const scenarioSection = document.createElement('div');
    scenarioSection.className = 'data-section-content';
    scenarioSection.style.display = this._dataSourceMode === 'scenario' ? '' : 'none';

    // Preset buttons
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
    scenarioSection.appendChild(presetRow);

    // Dataset chips + sliders (all controls except testSplit)
    scenarioControls.forEach(field => {
      const value = this.stateManager.get(field.name, defaults[field.name]);
      if (field.type === 'dataset') {
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
        scenarioSection.appendChild(chips);
        return;
      }
      scenarioSection.appendChild(this._buildSliderField(field, value));
    });
    box.appendChild(scenarioSection);

    // ── CSV section ──────────────────────────────────────────────
    const csvSection = document.createElement('div');
    csvSection.className = 'data-section-content';
    csvSection.style.display = this._dataSourceMode === 'csv' ? '' : 'none';
    csvSection.appendChild(this._buildCSVImport(sim));
    box.appendChild(csvSection);

    // ── Always visible: Test Split + Show Test toggle ─────────────
    const alwaysDiv = document.createElement('div');
    alwaysDiv.className = 'data-always-section';

    if (testSplitField) {
      const tsVal = this.stateManager.get('testSplit', defaults.testSplit ?? 0);
      alwaysDiv.appendChild(this._buildSliderField(testSplitField, tsVal));
    }
    alwaysDiv.appendChild(this._buildShowTestToggle());
    box.appendChild(alwaysDiv);
  }

  _applyPreset(params, sim) {
    Object.entries(params).forEach(([k, v]) => this.stateManager.setState({ [k]: v }));
    this.simulationManager.updateCurrentParams(params);
    this.renderDataParams(sim);
    this.setStatus('ready');
  }

  _buildShowTestToggle() {
    const sim = this.simulationManager.current;
    const current = sim?.showTestOverlay !== false; // default true

    const wrapper = document.createElement('div');
    wrapper.className = 'param-field';
    const row = document.createElement('div');
    row.className = 'toggle-row';

    const lblWrap = document.createElement('div');
    lblWrap.style.cssText = 'display:flex;align-items:center;gap:5px;';
    const lbl = document.createElement('label');
    lbl.textContent = 'Show Test Points';
    lbl.setAttribute('for', 'show-test-toggle');
    lblWrap.appendChild(lbl);
    // No tooltip for Show Test Points — label is self-explanatory

    const inp = document.createElement('input');
    inp.id       = 'show-test-toggle';
    inp.type     = 'checkbox';
    inp.className = 'toggle';
    inp.checked  = current;

    inp.addEventListener('change', () => {
      const s = this.simulationManager.current;
      if (s) {
        s.showTestOverlay = inp.checked;
        s.renderWithOverlays();
      }
    });

    row.appendChild(lblWrap);
    row.appendChild(inp);
    wrapper.appendChild(row);
    return wrapper;
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
    sampleLink.textContent = '2D sample';

    header.appendChild(label);
    header.appendChild(sampleLink);

    // Extra 3D sample link
    const s3dHref = sim.taskType === 'regression'
      ? 'samples/regression_3d_sample.csv'
      : 'samples/classification_3d_sample.csv';
    const s3d = document.createElement('a');
    s3d.className  = 'csv-sample-link';
    s3d.href       = s3dHref;
    s3d.download   = '';
    s3d.textContent = '3D sample';
    s3d.style.marginLeft = '6px';
    header.appendChild(s3d);
    wrap.appendChild(header);

    // Status / filename row
    const status = document.createElement('div');
    status.className = 'csv-status';
    if (dataStore.points && dataStore.type === sim.taskType) {
      const tag = dataStore.is3D ? ' · 3D'
        : dataStore.pcaInfo ? ` · PCA(${dataStore.nFeatures}→2)`
        : '';
      status.textContent = `✓ ${dataStore.filename || 'custom data'} (${dataStore.points.length} pts${tag})`;
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
      dataStore.points       = null;
      dataStore.type         = null;
      dataStore.filename     = null;
      dataStore.featureNames = [];
      dataStore.targetName   = null;
      dataStore.nFeatures    = 0;
      dataStore.xLabel       = null;
      dataStore.yLabel       = null;
      dataStore.zLabel       = null;
      dataStore.is3D         = false;
      dataStore.pcaInfo      = null;
      this._dataSourceMode   = 'scenario'; // go back to scenario tab after clear
      this.simulationManager.updateCurrentParams({});
      this.renderDataParams(sim);
    });

    fileRow.appendChild(fileInput);
    fileRow.appendChild(fileLabel);
    fileRow.appendChild(clearBtn);
    wrap.appendChild(fileRow);

    const fmt = document.createElement('p');
    fmt.className = 'csv-format-hint';
    if (sim.taskType === 'regression') {
      fmt.innerHTML = 'Format: <b>f1[,f2,...],target</b><br>'
        + '1 feature → 2D  |  2 features → <b>3D surface</b>  |  3+ → <b>PCA 1D</b>';
    } else {
      fmt.innerHTML = 'Format: <b>f1,f2[,f3,...],label</b><br>'
        + '2 features → 2D  |  3 features → <b>3D scatter</b>  |  4+ → <b>PCA 2D</b>';
    }
    wrap.appendChild(fmt);

    return wrap;
  }

  _handleCSVUpload(file, sim) {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const rows = e.target.result.trim().split('\n')
          .map(r => r.trim()).filter(r => r.length > 0);

        // Detect header: first row is a header if its first cell is non-numeric
        const hasHeader = isNaN(parseFloat(rows[0].split(',')[0].trim()));
        const headerRow = hasHeader ? rows[0].split(',').map(c => c.trim()) : null;
        const dataRows  = rows.slice(hasHeader ? 1 : 0)
          .map(r => r.split(',').map(v => parseFloat(v.trim())));

        // Helper: normalise an array of values to [-1, 1]
        const norm1D = (vals) => {
          const mn = Math.min(...vals), mx = Math.max(...vals);
          return mx === mn ? vals.map(() => 0) : vals.map(v => 2 * (v - mn) / (mx - mn) - 1);
        };

        // Reset metadata
        dataStore.featureNames = [];
        dataStore.targetName   = null;
        dataStore.nFeatures    = 0;
        dataStore.xLabel       = null;
        dataStore.yLabel       = null;
        dataStore.zLabel       = null;
        dataStore.is3D         = false;
        dataStore.pcaInfo      = null;

        if (sim.taskType === 'regression') {
          const nCols = dataRows[0]?.length ?? 0;
          if (nCols < 2) throw new Error('Need at least 2 columns (1 feature + target)');
          const nFeat = nCols - 1; // last column = target

          const raw = dataRows.filter(r =>
            r.length >= nCols && r.every(v => !isNaN(v))
          );
          if (!raw.length) throw new Error('No valid rows');

          const featNames = headerRow
            ? headerRow.slice(0, nFeat)
            : Array.from({ length: nFeat }, (_, i) => `x${i + 1}`);
          const tName = headerRow?.[nFeat] ?? 'y';

          dataStore.featureNames = featNames;
          dataStore.targetName   = tName;
          dataStore.nFeatures    = nFeat;

          const ny = norm1D(raw.map(r => r[nFeat])); // target always last col

          if (nFeat === 1) {
            // ── 2D regression: 1 feature + 1 target ────────────
            const nx = norm1D(raw.map(r => r[0]));
            dataStore.xLabel  = featNames[0];
            dataStore.yLabel  = tName;
            dataStore.points  = nx.map((x, i) => ({ x, y: ny[i] }));
          } else if (nFeat === 2) {
            // ── 3D regression: 2 features + 1 target ───────────
            const nx = norm1D(raw.map(r => r[0]));
            const nz = norm1D(raw.map(r => r[1]));
            dataStore.xLabel  = featNames[0];
            dataStore.zLabel  = featNames[1];
            dataStore.yLabel  = tName; // y-axis = target in 3D
            dataStore.is3D    = true;
            dataStore.points  = nx.map((x, i) => ({ x, y: ny[i], z: nz[i] }));
          } else {
            // ── >2 features: PCA → 1D (keep simplest approach) ─
            const X = raw.map(r => r.slice(0, nFeat));
            const { projected, varExplained } = computePCA(X);
            const pct = varExplained.map(v => `${(v * 100).toFixed(0)}%`);
            // Use first PC as single feature
            const pc1Vals = projected.map(p => p[0]);
            const npc = norm1D(pc1Vals);
            dataStore.xLabel  = `PC1 (${pct[0]})`;
            dataStore.yLabel  = tName;
            dataStore.pcaInfo = { varExplained };
            dataStore.points  = npc.map((x, i) => ({ x, y: ny[i] }));
          }

          dataStore.type     = 'regression';
          dataStore.filename = file.name;

        } else {
          // Classification: last column = label, all others = features
          const nCols = dataRows[0]?.length ?? 0;
          if (nCols < 2) throw new Error('Need at least 2 columns (1 feature + label)');
          const nFeat = nCols - 1;

          const raw = dataRows.filter(r =>
            r.length >= nCols && r.slice(0, nFeat).every(v => !isNaN(v)) && !isNaN(r[nFeat])
          );
          if (!raw.length) throw new Error('No valid rows');

          // Column names
          const featNames = headerRow
            ? headerRow.slice(0, nFeat)
            : Array.from({ length: nFeat }, (_, i) => `x${i + 1}`);
          const tName = headerRow?.[nFeat] ?? 'label';

          dataStore.featureNames = featNames;
          dataStore.targetName   = tName;
          dataStore.nFeatures    = nFeat;

          const labels = raw.map(r => r[nFeat] === 1 ? 1 : 0);

          if (nFeat === 2) {
            // ── 2D: direct mapping ──────────────────────────────
            const nx = norm1D(raw.map(r => r[0]));
            const ny = norm1D(raw.map(r => r[1]));
            dataStore.points = nx.map((x, i) => ({ x, y: ny[i], label: labels[i] }));
            dataStore.xLabel = featNames[0];
            dataStore.yLabel = featNames[1];
            dataStore.is3D   = false;

          } else if (nFeat === 3) {
            // ── 3D: three-feature scatter ───────────────────────
            const nx = norm1D(raw.map(r => r[0]));
            const ny = norm1D(raw.map(r => r[1]));
            const nz = norm1D(raw.map(r => r[2]));
            dataStore.points = nx.map((x, i) => ({ x, y: ny[i], z: nz[i], label: labels[i] }));
            dataStore.xLabel = featNames[0];
            dataStore.yLabel = featNames[1];
            dataStore.zLabel = featNames[2];
            dataStore.is3D   = true;

          } else {
            // ── >3 features: PCA → 2D ───────────────────────────
            const X = raw.map(r => r.slice(0, nFeat));
            const { projected, varExplained } = computePCA(X);
            const pct = varExplained.map(v => `${(v * 100).toFixed(0)}%`);
            dataStore.points   = projected.map(([x, y], i) => ({ x, y, label: labels[i] }));
            dataStore.xLabel   = `PC1 (${pct[0]})`;
            dataStore.yLabel   = `PC2 (${pct[1]})`;
            dataStore.pcaInfo  = { varExplained };
            dataStore.is3D     = false;
          }

          dataStore.type     = 'classification';
          dataStore.filename = file.name;
        }

        this._dataSourceMode = 'csv'; // auto-switch to CSV tab after load
        this.simulationManager.updateCurrentParams({});
        this.renderDataParams(sim);
        this.setStatus('ready');
      } catch (err) {
        alert(`CSV parse error: ${err.message}. Check the format hint below.`);
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
        const lblWrap = document.createElement('div');
        lblWrap.style.cssText = 'display:flex;align-items:center;gap:5px;';
        const lbl = document.createElement('label');
        lbl.textContent = field.label;
        lbl.setAttribute('for', `param-${field.name}`);
        lblWrap.appendChild(lbl);
        const tip = this._buildTooltipIcon(field.description);
        if (tip) lblWrap.appendChild(tip);
        const inp = document.createElement('input');
        inp.id = `param-${field.name}`;
        inp.type = 'checkbox';
        inp.className = 'toggle';
        inp.checked = Boolean(value);
        inp.setAttribute('data-param-name', field.name);
        inp.addEventListener('change', () => this._applyParam(field.name, inp.checked));
        row.appendChild(lblWrap);
        row.appendChild(inp);
        wrapper.appendChild(row);
        box.appendChild(wrapper);
        return;
      }

      if (field.type === 'select') {
        const wrapper = document.createElement('div');
        wrapper.className = 'param-field';
        const lblWrap = document.createElement('div');
        lblWrap.style.cssText = 'display:flex;align-items:center;gap:5px;margin-bottom:4px;';
        const lbl = document.createElement('label');
        lbl.textContent = field.label;
        lbl.setAttribute('for', `param-${field.name}`);
        lbl.style.cssText = 'font-size:.78rem;font-weight:600;color:var(--text-primary);';
        lblWrap.appendChild(lbl);
        const tip = this._buildTooltipIcon(field.description);
        if (tip) lblWrap.appendChild(tip);
        wrapper.appendChild(lblWrap);
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
    const lblWrap = document.createElement('div');
    lblWrap.style.cssText = 'display:flex;align-items:center;gap:5px;';
    const lbl = document.createElement('label');
    lbl.textContent = field.label;
    lbl.setAttribute('for', `param-${field.name}`);
    lblWrap.appendChild(lbl);
    const tip = this._buildTooltipIcon(field.description);
    if (tip) lblWrap.appendChild(tip);
    const valDisp = document.createElement('span');
    valDisp.className = 'param-value-display';
    valDisp.textContent = value;
    header.appendChild(lblWrap);
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
    // ── Train metrics ──────────────────────────────────────────────
    loss:         { label: 'Loss',      desc: 'Error minimised on training data.',                     impact: 'Lower = better fit. Plateau = converged.' },
    accuracy:     { label: 'Accuracy',  desc: 'Fraction correctly classified on training set.',        impact: 'High train + low test accuracy = overfitting.' },
    recall:       { label: 'Recall',    desc: 'True positives / (True pos + False neg).',              impact: 'Critical when missing positives is costly (e.g. disease detection).' },
    precision:    { label: 'Precision', desc: 'True positives / (True pos + False pos).',              impact: 'Critical when false alarms are costly (spam, fraud).' },
    f1:           { label: 'F1',        desc: 'Harmonic mean of Precision and Recall on train set.',   impact: 'Best single metric for imbalanced datasets.' },
    mae:          { label: 'MAE',       desc: 'Mean Absolute Error on training data.',                 impact: 'Interpretable in same units as target. Robust to outliers.' },
    rmse:         { label: 'RMSE',      desc: 'Root Mean Squared Error — penalises large errors.',     impact: 'Sensitive to outliers. Use when large errors are especially bad.' },
    mape:         { label: 'MAPE',      desc: 'Mean Absolute Percentage Error.',                       impact: 'Scale-independent. Compare across datasets.' },
    nmae:         { label: 'NMAE',      desc: 'Normalised MAE relative to mean target value.',         impact: 'Allows comparison when target scale varies.' },
    // ── Test metrics (classification) ──────────────────────────────
    testLoss:      { label: 'Loss',      desc: 'Error on held-out test set.',                           impact: 'Gap vs train loss = overfitting signal. High gap = bad generalisation.' },
    testAccuracy:  { label: 'Accuracy',  desc: 'Fraction correctly classified on test set.',            impact: 'Most reliable accuracy estimate. Match to train accuracy to check overfit.' },
    testRecall:    { label: 'Recall',    desc: 'True positives / (TP + FN) on test set.',               impact: 'Generalisation of recall. Low vs train = model memorised positives.' },
    testPrecision: { label: 'Precision', desc: 'True positives / (TP + FP) on test set.',               impact: 'Generalisation of precision. Large drop = model over-predicts positives.' },
    testF1:        { label: 'F1',        desc: 'Harmonic mean of precision and recall on test set.',    impact: 'Best test summary for imbalanced data. Compare to F1 (Train).' },
    // ── Test metrics (regression) ───────────────────────────────────
    testMAE:       { label: 'MAE',       desc: 'Mean Absolute Error on held-out test set.',             impact: 'Generalisation quality. Large gap vs train MAE = overfitting.' },
    testRMSE:      { label: 'RMSE',      desc: 'Root Mean Squared Error on held-out test set.',         impact: 'Large test RMSE vs train = outlier sensitivity or overfitting.' },
    testMAPE:      { label: 'MAPE',      desc: 'Mean Absolute Percentage Error on held-out test set.',  impact: 'Scale-independent generalisation check.' },
    testNMAE:      { label: 'NMAE',      desc: 'Normalised MAE on held-out test set.',                  impact: 'Comparable across different target scales.' },
  };

  setupMetrics(sim) {
    this.metricKeys = sim.metricKeys || ['loss'];
    if (!this.metricsContainer) return;
    this.metricsContainer.innerHTML = '';

    const allKeys    = this.metricKeys;
    const trainKeys  = allKeys.filter(k => !k.startsWith('test'));
    const testKeys   = allKeys.filter(k => k.startsWith('test'));
    const hasTest    = testKeys.length > 0;

    // ── Train/Test toggle (only when test keys exist) ─────────────
    if (hasTest) {
      const tabRow = document.createElement('div');
      tabRow.className = 'metric-tab-row';

      ['train', 'test'].forEach(mode => {
        const btn = document.createElement('button');
        btn.className  = `metric-tab${mode === 'train' ? ' active' : ''}`;
        btn.dataset.mode = mode;
        btn.textContent = mode === 'train' ? 'Train' : 'Test';
        btn.addEventListener('click', () => {
          tabRow.querySelectorAll('.metric-tab').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          const trainGroup = this.metricsContainer.querySelector('.metrics-train-group');
          const testGroup  = this.metricsContainer.querySelector('.metrics-test-group');
          if (trainGroup) trainGroup.style.display = mode === 'train' ? '' : 'none';
          if (testGroup)  testGroup.style.display  = mode === 'test'  ? '' : 'none';

          // When switching to Test tab → auto-activate show test overlay
          if (mode === 'test') {
            const s = this.simulationManager.current;
            if (s && s.showTestOverlay === false) {
              s.showTestOverlay = true;
              s.renderWithOverlays();
            }
            const chk = document.getElementById('show-test-toggle');
            if (chk) chk.checked = true;
          }
        });
        tabRow.appendChild(btn);
      });
      this.metricsContainer.appendChild(tabRow);
    }

    // ── Build a card for a single metric key ──────────────────────
    const buildCard = (key) => {
      const info = UIController.METRIC_INFO[key] || { label: key.toUpperCase(), desc: '', impact: '' };
      const card = document.createElement('div');
      card.className = 'metric-card';

      const header = document.createElement('div');
      header.className = 'metric-card-header';
      const nameWrap = document.createElement('div');
      nameWrap.className = 'metric-name-wrap';
      const name = document.createElement('span');
      name.className = 'metric-name';
      name.textContent = info.label.toUpperCase();
      nameWrap.appendChild(name);
      const tipText = [info.desc, info.impact].filter(Boolean).join(' — ');
      const tip = this._buildTooltipIcon(tipText);
      if (tip) nameWrap.appendChild(tip);
      const lastVal = document.createElement('span');
      lastVal.className = 'metric-last-value';
      lastVal.setAttribute('data-metric-val', key);
      lastVal.textContent = '—';
      header.appendChild(nameWrap);
      header.appendChild(lastVal);
      card.appendChild(header);

      const canvas = document.createElement('canvas');
      canvas.setAttribute('data-metric', key);
      canvas.width  = 500;
      canvas.height = 120;
      card.appendChild(canvas);
      return card;
    };

    // ── Train group ───────────────────────────────────────────────
    const trainGroup = document.createElement('div');
    trainGroup.className = 'metrics-train-group';
    trainKeys.forEach(key => trainGroup.appendChild(buildCard(key)));
    this.metricsContainer.appendChild(trainGroup);

    // ── Test group (hidden by default) ────────────────────────────
    if (hasTest) {
      const testGroup = document.createElement('div');
      testGroup.className = 'metrics-test-group';
      testGroup.style.display = 'none';
      testKeys.forEach(key => testGroup.appendChild(buildCard(key)));
      this.metricsContainer.appendChild(testGroup);
    }
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
      const isAcc  = ['accuracy', 'f1', 'recall', 'precision', 'testAccuracy', 'testF1'].includes(key);
      const isMape = key === 'mape' || key === 'testMAPE';
      const isLoss = key === 'loss' || key === 'testLoss';

      if (valSpan) {
        const disp = isAcc ? `${(latest * 100).toFixed(1)}%`
          : isMape ? `${latest.toFixed(1)}%`
          : latest.toFixed(4);
        valSpan.textContent = disp;
        valSpan.style.color = isLoss ? '#dc2626' : '#1d4ed8';
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
      const lineColor = isLoss ? '#dc2626' : '#1d4ed8';
      const areaColor = isLoss ? 'rgba(220,38,38,.08)' : 'rgba(29,78,216,.08)';

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
