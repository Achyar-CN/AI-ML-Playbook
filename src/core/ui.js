export class UIController {
  constructor({ controlsPanel, simulationManager, stateManager }) {
    this.controlsPanel = controlsPanel;
    this.simulationManager = simulationManager;
    this.stateManager = stateManager;

    this.statusText = document.getElementById('statusText');
    this.metricsContainer = document.getElementById('metrics');
    this.metricKeys = [];
    this.metricCanvasSize = { width: 420, height: 160 };

    this.controlsPanel.addEventListener('change', (event) => {
      const select = event.target.closest('select[data-sim-select]');
      if (select) {
        const selected = select.value;
        this.stateManager.setState({ sim: selected });
        this.simulationManager.selectSimulation(selected);
        this.setSelectedSim(selected);
      }

      const control = event.target.closest('input[data-param-name]');
      if (control && this.simulationManager.current) {
        const pName = control.getAttribute('data-param-name');
        const rawValue = control.value;
        const value = Number(rawValue);
        this.simulationManager.updateCurrentParams({ [pName]: Number.isNaN(value) ? rawValue : value });
        this.stateManager.setState({ [pName]: this.simulationManager.current.params[pName] });
      }
    });

    this.paramBox = document.getElementById('param-box');
  }

  setSelectedSim(id) {
    const select = this.controlsPanel.querySelector('select[data-sim-select]');
    if (select) {
      select.value = id;
    }
    const sim = this.simulationManager.simulations ? this.simulationManager.simulations.get(id) : null;
    if (sim) {
      this.renderParams(sim);
      this.setupMetrics(sim);
    }
  }

  setupMetrics(sim) {
    this.metricKeys = sim.metricKeys || ['loss', 'accuracy'];
    if (!this.metricsContainer) return;

    this.metricsContainer.innerHTML = '';
    this.metricKeys.forEach((key) => {
      const card = document.createElement('div');
      card.className = 'metric-card';
      const canvas = document.createElement('canvas');
      canvas.setAttribute('data-metric', key);
      canvas.width = this.metricCanvasSize.width;
      canvas.height = this.metricCanvasSize.height;
      card.innerHTML = `<h3>${key.toUpperCase()}</h3>`;
      card.appendChild(canvas);
      this.metricsContainer.appendChild(card);
    });
  }

  renderMenu(simulations) {
    const select = document.createElement('select');
    select.setAttribute('data-sim-select', '');
    simulations.forEach((sim) => {
      const option = document.createElement('option');
      option.value = sim.id;
      option.textContent = sim.title;
      select.appendChild(option);
    });
    this.controlsPanel.insertBefore(select, this.paramBox);

    const controls = document.createElement('div');
    controls.className = 'controls';
    controls.innerHTML = `
      <button id="start-btn">Start</button>
      <button id="pause-btn">Pause</button>
      <button id="reset-btn">Reset</button>
    `;
    this.controlsPanel.insertBefore(controls, this.paramBox);

    controls.querySelector('#start-btn').addEventListener('click', () => {
      this.setStatus('Running');
      this.simulationManager.start();
    });
    controls.querySelector('#pause-btn').addEventListener('click', () => {
      this.setStatus('Paused');
      this.simulationManager.stop();
    });
    controls.querySelector('#reset-btn').addEventListener('click', () => {
      this.setStatus('Ready');
      this.simulationManager.reset();
      this.renderMetrics([]);
    });

    const modelId = this.stateManager.get('sim', simulations[0]?.id);
    this.setSelectedSim(modelId);
  }

  renderParams(sim) {
    this.paramBox.innerHTML = '';

    const params = sim.defaultParams || {};
    const meta = sim.paramControls || [];

    meta.forEach((field) => {
      const value = this.stateManager.get(field.name, params[field.name]);
      const wrapper = document.createElement('div');
      wrapper.className = 'param-field';

      const label = document.createElement('label');
      label.textContent = field.label;
      label.setAttribute('for', `param-${field.name}`);

      const input = document.createElement('input');
      input.id = `param-${field.name}`;
      input.type = field.type || 'number';
      input.value = value;
      input.step = field.step || 'any';
      input.min = field.min ?? '';
      input.max = field.max ?? '';
      input.setAttribute('data-param-name', field.name);

      wrapper.appendChild(label);
      wrapper.appendChild(input);

      if (field.description) {
        const desc = document.createElement('div');
        desc.className = 'description';
        desc.textContent = field.description;
        wrapper.appendChild(desc);
      }

      this.paramBox.appendChild(wrapper);
    });
  }

  setStatus(text) {
    if (this.statusText) {
      this.statusText.textContent = `Status: ${text}`;
    }
  }

  renderMetrics(history) {
    if (!this.metricsContainer) return;

    this.metricKeys.forEach((key) => {
      const canvas = this.metricsContainer.querySelector(`canvas[data-metric="${key}"]`);
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const w = canvas.width;
      const h = canvas.height;
      const padding = 36;

      ctx.clearRect(0, 0, w, h);

      if (!history || history.length === 0) {
        ctx.fillStyle = '#64748b';
        ctx.font = '14px sans-serif';
        ctx.fillText('No data yet', w * 0.35, h / 2);
        return;
      }

      const values = history.map((item) => item[key] !== undefined ? item[key] : 0);
      const maxY = Math.max(...values, 1e-3);
      const minY = Math.min(...values, 0);
      const yRange = maxY - minY || 1;

      // axis lines
      ctx.strokeStyle = '#cbd5e1';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padding, padding);
      ctx.lineTo(padding, h - padding);
      ctx.lineTo(w - padding, h - padding);
      ctx.stroke();

      // tick labels
      ctx.fillStyle = '#334155';
      ctx.font = '11px sans-serif';
      ctx.fillText('Epoch', w * 0.78, h - 10);
      ctx.fillText(key.toUpperCase(), 8, 14);

      const steps = Math.min(history.length, 5);
      for (let i = 0; i <= steps; i += 1) {
        const yVal = minY + ((steps - i) / steps) * yRange;
        const yPos = padding + ((h - 2 * padding) * i / steps);

        ctx.beginPath();
        ctx.strokeStyle = '#e2e8f0';
        ctx.moveTo(padding, yPos);
        ctx.lineTo(w - padding, yPos);
        ctx.stroke();

        ctx.fillStyle = '#475569';
        ctx.fillText(yVal.toFixed(3), 4, yPos + 4);
      }

      const n = history.length;
      ctx.beginPath();
      ctx.strokeStyle = key === 'loss' ? '#ef4444' : '#2563eb';
      ctx.lineWidth = 2.5;

      for (let i = 0; i < n; i += 1) {
        const x = padding + ((w - 2 * padding) * (i) / Math.max(n - 1, 1));
        const y = h - padding - ((values[i] - minY) / yRange) * (h - 2 * padding);

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }

        ctx.fillStyle = '#1d4ed8';
        ctx.beginPath();
        ctx.arc(x, y, 2.2, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.stroke();

      const latest = values[values.length - 1];
      ctx.fillStyle = '#0f172a';
      ctx.font = '12px sans-serif';
      const textSuffix = key === 'accuracy' ? '%' : key === 'mape' ? '%' : '';
      const displayValue = key === 'accuracy' ? (latest * 100) : latest;
      ctx.fillText(`Last: ${displayValue.toFixed(3)}${textSuffix}`, w - padding - 100, padding + 12);
    });
  }
}

