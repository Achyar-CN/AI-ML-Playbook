export class UIController {
  constructor({ controlsPanel, simulationManager, stateManager }) {
    this.controlsPanel = controlsPanel;
    this.simulationManager = simulationManager;
    this.stateManager = stateManager;

    this.statusText = document.getElementById('statusText');
    this.metricsContainer = document.getElementById('metrics');
    this.metricKeys = ['accuracy', 'loss'];

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
      card.innerHTML = `
        <h3>${key.toUpperCase()}</h3>
        <div class="metric-value" data-metric="${key}">n/a</div>
      `;
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

    if (!history || !history.length) {
      this.metricsContainer.querySelectorAll('.metric-value').forEach((valueEl) => {
        valueEl.textContent = 'n/a';
      });
      return;
    }

    const latest = history[history.length - 1];

    this.metricKeys.forEach((key) => {
      const valueEl = this.metricsContainer.querySelector(`.metric-value[data-metric="${key}"]`);
      if (!valueEl) return;

      const value = latest[key];
      if (value === undefined || value === null || Number.isNaN(value)) {
        valueEl.textContent = 'n/a';
      } else {
        const format = ['loss', 'mape', 'mae', 'rmse', 'nmae'].includes(key) ? 4 : 3;
        const suffix = key === 'mape' ? '%' : key === 'accuracy' ? '%' : '';
        valueEl.textContent = `${(key === 'accuracy' ? value * 100 : value).toFixed(format)}${suffix}`;
      }
    });
  }
}

