export class UIController {
  constructor({ sidebar, simulationManager, stateManager }) {
    this.sidebar = sidebar;
    this.simulationManager = simulationManager;
    this.stateManager = stateManager;

    this.statusText = document.getElementById('statusText');
    this.accuracyCanvas = document.getElementById('accuracy-chart');
    this.lossCanvas = document.getElementById('loss-chart');

    this.sidebar.addEventListener('change', (event) => {
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

    this.paramBox = document.createElement('div');
    this.paramBox.className = 'param-box';
    this.sidebar.appendChild(this.paramBox);
  }

  setSelectedSim(id) {
    const select = this.sidebar.querySelector('select[data-sim-select]');
    if (select) {
      select.value = id;
    }
    const sim = this.simulationManager.simulations ? this.simulationManager.simulations.get(id) : null;
    if (sim) {
      this.renderParams(sim);
    }
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
    this.sidebar.appendChild(select);

    const controls = document.createElement('div');
    controls.className = 'controls';
    controls.innerHTML = `
      <button id="start-btn">Start</button>
      <button id="pause-btn">Pause</button>
      <button id="reset-btn">Reset</button>
      <button id="random-btn">New Data</button>
    `;
    this.sidebar.appendChild(controls);

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
    controls.querySelector('#random-btn').addEventListener('click', () => {
      this.setStatus('New dataset');
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
      this.paramBox.appendChild(wrapper);
    });
  }

  setStatus(text) {
    if (this.statusText) {
      this.statusText.textContent = `Status: ${text}`;
    }
  }

  renderMetrics(history) {
    if (!this.accuracyCanvas || !this.lossCanvas) return;

    const store = { accuracy: this.accuracyCanvas, loss: this.lossCanvas };

    const scaleData = (values, { min, max }) =>
      values.map((val) => ((val - min) / (max - min || 1)) * (80))
    ;

    ['accuracy', 'loss'].forEach((key) => {
      const canvas = store[key];
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (!history.length) {
        ctx.fillStyle = '#66788e';
        ctx.font = '12px sans-serif';
        ctx.fillText('No data yet', 10, 50);
        return;
      }

      const values = history.map((item) => item[key]);
      const min = Math.min(...values);
      const max = Math.max(...values);
      const scaled = scaleData(values, { min, max });

      // axis
      ctx.strokeStyle = '#d5e0ea';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, canvas.height - 18);
      ctx.lineTo(canvas.width, canvas.height - 18);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(35, 0);
      ctx.lineTo(35, canvas.height);
      ctx.stroke();

      // line
      ctx.strokeStyle = key === 'accuracy' ? '#2f6ee7' : '#d46f17';
      ctx.lineWidth = 2;
      ctx.beginPath();
      scaled.forEach((sv, idx) => {
        const x = 35 + ((canvas.width - 40) / Math.max(scaled.length - 1, 1)) * idx;
        const y = canvas.height - 18 - sv;
        if (idx === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();

      // values text
      ctx.fillStyle = '#495a74';
      ctx.font = '11px sans-serif';
      ctx.fillText(`${key === 'accuracy' ? 'Acc' : 'Loss'} min: ${min.toFixed(3)}`, 40, 14);
      ctx.fillText(`max: ${max.toFixed(3)}`, 40, 28);
    });
  }
}

