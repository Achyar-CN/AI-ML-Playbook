export class UIController {
  constructor({ sidebar, simulationManager, stateManager }) {
    this.sidebar = sidebar;
    this.simulationManager = simulationManager;
    this.stateManager = stateManager;

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
    `;
    this.sidebar.appendChild(controls);

    controls.querySelector('#start-btn').addEventListener('click', () => this.simulationManager.start());
    controls.querySelector('#pause-btn').addEventListener('click', () => this.simulationManager.stop());
    controls.querySelector('#reset-btn').addEventListener('click', () => this.simulationManager.reset());

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
}

