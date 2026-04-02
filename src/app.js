import { SimulationManager } from './core/simulationManager.js';
import { UIController } from './core/ui.js';
import { StateManager } from './core/state.js';
import { simulations } from './config/simulations.js';

export class App {
  constructor({ root, controlsPanel }) {
    this.root = root;
    this.controlsPanel = controlsPanel;
    this.state = new StateManager();
    this.simulationManager = new SimulationManager({ root });
    this.ui = new UIController({ controlsPanel, simulationManager: this.simulationManager, stateManager: this.state, simulations });
  }

  init() {
    this.simulationManager.registerSimulations(simulations);
    this.ui.renderMenu(simulations);

    this.simulationManager.onMetricsUpdate = (history) => {
      this.ui.renderMetrics(history);
    };

    const activeSim = this.state.get('sim', simulations[0]?.id || 'perceptron');

    // Collect any param values saved in URL hash so the simulation starts
    // with the exact same values shown in the sidebar.
    const activeMeta = simulations.find(s => s.id === activeSim);
    const savedParams = {};
    const allControls = [
      ...(activeMeta?.dataParamControls || []),
      ...(activeMeta?.paramControls     || []),
    ];
    allControls.forEach(f => {
      const v = this.state.get(f.name, undefined);
      if (v !== undefined) savedParams[f.name] = v;
    });

    this.simulationManager.selectSimulation(activeSim, savedParams);

    this.state.onUpdate = (state) => {
      if (state.sim && state.sim !== this.simulationManager.currentId) {
        this.simulationManager.selectSimulation(state.sim);
        this.ui.setSelectedSim(state.sim);
      }
    };
  }
}
