import { SimulationManager } from './core/simulationManager.js';
import { UIController } from './core/ui.js';
import { StateManager } from './core/state.js';
import { simulations } from './config/simulations.js';

export class App {
  constructor({ root, sidebar }) {
    this.root = root;
    this.sidebar = sidebar;
    this.state = new StateManager();
    this.simulationManager = new SimulationManager({ root });
    this.ui = new UIController({ sidebar, simulationManager: this.simulationManager, stateManager: this.state });
  }

  init() {
    this.simulationManager.registerSimulations(simulations);
    this.ui.renderMenu(simulations);

    const activeSim = this.state.get('sim', 'perceptron');
    this.simulationManager.selectSimulation(activeSim);

    this.state.onUpdate = (state) => {
      if (state.sim && state.sim !== this.simulationManager.currentId) {
        this.simulationManager.selectSimulation(state.sim);
        this.ui.setSelectedSim(state.sim);
      }
    };
  }
}
