import { BaseSimulation } from '../simulations/baseSimulation.js';

export class SimulationManager {
  constructor({ root }) {
    this.root = root;
    this.simulations = new Map();
    this.current = null;
    this.frame = null;
    this.onMetricsUpdate = null;
  }

  registerSimulations(simulations) {
    simulations.forEach((sim) => {
      this.simulations.set(sim.id, sim);
    });
  }

  selectSimulation(id) {
    if (!this.simulations.has(id)) {
      throw new Error(`Simulation ${id} not registered`);
    }

    this.stop();
    this.root.innerHTML = '';

    const meta = this.simulations.get(id);
    const instance = new meta.class({ container: this.root, params: meta.defaultParams });

    if (!(instance instanceof BaseSimulation)) {
      throw new Error(`Simulation ${id} must extend BaseSimulation`);
    }

    this.current = instance;
    this.currentId = id;
    this.currentMeta = meta;
    this.current.init();

    if (typeof this.onMetricsUpdate === 'function') {
      this.onMetricsUpdate(this.current.history || []);
    }

    return this.current;
  }

  updateCurrentParams(params) {
    if (!this.current) return;
    this.current.params = { ...this.current.params, ...params };
    if (typeof this.current.reset === 'function') {
      this.current.reset();
    }
  }

  start() {
    if (!this.current) return;
    const loop = () => {
      this.current.step();
      this.current.render();

      if (typeof this.onMetricsUpdate === 'function') {
        this.onMetricsUpdate(this.current.history || []);
      }

      this.frame = requestAnimationFrame(loop);
    };
    this.frame = requestAnimationFrame(loop);
  }

  stop() {
    if (this.frame) {
      cancelAnimationFrame(this.frame);
      this.frame = null;
    }
  }

  reset() {
    if (this.current) {
      this.current.reset();
    }
  }
}
