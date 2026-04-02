import { BaseSimulation } from '../simulations/baseSimulation.js';

export class SimulationManager {
  constructor({ root }) {
    this.root = root;
    this.simulations = new Map();
    this.current = null;
    this.currentId = null;
    this.currentMeta = null;
    this.frame = null;
    this.onMetricsUpdate = null;
    this.stepsPerSecond = 8; // default: 8 steps/second so animation is visible
    this._lastStepTime = 0;
    this._running = false;
  }

  registerSimulations(simulations) {
    simulations.forEach((sim) => {
      this.simulations.set(sim.id, sim);
    });
  }

  selectSimulation(id, initialParams = {}) {
    if (!this.simulations.has(id)) return;

    this.stop();
    this.root.innerHTML = '';

    const meta = this.simulations.get(id);
    // Merge defaultParams with any caller-supplied initialParams (e.g. from URL state)
    const params = { ...meta.defaultParams, ...initialParams };

    const instance = new meta.class({ container: this.root, params });

    if (!(instance instanceof BaseSimulation)) {
      throw new Error(`Simulation ${id} must extend BaseSimulation`);
    }

    this.current = instance;
    this.currentId = id;
    this.currentMeta = meta;
    this.current.init();
    this.current.render(); // show initial state

    if (typeof this.onMetricsUpdate === 'function') {
      this.onMetricsUpdate(this.current.history || []);
    }

    return this.current;
  }

  updateCurrentParams(params) {
    if (!this.current) return;
    this.current.params = { ...this.current.params, ...params };
    const wasRunning = this._running;
    this.stop();
    this.current.reset();
    this.current.render();
    if (typeof this.onMetricsUpdate === 'function') {
      this.onMetricsUpdate([]);
    }
    if (wasRunning) this.start();
  }

  setSpeed(stepsPerSecond) {
    this.stepsPerSecond = Math.max(1, Math.min(120, stepsPerSecond));
  }

  start() {
    if (!this.current || this._running) return;
    this._running = true;
    this._lastStepTime = 0;

    const stepInterval = 1000 / this.stepsPerSecond;

    const loop = (timestamp) => {
      if (!this._running) return;

      // Step only when enough time has elapsed (throttle to stepsPerSecond)
      if (this._lastStepTime === 0 || timestamp - this._lastStepTime >= stepInterval) {
        this.current.step();
        this._lastStepTime = timestamp;
        if (typeof this.onMetricsUpdate === 'function') {
          this.onMetricsUpdate(this.current.history || []);
        }
      }

      // Render every frame (smooth canvas)
      this.current.render();
      this.frame = requestAnimationFrame(loop);
    };

    this.frame = requestAnimationFrame(loop);
  }

  stop() {
    this._running = false;
    if (this.frame) {
      cancelAnimationFrame(this.frame);
      this.frame = null;
    }
  }

  reset() {
    const wasRunning = this._running;
    this.stop();
    if (this.current) {
      this.current.reset();
      this.current.render();
      if (typeof this.onMetricsUpdate === 'function') {
        this.onMetricsUpdate([]);
      }
    }
    if (wasRunning) this.start();
  }
}
