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
    this.stepsPerSecond = 8;
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

    const meta   = this.simulations.get(id);
    const params = { ...meta.defaultParams, ...initialParams };

    // Pass taskType so BaseSimulation can compute test metrics correctly
    const instance = new meta.class({ container: this.root, params, taskType: meta.taskType });

    if (!(instance instanceof BaseSimulation)) {
      throw new Error(`Simulation ${id} must extend BaseSimulation`);
    }

    this.current     = instance;
    this.currentId   = id;
    this.currentMeta = meta;
    this.current.init();
    this.current.renderWithOverlays();

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
    this.current.renderWithOverlays();
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

      if (this._lastStepTime === 0 || timestamp - this._lastStepTime >= stepInterval) {
        this.current.step();
        // Inject test-set metrics into the freshly pushed history entry
        this.current._injectTestMetrics();
        this._lastStepTime = timestamp;
        if (typeof this.onMetricsUpdate === 'function') {
          this.onMetricsUpdate(this.current.history || []);
        }
      }

      // Auto-rotate bubble chart when running (user can still drag to override)
      if (this.current._isBubbleChart && !this.current._3dDrag) {
        this.current._3dRotY = ((this.current._3dRotY ?? 0.62) + 0.006) % (Math.PI * 2);
      }

      this.current.renderWithOverlays();
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
      this.current.renderWithOverlays();
      if (typeof this.onMetricsUpdate === 'function') {
        this.onMetricsUpdate([]);
      }
    }
    if (wasRunning) this.start();
  }
}
