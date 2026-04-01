export class BaseSimulation {
  constructor({ container, params = {} }) {
    this.container = container;
    this.params = params;
    this.canvas = null;
    this.ctx = null;
    this.history = [];
  }

  init() {
    this.container.innerHTML = '';
    this.canvas = document.createElement('canvas');
    this.canvas.width = 620;
    this.canvas.height = 520;
    this.container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d');
    this.setup();
  }

  setup() {
    throw new Error('setup() not implemented');
  }

  reset() {
    this.setup();
  }

  step() {
    throw new Error('step() not implemented');
  }

  render() {
    throw new Error('render() not implemented');
  }

  computeMetrics() {
    return { accuracy: 0, loss: 0 };
  }
}
