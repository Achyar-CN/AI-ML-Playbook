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
    this.canvas.width = 600;
    this.canvas.height = 600;
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

  seededRandom(seed) {
    let x = Math.sin(seed) * 10000;
    return () => {
      x = Math.sin(x) * 10000;
      return x - Math.floor(x);
    };
  }

  randomBetween(min, max, seed) {
    const rand = this.seededRandom(seed);
    return rand() * (max - min) + min;
  }
}
