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
    return { loss: 0 };
  }

  computeClassificationMetrics(labels, preds) {
    const n = labels.length || 1;
    let tp = 0;
    let tn = 0;
    let fp = 0;
    let fn = 0;

    labels.forEach((trueLabel, i) => {
      const pred = preds[i];
      if (trueLabel === 1 && pred === 1) tp += 1;
      if (trueLabel === 0 && pred === 0) tn += 1;
      if (trueLabel === 0 && pred === 1) fp += 1;
      if (trueLabel === 1 && pred === 0) fn += 1;
    });

    const accuracy = (tp + tn) / n;
    const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
    const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
    const f1 = (precision + recall) === 0 ? 0 : (2 * precision * recall) / (precision + recall);
    const loss = 1 - accuracy;

    return { accuracy, recall, precision, f1, loss };
  }

  computeRegressionMetrics(trueValues, preds) {
    const n = trueValues.length || 1;
    let se = 0;
    let ae = 0;
    let ape = 0;
    let sumTrue = 0;

    trueValues.forEach((trueValue, i) => {
      const p = preds[i];
      const err = p - trueValue;
      se += err * err;
      ae += Math.abs(err);
      ape += trueValue === 0 ? 0 : Math.abs(err / trueValue);
      sumTrue += Math.abs(trueValue);
    });

    const mse = se / n;
    const rmse = Math.sqrt(mse);
    const mae = ae / n;
    const mape = (ape / n) * 100;
    const nmae = sumTrue === 0 ? 0 : mae / (sumTrue / n);
    const loss = mse;

    return { loss, mape, mae, rmse, nmae };
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
