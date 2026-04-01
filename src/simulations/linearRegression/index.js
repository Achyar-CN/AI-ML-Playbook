import { BaseSimulation } from '../baseSimulation.js';

export class LinearRegressionSimulation extends BaseSimulation {
  setup() {
    this.history = [];
    this.points = [];
    const { nPoints, seed } = this.params;
    this.m = 0; // slope
    this.b = 0; // intercept
    this.epoch = 0;

    for (let i = 0; i < nPoints; i += 1) {
      const x = this.randomBetween(-1, 1, seed + 10 + i * 2);
      const y = this.randomBetween(-1, 1, seed + 11 + i * 2);
      this.points.push({ x, y });
    }
  }

  predict(x) {
    return this.m * x + this.b;
  }

  step() {
    if (this.epoch >= this.params.epochs) return;

    const lr = this.params.learningRate;
    let dm = 0;
    let db = 0;

    this.points.forEach((pt) => {
      const pred = this.predict(pt.x);
      const error = pred - pt.y;
      dm += error * pt.x;
      db += error;
    });

    const n = this.points.length;
    this.m -= lr * (dm / n);
    this.b -= lr * (db / n);

    this.epoch += 1;

    const { loss } = this.computeMetrics();
    this.history.push({ epoch: this.epoch, loss });
  }

  render() {
    const { width, height } = this.canvas;
    this.ctx.clearRect(0, 0, width, height);

    // draw points
    this.points.forEach(({ x, y }) => {
      const px = ((x + 1) / 2) * width;
      const py = height - ((y + 1) / 2) * height;
      this.ctx.beginPath();
      this.ctx.arc(px, py, 5, 0, Math.PI * 2);
      this.ctx.fillStyle = '#1976d2';
      this.ctx.fill();
    });

    // draw regression line
    this.ctx.strokeStyle = '#e53935';
    this.ctx.lineWidth = 3;
    this.ctx.beginPath();
    const x1 = -1;
    const y1 = this.predict(x1);
    const px1 = ((x1 + 1) / 2) * width;
    const py1 = height - ((y1 + 1) / 2) * height;
    this.ctx.moveTo(px1, py1);

    const x2 = 1;
    const y2 = this.predict(x2);
    const px2 = ((x2 + 1) / 2) * width;
    const py2 = height - ((y2 + 1) / 2) * height;
    this.ctx.lineTo(px2, py2);
    this.ctx.stroke();

    this.ctx.fillStyle = '#333';
    this.ctx.font = '14px sans-serif';
    this.ctx.fillText(`Epoch: ${this.epoch}`, 10, 20);
    const metrics = this.computeMetrics();
    this.ctx.fillText(`Loss: ${metrics.loss.toFixed(4)}`, 10, 40);
  }

  computeMetrics() {
    let loss = 0;
    this.points.forEach((pt) => {
      const pred = this.predict(pt.x);
      loss += (pred - pt.y) ** 2;
    });
    loss /= this.points.length;
    return { loss };
  }
}