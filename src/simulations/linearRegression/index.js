import { BaseSimulation } from '../baseSimulation.js';

export class LinearRegressionSimulation extends BaseSimulation {
  setup() {
    this.history = [];
    this.points = [];
    const { nPoints, seed, noise } = this.params;
    this.m = 0;
    this.b = 0;
    this.epoch = 0;

    // Generate data following a noisy linear pattern: y = 0.65*x + 0.1 + noise
    const trueSlope = 0.65;
    const trueIntercept = 0.1;
    const noiseLevel = noise !== undefined ? noise : 0.3;

    for (let i = 0; i < nPoints; i++) {
      const x = this.randomBetween(-1, 1, seed + 10 + i * 2);
      const n = this.randomBetween(-noiseLevel, noiseLevel, seed + 11 + i * 2);
      const y = Math.max(-1, Math.min(1, trueSlope * x + trueIntercept + n));
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

    this.epoch++;
    const metrics = this.computeMetrics();
    this.history.push({ epoch: this.epoch, ...metrics });
  }

  render() {
    const { width, height } = this.canvas;
    this.ctx.clearRect(0, 0, width, height);

    // White background
    this.ctx.fillStyle = '#ffffff';
    this.ctx.fillRect(0, 0, width, height);

    // Axis lines (center cross)
    this.ctx.strokeStyle = '#e2e8f0';
    this.ctx.lineWidth = 1;
    this.ctx.beginPath();
    this.ctx.moveTo(width / 2, 0);
    this.ctx.lineTo(width / 2, height);
    this.ctx.moveTo(0, height / 2);
    this.ctx.lineTo(width, height / 2);
    this.ctx.stroke();

    // 1. Residual lines (vertical from point to regression line)
    this.points.forEach(({ x, y }) => {
      const px  = ((x + 1) / 2) * width;
      const py  = height - ((y + 1) / 2) * height;
      const yHat = this.predict(x);
      const pyHat = height - ((yHat + 1) / 2) * height;

      const residual = y - yHat;
      this.ctx.strokeStyle = residual > 0
        ? 'rgba(37, 99, 235, 0.35)'
        : 'rgba(220, 38, 38, 0.35)';
      this.ctx.lineWidth = 1.5;
      this.ctx.beginPath();
      this.ctx.moveTo(px, py);
      this.ctx.lineTo(px, pyHat);
      this.ctx.stroke();
    });

    // 2. Regression line
    const x1 = -1, x2 = 1;
    const y1 = this.predict(x1);
    const y2 = this.predict(x2);
    this.ctx.strokeStyle = '#dc2626';
    this.ctx.lineWidth = 2.5;
    this.ctx.beginPath();
    this.ctx.moveTo(((x1 + 1) / 2) * width, height - ((y1 + 1) / 2) * height);
    this.ctx.lineTo(((x2 + 1) / 2) * width, height - ((y2 + 1) / 2) * height);
    this.ctx.stroke();

    // 3. Data points (on top of residuals)
    this.points.forEach(({ x, y }) => {
      const px = ((x + 1) / 2) * width;
      const py = height - ((y + 1) / 2) * height;
      this.ctx.beginPath();
      this.ctx.arc(px, py, 4, 0, Math.PI * 2);
      this.ctx.fillStyle = '#1d4ed8';
      this.ctx.fill();
      this.ctx.strokeStyle = '#fff';
      this.ctx.lineWidth = 1;
      this.ctx.stroke();
    });

    // 4. Info panel (top-left)
    const metrics = this.computeMetrics();
    const r2 = this.computeR2();
    const bSign = this.b >= 0 ? '+' : '';

    this.ctx.save();
    this.ctx.fillStyle = 'rgba(255,255,255,0.93)';
    this.ctx.beginPath();
    this.ctx.roundRect(8, 8, 240, 100, 6);
    this.ctx.fill();
    this.ctx.strokeStyle = '#d1d5db';
    this.ctx.lineWidth = 1;
    this.ctx.stroke();

    this.ctx.fillStyle = '#1e293b';
    this.ctx.font = 'bold 12px sans-serif';
    this.ctx.textAlign = 'left';
    this.ctx.fillText(`Epoch: ${this.epoch} / ${this.params.epochs}`, 18, 28);
    this.ctx.font = '11px sans-serif';
    this.ctx.fillStyle = '#374151';
    this.ctx.fillText(`ŷ = ${this.m.toFixed(3)}x ${bSign}${this.b.toFixed(3)}`, 18, 46);
    this.ctx.fillText(`MSE (loss): ${metrics.loss.toFixed(5)}`, 18, 62);
    this.ctx.fillText(`MAE: ${metrics.mae.toFixed(5)}`, 18, 78);
    this.ctx.fillText(`R²: ${r2.toFixed(4)}`, 18, 94);
    this.ctx.restore();

    // 5. Residual legend (bottom-right)
    this.ctx.save();
    this.ctx.fillStyle = 'rgba(255,255,255,0.92)';
    this.ctx.beginPath();
    this.ctx.roundRect(width - 170, height - 58, 162, 50, 6);
    this.ctx.fill();
    this.ctx.strokeStyle = '#d1d5db';
    this.ctx.lineWidth = 1;
    this.ctx.stroke();

    this.ctx.fillStyle = '#374151';
    this.ctx.font = 'bold 10px sans-serif';
    this.ctx.textAlign = 'left';
    this.ctx.fillText('Residuals', width - 158, height - 43);

    this.ctx.strokeStyle = 'rgba(37,99,235,0.7)';
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();
    this.ctx.moveTo(width - 158, height - 28);
    this.ctx.lineTo(width - 140, height - 28);
    this.ctx.stroke();
    this.ctx.fillStyle = '#374151';
    this.ctx.font = '10px sans-serif';
    this.ctx.fillText('Above line (y > ŷ)', width - 136, height - 24);

    this.ctx.strokeStyle = 'rgba(220,38,38,0.7)';
    this.ctx.beginPath();
    this.ctx.moveTo(width - 158, height - 14);
    this.ctx.lineTo(width - 140, height - 14);
    this.ctx.stroke();
    this.ctx.fillText('Below line (y < ŷ)', width - 136, height - 10);
    this.ctx.restore();
  }

  computeR2() {
    const trueValues = this.points.map(pt => pt.y);
    const preds = this.points.map(pt => this.predict(pt.x));
    const mean = trueValues.reduce((s, v) => s + v, 0) / (trueValues.length || 1);
    let ssTot = 0, ssRes = 0;
    trueValues.forEach((v, i) => {
      ssTot += (v - mean) ** 2;
      ssRes += (v - preds[i]) ** 2;
    });
    return ssTot === 0 ? 0 : 1 - ssRes / ssTot;
  }

  computeMetrics() {
    const trueValues = this.points.map(pt => pt.y);
    const preds = this.points.map(pt => this.predict(pt.x));
    return this.computeRegressionMetrics(trueValues, preds);
  }
}
