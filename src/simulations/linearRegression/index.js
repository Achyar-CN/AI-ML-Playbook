import { BaseSimulation } from '../baseSimulation.js';

export class LinearRegressionSimulation extends BaseSimulation {
  setup() {
    this.history = [];
    this.epoch   = 0;
    const { nPoints, seed, noiseLevel, datasetType, degree } = this.params;
    const deg = Math.max(1, Math.round(degree || 1));

    this.points = this.generateRegressionDataset(datasetType || 'linear', nPoints, seed, noiseLevel ?? 0.25);

    // Weights: one per polynomial feature [x^0, x^1, ..., x^deg]
    this.weights = Array.from({ length: deg+1 }, (_, i) =>
      this.randomBetween(-0.1, 0.1, seed + i + 10)
    );
  }

  _polyFeatures(x) {
    // [1, x, x^2, ..., x^deg]
    return this.weights.map((_, i) => Math.pow(x, i));
  }

  predict(x) {
    const feats = this._polyFeatures(x);
    return this.weights.reduce((s, w, i) => s + w * feats[i], 0);
  }

  step() {
    if (this.epoch >= this.params.epochs) return;
    const lr = this.params.learningRate;
    const l2 = this.params.l2 || 0;
    const n  = this.points.length;
    const grads = new Array(this.weights.length).fill(0);

    this.points.forEach(pt => {
      const feats = this._polyFeatures(pt.x);
      const err   = this.predict(pt.x) - pt.y;
      feats.forEach((f, i) => { grads[i] += err * f; });
    });

    this.weights.forEach((w, i) => {
      this.weights[i] -= lr * (grads[i]/n + l2 * w);
    });

    this.epoch++;
    this.history.push({ epoch: this.epoch, ...this.computeMetrics() });
  }

  _drawCurve(W, H) {
    const steps = 200;
    this.ctx.strokeStyle = '#dc2626'; this.ctx.lineWidth = 2.5;
    this.ctx.beginPath();
    let started = false;
    for (let i = 0; i <= steps; i++) {
      const x  = -1 + (i/steps)*2;
      const y  = this.predict(x);
      const px = ((x+1)/2)*W;
      const py = H - ((y+1)/2)*H;
      if (py < -10 || py > H+10) { started = false; continue; } // clip
      started ? this.ctx.lineTo(px, py) : this.ctx.moveTo(px, py);
      started = true;
    }
    this.ctx.stroke();
  }

  render() {
    const { width: W, height: H } = this.canvas;
    this.ctx.clearRect(0, 0, W, H);
    this.ctx.fillStyle = '#fff'; this.ctx.fillRect(0, 0, W, H);

    // Axis guides
    this.ctx.strokeStyle = '#f1f5f9'; this.ctx.lineWidth = 1;
    this.ctx.beginPath();
    this.ctx.moveTo(W/2,0); this.ctx.lineTo(W/2,H);
    this.ctx.moveTo(0,H/2); this.ctx.lineTo(W,H/2);
    this.ctx.stroke();

    // Residual lines
    this.points.forEach(({ x, y }) => {
      const px    = ((x+1)/2)*W;
      const py    = H-((y+1)/2)*H;
      const yHat  = this.predict(x);
      const pyHat = H-((yHat+1)/2)*H;
      this.ctx.strokeStyle = y > yHat ? 'rgba(29,78,216,.35)' : 'rgba(220,38,38,.35)';
      this.ctx.lineWidth = 1.5;
      this.ctx.beginPath(); this.ctx.moveTo(px, py); this.ctx.lineTo(px, pyHat); this.ctx.stroke();
    });

    // Fitted curve
    this._drawCurve(W, H);

    // Data points
    this.points.forEach(({ x, y }) => {
      const px = ((x+1)/2)*W, py = H-((y+1)/2)*H;
      this.ctx.beginPath(); this.ctx.arc(px, py, 4, 0, Math.PI*2);
      this.ctx.fillStyle = '#1d4ed8'; this.ctx.fill();
      this.ctx.strokeStyle = '#fff'; this.ctx.lineWidth = 1; this.ctx.stroke();
    });

    // Info panel
    const m   = this.computeMetrics();
    const r2  = this._r2();
    const deg = Math.round(this.params.degree || 1);
    const eqParts = this.weights.map((w, i) => {
      if (i===0) return w.toFixed(2);
      if (i===1) return `${w>=0?'+':''}${w.toFixed(2)}x`;
      return `${w>=0?'+':''}${w.toFixed(2)}x^${i}`;
    });
    this.ctx.save();
    this.ctx.fillStyle = 'rgba(255,255,255,.93)';
    this.ctx.beginPath(); this.ctx.roundRect(8, 8, 248, 94, 6); this.ctx.fill();
    this.ctx.strokeStyle = '#e2e8f0'; this.ctx.lineWidth = 1; this.ctx.stroke();
    const lines = [
      `Epoch: ${this.epoch} / ${this.params.epochs}`,
      `Degree: ${deg}  |  L2: ${(this.params.l2||0).toFixed(3)}`,
      `MSE: ${m.loss.toFixed(5)}  |  R²: ${r2.toFixed(4)}`,
      `ŷ = ${eqParts.join(' ')}`,
    ];
    lines.forEach((line, i) => {
      this.ctx.font      = i===0 ? 'bold 12px sans-serif' : '11px sans-serif';
      this.ctx.fillStyle = i===0 ? '#1e293b' : '#374151';
      this.ctx.textAlign = 'left';
      this.ctx.fillText(line, 18, 26+i*17);
    });
    this.ctx.restore();

    // Residual legend (bottom-right)
    this.ctx.save();
    this.ctx.fillStyle = 'rgba(255,255,255,.92)';
    this.ctx.beginPath(); this.ctx.roundRect(W-162, H-54, 154, 46, 6); this.ctx.fill();
    this.ctx.strokeStyle = '#e2e8f0'; this.ctx.lineWidth = 1; this.ctx.stroke();
    this.ctx.font = 'bold 10px sans-serif'; this.ctx.fillStyle = '#374151'; this.ctx.textAlign = 'left';
    this.ctx.fillText('Residuals', W-150, H-38);
    [['rgba(29,78,216,.7)', 'y > ŷ (above)'], ['rgba(220,38,38,.7)', 'y < ŷ (below)']].forEach(([color, label], i) => {
      this.ctx.strokeStyle = color; this.ctx.lineWidth = 2;
      this.ctx.beginPath(); this.ctx.moveTo(W-150, H-24+i*12); this.ctx.lineTo(W-132, H-24+i*12); this.ctx.stroke();
      this.ctx.fillStyle = '#475569'; this.ctx.font = '10px sans-serif';
      this.ctx.fillText(label, W-128, H-20+i*12);
    });
    this.ctx.restore();
  }

  _r2() {
    const ys   = this.points.map(pt => pt.y);
    const preds = this.points.map(pt => this.predict(pt.x));
    const mean = ys.reduce((s,v)=>s+v,0)/(ys.length||1);
    let ssTot=0, ssRes=0;
    ys.forEach((v,i) => { ssTot += (v-mean)**2; ssRes += (v-preds[i])**2; });
    return ssTot===0 ? 0 : 1 - ssRes/ssTot;
  }

  computeMetrics() {
    const trueVals = this.points.map(pt => pt.y);
    const preds    = this.points.map(pt => this.predict(pt.x));
    return this.computeRegressionMetrics(trueVals, preds);
  }
}
