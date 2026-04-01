import { BaseSimulation } from '../baseSimulation.js';

export class DecisionTreeSimulation extends BaseSimulation {
  setup() {
    this.history = [];
    this.points = [];
    const { nPoints, seed } = this.params;
    this.tree = null;
    this.epoch = 0;

    for (let i = 0; i < nPoints; i += 1) {
      const x = this.randomBetween(-1, 1, seed + 10 + i * 2);
      const y = this.randomBetween(-1, 1, seed + 11 + i * 2);
      const label = x > 0 ? 1 : 0;
      this.points.push({ x, y, label });
    }

    this.currentThreshold = this.randomBetween(-0.8, 0.8, seed + 999);
    this.buildTree(this.currentThreshold);
  }

  buildTree(threshold) {
    this.tree = {
      feature: 'x',
      threshold,
      left: { label: 0 },
      right: { label: 1 }
    };
  }

  predict(x, y) {
    let node = this.tree;
    while (node.left && node.right) {
      if (node.feature === 'x') {
        node = x <= node.threshold ? node.left : node.right;
      } else {
        node = y <= node.threshold ? node.left : node.right;
      }
    }
    return node.label;
  }

  step() {
    if (this.epoch >= this.params.epochs) return;

    // Search best split threshold on x over points
    const xs = Array.from(new Set(this.points.map((pt) => pt.x))).sort((a, b) => a - b);
    let bestThreshold = this.tree.threshold;
    let bestAcc = -1;

    const candidates = [this.tree.threshold, ...xs.slice(1).map((x, i) => (x + xs[i]) / 2)];
    candidates.forEach((threshold) => {
      const acc = this.points.reduce((correct, pt) => {
        const pred = pt.x <= threshold ? 0 : 1;
        return correct + (pred === pt.label ? 1 : 0);
      }, 0) / this.points.length;
      if (acc > bestAcc) {
        bestAcc = acc;
        bestThreshold = threshold;
      }
    });

    // move current threshold gradually toward best candidate so we can animate the boundary
    this.currentThreshold += (bestThreshold - this.currentThreshold) * 0.23;
    this.currentThreshold += (this.randomBetween(-0.01, 0.01, this.epoch + 121) * 0.07);
    this.currentThreshold = Math.max(-1, Math.min(1, this.currentThreshold));

    this.buildTree(this.currentThreshold);
    this.epoch += 1;

    const metrics = this.computeMetrics();
    this.history.push({ epoch: this.epoch, ...metrics });
  }

  render() {
    const { width, height } = this.canvas;
    this.ctx.clearRect(0, 0, width, height);

    // draw points
    this.points.forEach(({ x, y, label }) => {
      const px = ((x + 1) / 2) * width;
      const py = height - ((y + 1) / 2) * height;
      this.ctx.beginPath();
      this.ctx.arc(px, py, 5, 0, Math.PI * 2);
      this.ctx.fillStyle = label === 1 ? '#1976d2' : '#e53935';
      this.ctx.fill();
    });

    const t = this.tree?.threshold ?? 0;
    const px = ((t + 1) / 2) * width;
    this.ctx.strokeStyle = '#0f172a';
    this.ctx.lineWidth = 2.5;
    this.ctx.beginPath();
    this.ctx.moveTo(px, 0);
    this.ctx.lineTo(px, height);
    this.ctx.stroke();

    this.ctx.fillStyle = '#0f172a';
    this.ctx.font = '12px sans-serif';
    this.ctx.fillText(`Threshold: ${t.toFixed(3)}`, 10, 70);

    this.ctx.fillStyle = '#333';
    this.ctx.font = '14px sans-serif';
    this.ctx.fillText(`Epoch: ${this.epoch}`, 10, 20);
    const metrics = this.computeMetrics();
    this.ctx.fillText(`Acc: ${(metrics.accuracy * 100).toFixed(1)}%`, 10, 40);
  }

  computeMetrics() {
    const labels = this.points.map((pt) => (pt.label === 1 ? 1 : 0));
    const preds = this.points.map((pt) => (this.predict(pt.x, pt.y) === 1 ? 1 : 0));
    return this.computeClassificationMetrics(labels, preds);
  }
}