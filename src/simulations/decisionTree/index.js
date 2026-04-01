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
      const label = x > 0 ? 1 : 0; // simple split
      this.points.push({ x, y, label });
    }

    this.buildTree();
  }

  buildTree() {
    // Simple decision tree: split on x at 0
    this.tree = {
      feature: 'x',
      threshold: 0,
      left: { label: 0 }, // x <= 0
      right: { label: 1 } // x > 0
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
    // For simplicity, tree is static, no training
    if (this.epoch >= this.params.epochs) return;
    this.epoch += 1;

    const { accuracy } = this.computeMetrics();
    this.history.push({ epoch: this.epoch, accuracy });
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

    // draw decision boundary (vertical line at x=0)
    const px = ((0 + 1) / 2) * width;
    this.ctx.strokeStyle = '#000';
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();
    this.ctx.moveTo(px, 0);
    this.ctx.lineTo(px, height);
    this.ctx.stroke();

    this.ctx.fillStyle = '#333';
    this.ctx.font = '14px sans-serif';
    this.ctx.fillText(`Epoch: ${this.epoch}`, 10, 20);
    const metrics = this.computeMetrics();
    this.ctx.fillText(`Acc: ${(metrics.accuracy * 100).toFixed(1)}%`, 10, 40);
  }

  computeMetrics() {
    let correct = 0;
    this.points.forEach((pt) => {
      const pred = this.predict(pt.x, pt.y);
      if (pred === pt.label) correct += 1;
    });
    const accuracy = correct / this.points.length;
    return { accuracy };
  }
}