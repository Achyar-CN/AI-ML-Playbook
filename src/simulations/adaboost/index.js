import { BaseSimulation } from '../baseSimulation.js';

export class AdaBoostSimulation extends BaseSimulation {
  setup() {
    this.history = [];
    this.points = [];
    const { nPoints, seed } = this.params;
    this.weights = Array(nPoints).fill(1 / nPoints);
    this.trees = [];
    this.alphas = [];
    this.epoch = 0;

    for (let i = 0; i < nPoints; i += 1) {
      const x = this.randomBetween(-1, 1, seed + 10 + i * 2);
      const y = this.randomBetween(-1, 1, seed + 11 + i * 2);
      const label = (x + y > 0) ? 1 : -1;
      this.points.push({ x, y, label, weight: this.weights[i] });
    }
  }

  fitWeakLearner() {
    // Fit a simple decision stump (1-level decision tree)
    let bestError = 1;
    let bestRule = null;

    // Try splits on x + y threshold
    const thresholds = [-0.8, -0.4, 0, 0.4, 0.8];

    thresholds.forEach((threshold) => {
      let error = 0;
      this.points.forEach((pt) => {
        const pred = (pt.x + pt.y > threshold) ? 1 : -1;
        if (pred !== pt.label) {
          error += pt.weight;
        }
      });

      if (error < bestError && error > 0) {
        bestError = error;
        bestRule = { threshold };
      }
    });

    if (!bestRule) {
      bestRule = { threshold: 0 };
      bestError = 0.5;
    }

    return { rule: bestRule, error: bestError };
  }

  step() {
    if (this.epoch >= this.params.epochs) return;

    const { rule, error } = this.fitWeakLearner();

    if (error >= 0.5) {
      this.epoch += 1;
      return;
    }

    const alpha = 0.5 * Math.log((1 - error) / (error + 1e-10));
    this.trees.push(rule);
    this.alphas.push(alpha);

    // Update weights
    let Z = 0;
    this.points.forEach((pt) => {
      const pred = (pt.x + pt.y > rule.threshold) ? 1 : -1;
      const exponent = -alpha * pt.label * pred;
      pt.weight *= Math.exp(exponent);
      Z += pt.weight;
    });

    // Normalize weights
    this.points.forEach((pt) => {
      pt.weight /= Z;
    });

    this.epoch += 1;

    const metrics = this.computeMetrics();
    this.history.push({ epoch: this.epoch, ...metrics });
  }

  predict(x, y) {
    let score = 0;
    this.trees.forEach((tree, idx) => {
      const pred = (x + y > tree.threshold) ? 1 : -1;
      score += this.alphas[idx] * pred;
    });
    return score > 0 ? 1 : -1;
  }

  render() {
    const { width, height } = this.canvas;
    this.ctx.clearRect(0, 0, width, height);

    // draw background prediction grid
    const grid = 40;
    for (let gx = 0; gx < grid; gx += 1) {
      for (let gy = 0; gy < grid; gy += 1) {
        const x = (gx / (grid - 1)) * 2 - 1;
        const y = (gy / (grid - 1)) * 2 - 1;
        const p = this.predict(x, y);
        this.ctx.fillStyle = p === 1 ? 'rgba(25, 118, 210, 0.15)' : 'rgba(229, 57, 53, 0.15)';
        this.ctx.fillRect((x + 1) / 2 * width, height - (y + 1) / 2 * height, width / grid + 1, height / grid + 1);
      }
    }

    // draw points
    this.points.forEach(({ x, y, label }) => {
      const px = ((x + 1) / 2) * width;
      const py = height - ((y + 1) / 2) * height;
      this.ctx.beginPath();
      this.ctx.arc(px, py, 5, 0, Math.PI * 2);
      this.ctx.fillStyle = label === 1 ? '#1565c0' : '#c62828';
      this.ctx.fill();
      this.ctx.strokeStyle = '#fff';
      this.ctx.lineWidth = 1.5;
      this.ctx.stroke();
    });

    // draw decision boundaries (lines from trees)
    this.ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
    this.ctx.lineWidth = 1;
    this.trees.forEach((tree) => {
      // x + y = threshold → y = threshold - x
      const y1 = tree.threshold - (-1);
      const y2 = tree.threshold - 1;
      const px1 = ((- 1 + 1) / 2) * width;
      const py1 = height - ((y1 + 1) / 2) * height;
      const px2 = ((1 + 1) / 2) * width;
      const py2 = height - ((y2 + 1) / 2) * height;

      this.ctx.beginPath();
      this.ctx.moveTo(px1, py1);
      this.ctx.lineTo(px2, py2);
      this.ctx.stroke();
    });

    this.ctx.fillStyle = '#333';
    this.ctx.font = '14px sans-serif';
    this.ctx.fillText(`Boosting Iteration: ${this.epoch}`, 10, 20);
    this.ctx.fillText(`Trees: ${this.trees.length}`, 10, 40);
    const metrics = this.computeMetrics();
    this.ctx.fillText(`Acc: ${(metrics.accuracy * 100).toFixed(1)}%`, 10, 58);
  }

  computeMetrics() {
    const labels = this.points.map((pt) => (pt.label === 1 ? 1 : 0));
    const preds = this.points.map((pt) => (this.predict(pt.x, pt.y) === 1 ? 1 : 0));
    return this.computeClassificationMetrics(labels, preds);
  }
}
