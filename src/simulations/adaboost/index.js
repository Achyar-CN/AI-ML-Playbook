import { BaseSimulation } from '../baseSimulation.js';

export class AdaBoostSimulation extends BaseSimulation {
  setup() {
    this.history = [];
    this.points = [];
    const { nPoints, seed } = this.params;
    this.stumps = [];  // { feature, threshold, polarity, alpha }
    this.epoch = 0;

    for (let i = 0; i < nPoints; i++) {
      const x = this.randomBetween(-1, 1, seed + 10 + i * 2);
      const y = this.randomBetween(-1, 1, seed + 11 + i * 2);
      // Slightly more complex boundary: uses both x and y
      const label = (x + y > 0) ? 1 : -1;
      this.points.push({ x, y, label, weight: 1 / nPoints });
    }
  }

  // --- Algorithm ---

  predictStump(pt, stump) {
    const val = stump.feature === 'x' ? pt.x : pt.y;
    return stump.polarity * (val <= stump.threshold ? 1 : -1);
  }

  fitWeakLearner() {
    let bestError = Infinity;
    let bestStump = null;

    for (const feature of ['x', 'y']) {
      // Data-driven thresholds: midpoints between consecutive unique values
      const vals = [...new Set(this.points.map(pt => pt[feature]))].sort((a, b) => a - b);

      for (let i = 0; i < vals.length - 1; i++) {
        const threshold = (vals[i] + vals[i + 1]) / 2;

        for (const polarity of [1, -1]) {
          let error = 0;
          this.points.forEach(pt => {
            const val = feature === 'x' ? pt.x : pt.y;
            const pred = polarity * (val <= threshold ? 1 : -1);
            if (pred !== pt.label) error += pt.weight;
          });

          if (error < bestError) {
            bestError = error;
            bestStump = { feature, threshold, polarity };
          }
        }
      }
    }

    return { stump: bestStump, error: bestError };
  }

  step() {
    if (this.epoch >= this.params.epochs) return;

    const { stump, error } = this.fitWeakLearner();

    // Skip if error >= 0.5 (worse than random)
    if (!stump || error >= 0.5 - 1e-10) {
      this.epoch++;
      return;
    }

    const alpha = 0.5 * Math.log((1 - error) / (error + 1e-10));
    stump.alpha = alpha;
    this.stumps.push(stump);

    // Update and normalize weights
    let Z = 0;
    this.points.forEach(pt => {
      const pred = this.predictStump(pt, stump);
      pt.weight *= Math.exp(-alpha * pt.label * pred);
      Z += pt.weight;
    });
    this.points.forEach(pt => { pt.weight /= Z; });

    this.epoch++;
    const metrics = this.computeMetrics();
    this.history.push({ epoch: this.epoch, ...metrics });
  }

  predict(x, y) {
    if (this.stumps.length === 0) return 1;
    let score = 0;
    this.stumps.forEach(s => {
      const val = s.feature === 'x' ? x : y;
      const pred = s.polarity * (val <= s.threshold ? 1 : -1);
      score += s.alpha * pred;
    });
    return score >= 0 ? 1 : -1;
  }

  // --- Rendering ---

  render() {
    const { width, height } = this.canvas;
    this.ctx.clearRect(0, 0, width, height);

    // White background
    this.ctx.fillStyle = '#ffffff';
    this.ctx.fillRect(0, 0, width, height);

    // 1. Decision region background
    const grid = 60;
    const cellW = width / grid;
    const cellH = height / grid;
    for (let gx = 0; gx < grid; gx++) {
      for (let gy = 0; gy < grid; gy++) {
        const x = (gx / (grid - 1)) * 2 - 1;
        const y = (gy / (grid - 1)) * 2 - 1;
        const p = this.predict(x, y);
        this.ctx.fillStyle = p === 1
          ? 'rgba(25, 118, 210, 0.14)'
          : 'rgba(229, 57, 53, 0.14)';
        this.ctx.fillRect(gx * cellW, height - (gy + 1) * cellH, cellW + 1, cellH + 1);
      }
    }

    // 2. Stump boundary lines
    const maxW = Math.max(...this.points.map(p => p.weight), 1e-10);
    this.stumps.forEach((s) => {
      const alpha = Math.min(1, 0.15 + (s.alpha / (this.stumps[0]?.alpha || 1)) * 0.45);
      this.ctx.strokeStyle = `rgba(245, 124, 0, ${alpha})`;
      this.ctx.lineWidth = 1.2;
      this.ctx.setLineDash([4, 3]);
      this.ctx.beginPath();

      if (s.feature === 'x') {
        const px = ((s.threshold + 1) / 2) * width;
        this.ctx.moveTo(px, 0);
        this.ctx.lineTo(px, height);
      } else {
        const py = height - ((s.threshold + 1) / 2) * height;
        this.ctx.moveTo(0, py);
        this.ctx.lineTo(width, py);
      }
      this.ctx.stroke();
      this.ctx.setLineDash([]);
    });

    // 3. Data points — radius proportional to sample weight
    this.points.forEach(({ x, y, label, weight }) => {
      const px = ((x + 1) / 2) * width;
      const py = height - ((y + 1) / 2) * height;
      const r = 3.5 + (weight / maxW) * 6.5; // 3.5 to 10 px

      this.ctx.beginPath();
      this.ctx.arc(px, py, r, 0, Math.PI * 2);
      this.ctx.fillStyle = label === 1 ? '#1565c0' : '#c62828';
      this.ctx.globalAlpha = 0.85;
      this.ctx.fill();
      this.ctx.globalAlpha = 1;
      this.ctx.strokeStyle = '#fff';
      this.ctx.lineWidth = 1.2;
      this.ctx.stroke();
    });

    // 4. Info panel (top-left)
    const metrics = this.computeMetrics();
    const lastAlpha = this.stumps.length > 0 ? this.stumps[this.stumps.length - 1].alpha : null;
    const lastStump = this.stumps[this.stumps.length - 1];

    this.ctx.save();
    this.ctx.fillStyle = 'rgba(255,255,255,0.93)';
    this.ctx.beginPath();
    this.ctx.roundRect(8, 8, 230, 100, 6);
    this.ctx.fill();
    this.ctx.strokeStyle = '#d1d5db';
    this.ctx.lineWidth = 1;
    this.ctx.stroke();

    this.ctx.fillStyle = '#1e293b';
    this.ctx.font = 'bold 12px sans-serif';
    this.ctx.textAlign = 'left';
    this.ctx.fillText(`Boosting Round: ${this.epoch} / ${this.params.epochs}`, 18, 28);
    this.ctx.font = '11px sans-serif';
    this.ctx.fillStyle = '#374151';
    this.ctx.fillText(`Weak learners: ${this.stumps.length}`, 18, 46);
    this.ctx.fillText(`Accuracy: ${(metrics.accuracy * 100).toFixed(1)}%`, 18, 62);
    if (lastAlpha !== null) {
      this.ctx.fillText(`Last alpha: ${lastAlpha.toFixed(3)}`, 18, 78);
      this.ctx.fillText(`Last stump: ${lastStump.feature} ${lastStump.polarity > 0 ? '<=' : '>'} ${lastStump.threshold.toFixed(3)}`, 18, 94);
    }
    this.ctx.restore();

    // 5. Confusion matrix (bottom-left)
    const labels = this.points.map(pt => (pt.label === 1 ? 1 : 0));
    const preds  = this.points.map(pt => (this.predict(pt.x, pt.y) === 1 ? 1 : 0));
    this.drawConfusionMatrix(this.ctx, labels, preds, 10, height - 142, 58);

    // 6. Legend: point size explanation (bottom-right)
    this.ctx.save();
    this.ctx.fillStyle = 'rgba(255,255,255,0.92)';
    this.ctx.beginPath();
    this.ctx.roundRect(width - 148, height - 68, 140, 60, 6);
    this.ctx.fill();
    this.ctx.strokeStyle = '#d1d5db';
    this.ctx.lineWidth = 1;
    this.ctx.stroke();

    this.ctx.fillStyle = '#374151';
    this.ctx.font = 'bold 10px sans-serif';
    this.ctx.textAlign = 'left';
    this.ctx.fillText('Point size = weight', width - 138, height - 53);

    [[12, '#1565c0', 'Class +1'], [30, '#c62828', 'Class -1']].forEach(([dy, color, text]) => {
      this.ctx.beginPath();
      this.ctx.arc(width - 132, height - 68 + dy, 5, 0, Math.PI * 2);
      this.ctx.fillStyle = color;
      this.ctx.fill();
      this.ctx.fillStyle = '#374151';
      this.ctx.font = '10px sans-serif';
      this.ctx.fillText(text, width - 122, height - 68 + dy + 4);
    });
    this.ctx.restore();
  }

  computeMetrics() {
    const labels = this.points.map(pt => (pt.label === 1 ? 1 : 0));
    const preds  = this.points.map(pt => (this.predict(pt.x, pt.y) === 1 ? 1 : 0));
    return this.computeClassificationMetrics(labels, preds);
  }
}
