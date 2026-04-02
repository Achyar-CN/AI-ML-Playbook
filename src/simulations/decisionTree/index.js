import { BaseSimulation } from '../baseSimulation.js';

export class DecisionTreeSimulation extends BaseSimulation {
  setup() {
    this.history = [];
    this.points = [];
    const { nPoints, seed } = this.params;
    this.tree = null;
    this.epoch = 0;

    for (let i = 0; i < nPoints; i++) {
      const x = this.randomBetween(-1, 1, seed + 10 + i * 2);
      const y = this.randomBetween(-1, 1, seed + 11 + i * 2);
      const noise = this.randomBetween(-0.28, 0.28, seed + 1000 + i);
      const label = (x + 0.6 * y + noise > 0) ? 1 : 0;
      this.points.push({ x, y, label });
    }
  }

  // --- Algorithm helpers ---

  entropy(data) {
    if (data.length === 0) return 0;
    let c1 = 0;
    data.forEach(pt => { if (pt.label === 1) c1++; });
    const c0 = data.length - c1;
    let h = 0;
    if (c0 > 0) { const p = c0 / data.length; h -= p * Math.log2(p); }
    if (c1 > 0) { const p = c1 / data.length; h -= p * Math.log2(p); }
    return h;
  }

  majorityLabel(data) {
    let c1 = 0;
    data.forEach(pt => { if (pt.label === 1) c1++; });
    return c1 * 2 >= data.length ? 1 : 0;
  }

  findBestSplit(data) {
    const parentEntropy = this.entropy(data);
    let bestGain = 1e-10; // min gain threshold to avoid trivial splits
    let bestSplit = null;

    for (const feature of ['x', 'y']) {
      const vals = data.map(pt => pt[feature]);
      const sorted = [...new Set(vals)].sort((a, b) => a - b);

      for (let i = 0; i < sorted.length - 1; i++) {
        const threshold = (sorted[i] + sorted[i + 1]) / 2;
        const left = data.filter(pt => pt[feature] <= threshold);
        const right = data.filter(pt => pt[feature] > threshold);
        if (left.length === 0 || right.length === 0) continue;

        const weightedH =
          (left.length / data.length) * this.entropy(left) +
          (right.length / data.length) * this.entropy(right);
        const gain = parentEntropy - weightedH;

        if (gain > bestGain) {
          bestGain = gain;
          bestSplit = { feature, threshold, gain, left, right };
        }
      }
    }
    return bestSplit;
  }

  buildNode(data, depth, maxDepth, minLeaf) {
    if (data.length === 0) return { label: 0, size: 0 };
    if (depth >= maxDepth || data.length < minLeaf || this.entropy(data) === 0) {
      return { label: this.majorityLabel(data), size: data.length };
    }

    const split = this.findBestSplit(data);
    if (!split) {
      return { label: this.majorityLabel(data), size: data.length };
    }

    return {
      feature: split.feature,
      threshold: split.threshold,
      gain: split.gain,
      size: data.length,
      left: this.buildNode(split.left, depth + 1, maxDepth, minLeaf),
      right: this.buildNode(split.right, depth + 1, maxDepth, minLeaf),
    };
  }

  predictNode(node, x, y) {
    if (!node) return 0;
    if (!node.feature) return node.label;
    const val = node.feature === 'x' ? x : y;
    return val <= node.threshold
      ? this.predictNode(node.left, x, y)
      : this.predictNode(node.right, x, y);
  }

  predict(x, y) {
    if (!this.tree) return 0;
    return this.predictNode(this.tree, x, y);
  }

  countNodes(node) {
    if (!node || !node.feature) return 1;
    return 1 + this.countNodes(node.left) + this.countNodes(node.right);
  }

  step() {
    const maxDepth = this.params.maxDepth || 5;
    if (this.epoch >= maxDepth) return;

    this.epoch++;
    const minLeaf = this.params.minLeafSize || 5;
    this.tree = this.buildNode(this.points, 0, this.epoch, minLeaf);

    const metrics = this.computeMetrics();
    this.history.push({ epoch: this.epoch, ...metrics });
  }

  // --- Rendering ---

  // Draw colored background decision regions via grid sampling
  drawDecisionRegions(ctx, width, height) {
    const grid = 60;
    const cellW = width / grid;
    const cellH = height / grid;
    for (let gx = 0; gx < grid; gx++) {
      for (let gy = 0; gy < grid; gy++) {
        const x = (gx / (grid - 1)) * 2 - 1;
        const y = (gy / (grid - 1)) * 2 - 1;
        const p = this.predict(x, y);
        ctx.fillStyle = p === 1
          ? 'rgba(25, 118, 210, 0.13)'
          : 'rgba(229, 57, 53, 0.13)';
        ctx.fillRect(gx * cellW, height - (gy + 1) * cellH, cellW + 1, cellH + 1);
      }
    }
  }

  // Recursively draw split lines, each constrained to its sub-region
  drawSplitLines(ctx, node, bounds, width, height, depth) {
    if (!node || !node.feature) return;

    const { xMin, xMax, yMin, yMax } = bounds;

    // Color depth: darker lines at root, lighter deeper
    const alpha = Math.max(0.25, 1 - depth * 0.18);
    ctx.strokeStyle = `rgba(15, 23, 42, ${alpha})`;
    ctx.lineWidth = Math.max(1, 3 - depth * 0.5);

    if (node.feature === 'x') {
      const t = node.threshold;
      const px  = ((t + 1) / 2) * width;
      const cy0 = height - ((yMax + 1) / 2) * height;
      const cy1 = height - ((yMin + 1) / 2) * height;
      ctx.beginPath();
      ctx.moveTo(px, cy0);
      ctx.lineTo(px, cy1);
      ctx.stroke();
      this.drawSplitLines(ctx, node.left,  { xMin, xMax: t,    yMin, yMax }, width, height, depth + 1);
      this.drawSplitLines(ctx, node.right, { xMin: t,  xMax,   yMin, yMax }, width, height, depth + 1);
    } else {
      const t = node.threshold;
      const py  = height - ((t + 1) / 2) * height;
      const cx0 = ((xMin + 1) / 2) * width;
      const cx1 = ((xMax + 1) / 2) * width;
      ctx.beginPath();
      ctx.moveTo(cx0, py);
      ctx.lineTo(cx1, py);
      ctx.stroke();
      this.drawSplitLines(ctx, node.left,  { xMin, xMax, yMin, yMax: t  }, width, height, depth + 1);
      this.drawSplitLines(ctx, node.right, { xMin, xMax, yMin: t, yMax  }, width, height, depth + 1);
    }
  }

  render() {
    const { width, height } = this.canvas;
    this.ctx.clearRect(0, 0, width, height);

    // White background
    this.ctx.fillStyle = '#ffffff';
    this.ctx.fillRect(0, 0, width, height);

    // 1. Decision regions
    if (this.tree) {
      this.drawDecisionRegions(this.ctx, width, height);
    }

    // 2. Split lines
    if (this.tree) {
      this.drawSplitLines(this.ctx, this.tree,
        { xMin: -1, xMax: 1, yMin: -1, yMax: 1 },
        width, height, 0);
    }

    // 3. Data points
    this.points.forEach(({ x, y, label }) => {
      const px = ((x + 1) / 2) * width;
      const py = height - ((y + 1) / 2) * height;
      this.ctx.beginPath();
      this.ctx.arc(px, py, 4.5, 0, Math.PI * 2);
      this.ctx.fillStyle = label === 1 ? '#1565c0' : '#c62828';
      this.ctx.fill();
      this.ctx.strokeStyle = '#fff';
      this.ctx.lineWidth = 1.2;
      this.ctx.stroke();
    });

    // 4. Info text (top-left)
    const maxDepth = this.params.maxDepth || 5;
    const nodeCount = this.tree ? this.countNodes(this.tree) : 0;
    const metrics = this.tree ? this.computeMetrics() : null;

    this.ctx.save();
    this.ctx.fillStyle = 'rgba(255,255,255,0.92)';
    this.ctx.beginPath();
    this.ctx.roundRect(8, 8, 220, 80, 6);
    this.ctx.fill();
    this.ctx.strokeStyle = '#d1d5db';
    this.ctx.lineWidth = 1;
    this.ctx.stroke();

    this.ctx.fillStyle = '#1e293b';
    this.ctx.font = 'bold 12px sans-serif';
    this.ctx.textAlign = 'left';
    this.ctx.fillText(`Depth: ${this.epoch} / ${maxDepth}`, 18, 30);
    this.ctx.font = '11px sans-serif';
    this.ctx.fillStyle = '#374151';
    this.ctx.fillText(`Nodes: ${nodeCount}`, 18, 48);
    if (metrics) {
      this.ctx.fillText(`Accuracy: ${(metrics.accuracy * 100).toFixed(1)}%`, 18, 65);
      this.ctx.fillText(`Loss: ${metrics.loss.toFixed(3)}`, 18, 81);
    } else {
      this.ctx.fillText('Press Start to build the tree', 18, 65);
    }
    this.ctx.restore();

    // 5. Confusion matrix (bottom-left)
    if (metrics) {
      const labels = this.points.map(pt => pt.label);
      const preds  = this.points.map(pt => this.predict(pt.x, pt.y));
      this.drawConfusionMatrix(this.ctx, labels, preds, 10, height - 142, 58);
    }

    // 6. Legend (bottom-right)
    this.ctx.save();
    this.ctx.fillStyle = 'rgba(255,255,255,0.92)';
    this.ctx.beginPath();
    this.ctx.roundRect(width - 110, height - 52, 102, 44, 6);
    this.ctx.fill();
    this.ctx.strokeStyle = '#d1d5db';
    this.ctx.lineWidth = 1;
    this.ctx.stroke();

    [[10, '#1565c0', 'Class 1'], [28, '#c62828', 'Class 0']].forEach(([dy, color, text]) => {
      this.ctx.beginPath();
      this.ctx.arc(width - 96, height - 52 + dy, 5, 0, Math.PI * 2);
      this.ctx.fillStyle = color;
      this.ctx.fill();
      this.ctx.fillStyle = '#374151';
      this.ctx.font = '10px sans-serif';
      this.ctx.textAlign = 'left';
      this.ctx.fillText(text, width - 86, height - 52 + dy + 4);
    });
    this.ctx.restore();
  }

  computeMetrics() {
    const labels = this.points.map(pt => pt.label);
    const preds  = this.points.map(pt => this.predict(pt.x, pt.y));
    return this.computeClassificationMetrics(labels, preds);
  }
}
