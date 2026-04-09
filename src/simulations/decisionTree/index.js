import { BaseSimulation } from '../baseSimulation.js';

export class DecisionTreeSimulation extends BaseSimulation {
  setup() {
    this.history = [];
    this.epoch   = 0;
    this.tree    = null;
    this._3d     = this._is3DClass;
    const { nPoints, seed, noiseLevel, datasetType } = this.params;
    this.points = this.generateClassDataset(datasetType || 'xor', nPoints, seed, noiseLevel ?? 0.08);
  }

  // ── Impurity ────────────────────────────────────────────────
  _impurity(data) {
    if (data.length === 0) return 0;
    let c1 = 0; data.forEach(pt => { if (pt.label === 1) c1++; });
    const c0 = data.length - c1;
    const p1 = c1/data.length, p0 = c0/data.length;
    if (this.params.useGini) {
      return 1 - p1*p1 - p0*p0; // Gini
    }
    let h = 0;
    if (p0 > 0) h -= p0 * Math.log2(p0);
    if (p1 > 0) h -= p1 * Math.log2(p1);
    return h; // Entropy
  }

  _majorityLabel(data) {
    let c1 = 0; data.forEach(pt => { if (pt.label===1) c1++; });
    return c1*2 >= data.length ? 1 : 0;
  }

  _findBestSplit(data) {
    const parentImp = this._impurity(data);
    let bestGain = 1e-10, bestSplit = null;
    const feats = this._3d ? ['x','y','z'] : ['x','y'];

    for (const feature of feats) {
      const vals = [...new Set(data.map(pt => pt[feature] ?? 0))].sort((a,b) => a-b);
      for (let i = 0; i < vals.length-1; i++) {
        const threshold = (vals[i]+vals[i+1])/2;
        const left  = data.filter(pt => (pt[feature] ?? 0) <= threshold);
        const right = data.filter(pt => (pt[feature] ?? 0) >  threshold);
        if (left.length===0 || right.length===0) continue;
        const wImp = (left.length/data.length)*this._impurity(left)
                   + (right.length/data.length)*this._impurity(right);
        const gain = parentImp - wImp;
        if (gain > bestGain) { bestGain = gain; bestSplit = { feature, threshold, gain, left, right }; }
      }
    }
    return bestSplit;
  }

  _buildNode(data, depth, maxDepth, minLeaf) {
    if (data.length===0) return { label: 0, size: 0 };
    if (depth >= maxDepth || data.length < minLeaf || this._impurity(data) < 1e-6)
      return { label: this._majorityLabel(data), size: data.length };
    const split = this._findBestSplit(data);
    if (!split) return { label: this._majorityLabel(data), size: data.length };
    return {
      feature: split.feature, threshold: split.threshold,
      gain: split.gain, size: data.length,
      left:  this._buildNode(split.left,  depth+1, maxDepth, minLeaf),
      right: this._buildNode(split.right, depth+1, maxDepth, minLeaf),
    };
  }

  _predictNode(node, x, y, z) {
    if (!node || !node.feature) return node ? node.label : 0;
    const val = node.feature==='x' ? x : node.feature==='y' ? y : (z ?? 0);
    return val <= node.threshold
      ? this._predictNode(node.left, x, y, z)
      : this._predictNode(node.right, x, y, z);
  }

  predict(x, y, z) { return this.tree ? this._predictNode(this.tree, x, y, z) : 0; }

  _countNodes(node) {
    if (!node || !node.feature) return 1;
    return 1 + this._countNodes(node.left) + this._countNodes(node.right);
  }

  step() {
    const maxDepth = this.params.maxDepth || 6;
    if (this.epoch >= maxDepth) return;
    this.epoch++;
    this.tree = this._buildNode(this.points, 0, this.epoch, this.params.minLeafSize || 5);
    this.history.push({ epoch: this.epoch, ...this.computeMetrics() });
  }

  _drawRegions(W, H) {
    const G = 60;
    for (let gx = 0; gx < G; gx++) {
      for (let gy = 0; gy < G; gy++) {
        const x = (gx/(G-1))*2-1, y = (gy/(G-1))*2-1;
        this.ctx.fillStyle = this.predict(x,y)===1 ? 'rgba(29,78,216,.12)' : 'rgba(220,38,38,.12)';
        this.ctx.fillRect(gx*(W/G), H-(gy+1)*(H/G), W/G+1, H/G+1);
      }
    }
  }

  _drawSplits(node, bounds, W, H, depth) {
    if (!node || !node.feature) return;
    const { xMin, xMax, yMin, yMax } = bounds;
    const alpha = Math.max(.2, 1-depth*.18);
    this.ctx.strokeStyle = `rgba(15,23,42,${alpha})`;
    this.ctx.lineWidth = Math.max(1, 2.5-depth*.4);
    if (node.feature === 'x') {
      const px = ((node.threshold+1)/2)*W;
      const cy0 = H-((yMax+1)/2)*H, cy1 = H-((yMin+1)/2)*H;
      this.ctx.beginPath(); this.ctx.moveTo(px, cy0); this.ctx.lineTo(px, cy1); this.ctx.stroke();
      this._drawSplits(node.left,  { xMin, xMax: node.threshold, yMin, yMax }, W, H, depth+1);
      this._drawSplits(node.right, { xMin: node.threshold, xMax, yMin, yMax }, W, H, depth+1);
    } else {
      const py = H-((node.threshold+1)/2)*H;
      const cx0 = ((xMin+1)/2)*W, cx1 = ((xMax+1)/2)*W;
      this.ctx.beginPath(); this.ctx.moveTo(cx0, py); this.ctx.lineTo(cx1, py); this.ctx.stroke();
      this._drawSplits(node.left,  { xMin, xMax, yMin, yMax: node.threshold }, W, H, depth+1);
      this._drawSplits(node.right, { xMin, xMax, yMin: node.threshold, yMax }, W, H, depth+1);
    }
  }

  render() {
    const { width: W, height: H } = this.canvas;
    this.ctx.clearRect(0, 0, W, H);
    this.ctx.fillStyle = '#fff'; this.ctx.fillRect(0, 0, W, H);

    if (this.tree) {
      this._drawRegions(W, H);
      this._drawSplits(this.tree, { xMin:-1, xMax:1, yMin:-1, yMax:1 }, W, H, 0);
    }

    this.points.forEach(({ x, y, label }) => {
      const px = ((x+1)/2)*W, py = H-((y+1)/2)*H;
      this.ctx.beginPath(); this.ctx.arc(px, py, 4.5, 0, Math.PI*2);
      this.ctx.fillStyle = label===1 ? '#1565c0' : '#c62828'; this.ctx.fill();
      this.ctx.strokeStyle = '#fff'; this.ctx.lineWidth = 1.2; this.ctx.stroke();
    });

    // Info panel
    const maxDepth = this.params.maxDepth || 6;
    const criterion = this.params.useGini ? 'Gini' : 'Entropy';
    const m = this.tree ? this.computeMetrics() : null;
    const nodeCount = this.tree ? this._countNodes(this.tree) : 0;
    this.ctx.save();
    this.ctx.fillStyle = 'rgba(255,255,255,.93)';
    this.ctx.beginPath(); this.ctx.roundRect(8, 8, 230, 90, 6); this.ctx.fill();
    this.ctx.strokeStyle = '#e2e8f0'; this.ctx.lineWidth = 1; this.ctx.stroke();
    const lines = [
      `Depth: ${this.epoch} / ${maxDepth}`,
      `Criterion: ${criterion}  |  Nodes: ${nodeCount}`,
      m ? `Accuracy: ${(m.accuracy*100).toFixed(1)}%` : 'Press Run to build tree',
      m ? `Loss: ${m.loss.toFixed(3)}` : '',
    ];
    lines.forEach((line, i) => {
      if (!line) return;
      this.ctx.font      = i===0 ? 'bold 12px sans-serif' : '11px sans-serif';
      this.ctx.fillStyle = i===0 ? '#1e293b' : '#374151';
      this.ctx.textAlign = 'left';
      this.ctx.fillText(line, 18, 26 + i*17);
    });
    this.ctx.restore();

    if (m) {
      const labels = this.points.map(pt => pt.label);
      const preds  = this.points.map(pt => this.predict(pt.x, pt.y, pt.z));
      this.drawConfusionMatrix(this.ctx, labels, preds, 10, H-142, 58);
    }
  }

  computeMetrics() {
    const labels = this.points.map(pt => pt.label);
    const preds  = this.points.map(pt => this.predict(pt.x, pt.y, pt.z));
    return this.computeClassificationMetrics(labels, preds);
  }
}

// ── Decision Tree Regression ──────────────────────────────────────
export class DecisionTreeRegressionSimulation extends BaseSimulation {
  setup() {
    this.history = [];
    this.epoch   = 0;
    this.tree    = null;
    this._3d     = this._is3DReg;
    const { nPoints, seed, noiseLevel, datasetType } = this.params;
    this.points = this.generateRegressionDataset(datasetType || 'sine', nPoints, seed, noiseLevel ?? 0.2);
  }

  _mse(data) {
    if (!data.length) return 0;
    const m = data.reduce((s, pt) => s + pt.y, 0) / data.length;
    return data.reduce((s, pt) => s + (pt.y - m) ** 2, 0) / data.length;
  }

  _buildNode(data, depth, maxDepth, minLeaf) {
    if (!data.length) return { value: 0 };
    const meanY = data.reduce((s, pt) => s + pt.y, 0) / data.length;
    if (depth >= maxDepth || data.length < minLeaf) return { value: meanY };
    const pMse = this._mse(data);
    let bestGain = 1e-10, bestSplit = null;
    const feats = this._3d ? ['x', 'z'] : ['x'];
    for (const feat of feats) {
      const vals = [...new Set(data.map(pt => pt[feat] ?? 0))].sort((a, b) => a - b);
      for (let i = 0; i < vals.length - 1; i++) {
        const t = (vals[i] + vals[i + 1]) / 2;
        const L = data.filter(pt => (pt[feat] ?? 0) <= t);
        const R = data.filter(pt => (pt[feat] ?? 0) > t);
        if (!L.length || !R.length) continue;
        const wMse = (L.length * this._mse(L) + R.length * this._mse(R)) / data.length;
        const gain = pMse - wMse;
        if (gain > bestGain) { bestGain = gain; bestSplit = { feat, t, L, R }; }
      }
    }
    if (!bestSplit) return { value: meanY };
    return {
      feature: bestSplit.feat, threshold: bestSplit.t,
      left:  this._buildNode(bestSplit.L, depth + 1, maxDepth, minLeaf),
      right: this._buildNode(bestSplit.R, depth + 1, maxDepth, minLeaf),
    };
  }

  _predictNode(node, x, z) {
    if (node.value !== undefined) return node.value;
    const v = node.feature === 'z' ? (z ?? 0) : x;
    return v <= node.threshold ? this._predictNode(node.left, x, z) : this._predictNode(node.right, x, z);
  }

  predict(x, z) { return this.tree ? this._predictNode(this.tree, x, z) : 0; }

  step() {
    const maxDepth = this.params.maxDepth || 5;
    if (this.epoch >= maxDepth) return;
    this.epoch++;
    this.tree = this._buildNode(this.points, 0, this.epoch, this.params.minLeafSize || 3);
    this.history.push({ epoch: this.epoch, ...this.computeMetrics() });
  }

  computeMetrics() {
    return this.computeRegressionMetrics(this.points.map(pt => pt.y), this.points.map(pt => this.predict(pt.x, pt.z)));
  }

  render() {
    const { width: W, height: H } = this.canvas;
    const PAD = 36;
    this.ctx.clearRect(0, 0, W, H);
    this.ctx.fillStyle = '#fff'; this.ctx.fillRect(0, 0, W, H);
    const toX = x => PAD + ((x + 1) / 2) * (W - 2 * PAD);
    const toY = y => H - PAD - ((y + 1.2) / 2.4) * (H - 2 * PAD);
    const clamp = y => Math.max(-1.2, Math.min(1.2, y));

    this.ctx.strokeStyle = '#e2e8f0'; this.ctx.lineWidth = 1;
    this.ctx.beginPath(); this.ctx.moveTo(PAD, toY(0)); this.ctx.lineTo(W - PAD, toY(0)); this.ctx.stroke();
    this.ctx.beginPath(); this.ctx.moveTo(toX(0), PAD); this.ctx.lineTo(toX(0), H - PAD); this.ctx.stroke();

    if (this.tree) {
      this.ctx.beginPath(); this.ctx.strokeStyle = '#dc2626'; this.ctx.lineWidth = 2.5;
      for (let i = 0; i <= 400; i++) {
        const x = -1 + (i / 400) * 2, y = clamp(this.predict(x));
        i === 0 ? this.ctx.moveTo(toX(x), toY(y)) : this.ctx.lineTo(toX(x), toY(y));
      }
      this.ctx.stroke();
    }

    this.points.forEach(({ x, y }) => {
      this.ctx.beginPath(); this.ctx.arc(toX(x), toY(clamp(y)), 3.5, 0, Math.PI * 2);
      this.ctx.fillStyle = '#64748b'; this.ctx.fill();
      this.ctx.strokeStyle = '#fff'; this.ctx.lineWidth = 1; this.ctx.stroke();
    });

    const m = this.tree ? this.computeMetrics() : null;
    this.ctx.save();
    this.ctx.fillStyle = 'rgba(255,255,255,.93)';
    this.ctx.beginPath(); this.ctx.roundRect(8, 8, 270, 70, 6); this.ctx.fill();
    this.ctx.strokeStyle = '#e2e8f0'; this.ctx.lineWidth = 1; this.ctx.stroke();
    [`Depth: ${this.epoch} / ${this.params.maxDepth || 5}  |  Min Leaf: ${this.params.minLeafSize || 3}`,
     m ? `MAE: ${m.mae.toFixed(3)}  RMSE: ${m.rmse.toFixed(3)}` : 'Press Run to build tree',
    ].forEach((line, i) => {
      this.ctx.font = i === 0 ? 'bold 12px sans-serif' : '11px sans-serif';
      this.ctx.fillStyle = i === 0 ? '#1e293b' : '#374151';
      this.ctx.textAlign = 'left'; this.ctx.fillText(line, 18, 26 + i * 17);
    });
    this.ctx.restore();
  }
}
