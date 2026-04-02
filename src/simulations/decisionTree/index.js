import { BaseSimulation } from '../baseSimulation.js';

export class DecisionTreeSimulation extends BaseSimulation {
  setup() {
    this.history = [];
    this.epoch   = 0;
    this.tree    = null;
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

    for (const feature of ['x','y']) {
      const vals = [...new Set(data.map(pt => pt[feature]))].sort((a,b) => a-b);
      for (let i = 0; i < vals.length-1; i++) {
        const threshold = (vals[i]+vals[i+1])/2;
        const left  = data.filter(pt => pt[feature] <= threshold);
        const right = data.filter(pt => pt[feature] >  threshold);
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

  _predictNode(node, x, y) {
    if (!node || !node.feature) return node ? node.label : 0;
    const val = node.feature==='x' ? x : y;
    return val <= node.threshold
      ? this._predictNode(node.left, x, y)
      : this._predictNode(node.right, x, y);
  }

  predict(x, y) { return this.tree ? this._predictNode(this.tree, x, y) : 0; }

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
      const preds  = this.points.map(pt => this.predict(pt.x, pt.y));
      this.drawConfusionMatrix(this.ctx, labels, preds, 10, H-142, 58);
    }
  }

  computeMetrics() {
    const labels = this.points.map(pt => pt.label);
    const preds  = this.points.map(pt => this.predict(pt.x, pt.y));
    return this.computeClassificationMetrics(labels, preds);
  }
}
