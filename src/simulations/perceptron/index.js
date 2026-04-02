import { BaseSimulation } from '../baseSimulation.js';

export class PerceptronSimulation extends BaseSimulation {
  setup() {
    this.history = [];
    this.epoch   = 0;
    const { nPoints, seed, noiseLevel, datasetType } = this.params;

    // Use shared generator; perceptron needs ±1 labels
    this.points = this.generateClassDataset(datasetType || 'linear', nPoints, seed, noiseLevel ?? 0.08)
      .map(pt => ({ ...pt, label: pt.label === 1 ? 1 : -1 }));

    this.weights = [
      this.randomBetween(-0.5, 0.5, seed + 1),
      this.randomBetween(-0.5, 0.5, seed + 2),
      this.randomBetween(-0.5, 0.5, seed + 3),
    ];
  }

  predict(x, y) {
    return this.weights[0]*x + this.weights[1]*y + this.weights[2] >= 0 ? 1 : -1;
  }

  step() {
    if (this.epoch >= this.params.epochs) return;
    const lr = this.params.learningRate;
    this.points.forEach(pt => {
      const err = pt.label - this.predict(pt.x, pt.y);
      this.weights[0] += lr * err * pt.x;
      this.weights[1] += lr * err * pt.y;
      this.weights[2] += lr * err;
    });
    this.epoch++;
    this.history.push({ epoch: this.epoch, ...this.computeMetrics() });
  }

  render() {
    const { width: W, height: H } = this.canvas;
    this.ctx.clearRect(0, 0, W, H);
    this.ctx.fillStyle = '#fff'; this.ctx.fillRect(0, 0, W, H);

    // Grid guides
    this.ctx.strokeStyle = '#f1f5f9'; this.ctx.lineWidth = 1;
    this.ctx.beginPath();
    this.ctx.moveTo(W/2, 0); this.ctx.lineTo(W/2, H);
    this.ctx.moveTo(0, H/2); this.ctx.lineTo(W, H/2);
    this.ctx.stroke();

    // Decision regions
    const G = 60;
    for (let gx = 0; gx < G; gx++) {
      for (let gy = 0; gy < G; gy++) {
        const x = (gx/(G-1))*2-1, y = (gy/(G-1))*2-1;
        this.ctx.fillStyle = this.predict(x, y) === 1
          ? 'rgba(29,78,216,.10)' : 'rgba(220,38,38,.10)';
        this.ctx.fillRect(gx*(W/G), H-(gy+1)*(H/G), W/G+1, H/G+1);
      }
    }

    // Decision boundary
    if (Math.abs(this.weights[1]) > 1e-6) {
      const yL = -(this.weights[2] + this.weights[0]*-1) / this.weights[1];
      const yR = -(this.weights[2] + this.weights[0]*1)  / this.weights[1];
      this.ctx.strokeStyle = '#0f172a'; this.ctx.lineWidth = 2.5;
      this.ctx.beginPath();
      this.ctx.moveTo(0, H-((yL+1)/2)*H);
      this.ctx.lineTo(W, H-((yR+1)/2)*H);
      this.ctx.stroke();
    }

    // Points
    this.points.forEach(({ x, y, label }) => {
      const px = ((x+1)/2)*W, py = H-((y+1)/2)*H;
      this.ctx.beginPath(); this.ctx.arc(px, py, 4.5, 0, Math.PI*2);
      this.ctx.fillStyle = label===1 ? '#1565c0' : '#c62828'; this.ctx.fill();
      this.ctx.strokeStyle = '#fff'; this.ctx.lineWidth = 1.2; this.ctx.stroke();
    });

    // Info panel
    const m = this.computeMetrics();
    this._infoPanel([
      `Epoch: ${this.epoch} / ${this.params.epochs}`,
      `Accuracy: ${(m.accuracy*100).toFixed(1)}%`,
      `Loss: ${m.loss.toFixed(3)}`,
      `w: [${this.weights.map(w=>w.toFixed(2)).join(', ')}]`,
    ]);

    // Confusion matrix
    const labels = this.points.map(pt => pt.label===1?1:0);
    const preds  = this.points.map(pt => this.predict(pt.x,pt.y)===1?1:0);
    this.drawConfusionMatrix(this.ctx, labels, preds, 10, H-142, 58);
  }

  _infoPanel(lines) {
    const panelH = lines.length*17 + 14;
    this.ctx.save();
    this.ctx.fillStyle = 'rgba(255,255,255,.93)';
    this.ctx.beginPath(); this.ctx.roundRect(8, 8, 230, panelH, 6); this.ctx.fill();
    this.ctx.strokeStyle = '#e2e8f0'; this.ctx.lineWidth = 1; this.ctx.stroke();
    lines.forEach((line, i) => {
      this.ctx.font      = i===0 ? 'bold 12px sans-serif' : '11px sans-serif';
      this.ctx.fillStyle = i===0 ? '#1e293b' : '#374151';
      this.ctx.textAlign = 'left';
      this.ctx.fillText(line, 18, 24 + i*17);
    });
    this.ctx.restore();
  }

  computeMetrics() {
    const labels = this.points.map(pt => pt.label===1?1:0);
    const preds  = this.points.map(pt => this.predict(pt.x,pt.y)===1?1:0);
    return this.computeClassificationMetrics(labels, preds);
  }
}
