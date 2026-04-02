import { BaseSimulation } from '../baseSimulation.js';

function tanh(x)       { return Math.tanh(x); }
function tanhDeriv(x)  { const t = Math.tanh(x); return 1 - t * t; }
function sigmoid(x)    { return 1 / (1 + Math.exp(-x)); }
function sigmoidDeriv(x) { const s = sigmoid(x); return s * (1 - s); }

export class NNSimulation extends BaseSimulation {
  setup() {
    this.history = [];
    this.points = [];
    const { nPoints, seed } = this.params;
    this.epoch = 0;

    const hidden = this.params.hiddenUnits || 4;
    this.w1 = Array.from({ length: hidden }, (_, i) => [
      this.randomBetween(-1, 1, seed + 100 + i * 3),
      this.randomBetween(-1, 1, seed + 101 + i * 3),
      this.randomBetween(-1, 1, seed + 102 + i * 3),
    ]);
    this.w2 = Array.from({ length: hidden + 1 }, (_, i) =>
      this.randomBetween(-1, 1, seed + 200 + i)
    );

    for (let i = 0; i < nPoints; i++) {
      const x = this.randomBetween(-1, 1, seed + 300 + i * 2);
      const y = this.randomBetween(-1, 1, seed + 301 + i * 2);
      const label = x * x + y * y < 0.5 ? 1 : 0;
      this.points.push({ x, y, label });
    }
  }

  reset() { this.setup(); }

  forward(x, y) {
    const hiddenIn  = this.w1.map(ws => ws[0] * x + ws[1] * y + ws[2]);
    const hidden    = hiddenIn.map(h => tanh(h));
    const body      = [...hidden, 1];
    const outputIn  = this.w2.reduce((sum, w, i) => sum + w * body[i], 0);
    const output    = sigmoid(outputIn);
    return { hidden, hiddenIn, output, body };
  }

  step() {
    if (this.epoch >= this.params.epochs) return;

    const lr = this.params.learningRate;
    this.points.forEach((pt) => {
      const { x, y, label } = pt;
      const { hiddenIn, output, body } = this.forward(x, y);
      const errorOut  = output - label;
      const gradOut   = errorOut * sigmoidDeriv(output);

      for (let i = 0; i < this.w2.length; i++) {
        this.w2[i] -= lr * gradOut * body[i];
      }
      for (let j = 0; j < this.w1.length; j++) {
        const gradHid = gradOut * this.w2[j] * tanhDeriv(hiddenIn[j]);
        this.w1[j][0] -= lr * gradHid * x;
        this.w1[j][1] -= lr * gradHid * y;
        this.w1[j][2] -= lr * gradHid;
      }
    });

    this.epoch++;
    const metrics = this.computeMetrics();
    this.history.push({ epoch: this.epoch, ...metrics });
  }

  predict(x, y) {
    return this.forward(x, y).output >= 0.5 ? 1 : 0;
  }

  // Draw mini network diagram in a panel area
  drawNetworkDiagram(ctx, panelX, panelY, panelW, panelH) {
    const hidden = this.w1.length;
    const displayH = Math.min(hidden, 8); // cap at 8 for readability

    ctx.save();
    // Panel background
    ctx.fillStyle = 'rgba(255,255,255,0.93)';
    ctx.beginPath();
    ctx.roundRect(panelX, panelY, panelW, panelH, 6);
    ctx.fill();
    ctx.strokeStyle = '#d1d5db';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Title
    ctx.fillStyle = '#374151';
    ctx.font = 'bold 10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Network Weights', panelX + panelW / 2, panelY + 13);

    const margin  = 18;
    const netW    = panelW - margin * 2;
    const netH    = panelH - 30;
    const top     = panelY + 22;

    // Layer x positions
    const xIn  = panelX + margin + netW * 0.1;
    const xHid = panelX + margin + netW * 0.5;
    const xOut = panelX + margin + netW * 0.9;

    // Node y positions
    const inputNodes  = [top + netH * 0.3, top + netH * 0.7];
    const hiddenNodes = Array.from({ length: displayH }, (_, i) =>
      top + (netH * (i + 1)) / (displayH + 1)
    );
    const outputNode  = top + netH * 0.5;

    // Max weight magnitudes for scaling
    const maxW1 = Math.max(...this.w1.map(ws => Math.max(Math.abs(ws[0]), Math.abs(ws[1]))), 1e-6);
    const maxW2 = Math.max(...this.w2.slice(0, hidden).map(w => Math.abs(w)), 1e-6);

    // Draw input→hidden connections
    for (let j = 0; j < displayH; j++) {
      for (let inp = 0; inp < 2; inp++) {
        const w = this.w1[j][inp];
        const thickness = Math.max(0.5, (Math.abs(w) / maxW1) * 3);
        ctx.strokeStyle = w >= 0
          ? `rgba(37,99,235,${0.15 + Math.abs(w) / maxW1 * 0.7})`
          : `rgba(220,38,38,${0.15 + Math.abs(w) / maxW1 * 0.7})`;
        ctx.lineWidth = thickness;
        ctx.beginPath();
        ctx.moveTo(xIn, inputNodes[inp]);
        ctx.lineTo(xHid, hiddenNodes[j]);
        ctx.stroke();
      }
    }

    // Draw hidden→output connections
    for (let j = 0; j < displayH; j++) {
      const w = this.w2[j];
      const thickness = Math.max(0.5, (Math.abs(w) / maxW2) * 3);
      ctx.strokeStyle = w >= 0
        ? `rgba(37,99,235,${0.15 + Math.abs(w) / maxW2 * 0.7})`
        : `rgba(220,38,38,${0.15 + Math.abs(w) / maxW2 * 0.7})`;
      ctx.lineWidth = thickness;
      ctx.beginPath();
      ctx.moveTo(xHid, hiddenNodes[j]);
      ctx.lineTo(xOut, outputNode);
      ctx.stroke();
    }

    // Draw nodes
    const drawNode = (x, y, color, label) => {
      ctx.beginPath();
      ctx.arc(x, y, 7, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 7px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(label, x, y + 2.5);
    };

    inputNodes.forEach((ny, i) => drawNode(xIn, ny, '#0369a1', i === 0 ? 'x' : 'y'));
    hiddenNodes.forEach((ny, i) => drawNode(xHid, ny, '#7c3aed', `h${i + 1}`));
    drawNode(xOut, outputNode, '#059669', 'out');

    // Overflow indicator
    if (hidden > 8) {
      ctx.fillStyle = '#6b7280';
      ctx.font = '9px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`+${hidden - 8} more`, xHid, top + netH - 2);
    }

    ctx.restore();
  }

  render() {
    const { width, height } = this.canvas;
    this.ctx.clearRect(0, 0, width, height);
    this.ctx.fillStyle = '#ffffff';
    this.ctx.fillRect(0, 0, width, height);

    // 1. Decision region background
    const grid = 80;
    for (let gx = 0; gx < grid; gx++) {
      for (let gy = 0; gy < grid; gy++) {
        const x = (gx / (grid - 1)) * 2 - 1;
        const y = (gy / (grid - 1)) * 2 - 1;
        const p = this.predict(x, y);
        this.ctx.fillStyle = p === 1
          ? 'rgba(25, 118, 210, 0.18)'
          : 'rgba(229, 57, 53, 0.18)';
        this.ctx.fillRect(
          (x + 1) / 2 * width,
          height - (y + 1) / 2 * height,
          width / grid + 1,
          height / grid + 1
        );
      }
    }

    // True boundary circle guide
    this.ctx.strokeStyle = 'rgba(148, 163, 184, 0.5)';
    this.ctx.lineWidth = 1.5;
    this.ctx.setLineDash([5, 4]);
    this.ctx.beginPath();
    this.ctx.arc(width / 2, height / 2, Math.sqrt(0.5) * width / 2, 0, Math.PI * 2);
    this.ctx.stroke();
    this.ctx.setLineDash([]);

    // 2. Data points
    this.points.forEach(({ x, y, label }) => {
      const px = ((x + 1) / 2) * width;
      const py = height - ((y + 1) / 2) * height;
      this.ctx.beginPath();
      this.ctx.arc(px, py, 4.5, 0, Math.PI * 2);
      this.ctx.fillStyle = label === 1 ? '#1565c0' : '#c62828';
      this.ctx.fill();
      this.ctx.strokeStyle = '#fff';
      this.ctx.lineWidth = 1.5;
      this.ctx.stroke();
    });

    // 3. Info panel (top-left)
    const metrics = this.computeMetrics();
    this.ctx.save();
    this.ctx.fillStyle = 'rgba(255,255,255,0.93)';
    this.ctx.beginPath();
    this.ctx.roundRect(8, 8, 230, 90, 6);
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
    this.ctx.fillText(`Hidden units: ${this.w1.length}`, 18, 46);
    this.ctx.fillText(`Accuracy: ${(metrics.accuracy * 100).toFixed(1)}%`, 18, 62);
    this.ctx.fillText(`Loss: ${metrics.loss.toFixed(4)}`, 18, 78);
    this.ctx.fillText(`F1: ${metrics.f1.toFixed(3)}`, 18, 94);
    this.ctx.restore();

    // 4. Network weight diagram (top-right)
    const diagH = Math.min(180, 40 + Math.min(this.w1.length, 8) * 20);
    this.drawNetworkDiagram(this.ctx, width - 170, 8, 162, diagH);

    // 5. Confusion matrix (bottom-left)
    const labels = this.points.map(pt => pt.label);
    const preds  = this.points.map(pt => this.predict(pt.x, pt.y));
    this.drawConfusionMatrix(this.ctx, labels, preds, 10, height - 142, 58);

    // 6. Legend (bottom-right)
    this.ctx.save();
    this.ctx.fillStyle = 'rgba(255,255,255,0.92)';
    this.ctx.beginPath();
    this.ctx.roundRect(width - 180, height - 72, 172, 64, 6);
    this.ctx.fill();
    this.ctx.strokeStyle = '#d1d5db';
    this.ctx.lineWidth = 1;
    this.ctx.stroke();

    this.ctx.fillStyle = '#374151';
    this.ctx.font = 'bold 10px sans-serif';
    this.ctx.textAlign = 'left';
    this.ctx.fillText('Legend', width - 166, height - 57);

    [[13, '#1565c0', 'Class 1 (x²+y²<0.5)'],
     [29, '#c62828', 'Class 0 (outside)']].forEach(([dy, color, text]) => {
      this.ctx.beginPath();
      this.ctx.arc(width - 162, height - 72 + dy, 5, 0, Math.PI * 2);
      this.ctx.fillStyle = color;
      this.ctx.fill();
      this.ctx.fillStyle = '#374151';
      this.ctx.font = '10px sans-serif';
      this.ctx.fillText(text, width - 152, height - 72 + dy + 4);
    });

    this.ctx.strokeStyle = 'rgba(148,163,184,0.6)';
    this.ctx.lineWidth = 1.5;
    this.ctx.setLineDash([4, 3]);
    this.ctx.beginPath();
    this.ctx.moveTo(width - 162, height - 72 + 45);
    this.ctx.lineTo(width - 144, height - 72 + 45);
    this.ctx.stroke();
    this.ctx.setLineDash([]);
    this.ctx.fillStyle = '#374151';
    this.ctx.font = '10px sans-serif';
    this.ctx.fillText('True boundary', width - 140, height - 72 + 49);
    this.ctx.restore();
  }

  computeMetrics() {
    const labels = [];
    const preds  = [];
    let lossTotal = 0;

    this.points.forEach((pt) => {
      const { output } = this.forward(pt.x, pt.y);
      const prediction = output >= 0.5 ? 1 : 0;
      labels.push(pt.label);
      preds.push(prediction);
      lossTotal += 0.5 * (output - pt.label) ** 2;
    });

    const classMetrics = this.computeClassificationMetrics(labels, preds);
    return { ...classMetrics, loss: lossTotal / (this.points.length || 1) };
  }
}
