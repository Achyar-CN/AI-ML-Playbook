import { BaseSimulation } from '../baseSimulation.js';
import { dataStore } from '../core/dataStore.js';

// Activation functions
function applyAct(x, act) {
  if (act === 'relu')    return Math.max(0, x);
  if (act === 'sigmoid') return 1 / (1 + Math.exp(-x));
  return Math.tanh(x); // default tanh
}
function actDeriv(x, act) {
  if (act === 'relu')    return x > 0 ? 1 : 0;
  if (act === 'sigmoid') { const s = 1/(1+Math.exp(-x)); return s*(1-s); }
  const t = Math.tanh(x); return 1 - t*t;
}
function sigmoid(x)  { return 1 / (1 + Math.exp(-x)); }

export class NNSimulation extends BaseSimulation {
  setup() {
    this.history = [];
    this.epoch   = 0;
    const { nPoints, seed, noiseLevel, datasetType, hiddenUnits } = this.params;
    const H = hiddenUnits || 6;

    this.points = this.generateClassDataset(datasetType || 'circle', nPoints, seed, noiseLevel ?? 0.08);

    // Xavier-ish init
    this.w1 = Array.from({ length: H }, (_, i) => [
      this.randomBetween(-1, 1, seed+100+i*3),
      this.randomBetween(-1, 1, seed+101+i*3),
      this.randomBetween(-0.1, 0.1, seed+102+i*3), // bias
    ]);
    this.w2 = Array.from({ length: H+1 }, (_, i) => this.randomBetween(-1, 1, seed+200+i));
  }

  reset() { this.setup(); }

  forward(x, y) {
    const act = this.params.activation || 'tanh';
    const hiddenIn = this.w1.map(ws => ws[0]*x + ws[1]*y + ws[2]);
    const hidden   = hiddenIn.map(h => applyAct(h, act));
    const body     = [...hidden, 1]; // +bias
    const outIn    = this.w2.reduce((s, w, i) => s + w*body[i], 0);
    const output   = sigmoid(outIn);
    return { hidden, hiddenIn, output, body };
  }

  step() {
    if (this.epoch >= this.params.epochs) return;
    const lr  = this.params.learningRate;
    const l2  = this.params.l2  || 0;
    const act = this.params.activation || 'tanh';
    const H   = this.w1.length;

    this.points.forEach(pt => {
      const { hiddenIn, output, body } = this.forward(pt.x, pt.y);
      const eOut   = output - pt.label;
      const gOut   = eOut * (output*(1-output)); // sigmoid deriv

      // w2 update (with L2)
      for (let i = 0; i < this.w2.length; i++) {
        this.w2[i] -= lr * (gOut * body[i] + l2 * this.w2[i]);
      }
      // w1 backprop
      for (let j = 0; j < H; j++) {
        const gHid = gOut * this.w2[j] * actDeriv(hiddenIn[j], act);
        this.w1[j][0] -= lr * (gHid * pt.x + l2 * this.w1[j][0]);
        this.w1[j][1] -= lr * (gHid * pt.y + l2 * this.w1[j][1]);
        this.w1[j][2] -= lr * gHid;
      }
    });

    this.epoch++;
    this.history.push({ epoch: this.epoch, ...this.computeMetrics() });
  }

  predict(x, y) { return this.forward(x, y).output >= 0.5 ? 1 : 0; }

  // Network weight diagram (top-right panel)
  _drawWeightDiagram(panelX, panelY, panelW, panelH) {
    const H = this.w1.length;
    const display = Math.min(H, 8);
    const ctx = this.ctx;
    ctx.save();
    ctx.fillStyle = 'rgba(255,255,255,.93)';
    ctx.beginPath(); ctx.roundRect(panelX, panelY, panelW, panelH, 6); ctx.fill();
    ctx.strokeStyle = '#e2e8f0'; ctx.lineWidth = 1; ctx.stroke();

    ctx.fillStyle = '#64748b'; ctx.font = 'bold 9px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('Weights', panelX + panelW/2, panelY + 12);

    const m = 16, netW = panelW - m*2, netH = panelH - 22, top = panelY + 18;
    const xIn = panelX + m + netW*0.1;
    const xHid = panelX + m + netW*0.5;
    const xOut = panelX + m + netW*0.9;
    const inNodes  = [top + netH*0.3, top + netH*0.7];
    const hidNodes = Array.from({ length: display }, (_, i) => top + (netH*(i+1))/(display+1));
    const outNode  = top + netH*0.5;

    const maxW1 = Math.max(...this.w1.map(ws => Math.max(Math.abs(ws[0]), Math.abs(ws[1]))), 1e-6);
    const maxW2 = Math.max(...this.w2.slice(0, H).map(w => Math.abs(w)), 1e-6);

    // Connections
    for (let j = 0; j < display; j++) {
      for (let inp = 0; inp < 2; inp++) {
        const w = this.w1[j][inp];
        ctx.strokeStyle = w >= 0 ? `rgba(29,78,216,${.12+Math.abs(w)/maxW1*.6})` : `rgba(220,38,38,${.12+Math.abs(w)/maxW1*.6})`;
        ctx.lineWidth = Math.max(.5, Math.abs(w)/maxW1*2.5);
        ctx.beginPath(); ctx.moveTo(xIn, inNodes[inp]); ctx.lineTo(xHid, hidNodes[j]); ctx.stroke();
      }
      const w2 = this.w2[j];
      ctx.strokeStyle = w2 >= 0 ? `rgba(29,78,216,${.12+Math.abs(w2)/maxW2*.6})` : `rgba(220,38,38,${.12+Math.abs(w2)/maxW2*.6})`;
      ctx.lineWidth = Math.max(.5, Math.abs(w2)/maxW2*2.5);
      ctx.beginPath(); ctx.moveTo(xHid, hidNodes[j]); ctx.lineTo(xOut, outNode); ctx.stroke();
    }

    const drawNode = (x, y, color, lbl) => {
      ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI*2);
      ctx.fillStyle = color; ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
      ctx.fillStyle = '#fff'; ctx.font = 'bold 7px sans-serif'; ctx.textAlign = 'center';
      ctx.fillText(lbl, x, y+2.5);
    };
    inNodes.forEach((ny, i) => drawNode(xIn, ny, '#0369a1', i===0?'x':'y'));
    hidNodes.forEach((ny, i) => drawNode(xHid, ny, '#7c3aed', `h${i+1}`));
    drawNode(xOut, outNode, '#059669', 'out');
    if (H > 8) {
      ctx.fillStyle = '#94a3b8'; ctx.font = '9px sans-serif'; ctx.textAlign = 'center';
      ctx.fillText(`+${H-8}`, xHid, top+netH-2);
    }
    ctx.restore();
  }

  render() {
    const { width: W, height: H } = this.canvas;
    this.ctx.clearRect(0, 0, W, H);
    this.ctx.fillStyle = '#fff'; this.ctx.fillRect(0, 0, W, H);

    // Decision region grid
    const G = 70;
    for (let gx = 0; gx < G; gx++) {
      for (let gy = 0; gy < G; gy++) {
        const x = (gx/(G-1))*2-1, y = (gy/(G-1))*2-1;
        this.ctx.fillStyle = this.predict(x, y)===1 ? 'rgba(29,78,216,.15)' : 'rgba(220,38,38,.15)';
        this.ctx.fillRect(gx*(W/G), H-(gy+1)*(H/G), W/G+1, H/G+1);
      }
    }

    // Points
    this.points.forEach(({ x, y, label }) => {
      const px = ((x+1)/2)*W, py = H-((y+1)/2)*H;
      this.ctx.beginPath(); this.ctx.arc(px, py, 4.5, 0, Math.PI*2);
      this.ctx.fillStyle = label===1 ? '#1565c0' : '#c62828'; this.ctx.fill();
      this.ctx.strokeStyle = '#fff'; this.ctx.lineWidth = 1.5; this.ctx.stroke();
    });

    // Info panel
    const m = this.computeMetrics();
    const act = this.params.activation || 'tanh';
    this.ctx.save();
    this.ctx.fillStyle = 'rgba(255,255,255,.93)';
    this.ctx.beginPath(); this.ctx.roundRect(8, 8, 220, 90, 6); this.ctx.fill();
    this.ctx.strokeStyle = '#e2e8f0'; this.ctx.lineWidth = 1; this.ctx.stroke();
    [
      { t: `Epoch: ${this.epoch} / ${this.params.epochs}`, bold: true },
      { t: `Hidden: ${this.w1.length} × ${act}` },
      { t: `Accuracy: ${(m.accuracy*100).toFixed(1)}%` },
      { t: `Loss: ${m.loss.toFixed(4)}` },
    ].forEach(({ t, bold }, i) => {
      this.ctx.font      = bold ? 'bold 12px sans-serif' : '11px sans-serif';
      this.ctx.fillStyle = bold ? '#1e293b' : '#374151';
      this.ctx.textAlign = 'left';
      this.ctx.fillText(t, 18, 26 + i*17);
    });
    this.ctx.restore();

    // Weight diagram (top-right)
    const diagH = Math.min(190, 40 + Math.min(this.w1.length,8)*20);
    this._drawWeightDiagram(W-165, 8, 157, diagH);

    // Confusion matrix
    const labels = this.points.map(pt => pt.label);
    const preds  = this.points.map(pt => this.predict(pt.x, pt.y));
    this.drawConfusionMatrix(this.ctx, labels, preds, 10, H-142, 58);
  }

  computeMetrics() {
    let lossTotal = 0;
    const labels = [], preds = [];
    this.points.forEach(pt => {
      const { output } = this.forward(pt.x, pt.y);
      labels.push(pt.label);
      preds.push(output >= 0.5 ? 1 : 0);
      lossTotal += 0.5*(output - pt.label)**2;
    });
    return { ...this.computeClassificationMetrics(labels, preds), loss: lossTotal/(this.points.length||1) };
  }
}

// ── Neural Network Regression ─────────────────────────────────────

export class NNRegressionSimulation extends BaseSimulation {
  setup() {
    this.history = [];
    this.epoch   = 0;
    this._is3D   = dataStore.is3D && dataStore.type === 'regression';
    const { nPoints, seed, noiseLevel, datasetType, hiddenUnits } = this.params;
    const H = hiddenUnits || 8;
    this.points = this.generateRegressionDataset(datasetType || 'sine', nPoints, seed, noiseLevel ?? 0.2);

    // w1[j] = [w_x, (w_z if 3D,) bias]
    this.w1 = Array.from({ length: H }, (_, i) => {
      const ws = [this.randomBetween(-1, 1, seed + 100 + i * 3)];
      if (this._is3D) ws.push(this.randomBetween(-1, 1, seed + 101 + i * 3));
      ws.push(this.randomBetween(-0.1, 0.1, seed + 102 + i * 3)); // bias
      return ws;
    });
    this.w2 = Array.from({ length: H + 1 }, (_, i) => this.randomBetween(-0.5, 0.5, seed + 200 + i));
  }

  reset() { this.setup(); }

  _forward(x, z) {
    const act    = this.params.activation || 'tanh';
    const hidIn  = this._is3D
      ? this.w1.map(ws => ws[0] * x + ws[1] * (z ?? 0) + ws[2])
      : this.w1.map(ws => ws[0] * x + ws[1]);
    const hidden = hidIn.map(h => applyAct(h, act));
    const body   = [...hidden, 1];
    const out    = this.w2.reduce((s, w, i) => s + w * body[i], 0); // linear output
    return { hidIn, hidden, out, body };
  }

  predict(x, z) { return this._forward(x, z).out; }

  step() {
    if (this.epoch >= this.params.epochs) return;
    const lr  = this.params.learningRate;
    const l2  = this.params.l2 || 0;
    const act = this.params.activation || 'tanh';
    const H   = this.w1.length;
    this.points.forEach(pt => {
      const { hidIn, out, body } = this._forward(pt.x, pt.z);
      const eOut = out - pt.y; // MSE gradient
      for (let i = 0; i < this.w2.length; i++)
        this.w2[i] -= lr * (eOut * body[i] + l2 * this.w2[i]);
      for (let j = 0; j < H; j++) {
        const gHid = eOut * this.w2[j] * actDeriv(hidIn[j], act);
        this.w1[j][0] -= lr * (gHid * pt.x + l2 * this.w1[j][0]);
        if (this._is3D) {
          this.w1[j][1] -= lr * (gHid * (pt.z ?? 0) + l2 * this.w1[j][1]);
          this.w1[j][2] -= lr * gHid;
        } else {
          this.w1[j][1] -= lr * gHid;
        }
      }
    });
    this.epoch++;
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

    if (this.epoch > 0) {
      this.ctx.beginPath(); this.ctx.strokeStyle = '#1d4ed8'; this.ctx.lineWidth = 2;
      for (let i = 0; i <= 200; i++) {
        const x = -1 + (i / 200) * 2, y = clamp(this.predict(x));
        i === 0 ? this.ctx.moveTo(toX(x), toY(y)) : this.ctx.lineTo(toX(x), toY(y));
      }
      this.ctx.stroke();
    }

    this.points.forEach(({ x, y }) => {
      this.ctx.beginPath(); this.ctx.arc(toX(x), toY(clamp(y)), 3.5, 0, Math.PI * 2);
      this.ctx.fillStyle = '#64748b'; this.ctx.fill();
      this.ctx.strokeStyle = '#fff'; this.ctx.lineWidth = 1; this.ctx.stroke();
    });

    const m = this.epoch > 0 ? this.computeMetrics() : null;
    this.ctx.save();
    this.ctx.fillStyle = 'rgba(255,255,255,.93)';
    this.ctx.beginPath(); this.ctx.roundRect(8, 8, 270, 70, 6); this.ctx.fill();
    this.ctx.strokeStyle = '#e2e8f0'; this.ctx.lineWidth = 1; this.ctx.stroke();
    [`Epoch: ${this.epoch} / ${this.params.epochs}  |  H: ${this.w1.length}  ${this.params.activation || 'tanh'}`,
     m ? `MAE: ${m.mae.toFixed(3)}  RMSE: ${m.rmse.toFixed(3)}` : 'Press Run to train',
    ].forEach((line, i) => {
      this.ctx.font = i === 0 ? 'bold 12px sans-serif' : '11px sans-serif';
      this.ctx.fillStyle = i === 0 ? '#1e293b' : '#374151';
      this.ctx.textAlign = 'left'; this.ctx.fillText(line, 18, 26 + i * 17);
    });
    this.ctx.restore();
  }
}

