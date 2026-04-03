import { dataStore } from '../core/dataStore.js';

export class BaseSimulation {
  constructor({ container, params = {} }) {
    this.container = container;
    this.params = params;
    this.canvas = null;
    this.ctx = null;
    this.history = [];
  }

  init() {
    this.container.innerHTML = '';
    this.canvas = document.createElement('canvas');
    this.canvas.width = 600;
    this.canvas.height = 600;
    this.container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d');
    this.setup();
  }

  setup() {
    throw new Error('setup() not implemented');
  }

  reset() {
    this.setup();
  }

  step() {
    throw new Error('step() not implemented');
  }

  render() {
    throw new Error('render() not implemented');
  }

  computeMetrics() {
    return { loss: 0 };
  }

  computeClassificationMetrics(labels, preds) {
    const n = labels.length || 1;
    let tp = 0;
    let tn = 0;
    let fp = 0;
    let fn = 0;

    labels.forEach((trueLabel, i) => {
      const pred = preds[i];
      if (trueLabel === 1 && pred === 1) tp += 1;
      if (trueLabel === 0 && pred === 0) tn += 1;
      if (trueLabel === 0 && pred === 1) fp += 1;
      if (trueLabel === 1 && pred === 0) fn += 1;
    });

    const accuracy = (tp + tn) / n;
    const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
    const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
    const f1 = (precision + recall) === 0 ? 0 : (2 * precision * recall) / (precision + recall);
    const loss = 1 - accuracy;

    return { accuracy, recall, precision, f1, loss };
  }

  computeRegressionMetrics(trueValues, preds) {
    const n = trueValues.length || 1;
    let se = 0;
    let ae = 0;
    let ape = 0;
    let sumTrue = 0;

    trueValues.forEach((trueValue, i) => {
      const p = preds[i];
      const err = p - trueValue;
      se += err * err;
      ae += Math.abs(err);
      ape += trueValue === 0 ? 0 : Math.abs(err / trueValue);
      sumTrue += Math.abs(trueValue);
    });

    const mse = se / n;
    const rmse = Math.sqrt(mse);
    const mae = ae / n;
    const mape = (ape / n) * 100;
    const nmae = sumTrue === 0 ? 0 : mae / (sumTrue / n);
    const loss = mse;

    return { loss, mape, mae, rmse, nmae };
  }

  seededRandom(seed) {
    let x = Math.sin(seed) * 10000;
    return () => {
      x = Math.sin(x) * 10000;
      return x - Math.floor(x);
    };
  }

  randomBetween(min, max, seed) {
    const rand = this.seededRandom(seed);
    return rand() * (max - min) + min;
  }

  // ── Dataset Generators ─────────────────────────────────────
  // Returns [{x, y, label}] — x,y in roughly [-1,1], label in {0,1}
  generateClassDataset(type, nPoints, seed, noiseLevel = 0.08) {
    if (dataStore.points && dataStore.type === 'classification') return [...dataStore.points];
    const pts = [];
    const rng  = this.seededRandom(seed);
    const rand = () => rng() * 2 - 1; // [-1,1]
    const rn   = () => (rng() - 0.5) * 2 * noiseLevel;

    if (type === 'moons') {
      const n = Math.floor(nPoints / 2);
      // Upper moon: arc 0..π, centered ~(-0.5,0.25)
      for (let i = 0; i < n; i++) {
        const t = Math.PI * i / Math.max(n - 1, 1);
        pts.push({ x: Math.cos(t) * 0.75 + rn(), y:  Math.sin(t) * 0.5 - 0.1 + rn(), label: 1 });
      }
      // Lower moon: arc 0..π, mirrored & offset
      for (let i = 0; i < nPoints - n; i++) {
        const t = Math.PI * i / Math.max(nPoints - n - 1, 1);
        pts.push({ x: (1 - Math.cos(t)) * 0.75 - 0.375 + rn(), y: -Math.sin(t) * 0.5 + 0.1 + rn(), label: 0 });
      }
      return pts;
    }

    if (type === 'spiral') {
      const n = Math.floor(nPoints / 2);
      for (let c = 0; c < 2; c++) {
        const count = c === 0 ? n : nPoints - n;
        for (let i = 0; i < count; i++) {
          const r = (i / count) * 0.88;
          const t = (i / count) * 4.5 * Math.PI + c * Math.PI;
          pts.push({ x: r * Math.cos(t) + rn(), y: r * Math.sin(t) + rn(), label: c });
        }
      }
      return pts;
    }

    // Point-based datasets
    for (let i = 0; i < nPoints; i++) {
      const x = rand(), y = rand();
      let label;
      switch (type) {
        case 'xor':         label = (x * y > 0) ? 1 : 0; break;
        case 'circle':      label = (x * x + y * y < 0.45) ? 1 : 0; break;
        case 'checkerboard': label = ((Math.floor((x + 1) * 2) + Math.floor((y + 1) * 2)) % 2 === 0) ? 1 : 0; break;
        case 'diagonal':    label = (x + 0.6 * y + (rng() - 0.5) * 0.28 > 0) ? 1 : 0; break;
        default:            label = (y > x) ? 1 : 0; // 'linear'
      }
      pts.push({ x, y, label });
    }
    return pts;
  }

  // Returns [{x, y}] — x in [-1,1], y approx in [-1,1]
  generateRegressionDataset(type, nPoints, seed, noiseLevel = 0.3) {
    if (dataStore.points && dataStore.type === 'regression') return [...dataStore.points];
    const pts = [];
    const rng  = this.seededRandom(seed);

    for (let i = 0; i < nPoints; i++) {
      const x = rng() * 2 - 1;  // [-1,1]
      const noise = (rng() - 0.5) * 2 * noiseLevel;
      let y;
      switch (type) {
        case 'quadratic': y = 1.2 * x * x - 0.3 + noise; break;
        case 'sine':      y = Math.sin(x * Math.PI * 1.2) * 0.7 + noise; break;
        case 'cubic':     y = 1.0 * x * x * x + 0.2 * x + noise; break;
        case 'noisy':     y = 0.5 * x + 0.1 + (rng() - 0.5) * 2 * 0.7; break;
        default:          y = 0.65 * x + 0.1 + noise; // 'linear'
      }
      pts.push({ x, y: Math.max(-1.2, Math.min(1.2, y)) });
    }
    return pts;
  }

  // Draw a 2x2 confusion matrix at (x0, y0) with given cellSize.
  // labels and preds are arrays of 0/1.
  drawConfusionMatrix(ctx, labels, preds, x0, y0, cellSize = 56) {
    let tp = 0, tn = 0, fp = 0, fn = 0;
    labels.forEach((l, i) => {
      const p = preds[i];
      if (l === 1 && p === 1) tp++;
      else if (l === 0 && p === 0) tn++;
      else if (l === 0 && p === 1) fp++;
      else fn++;
    });

    const boxW = cellSize * 2 + 2;
    const boxH = cellSize * 2 + 2;

    ctx.save();
    // Background panel
    ctx.fillStyle = 'rgba(255,255,255,0.93)';
    ctx.beginPath();
    ctx.roundRect(x0 - 6, y0 - 26, boxW + 12, boxH + 30, 6);
    ctx.fill();
    ctx.strokeStyle = '#d1d5db';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Title
    ctx.fillStyle = '#374151';
    ctx.font = 'bold 10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Confusion Matrix', x0 + cellSize, y0 - 12);

    // Column headers
    ctx.font = '9px sans-serif';
    ctx.fillStyle = '#6b7280';
    ctx.fillText('Pred 0', x0 + cellSize * 0.5, y0 - 2);
    ctx.fillText('Pred 1', x0 + cellSize * 1.5, y0 - 2);

    const cells = [
      { label: 'TN', value: tn, correct: true,  row: 0, col: 0 },
      { label: 'FP', value: fp, correct: false, row: 0, col: 1 },
      { label: 'FN', value: fn, correct: false, row: 1, col: 0 },
      { label: 'TP', value: tp, correct: true,  row: 1, col: 1 },
    ];

    cells.forEach(({ label, value, correct, row, col }) => {
      const cx = x0 + col * cellSize;
      const cy = y0 + row * cellSize;
      ctx.fillStyle = correct ? 'rgba(34,197,94,0.78)' : 'rgba(239,68,68,0.72)';
      ctx.fillRect(cx, cy, cellSize, cellSize);
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(cx, cy, cellSize, cellSize);

      ctx.fillStyle = '#fff';
      ctx.font = 'bold 10px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(label, cx + cellSize / 2, cy + cellSize / 2 - 5);
      ctx.font = '11px sans-serif';
      ctx.fillText(value, cx + cellSize / 2, cy + cellSize / 2 + 9);

      // Row label (left side)
      if (col === 0) {
        ctx.fillStyle = '#6b7280';
        ctx.font = '9px sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(row === 0 ? 'Act 0' : 'Act 1', cx - 3, cy + cellSize / 2 + 3);
      }
    });

    ctx.restore();
  }
}
