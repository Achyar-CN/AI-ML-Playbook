import { dataStore } from '../core/dataStore.js';

export class BaseSimulation {
  constructor({ container, params = {}, taskType = null }) {
    this.container = container;
    this.params    = params;
    this.taskType  = taskType;   // 'classification' | 'regression' — set by SimulationManager
    this.canvas    = null;
    this.ctx       = null;
    this.history   = [];
    this.testPoints = [];        // held-out test set (set by _splitAndStore)
  }

  init() {
    this.container.innerHTML = '';
    this.canvas = document.createElement('canvas');
    this.canvas.width  = 600;
    this.canvas.height = 600;
    this.container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d');
    // 3D rotation state (reset on each init so re-selecting resets view)
    this._3dRotX  = 0.38;   // elevation  (~22°)
    this._3dRotY  = 0.62;   // azimuth    (~35°)
    this._3dBound = false;
    this._3dDrag  = null;
    this.setup();
  }

  setup()  { throw new Error('setup() not implemented'); }
  reset()  { this.setup(); }
  step()   { throw new Error('step() not implemented'); }
  render() { throw new Error('render() not implemented'); }

  computeMetrics() { return { loss: 0 }; }

  // ── Public render entry-point called by SimulationManager ───────
  // Calls the subclass render(), then draws overlays (axis labels, test
  // points, 3D scatter, PCA badge, train/test legend).
  renderWithOverlays() {
    const W = this.canvas.width, H = this.canvas.height;
    const ctx = this.ctx;

    if (dataStore.is3D) {
      // Bind drag-to-rotate once per canvas instance
      if (!this._3dBound) this._bind3DInteraction();
      // 3D mode: replace normal viz with 3D scatter (draws its own axes + legend)
      this._draw3DScatter(ctx, W, H);
      return;
    }

    this.render();
    // Overlay hollow test-point markers on top of the simulation canvas
    if (this.testPoints?.length > 0 && this.showTestOverlay !== false) {
      if (this.taskType === 'classification') {
        this._drawTestPointsClass(ctx, W, H);
      } else if (this.taskType === 'regression') {
        this._drawTestPointsReg(ctx, W, H);
      }
    }

    // Axis labels: use CSV names if available, else sensible defaults
    const xLbl = dataStore.xLabel
      || (this.taskType === 'regression' ? 'x' : 'x₁');
    const yLbl = dataStore.yLabel
      || (this.taskType === 'regression' ? 'y' : 'x₂');
    this._drawAxisLabels(ctx, W, H, xLbl, yLbl);

    // PCA badge
    if (dataStore.pcaInfo) {
      this._drawPCABadge(ctx, W, H);
    }

    // Train / Test legend
    if (this.testPoints?.length > 0 && this.showTestOverlay !== false) {
      this._drawTrainTestLegend(ctx, W, H);
    }
  }

  // ── Inject test metrics into the last history entry ─────────────
  // Called by SimulationManager after each step().
  _injectTestMetrics() {
    if (!this.testPoints?.length || !this.history.length) return;
    const last = this.history[this.history.length - 1];
    if (last._testDone) return;
    last._testDone = true;

    if (typeof this.predict !== 'function') return;

    if (this.taskType === 'classification') {
      const labels = this.testPoints.map(pt => pt.label === 1 ? 1 : 0);
      const preds  = this.testPoints.map(pt => this.predict(pt.x, pt.y) === 1 ? 1 : 0);
      const m = this.computeClassificationMetrics(labels, preds);
      Object.assign(last, {
        testLoss: m.loss, testAccuracy: m.accuracy,
        testRecall: m.recall, testPrecision: m.precision, testF1: m.f1,
      });
    } else if (this.taskType === 'regression') {
      const trues = this.testPoints.map(pt => pt.y);
      const preds = dataStore.is3D
        ? this.testPoints.map(pt => this.predict(pt.x, pt.z))
        : this.testPoints.map(pt => this.predict(pt.x));
      const m = this.computeRegressionMetrics(trues, preds);
      Object.assign(last, {
        testLoss: m.mse, testMAE: m.mae,
        testRMSE: m.rmse, testMAPE: m.mape, testNMAE: m.nmae,
      });
    }
  }

  // ── Train / Test split ──────────────────────────────────────────
  // Shuffles `all` deterministically, stores the test fraction in
  // this.testPoints, and returns the train fraction.
  _splitAndStore(all) {
    const ratio = Number(this.params?.testSplit ?? 0.2);
    if (ratio <= 0 || all.length < 4) {
      this.testPoints = [];
      return all;
    }
    const shuffled = this._shuffle(all, (this.params?.seed ?? 42) + 77777);
    const trainEnd = Math.max(2, Math.floor(shuffled.length * (1 - ratio)));
    this.testPoints = shuffled.slice(trainEnd);
    return shuffled.slice(0, trainEnd);
  }

  // Deterministic Fisher-Yates using seededRandom
  _shuffle(arr, seed) {
    const a = [...arr];
    const rng = this.seededRandom(seed);
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  // ── Canvas overlays ─────────────────────────────────────────────

  // Hollow square markers for test points (classification)
  _drawTestPointsClass(ctx, W, H) {
    if (!this.testPoints?.length) return;
    ctx.save();
    this.testPoints.forEach(({ x, y, label }) => {
      const px = ((x + 1) / 2) * W;
      const py = H - ((y + 1) / 2) * H;
      const r  = 5;
      ctx.strokeStyle = label === 1 ? '#1565c0' : '#c62828';
      ctx.lineWidth   = 2;
      ctx.fillStyle   = 'rgba(255,255,255,0.55)';
      ctx.beginPath();
      ctx.rect(px - r, py - r, r * 2, r * 2);
      ctx.fill();
      ctx.stroke();
    });
    ctx.restore();
  }

  // Hollow diamond markers for test points (regression)
  _drawTestPointsReg(ctx, W, H, PAD = 36) {
    if (!this.testPoints?.length) return;
    ctx.save();
    this.testPoints.forEach(({ x, y }) => {
      const px = PAD + ((x + 1) / 2) * (W - 2 * PAD);
      const py = H - PAD - ((y + 1.2) / 2.4) * (H - 2 * PAD);
      const r  = 4;
      ctx.strokeStyle = '#1d4ed8';
      ctx.lineWidth   = 1.5;
      ctx.fillStyle   = 'rgba(255,255,255,0.55)';
      ctx.beginPath();
      ctx.moveTo(px,     py - r);
      ctx.lineTo(px + r, py);
      ctx.lineTo(px,     py + r);
      ctx.lineTo(px - r, py);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    });
    ctx.restore();
  }

  // Axis labels (bottom = xLabel, left side = yLabel rotated)
  _drawAxisLabels(ctx, W, H, xLabel, yLabel) {
    const dark = document.documentElement.dataset.theme === 'dark';
    const col  = dark ? '#94a3b8' : '#64748b';
    ctx.save();
    ctx.font      = 'bold 11px sans-serif';
    ctx.fillStyle = col;

    if (xLabel) {
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(xLabel, W / 2, H - 2);
    }
    if (yLabel) {
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'top';
      ctx.save();
      ctx.translate(12, H / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(yLabel, 0, 0);
      ctx.restore();
    }
    ctx.restore();
  }

  // Small "PCA" badge in bottom-right
  _drawPCABadge(ctx, W, H) {
    const dark = document.documentElement.dataset.theme === 'dark';
    ctx.save();
    ctx.font      = 'bold 10px sans-serif';
    ctx.fillStyle = dark ? '#334155' : '#f0f9ff';
    ctx.strokeStyle = dark ? '#475569' : '#bae6fd';
    ctx.lineWidth = 1;
    const label = 'PCA';
    const tw = ctx.measureText(label).width;
    const bx = W - tw - 20, by = H - 22, bw = tw + 12, bh = 16;
    ctx.beginPath(); ctx.roundRect(bx, by, bw, bh, 4); ctx.fill(); ctx.stroke();
    ctx.fillStyle = dark ? '#7dd3fc' : '#0369a1';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(label, bx + bw / 2, by + bh / 2);
    ctx.restore();
  }

  // Train/Test legend (bottom-right or top-right area)
  _drawTrainTestLegend(ctx, W, H) {
    if (!this.testPoints?.length) return;
    const dark = document.documentElement.dataset.theme === 'dark';
    const bg   = dark ? 'rgba(15,23,42,0.88)' : 'rgba(255,255,255,0.88)';
    const txt  = dark ? '#e2e8f0' : '#374151';

    ctx.save();
    const bx = W - 110, by = H - 56, bw = 104, bh = 48;
    ctx.fillStyle = bg;
    ctx.strokeStyle = dark ? '#334155' : '#e2e8f0';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.roundRect(bx, by, bw, bh, 5); ctx.fill(); ctx.stroke();

    // Train dot (filled circle)
    ctx.fillStyle = '#64748b';
    ctx.beginPath(); ctx.arc(bx + 12, by + 14, 4, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = txt; ctx.font = '10px sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText(`Train (${this.points?.length ?? '?'})`, bx + 20, by + 14);

    // Test marker (hollow square)
    ctx.strokeStyle = '#64748b'; ctx.lineWidth = 1.5; ctx.fillStyle = 'rgba(255,255,255,0.55)';
    ctx.beginPath(); ctx.rect(bx + 8, by + 30, 8, 8); ctx.fill(); ctx.stroke();
    ctx.fillStyle = txt;
    ctx.fillText(`Test  (${this.testPoints.length})`, bx + 20, by + 34);

    ctx.restore();
  }

  // ── Drag-to-rotate event binding ───────────────────────────────
  _bind3DInteraction() {
    this._3dBound = true;
    const canvas  = this.canvas;
    canvas.style.cursor = 'grab';

    const onDown = (clientX, clientY) => {
      const rect = canvas.getBoundingClientRect();
      this._3dDrag = {
        x: clientX - rect.left, y: clientY - rect.top,
        rotX: this._3dRotX, rotY: this._3dRotY,
      };
      canvas.style.cursor = 'grabbing';
    };

    const onMove = (clientX, clientY) => {
      if (!this._3dDrag) return;
      const rect = canvas.getBoundingClientRect();
      const dx   = (clientX - rect.left) - this._3dDrag.x;
      const dy   = (clientY - rect.top)  - this._3dDrag.y;
      this._3dRotY = this._3dDrag.rotY + dx * 0.012;
      this._3dRotX = Math.max(-Math.PI / 2 + 0.05,
        Math.min(Math.PI / 2 - 0.05, this._3dDrag.rotX - dy * 0.012));
      this._draw3DScatter(this.ctx, canvas.width, canvas.height);
    };

    const onUp = () => { this._3dDrag = null; canvas.style.cursor = 'grab'; };

    canvas.addEventListener('mousedown',  e => onDown(e.clientX, e.clientY));
    canvas.addEventListener('mousemove',  e => onMove(e.clientX, e.clientY));
    canvas.addEventListener('mouseup',    onUp);
    canvas.addEventListener('mouseleave', onUp);

    canvas.addEventListener('touchstart', e => {
      e.preventDefault();
      onDown(e.touches[0].clientX, e.touches[0].clientY);
    }, { passive: false });
    canvas.addEventListener('touchmove', e => {
      e.preventDefault();
      onMove(e.touches[0].clientX, e.touches[0].clientY);
    }, { passive: false });
    canvas.addEventListener('touchend', onUp);
  }

  // ── 3D Scatter (replaces normal render when dataStore.is3D=true) ──
  _draw3DScatter(ctx, W, H) {
    const dark  = document.documentElement.dataset.theme === 'dark';
    const bg    = dark ? '#0f172a' : '#ffffff';
    const gridC = dark ? '#1e293b' : '#f1f5f9';
    const axC   = dark ? '#475569' : '#94a3b8';
    const txtC  = dark ? '#94a3b8' : '#64748b';

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);

    const scale  = Math.min(W, H) * 0.28;
    const cx     = W * 0.5;
    const cy     = H * 0.5;
    const rotX   = this._3dRotX ?? 0.38;
    const rotY   = this._3dRotY ?? 0.62;

    // Rotation-matrix projection:
    //  1. Rotate around Y (azimuth):  x' = x·cosY + z·sinY,  z' = -x·sinY + z·cosY
    //  2. Rotate around X (elevation): y'' = y·cosX - z'·sinX, z'' = y·sinX + z'·cosX
    //  3. Orthographic project: sx = cx + x'·scale,  sy = cy - y''·scale
    const project = (x, y, z) => {
      const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
      const cosX = Math.cos(rotX), sinX = Math.sin(rotX);
      const x1  =  x * cosY + z * sinY;
      const z1  = -x * sinY + z * cosY;
      const y2  =  y * cosX - z1 * sinX;
      const z2  =  y * sinX + z1 * cosX;
      return { sx: cx + x1 * scale, sy: cy - y2 * scale, depth: z2 };
    };

    // Floor grid on y = -1 plane
    ctx.strokeStyle = gridC; ctx.lineWidth = 0.8;
    for (let t = -1; t <= 1; t += 0.5) {
      const a = project(t, -1, -1), b = project(t, -1, 1);
      ctx.beginPath(); ctx.moveTo(a.sx, a.sy); ctx.lineTo(b.sx, b.sy); ctx.stroke();
      const c = project(-1, -1, t), d = project(1, -1, t);
      ctx.beginPath(); ctx.moveTo(c.sx, c.sy); ctx.lineTo(d.sx, d.sy); ctx.stroke();
    }

    // ── Classification: decision boundary on floor (y = -1) ──────
    const isClassification = this.taskType === 'classification';
    const isRegression     = this.taskType === 'regression';
    const hasTrained       = typeof this.predict === 'function' && this.epoch > 0;

    if (isClassification && hasTrained) {
      const G = 24; // grid resolution for floor boundary
      const step = 2 / G;
      for (let gx = 0; gx < G; gx++) {
        for (let gz = 0; gz < G; gz++) {
          const fx = -1 + gx * step;
          const fz = -1 + gz * step;
          // Model predicts using x,y (first 2 features); in 3D, x→x, y→y of data
          // but floor uses x→feature1, z→feature2; data.y = feature2
          const pred = this.predict(fx, fz);
          const col  = pred === 1
            ? 'rgba(29,78,216,0.18)' : 'rgba(220,38,38,0.18)';
          // Draw filled quad on y=-1 floor
          const c0 = project(fx,        -1, fz);
          const c1 = project(fx + step,  -1, fz);
          const c2 = project(fx + step,  -1, fz + step);
          const c3 = project(fx,         -1, fz + step);
          ctx.fillStyle = col;
          ctx.beginPath();
          ctx.moveTo(c0.sx, c0.sy);
          ctx.lineTo(c1.sx, c1.sy);
          ctx.lineTo(c2.sx, c2.sy);
          ctx.lineTo(c3.sx, c3.sy);
          ctx.closePath();
          ctx.fill();
        }
      }
    }

    // ── Regression: prediction surface mesh ──────────────────────
    if (isRegression && hasTrained) {
      const G = 20;
      const step = 2 / G;
      // Build grid of predicted y values
      const grid = [];
      for (let gx = 0; gx <= G; gx++) {
        grid[gx] = [];
        for (let gz = 0; gz <= G; gz++) {
          const fx = -1 + gx * step;
          const fz = -1 + gz * step;
          const py = Math.max(-1, Math.min(1, this.predict(fx, fz)));
          grid[gx][gz] = py;
        }
      }
      // Draw surface quads (semi-transparent blue)
      for (let gx = 0; gx < G; gx++) {
        for (let gz = 0; gz < G; gz++) {
          const fx = -1 + gx * step;
          const fz = -1 + gz * step;
          const c0 = project(fx,        grid[gx][gz],       fz);
          const c1 = project(fx + step,  grid[gx+1][gz],     fz);
          const c2 = project(fx + step,  grid[gx+1][gz+1],   fz + step);
          const c3 = project(fx,         grid[gx][gz+1],     fz + step);
          const avgDepth = (c0.depth + c1.depth + c2.depth + c3.depth) / 4;
          // Color by average height
          const avgY = (grid[gx][gz] + grid[gx+1][gz] + grid[gx+1][gz+1] + grid[gx][gz+1]) / 4;
          const t = (avgY + 1) / 2; // 0..1
          const r = Math.round(30 + t * 190);
          const g = Math.round(60 + (1 - Math.abs(t - 0.5) * 2) * 100);
          const b = Math.round(220 - t * 190);
          ctx.fillStyle = `rgba(${r},${g},${b},0.25)`;
          ctx.strokeStyle = `rgba(${r},${g},${b},0.35)`;
          ctx.lineWidth = 0.5;
          ctx.beginPath();
          ctx.moveTo(c0.sx, c0.sy);
          ctx.lineTo(c1.sx, c1.sy);
          ctx.lineTo(c2.sx, c2.sy);
          ctx.lineTo(c3.sx, c3.sy);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
        }
      }
    }

    // Axes from origin (-1,-1,-1)
    const O  = project(-1, -1, -1);
    const Xe = project( 1, -1, -1);
    const Ye = project(-1,  1, -1);
    const Ze = project(-1, -1,  1);

    // Classification: swap y↔z so feature2 goes to z-axis (floor) for boundary viz
    //   X-axis = feature1, Z-axis = feature2, Y-axis(vertical) = feature3
    // Regression: x=f1, y(vertical)=target, z=f2
    const axLabels = isRegression
      ? [[Xe, dataStore.xLabel || 'x₁'],
         [Ye, dataStore.targetName || 'y'],
         [Ze, dataStore.zLabel || 'x₂']]
      : [[Xe, dataStore.xLabel || 'x₁'],
         [Ye, dataStore.zLabel || 'x₃'],
         [Ze, dataStore.yLabel || 'x₂']];

    axLabels.forEach(([end, label]) => {
      ctx.strokeStyle = axC; ctx.lineWidth = 1.8;
      ctx.beginPath(); ctx.moveTo(O.sx, O.sy); ctx.lineTo(end.sx, end.sy); ctx.stroke();

      const ang = Math.atan2(end.sy - O.sy, end.sx - O.sx);
      const al  = 7;
      ctx.fillStyle = axC;
      ctx.beginPath();
      ctx.moveTo(end.sx, end.sy);
      ctx.lineTo(end.sx - al * Math.cos(ang - 0.4), end.sy - al * Math.sin(ang - 0.4));
      ctx.lineTo(end.sx - al * Math.cos(ang + 0.4), end.sy - al * Math.sin(ang + 0.4));
      ctx.closePath(); ctx.fill();

      const lx = end.sx + (end.sx - O.sx) * 0.14;
      const ly = end.sy + (end.sy - O.sy) * 0.14;
      ctx.fillStyle = txtC; ctx.font = 'bold 11px sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(label, lx, ly);
    });

    // Collect points, sort back-to-front by projected depth (painter's algorithm)
    const trainPts = (this.points || []).map(pt => ({ ...pt, _train: true }));
    const testPts  = (this.testPoints || []).map(pt => ({ ...pt, _train: false }));
    const all = [...trainPts, ...testPts];

    const projected = all.map(pt => {
      const px = pt.x;
      // Classification: swap y↔z so feature2→z-axis, feature3→vertical
      const py = isClassification ? (pt.z ?? 0) : (pt.y ?? 0);
      const pz = isClassification ? (pt.y ?? 0) : (pt.z ?? 0);
      const { sx, sy, depth } = project(px, py, pz);
      return { pt, sx, sy, depth };
    });
    projected.sort((a, b) => a.depth - b.depth); // draw far → near

    projected.forEach(({ pt, sx, sy }) => {
      if (isClassification) {
        // Classification: colored by label, shapes differ for train/test
        const col1 = pt.label === 1 ? '#1565c0' : '#c62828';
        const col0 = pt.label === 1 ? 'rgba(21,101,192,0.28)' : 'rgba(198,40,40,0.28)';

        if (pt._train) {
          // Train: filled circle
          ctx.beginPath(); ctx.arc(sx, sy, 4.5, 0, Math.PI * 2);
          ctx.fillStyle   = col1; ctx.fill();
          ctx.strokeStyle = 'rgba(255,255,255,0.6)'; ctx.lineWidth = 0.8; ctx.stroke();
        } else {
          // Test: hollow square (consistent with 2D)
          const r = 5;
          ctx.fillStyle   = 'rgba(255,255,255,0.55)';
          ctx.strokeStyle = col1; ctx.lineWidth = 2;
          ctx.beginPath(); ctx.rect(sx - r, sy - r, r * 2, r * 2);
          ctx.fill(); ctx.stroke();
        }
      } else {
        // Regression: single color, shapes differ for train/test
        const col = '#1d4ed8';
        if (pt._train) {
          ctx.beginPath(); ctx.arc(sx, sy, 4, 0, Math.PI * 2);
          ctx.fillStyle = '#64748b'; ctx.fill();
          ctx.strokeStyle = 'rgba(255,255,255,0.6)'; ctx.lineWidth = 0.8; ctx.stroke();
        } else {
          // Test: hollow diamond (consistent with 2D)
          const r = 4.5;
          ctx.fillStyle   = 'rgba(255,255,255,0.55)';
          ctx.strokeStyle = col; ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(sx,     sy - r);
          ctx.lineTo(sx + r, sy);
          ctx.lineTo(sx,     sy + r);
          ctx.lineTo(sx - r, sy);
          ctx.closePath();
          ctx.fill(); ctx.stroke();
        }
      }
    });

    // Info + drag hint badge
    ctx.save();
    ctx.fillStyle   = dark ? 'rgba(15,23,42,0.88)' : 'rgba(255,255,255,0.88)';
    ctx.strokeStyle = dark ? '#334155' : '#e2e8f0'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.roundRect(8, 8, 320, 48, 6); ctx.fill(); ctx.stroke();
    ctx.fillStyle = dark ? '#94a3b8' : '#475569';
    ctx.font = '11px sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    if (isRegression) {
      ctx.fillText('3D Scatter  •  Model trains on ' +
        (dataStore.xLabel || 'x₁') + ' & ' + (dataStore.zLabel || 'x₂') +
        ' → ' + (dataStore.targetName || 'y'), 14, 22);
    } else {
      ctx.fillText('3D Scatter  •  Model trains on ' +
        (dataStore.xLabel || 'x₁') + ' & ' + (dataStore.yLabel || 'x₂'), 14, 22);
    }
    ctx.fillStyle = dark ? '#64748b' : '#94a3b8';
    ctx.font = '10px sans-serif';
    ctx.fillText('Drag to rotate  •  Touch supported', 14, 40);
    ctx.restore();

    // Train / Test legend for 3D
    if (this.testPoints?.length > 0) {
      this._drawTrainTestLegend(ctx, W, H);
    }
  }

  // ── Classification metrics ──────────────────────────────────────
  computeClassificationMetrics(labels, preds) {
    const n = labels.length || 1;
    let tp = 0, tn = 0, fp = 0, fn = 0;
    labels.forEach((trueLabel, i) => {
      const pred = preds[i];
      if (trueLabel === 1 && pred === 1) tp++;
      else if (trueLabel === 0 && pred === 0) tn++;
      else if (trueLabel === 0 && pred === 1) fp++;
      else fn++;
    });
    const accuracy  = (tp + tn) / n;
    const recall    = tp + fn === 0 ? 0 : tp / (tp + fn);
    const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
    const f1        = (precision + recall) === 0 ? 0 : (2 * precision * recall) / (precision + recall);
    return { accuracy, recall, precision, f1, loss: 1 - accuracy };
  }

  // ── Regression metrics ──────────────────────────────────────────
  computeRegressionMetrics(trueValues, preds) {
    const n = trueValues.length || 1;
    let se = 0, ae = 0, ape = 0, sumTrue = 0;
    trueValues.forEach((tv, i) => {
      const err   = preds[i] - tv;
      const absTv = Math.abs(tv);
      se      += err * err;
      ae      += Math.abs(err);
      // Skip near-zero values to avoid MAPE explosion (normalised data crosses 0)
      ape     += absTv < 0.1 ? 0 : Math.abs(err) / absTv;
      sumTrue += absTv;
    });
    const mse  = se / n;
    const mae  = ae / n;
    return {
      loss: mse, mse,
      rmse: Math.sqrt(mse),
      mae,
      mape: (ape / n) * 100,
      nmae: sumTrue === 0 ? 0 : mae / (sumTrue / n),
    };
  }

  // ── Dataset generators ──────────────────────────────────────────
  // Returns trainPoints (sets this.testPoints as side-effect).

  generateClassDataset(type, nPoints, seed, noiseLevel = 0.08) {
    const raw = dataStore.points && dataStore.type === 'classification'
      ? [...dataStore.points]
      : this._generateClassRaw(type, nPoints, seed, noiseLevel);
    return this._splitAndStore(raw);
  }

  generateRegressionDataset(type, nPoints, seed, noiseLevel = 0.3) {
    const raw = dataStore.points && dataStore.type === 'regression'
      ? [...dataStore.points]
      : this._generateRegRaw(type, nPoints, seed, noiseLevel);
    return this._splitAndStore(raw);
  }

  // ── Raw dataset generators (internal) ──────────────────────────
  _generateClassRaw(type, nPoints, seed, noiseLevel) {
    const pts  = [];
    const rng  = this.seededRandom(seed);
    const rand = () => rng() * 2 - 1;
    const rn   = () => (rng() - 0.5) * 2 * noiseLevel;

    if (type === 'moons') {
      const n = Math.floor(nPoints / 2);
      for (let i = 0; i < n; i++) {
        const t = Math.PI * i / Math.max(n - 1, 1);
        pts.push({ x: Math.cos(t) * 0.75 + rn(), y: Math.sin(t) * 0.5 - 0.1 + rn(), label: 1 });
      }
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
    for (let i = 0; i < nPoints; i++) {
      const x = rand(), y = rand();
      let label;
      switch (type) {
        case 'xor':          label = (x * y > 0) ? 1 : 0; break;
        case 'circle':       label = (x * x + y * y < 0.45) ? 1 : 0; break;
        case 'checkerboard': label = ((Math.floor((x + 1) * 2) + Math.floor((y + 1) * 2)) % 2 === 0) ? 1 : 0; break;
        case 'diagonal':     label = (x + 0.6 * y + (rng() - 0.5) * 0.28 > 0) ? 1 : 0; break;
        default:             label = (y > x) ? 1 : 0;
      }
      pts.push({ x, y, label });
    }
    return pts;
  }

  _generateRegRaw(type, nPoints, seed, noiseLevel) {
    const pts = [];
    const rng = this.seededRandom(seed);
    for (let i = 0; i < nPoints; i++) {
      const x = rng() * 2 - 1;
      const noise = (rng() - 0.5) * 2 * noiseLevel;
      let y;
      switch (type) {
        case 'quadratic': y = 1.2 * x * x - 0.3 + noise; break;
        case 'sine':      y = Math.sin(x * Math.PI * 1.2) * 0.7 + noise; break;
        case 'cubic':     y = x * x * x + 0.2 * x + noise; break;
        case 'noisy':     y = 0.5 * x + 0.1 + (rng() - 0.5) * 2 * 0.7; break;
        default:          y = 0.65 * x + 0.1 + noise;
      }
      pts.push({ x, y: Math.max(-1.2, Math.min(1.2, y)) });
    }
    return pts;
  }

  // ── Seeded RNG ──────────────────────────────────────────────────
  seededRandom(seed) {
    let x = Math.sin(seed) * 10000;
    return () => { x = Math.sin(x) * 10000; return x - Math.floor(x); };
  }

  randomBetween(min, max, seed) {
    return this.seededRandom(seed)() * (max - min) + min;
  }

  // ── Confusion matrix ────────────────────────────────────────────
  drawConfusionMatrix(ctx, labels, preds, x0, y0, cellSize = 56) {
    let tp = 0, tn = 0, fp = 0, fn = 0;
    labels.forEach((l, i) => {
      const p = preds[i];
      if (l === 1 && p === 1) tp++;
      else if (l === 0 && p === 0) tn++;
      else if (l === 0 && p === 1) fp++;
      else fn++;
    });

    const boxW = cellSize * 2 + 2, boxH = cellSize * 2 + 2;
    ctx.save();
    ctx.fillStyle = 'rgba(255,255,255,0.93)';
    ctx.beginPath(); ctx.roundRect(x0 - 6, y0 - 26, boxW + 12, boxH + 30, 6); ctx.fill();
    ctx.strokeStyle = '#d1d5db'; ctx.lineWidth = 1; ctx.stroke();

    ctx.fillStyle = '#374151'; ctx.font = 'bold 10px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('Confusion Matrix', x0 + cellSize, y0 - 12);

    ctx.font = '9px sans-serif'; ctx.fillStyle = '#6b7280';
    ctx.fillText('Pred 0', x0 + cellSize * 0.5, y0 - 2);
    ctx.fillText('Pred 1', x0 + cellSize * 1.5, y0 - 2);

    [
      { label: 'TN', value: tn, correct: true,  row: 0, col: 0 },
      { label: 'FP', value: fp, correct: false, row: 0, col: 1 },
      { label: 'FN', value: fn, correct: false, row: 1, col: 0 },
      { label: 'TP', value: tp, correct: true,  row: 1, col: 1 },
    ].forEach(({ label, value, correct, row, col }) => {
      const cx = x0 + col * cellSize, cy = y0 + row * cellSize;
      ctx.fillStyle = correct ? 'rgba(34,197,94,0.78)' : 'rgba(239,68,68,0.72)';
      ctx.fillRect(cx, cy, cellSize, cellSize);
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.strokeRect(cx, cy, cellSize, cellSize);
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 10px sans-serif'; ctx.textAlign = 'center';
      ctx.fillText(label, cx + cellSize / 2, cy + cellSize / 2 - 5);
      ctx.font = '11px sans-serif';
      ctx.fillText(value, cx + cellSize / 2, cy + cellSize / 2 + 9);
      if (col === 0) {
        ctx.fillStyle = '#6b7280'; ctx.font = '9px sans-serif'; ctx.textAlign = 'right';
        ctx.fillText(row === 0 ? 'Act 0' : 'Act 1', cx - 3, cy + cellSize / 2 + 3);
      }
    });
    ctx.restore();
  }
}
