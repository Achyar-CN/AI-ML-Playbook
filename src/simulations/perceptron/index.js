import { BaseSimulation } from '../baseSimulation.js';

export class PerceptronSimulation extends BaseSimulation {
  setup() {
    this.history = [];
    this.points = [];
    const { nPoints, seed } = this.params;
    this.weights = [
      this.randomBetween(-1, 1, seed + 1),
      this.randomBetween(-1, 1, seed + 2),
      this.randomBetween(-1, 1, seed + 3),
    ];
    this.epoch = 0;

    for (let i = 0; i < nPoints; i++) {
      const x = this.randomBetween(-1, 1, seed + 10 + i * 2);
      const y = this.randomBetween(-1, 1, seed + 11 + i * 2);
      const label = y > x ? 1 : -1;
      this.points.push({ x, y, label });
    }
  }

  predict(x, y) {
    const sum = this.weights[0] * x + this.weights[1] * y + this.weights[2];
    return sum >= 0 ? 1 : -1;
  }

  step() {
    if (this.epoch >= this.params.epochs) return;

    const lr = this.params.learningRate;
    this.points.forEach((pt) => {
      const guess = this.predict(pt.x, pt.y);
      const error = pt.label - guess;
      this.weights[0] += lr * error * pt.x;
      this.weights[1] += lr * error * pt.y;
      this.weights[2] += lr * error;
    });
    this.epoch++;

    const metrics = this.computeMetrics();
    this.history.push({ epoch: this.epoch, ...metrics });
  }

  render() {
    const { width, height } = this.canvas;
    this.ctx.clearRect(0, 0, width, height);
    this.ctx.fillStyle = '#ffffff';
    this.ctx.fillRect(0, 0, width, height);

    // Axis guide lines
    this.ctx.strokeStyle = '#e2e8f0';
    this.ctx.lineWidth = 1;
    this.ctx.beginPath();
    this.ctx.moveTo(width / 2, 0);
    this.ctx.lineTo(width / 2, height);
    this.ctx.moveTo(0, height / 2);
    this.ctx.lineTo(width, height / 2);
    this.ctx.stroke();

    // True boundary (y = x, dashed gray)
    this.ctx.strokeStyle = 'rgba(148, 163, 184, 0.6)';
    this.ctx.lineWidth = 1.5;
    this.ctx.setLineDash([6, 4]);
    this.ctx.beginPath();
    this.ctx.moveTo(0, height);
    this.ctx.lineTo(width, 0);
    this.ctx.stroke();
    this.ctx.setLineDash([]);

    // Data points
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

    // Learned decision boundary
    if (this.weights[1] !== 0) {
      const xLeft = -1, xRight = 1;
      const yLeft  = -(this.weights[2] + this.weights[0] * xLeft)  / this.weights[1];
      const yRight = -(this.weights[2] + this.weights[0] * xRight) / this.weights[1];
      this.ctx.strokeStyle = '#0f172a';
      this.ctx.lineWidth = 2.5;
      this.ctx.beginPath();
      this.ctx.moveTo(((xLeft  + 1) / 2) * width, height - ((yLeft  + 1) / 2) * height);
      this.ctx.lineTo(((xRight + 1) / 2) * width, height - ((yRight + 1) / 2) * height);
      this.ctx.stroke();
    }

    // Info panel (top-left)
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
    this.ctx.fillText(`Accuracy: ${(metrics.accuracy * 100).toFixed(1)}%`, 18, 46);
    this.ctx.fillText(`Loss: ${metrics.loss.toFixed(3)}`, 18, 62);
    this.ctx.fillText(`F1: ${metrics.f1.toFixed(3)}`, 18, 78);
    const w = this.weights;
    this.ctx.fillText(`w: [${w[0].toFixed(2)}, ${w[1].toFixed(2)}, ${w[2].toFixed(2)}]`, 18, 94);
    this.ctx.restore();

    // Confusion matrix (bottom-left)
    const labels = this.points.map(pt => (pt.label === 1 ? 1 : 0));
    const preds  = this.points.map(pt => (this.predict(pt.x, pt.y) === 1 ? 1 : 0));
    this.drawConfusionMatrix(this.ctx, labels, preds, 10, height - 142, 58);

    // Legend (bottom-right)
    this.ctx.save();
    this.ctx.fillStyle = 'rgba(255,255,255,0.92)';
    this.ctx.beginPath();
    this.ctx.roundRect(width - 168, height - 74, 160, 66, 6);
    this.ctx.fill();
    this.ctx.strokeStyle = '#d1d5db';
    this.ctx.lineWidth = 1;
    this.ctx.stroke();

    this.ctx.fillStyle = '#374151';
    this.ctx.font = 'bold 10px sans-serif';
    this.ctx.textAlign = 'left';
    this.ctx.fillText('Legend', width - 154, height - 58);

    [[14, '#1565c0', 'Class +1 (y > x)'],
     [30, '#c62828', 'Class -1 (y ≤ x)']].forEach(([dy, color, text]) => {
      this.ctx.beginPath();
      this.ctx.arc(width - 150, height - 74 + dy, 5, 0, Math.PI * 2);
      this.ctx.fillStyle = color;
      this.ctx.fill();
      this.ctx.fillStyle = '#374151';
      this.ctx.font = '10px sans-serif';
      this.ctx.fillText(text, width - 140, height - 74 + dy + 4);
    });

    // Boundary lines in legend
    this.ctx.strokeStyle = '#0f172a';
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();
    this.ctx.moveTo(width - 150, height - 74 + 46);
    this.ctx.lineTo(width - 130, height - 74 + 46);
    this.ctx.stroke();
    this.ctx.fillStyle = '#374151';
    this.ctx.font = '10px sans-serif';
    this.ctx.fillText('Learned boundary', width - 126, height - 74 + 50);

    this.ctx.restore();
  }

  computeMetrics() {
    const labels = this.points.map(pt => (pt.label === 1 ? 1 : 0));
    const preds  = this.points.map(pt => (this.predict(pt.x, pt.y) === 1 ? 1 : 0));
    return this.computeClassificationMetrics(labels, preds);
  }
}
