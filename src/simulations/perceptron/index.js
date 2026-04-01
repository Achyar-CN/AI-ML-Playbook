import { BaseSimulation } from '../baseSimulation.js';

function randomBetween(min, max) {
  return Math.random() * (max - min) + min;
}

export class PerceptronSimulation extends BaseSimulation {
  setup() {
    this.points = [];
    const { nPoints } = this.params;
    this.weights = [randomBetween(-1, 1), randomBetween(-1, 1), randomBetween(-1, 1)];
    this.epoch = 0;

    for (let i = 0; i < nPoints; i += 1) {
      const x = randomBetween(-1, 1);
      const y = randomBetween(-1, 1);
      const label = y > x ? 1 : -1;
      this.points.push({ x, y, label });
    }
  }

  predict(x, y) {
    const sum = this.weights[0] * x + this.weights[1] * y + this.weights[2] * 1;
    return sum >= 0 ? 1 : -1;
  }

  step() {
    if (this.epoch >= this.params.epochs) {
      return;
    }

    const lr = this.params.learningRate;
    this.points.forEach((pt) => {
      const guess = this.predict(pt.x, pt.y);
      const error = pt.label - guess;
      this.weights[0] += lr * error * pt.x;
      this.weights[1] += lr * error * pt.y;
      this.weights[2] += lr * error * 1;
    });
    this.epoch += 1;
  }

  render() {
    const { width, height } = this.canvas;
    this.ctx.clearRect(0, 0, width, height);
    this.ctx.fillStyle = '#fff';
    this.ctx.fillRect(0, 0, width, height);

    // draw points
    this.points.forEach(({ x, y, label }) => {
      const px = ((x + 1) / 2) * width;
      const py = height - ((y + 1) / 2) * height;
      this.ctx.beginPath();
      this.ctx.arc(px, py, 5, 0, Math.PI * 2);
      this.ctx.fillStyle = label === 1 ? '#1976d2' : '#e53935';
      this.ctx.fill();
    });

    // decision boundary
    const left = -1;
    const right = 1;
    const y1 = -(this.weights[2] + this.weights[0] * left) / this.weights[1];
    const y2 = -(this.weights[2] + this.weights[0] * right) / this.weights[1];
    const x1 = ((left + 1) / 2) * width;
    const x2 = ((right + 1) / 2) * width;
    const dy1 = height - ((y1 + 1) / 2) * height;
    const dy2 = height - ((y2 + 1) / 2) * height;

    this.ctx.strokeStyle = '#000';
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();
    this.ctx.moveTo(x1, dy1);
    this.ctx.lineTo(x2, dy2);
    this.ctx.stroke();

    this.ctx.fillStyle = '#333';
    this.ctx.font = '14px sans-serif';
    this.ctx.fillText(`Epoch: ${this.epoch}`, 10, 20);
  }
}
