import { BaseSimulation } from '../baseSimulation.js';

function randomBetween(min, max) {
  return Math.random() * (max - min) + min;
}

function tanh(x) {
  return Math.tanh(x);
}

function tanhDeriv(x) {
  const t = Math.tanh(x);
  return 1 - t * t;
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function sigmoidDeriv(x) {
  const s = sigmoid(x);
  return s * (1 - s);
}

export class NNSimulation extends BaseSimulation {
  setup() {
    this.points = [];
    const { nPoints } = this.params;
    this.epoch = 0;

    const hidden = this.params.hiddenUnits || 4;
    this.w1 = Array.from({ length: hidden }, () => [randomBetween(-1, 1), randomBetween(-1, 1), randomBetween(-1, 1)]);
    this.w2 = Array.from({ length: hidden + 1 }, () => randomBetween(-1, 1));

    for (let i = 0; i < nPoints; i += 1) {
      const x = randomBetween(-1, 1);
      const y = randomBetween(-1, 1);
      const label = x * x + y * y < 0.5 ? 1 : 0;
      this.points.push({ x, y, label });
    }
  }

  reset() {
    this.setup();
  }

  forward(x, y) {
    const input = [x, y, 1];
    const hidden = this.w1.map((ws) => tanh(ws[0] * x + ws[1] * y + ws[2] * 1));
    const body = [...hidden, 1];
    const output = sigmoid(this.w2.reduce((sum, w, i) => sum + w * body[i], 0));
    return { hidden, output, body }; 
  }

  step() {
    if (this.epoch >= this.params.epochs) return;

    const lr = this.params.learningRate;
    this.points.forEach((pt) => {
      const { x, y, label } = pt;
      const { hidden, output, body } = this.forward(x, y);
      const errorOut = output - label;
      const gradOut = errorOut * sigmoidDeriv(output);

      // update hidden->output
      for (let i = 0; i < this.w2.length; i += 1) {
        this.w2[i] -= lr * gradOut * body[i];
      }

      // backprop hidden
      for (let j = 0; j < this.w1.length; j += 1) {
        const w2j = this.w2[j];
        const gradHidden = gradOut * w2j * tanhDeriv(this.w1[j][0] * x + this.w1[j][1] * y + this.w1[j][2]);
        this.w1[j][0] -= lr * gradHidden * x;
        this.w1[j][1] -= lr * gradHidden * y;
        this.w1[j][2] -= lr * gradHidden * 1;
      }
    });

    this.epoch += 1;
  }

  predict(x, y) {
    const { output } = this.forward(x, y);
    return output >= 0.5 ? 1 : 0;
  }

  render() {
    const { width, height } = this.canvas;
    this.ctx.clearRect(0, 0, width, height);

    // background grid decisions
    const grid = 80;
    for (let gx = 0; gx < grid; gx += 1) {
      for (let gy = 0; gy < grid; gy += 1) {
        const x = (gx / (grid - 1)) * 2 - 1;
        const y = (gy / (grid - 1)) * 2 - 1;
        const p = this.predict(x, y);
        this.ctx.fillStyle = p === 1 ? 'rgba(25, 118, 210, 0.2)' : 'rgba(229, 57, 53, 0.2)';
        this.ctx.fillRect((x + 1) / 2 * width, height - (y + 1) / 2 * height, width / grid + 1, height / grid + 1);
      }
    }

    this.points.forEach(({ x, y, label }) => {
      const px = ((x + 1) / 2) * width;
      const py = height - ((y + 1) / 2) * height;
      this.ctx.beginPath();
      this.ctx.arc(px, py, 5, 0, Math.PI * 2);
      this.ctx.fillStyle = label === 1 ? '#1565c0' : '#c62828';
      this.ctx.fill();
      this.ctx.strokeStyle = '#fff';
      this.ctx.lineWidth = 2;
      this.ctx.stroke();
    });

    this.ctx.fillStyle = '#333';
    this.ctx.font = '14px sans-serif';
    this.ctx.fillText(`Epoch: ${this.epoch}`, 10, 20);
  }
}
