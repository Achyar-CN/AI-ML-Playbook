import { BaseSimulation } from '../baseSimulation.js';

export class DecisionTreeSimulation extends BaseSimulation {
  setup() {
    this.history = [];
    this.points = [];
    const { nPoints, seed } = this.params;
    this.tree = null;
    this.epoch = 0;
    this.currentStep = 0; // for step-by-step visualization
    this.bestSplit = null;
    this.impurityBefore = 0;
    this.impurityAfter = 0;
    this.informationGain = 0;
    this.stepTimer = 0; // timer for step-by-step mode

    for (let i = 0; i < nPoints; i += 1) {
      const x = this.randomBetween(-1, 1, seed + 10 + i * 2);
      const y = this.randomBetween(-1, 1, seed + 11 + i * 2);
      const label = x > 0 ? 1 : 0;
      this.points.push({ x, y, label });
    }

    this.currentThreshold = this.randomBetween(-0.8, 0.8, seed + 999);
    this.buildTree(this.currentThreshold);
  }

  // Calculate entropy for a set of labels
  calculateEntropy(labels) {
    const total = labels.length;
    if (total === 0) return 0;

    const counts = {};
    labels.forEach(label => {
      counts[label] = (counts[label] || 0) + 1;
    });

    let entropy = 0;
    Object.values(counts).forEach(count => {
      const p = count / total;
      entropy -= p * Math.log2(p);
    });

    return entropy;
  }

  // Calculate information gain for a potential split
  calculateInformationGain(threshold) {
    const leftLabels = [];
    const rightLabels = [];

    this.points.forEach(pt => {
      if (pt.x <= threshold) {
        leftLabels.push(pt.label);
      } else {
        rightLabels.push(pt.label);
      }
    });

    const totalEntropy = this.calculateEntropy(this.points.map(pt => pt.label));
    const leftEntropy = this.calculateEntropy(leftLabels);
    const rightEntropy = this.calculateEntropy(rightLabels);

    const leftWeight = leftLabels.length / this.points.length;
    const rightWeight = rightLabels.length / this.points.length;

    const weightedEntropy = leftWeight * leftEntropy + rightWeight * rightEntropy;
    const informationGain = totalEntropy - weightedEntropy;

    return {
      threshold,
      informationGain,
      totalEntropy,
      weightedEntropy,
      leftLabels,
      rightLabels
    };
  }

  buildTree(threshold) {
    this.tree = {
      feature: 'x',
      threshold,
      left: { label: 0 },
      right: { label: 1 }
    };
  }

  predict(x, y) {
    let node = this.tree;
    while (node.left && node.right) {
      if (node.feature === 'x') {
        node = x <= node.threshold ? node.left : node.right;
      } else {
        node = y <= node.threshold ? node.left : node.right;
      }
    }
    return node.label;
  }

  step() {
    if (this.epoch >= this.params.epochs) return;

    // Handle step-by-step mode with timing
    if (this.params.stepMode) {
      this.stepTimer += 16; // assuming ~60fps, 16ms per frame
      if (this.stepTimer < this.params.stepSpeed) {
        return; // Wait for step speed delay
      }
      this.stepTimer = 0; // Reset timer
    }

    // Step-by-step tree building process
    if (this.currentStep === 0) {
      // Calculate initial entropy
      const allLabels = this.points.map(pt => pt.label);
      this.impurityBefore = this.calculateEntropy(allLabels);
      this.currentStep = 1;
      if (this.params.stepMode) return; // Pause for visualization in step mode
    }

    if (this.currentStep === 1) {
      // Find best split using information gain
      const xs = Array.from(new Set(this.points.map((pt) => pt.x))).sort((a, b) => a - b);
      const candidates = xs.slice(1).map((x, i) => (x + xs[i]) / 2);

      let bestSplit = null;
      let maxGain = -1;

      candidates.forEach(threshold => {
        const splitInfo = this.calculateInformationGain(threshold);
        if (splitInfo.informationGain > maxGain) {
          maxGain = splitInfo.informationGain;
          bestSplit = splitInfo;
        }
      });

      this.bestSplit = bestSplit;
      this.impurityAfter = bestSplit.weightedEntropy;
      this.informationGain = bestSplit.informationGain;
      this.currentStep = 2;
      if (this.params.stepMode) return; // Pause for visualization in step mode
    }

    if (this.currentStep === 2) {
      // Apply the best split
      this.buildTree(this.bestSplit.threshold);
      this.currentStep = 0; // Reset for next epoch
      this.epoch += 1;

      const metrics = this.computeMetrics();
      this.history.push({ epoch: this.epoch, ...metrics });
    }
  }

  render() {
    const { width, height } = this.canvas;
    this.ctx.clearRect(0, 0, width, height);

    // draw points
    this.points.forEach(({ x, y, label }) => {
      const px = ((x + 1) / 2) * width;
      const py = height - ((y + 1) / 2) * height;
      this.ctx.beginPath();
      this.ctx.arc(px, py, 5, 0, Math.PI * 2);
      this.ctx.fillStyle = label === 1 ? '#1976d2' : '#e53935';
      this.ctx.fill();
    });

    // Draw current split line
    if (this.tree) {
      const t = this.tree.threshold;
      const px = ((t + 1) / 2) * width;
      this.ctx.strokeStyle = '#0f172a';
      this.ctx.lineWidth = 2.5;
      this.ctx.beginPath();
      this.ctx.moveTo(px, 0);
      this.ctx.lineTo(px, height);
      this.ctx.stroke();
    }

    // Draw best split candidate if in step 1
    if (this.currentStep === 1 && this.bestSplit) {
      const t = this.bestSplit.threshold;
      const px = ((t + 1) / 2) * width;
      this.ctx.strokeStyle = '#ff9800';
      this.ctx.lineWidth = 2;
      this.ctx.setLineDash([5, 5]);
      this.ctx.beginPath();
      this.ctx.moveTo(px, 0);
      this.ctx.lineTo(px, height);
      this.ctx.stroke();
      this.ctx.setLineDash([]);
    }

    // Display step-by-step information
    this.ctx.fillStyle = '#333';
    this.ctx.font = '14px sans-serif';

    let yOffset = 20;
    this.ctx.fillText(`Epoch: ${this.epoch}`, 10, yOffset);
    yOffset += 20;

    if (this.currentStep === 0) {
      this.ctx.fillText('Step: Calculating initial entropy...', 10, yOffset);
    } else if (this.currentStep === 1) {
      this.ctx.fillText('Step: Finding best split...', 10, yOffset);
    } else if (this.currentStep === 2) {
      this.ctx.fillText('Step: Applying split', 10, yOffset);
    }
    yOffset += 20;

    // Display entropy and information gain
    if (this.impurityBefore !== undefined) {
      this.ctx.fillText(`Initial Entropy: ${this.impurityBefore.toFixed(3)}`, 10, yOffset);
      yOffset += 18;
    }

    if (this.bestSplit) {
      this.ctx.fillText(`Best Threshold: ${this.bestSplit.threshold.toFixed(3)}`, 10, yOffset);
      yOffset += 18;
      this.ctx.fillText(`Information Gain: ${this.informationGain.toFixed(3)}`, 10, yOffset);
      yOffset += 18;
    }

    const metrics = this.computeMetrics();
    this.ctx.fillText(`Accuracy: ${(metrics.accuracy * 100).toFixed(1)}%`, 10, yOffset);
  }

  computeMetrics() {
    const labels = this.points.map((pt) => (pt.label === 1 ? 1 : 0));
    const preds = this.points.map((pt) => (this.predict(pt.x, pt.y) === 1 ? 1 : 0));
    return this.computeClassificationMetrics(labels, preds);
  }
}