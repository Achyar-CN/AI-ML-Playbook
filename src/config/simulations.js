import { PerceptronSimulation } from '../simulations/perceptron/index.js';
import { NNSimulation } from '../simulations/nn/index.js';

export const simulations = [
  {
    id: 'perceptron',
    title: 'Perceptron (linear separable)',
    class: PerceptronSimulation,
    defaultParams: {
      learningRate: 0.1,
      epochs: 100,
      nPoints: 120,
      lineFactor: 0.3,
    },
    paramControls: [
      { name: 'learningRate', label: 'Learning Rate', type: 'number', min: 0.001, max: 1, step: 0.001 },
      { name: 'epochs', label: 'Epochs', type: 'number', min: 1, max: 500, step: 1 },
      { name: 'nPoints', label: 'Points', type: 'number', min: 20, max: 1000, step: 10 },
    ],
  },
  {
    id: 'nn',
    title: 'Simple Neural Network',
    class: NNSimulation,
    defaultParams: {
      learningRate: 0.1,
      epochs: 200,
      nPoints: 150,
      hiddenUnits: 4,
    },
    paramControls: [
      { name: 'learningRate', label: 'Learning Rate', type: 'number', min: 0.001, max: 1, step: 0.001 },
      { name: 'epochs', label: 'Epochs', type: 'number', min: 1, max: 1000, step: 1 },
      { name: 'hiddenUnits', label: 'Hidden Units', type: 'number', min: 1, max: 20, step: 1 },
      { name: 'nPoints', label: 'Points', type: 'number', min: 20, max: 1000, step: 10 },
    ],
  },
];
