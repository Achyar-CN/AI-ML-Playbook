import { PerceptronSimulation } from '../simulations/perceptron/index.js';
import { NNSimulation } from '../simulations/nn/index.js';
import { LinearRegressionSimulation } from '../simulations/linearRegression/index.js';
import { DecisionTreeSimulation } from '../simulations/decisionTree/index.js';
import { AdaBoostSimulation } from '../simulations/adaboost/index.js';

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
      seed: 42,
    },
    paramControls: [
      { name: 'learningRate', label: 'Learning Rate', type: 'number', min: 0.001, max: 1, step: 0.001 },
      { name: 'epochs', label: 'Epochs', type: 'number', min: 1, max: 500, step: 1 },
      { name: 'nPoints', label: 'Points', type: 'number', min: 20, max: 1000, step: 10 },
      { name: 'seed', label: 'Random Seed', type: 'number', min: 0, max: 9999, step: 1 },
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
      seed: 42,
    },
    paramControls: [
      { name: 'learningRate', label: 'Learning Rate', type: 'number', min: 0.001, max: 1, step: 0.001 },
      { name: 'epochs', label: 'Epochs', type: 'number', min: 1, max: 1000, step: 1 },
      { name: 'hiddenUnits', label: 'Hidden Units', type: 'number', min: 1, max: 20, step: 1 },
      { name: 'nPoints', label: 'Points', type: 'number', min: 20, max: 1000, step: 10 },
      { name: 'seed', label: 'Random Seed', type: 'number', min: 0, max: 9999, step: 1 },
    ],
  },
  {
    id: 'linearRegression',
    title: 'Linear Regression',
    class: LinearRegressionSimulation,
    defaultParams: {
      learningRate: 0.1,
      epochs: 100,
      nPoints: 100,
      seed: 42,
    },
    paramControls: [
      { name: 'learningRate', label: 'Learning Rate', type: 'number', min: 0.001, max: 1, step: 0.001 },
      { name: 'epochs', label: 'Epochs', type: 'number', min: 1, max: 500, step: 1 },
      { name: 'nPoints', label: 'Points', type: 'number', min: 20, max: 1000, step: 10 },
      { name: 'seed', label: 'Random Seed', type: 'number', min: 0, max: 9999, step: 1 },
    ],
  },
  {
    id: 'decisionTree',
    title: 'Decision Tree (simple)',
    class: DecisionTreeSimulation,
    defaultParams: {
      epochs: 50,
      nPoints: 120,
      seed: 42,
    },
    paramControls: [
      { name: 'epochs', label: 'Epochs', type: 'number', min: 1, max: 200, step: 1 },
      { name: 'nPoints', label: 'Points', type: 'number', min: 20, max: 1000, step: 10 },
      { name: 'seed', label: 'Random Seed', type: 'number', min: 0, max: 9999, step: 1 },
    ],
  },
  {
    id: 'adaboost',
    title: 'AdaBoost (Ensemble)',
    class: AdaBoostSimulation,
    defaultParams: {
      epochs: 20,
      nPoints: 120,
      seed: 42,
    },
    paramControls: [
      { name: 'epochs', label: 'Boosting Rounds', type: 'number', min: 1, max: 50, step: 1 },
      { name: 'nPoints', label: 'Points', type: 'number', min: 20, max: 1000, step: 10 },
      { name: 'seed', label: 'Random Seed', type: 'number', min: 0, max: 9999, step: 1 },
    ],
  },
];
