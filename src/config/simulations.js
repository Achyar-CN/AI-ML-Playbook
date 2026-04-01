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
    taskType: 'classification',
    metricKeys: ['loss','accuracy','recall','precision','f1'],
    defaultParams: {
      learningRate: 0.1,
      epochs: 100,
      nPoints: 120,
      lineFactor: 0.3,
      seed: 42,
    },
    paramControls: [
      { name: 'learningRate', label: 'Learning Rate', type: 'number', min: 0.001, max: 1, step: 0.001, description: 'Controls how much the weights are updated each step. Higher values learn faster but may overshoot.' },
      { name: 'epochs', label: 'Epochs', type: 'number', min: 1, max: 500, step: 1, description: 'Number of complete passes through the training data.' },
      { name: 'nPoints', label: 'Points', type: 'number', min: 20, max: 1000, step: 10, description: 'Number of data points to generate for training.' },
      { name: 'seed', label: 'Random Seed', type: 'number', min: 0, max: 9999, step: 1, description: 'Seed for random number generation to ensure reproducible results.' },
    ],
  },
  {
    id: 'nn',
    title: 'Simple Neural Network',
    class: NNSimulation,
    taskType: 'classification',
    metricKeys: ['loss','accuracy','recall','precision','f1'],
    defaultParams: {
      learningRate: 0.1,
      epochs: 200,
      nPoints: 150,
      hiddenUnits: 4,
      seed: 42,
    },
    paramControls: [
      { name: 'learningRate', label: 'Learning Rate', type: 'number', min: 0.001, max: 1, step: 0.001, description: 'Controls how much the weights are updated each step. Higher values learn faster but may overshoot.' },
      { name: 'epochs', label: 'Epochs', type: 'number', min: 1, max: 1000, step: 1, description: 'Number of complete passes through the training data.' },
      { name: 'hiddenUnits', label: 'Hidden Units', type: 'number', min: 1, max: 20, step: 1, description: 'Number of neurons in the hidden layer. More units can learn complex patterns.' },
      { name: 'nPoints', label: 'Points', type: 'number', min: 20, max: 1000, step: 10, description: 'Number of data points to generate for training.' },
      { name: 'seed', label: 'Random Seed', type: 'number', min: 0, max: 9999, step: 1, description: 'Seed for random number generation to ensure reproducible results.' },
    ],
  },
  {
    id: 'linearRegression',
    title: 'Linear Regression',
    class: LinearRegressionSimulation,
    taskType: 'regression',
    metricKeys: ['loss','mape','mae','rmse','nmae'],
    defaultParams: {
      learningRate: 0.1,
      epochs: 100,
      nPoints: 100,
      seed: 42,
    },
    paramControls: [
      { name: 'learningRate', label: 'Learning Rate', type: 'number', min: 0.001, max: 1, step: 0.001, description: 'Controls how much the weights are updated each step. Higher values learn faster but may overshoot.' },
      { name: 'epochs', label: 'Epochs', type: 'number', min: 1, max: 500, step: 1, description: 'Number of complete passes through the training data.' },
      { name: 'nPoints', label: 'Points', type: 'number', min: 20, max: 1000, step: 10, description: 'Number of data points to generate for training.' },
      { name: 'seed', label: 'Random Seed', type: 'number', min: 0, max: 9999, step: 1, description: 'Seed for random number generation to ensure reproducible results.' },
    ],
  },
  {
    id: 'decisionTree',
    title: 'Decision Tree (simple)',
    class: DecisionTreeSimulation,
    taskType: 'classification',
    metricKeys: ['loss','accuracy','recall','precision','f1'],
    defaultParams: {
      epochs: 50,
      nPoints: 120,
      seed: 42,
      stepMode: false,
      stepSpeed: 1000,
    },
    paramControls: [
      { name: 'epochs', label: 'Epochs', type: 'number', min: 1, max: 200, step: 1, description: 'Number of complete passes through the training data.' },
      { name: 'nPoints', label: 'Points', type: 'number', min: 20, max: 1000, step: 10, description: 'Number of data points to generate for training.' },
      { name: 'seed', label: 'Random Seed', type: 'number', min: 0, max: 9999, step: 1, description: 'Seed for random number generation to ensure reproducible results.' },
      { name: 'stepMode', label: 'Step-by-Step Mode', type: 'boolean', description: 'Enable step-by-step visualization of the decision tree building process.' },
      { name: 'stepSpeed', label: 'Step Speed (ms)', type: 'number', min: 100, max: 5000, step: 100, description: 'Speed of step-by-step visualization in milliseconds.' },
    ],
  },
  {
    id: 'adaboost',
    title: 'AdaBoost (Ensemble)',
    class: AdaBoostSimulation,
    taskType: 'classification',
    metricKeys: ['loss','accuracy','recall','precision','f1'],
    defaultParams: {
      epochs: 20,
      nPoints: 120,
      seed: 42,
    },
    paramControls: [
      { name: 'epochs', label: 'Boosting Rounds', type: 'number', min: 1, max: 50, step: 1, description: 'Number of weak learners to combine in the ensemble.' },
      { name: 'nPoints', label: 'Points', type: 'number', min: 20, max: 1000, step: 10, description: 'Number of data points to generate for training.' },
      { name: 'seed', label: 'Random Seed', type: 'number', min: 0, max: 9999, step: 1, description: 'Seed for random number generation to ensure reproducible results.' },
    ],
  },
];
