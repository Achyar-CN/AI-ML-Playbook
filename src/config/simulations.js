import { PerceptronSimulation }       from '../simulations/perceptron/index.js';
import { NNSimulation }               from '../simulations/nn/index.js';
import { LinearRegressionSimulation }  from '../simulations/linearRegression/index.js';
import { DecisionTreeSimulation }     from '../simulations/decisionTree/index.js';
import { AdaBoostSimulation }         from '../simulations/adaboost/index.js';

const CLASS_DATASETS = [
  { id: 'linear',       label: 'Linear' },
  { id: 'diagonal',     label: 'Diagonal' },
  { id: 'xor',          label: 'XOR' },
  { id: 'moons',        label: 'Moons' },
  { id: 'circle',       label: 'Circle' },
  { id: 'spiral',       label: 'Spiral' },
  { id: 'checkerboard', label: 'Grid' },
];

const REG_DATASETS = [
  { id: 'linear',    label: 'Linear' },
  { id: 'quadratic', label: 'Quadratic' },
  { id: 'sine',      label: 'Sine' },
  { id: 'cubic',     label: 'Cubic' },
  { id: 'noisy',     label: 'Noisy' },
];

const classDataParams = (datasets, defaultDataset = 'linear') => [
  { name: 'datasetType', label: 'Dataset Shape', type: 'dataset', options: datasets, default: defaultDataset },
  { name: 'nPoints',     label: 'Points',        type: 'number', min: 20, max: 400, step: 10,
    description: 'Number of training samples.' },
  { name: 'noiseLevel',  label: 'Noise',         type: 'number', min: 0, max: 0.5, step: 0.02,
    description: 'Random noise added to class boundaries.' },
  { name: 'seed',        label: 'Random Seed',   type: 'number', min: 0, max: 9999, step: 1,
    description: 'Same seed = same data. Change to try different splits.' },
];

const regDataParams = [
  { name: 'datasetType', label: 'Dataset Shape', type: 'dataset', options: REG_DATASETS, default: 'linear' },
  { name: 'nPoints',     label: 'Points',        type: 'number', min: 20, max: 400, step: 10,
    description: 'Number of training samples.' },
  { name: 'noiseLevel',  label: 'Noise',         type: 'number', min: 0.02, max: 0.8, step: 0.02,
    description: 'Noise amplitude around the true function.' },
  { name: 'seed',        label: 'Random Seed',   type: 'number', min: 0, max: 9999, step: 1,
    description: 'Same seed = same data.' },
];

export const simulations = [
  // ── Classification ───────────────────────────────────────────
  {
    id: 'perceptron',
    title: 'Perceptron',
    taskType: 'classification',
    class: PerceptronSimulation,
    metricKeys: ['loss', 'accuracy', 'recall', 'precision', 'f1'],
    info: {
      tagline: 'Linear binary classifier',
      description: 'The simplest learning unit. Iterates through training data and nudges weights whenever a point is misclassified using the perceptron update rule.',
      insights: [
        'Only converges on linearly separable data',
        'High LR → unstable; low LR → slow convergence',
        'No probabilistic output — purely hard boundary',
      ],
    },
    defaultParams: {
      datasetType: 'linear', nPoints: 120, noiseLevel: 0.08, seed: 42,
      learningRate: 0.1, epochs: 100,
    },
    dataParamControls: classDataParams(
      [{ id:'linear',label:'Linear'},{id:'diagonal',label:'Diagonal'},{id:'moons',label:'Moons'},{id:'xor',label:'XOR'}],
      'linear'
    ),
    paramControls: [
      { name: 'learningRate', label: 'Learning Rate', type: 'number', min: 0.001, max: 1, step: 0.001,
        description: 'Step size for weight updates. Too high = unstable, too low = slow.' },
      { name: 'epochs', label: 'Epochs', type: 'number', min: 1, max: 500, step: 1,
        description: 'Full passes over all training data.' },
    ],
  },

  {
    id: 'nn',
    title: 'Neural Network',
    taskType: 'classification',
    class: NNSimulation,
    metricKeys: ['loss', 'accuracy', 'recall', 'precision', 'f1'],
    info: {
      tagline: 'Universal function approximator',
      description: 'Stacked layers of nonlinear units trained via backpropagation (chain rule). Can learn arbitrarily complex decision boundaries given enough capacity.',
      insights: [
        'More hidden units → more complex boundary, higher overfitting risk',
        'Tanh & ReLU outperform Sigmoid for hidden layers',
        'L2 regularization shrinks weights to reduce overfitting',
      ],
    },
    defaultParams: {
      datasetType: 'circle', nPoints: 150, noiseLevel: 0.08, seed: 42,
      learningRate: 0.05, epochs: 300, hiddenUnits: 6, activation: 'tanh', l2: 0,
    },
    dataParamControls: classDataParams(CLASS_DATASETS, 'circle'),
    paramControls: [
      { name: 'learningRate', label: 'Learning Rate', type: 'number', min: 0.001, max: 0.5, step: 0.001,
        description: 'Gradient descent step size. Typical range: 0.01–0.1.' },
      { name: 'epochs', label: 'Epochs', type: 'number', min: 1, max: 1000, step: 10,
        description: 'Training iterations over all data.' },
      { name: 'hiddenUnits', label: 'Hidden Units', type: 'number', min: 1, max: 24, step: 1,
        description: 'Neurons in the hidden layer. More = more complex boundaries.' },
      { name: 'activation', label: 'Activation', type: 'select',
        options: [{value:'tanh',label:'Tanh'},{value:'relu',label:'ReLU'},{value:'sigmoid',label:'Sigmoid'}],
        description: 'Nonlinearity for hidden neurons. Tanh: symmetric, ReLU: fast, Sigmoid: saturates.' },
      { name: 'l2', label: 'L2 Regularization', type: 'number', min: 0, max: 0.1, step: 0.002,
        description: 'Penalizes large weights to reduce overfitting.' },
    ],
  },

  {
    id: 'decisionTree',
    title: 'Decision Tree',
    taskType: 'classification',
    class: DecisionTreeSimulation,
    metricKeys: ['loss', 'accuracy', 'recall', 'precision', 'f1'],
    info: {
      tagline: 'Recursive partition learner',
      description: 'Greedily splits feature space by finding thresholds that maximize class purity. Each Run epoch grows the tree one level deeper.',
      insights: [
        'Max depth directly controls bias-variance tradeoff',
        'Gini is faster; Entropy can yield marginally better splits',
        'Min leaf size acts as implicit pruning',
      ],
    },
    defaultParams: {
      datasetType: 'xor', nPoints: 120, noiseLevel: 0.08, seed: 42,
      maxDepth: 6, minLeafSize: 5, useGini: false,
    },
    dataParamControls: classDataParams(CLASS_DATASETS, 'xor'),
    paramControls: [
      { name: 'maxDepth', label: 'Max Depth', type: 'number', min: 1, max: 10, step: 1,
        description: 'Max recursive levels. Each Run epoch adds 1 more level.' },
      { name: 'minLeafSize', label: 'Min Leaf Size', type: 'number', min: 2, max: 40, step: 1,
        description: 'Minimum samples to split a node. Higher = simpler, more pruned tree.' },
      { name: 'useGini', label: 'Use Gini (vs Entropy)', type: 'boolean',
        description: 'Gini impurity vs Information Gain (entropy) as the split criterion.' },
    ],
  },

  {
    id: 'adaboost',
    title: 'AdaBoost',
    taskType: 'classification',
    class: AdaBoostSimulation,
    metricKeys: ['loss', 'accuracy', 'recall', 'precision', 'f1'],
    info: {
      tagline: 'Boosting ensemble of decision stumps',
      description: 'Trains weak learners (depth-1 trees) sequentially. Each round re-weights training data so the next stump focuses harder on previously misclassified points.',
      insights: [
        'More boosting rounds = more complex ensemble',
        'Shrinkage < 1 scales each stump's vote → more stable training',
        'Sensitive to noisy labels — outliers get repeatedly up-weighted',
      ],
    },
    defaultParams: {
      datasetType: 'linear', nPoints: 120, noiseLevel: 0.08, seed: 42,
      epochs: 20, learningRate: 1.0,
    },
    dataParamControls: classDataParams(
      [{id:'linear',label:'Linear'},{id:'xor',label:'XOR'},{id:'moons',label:'Moons'},{id:'circle',label:'Circle'},{id:'diagonal',label:'Diagonal'}],
      'linear'
    ),
    paramControls: [
      { name: 'epochs', label: 'Boosting Rounds', type: 'number', min: 1, max: 60, step: 1,
        description: 'Number of weak learners (decision stumps) in the ensemble.' },
      { name: 'learningRate', label: 'Shrinkage', type: 'number', min: 0.05, max: 1, step: 0.05,
        description: 'Scales each stump\'s weight. Lower = slower but more stable ensemble.' },
    ],
  },

  // ── Regression ───────────────────────────────────────────────
  {
    id: 'linearRegression',
    title: 'Linear / Poly Regression',
    taskType: 'regression',
    class: LinearRegressionSimulation,
    metricKeys: ['loss', 'mae', 'rmse', 'mape', 'nmae'],
    info: {
      tagline: 'Gradient descent curve fitter',
      description: 'Minimizes Mean Squared Error by repeatedly adjusting weights in the direction that reduces the loss. Polynomial features (x², x³…) allow fitting nonlinear curves.',
      insights: [
        'Higher degree → more flexible fit but risk of overfitting',
        'L2 (Ridge) regularization keeps weights small and curves smoother',
        'If loss diverges, reduce learning rate',
      ],
    },
    defaultParams: {
      datasetType: 'linear', nPoints: 100, noiseLevel: 0.25, seed: 42,
      learningRate: 0.05, epochs: 200, degree: 1, l2: 0,
    },
    dataParamControls: regDataParams,
    paramControls: [
      { name: 'learningRate', label: 'Learning Rate', type: 'number', min: 0.001, max: 0.5, step: 0.001,
        description: 'Gradient descent step size. Too high = divergence.' },
      { name: 'epochs', label: 'Epochs', type: 'number', min: 1, max: 1000, step: 10,
        description: 'Gradient descent iterations.' },
      { name: 'degree', label: 'Polynomial Degree', type: 'number', min: 1, max: 6, step: 1,
        description: 'Degree 1 = line, 2 = parabola, 3–6 = higher-order curves.' },
      { name: 'l2', label: 'L2 (Ridge) Regularization', type: 'number', min: 0, max: 0.2, step: 0.005,
        description: 'Penalizes large weights. Helps prevent overfitting on high-degree poly.' },
    ],
  },
];
