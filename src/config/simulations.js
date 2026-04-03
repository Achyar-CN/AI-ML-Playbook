import { PerceptronSimulation }              from '../simulations/perceptron/index.js';
import { NNSimulation }                      from '../simulations/nn/index.js';
import { LinearRegressionSimulation }        from '../simulations/linearRegression/index.js';
import { DecisionTreeSimulation }            from '../simulations/decisionTree/index.js';
import { AdaBoostSimulation }               from '../simulations/adaboost/index.js';
import { KNNClassificationSimulation,
         KNNRegressionSimulation }           from '../simulations/knn/index.js';
import { RandomForestClassificationSimulation,
         RandomForestRegressionSimulation }  from '../simulations/randomForest/index.js';
import { SVMClassificationSimulation,
         SVRSimulation }                     from '../simulations/svm/index.js';

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
        'High LR = unstable; low LR = slow convergence',
        'No probability output — purely a hard decision boundary',
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
        'More hidden units = more complex boundary, higher overfitting risk',
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
        'Max depth directly controls the bias-variance tradeoff',
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
      description: 'Trains weak learners (depth-1 trees) sequentially. Each round re-weights data so the next stump focuses harder on previously misclassified points.',
      insights: [
        'More boosting rounds = more complex ensemble',
        "Shrinkage < 1 scales each stump's vote — more stable training",
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

  {
    id: 'knn',
    title: 'K-Nearest Neighbors',
    taskType: 'classification',
    class: KNNClassificationSimulation,
    metricKeys: ['loss', 'accuracy', 'recall', 'precision', 'f1'],
    info: {
      tagline: 'Lazy instance-based learner',
      description: 'No training phase — classifies each point by majority vote among its K nearest neighbors. The decision boundary adapts to the data density.',
      insights: [
        'Small k = complex wiggly boundary (low bias, high variance)',
        'Large k = smoother boundary (high bias, low variance)',
        'Manhattan distance works better when features have different scales',
      ],
    },
    defaultParams: {
      datasetType: 'moons', nPoints: 150, noiseLevel: 0.1, seed: 42, k: 5,
      distanceMetric: 'euclidean',
    },
    dataParamControls: classDataParams(CLASS_DATASETS, 'moons'),
    paramControls: [
      { name: 'k', label: 'K (Neighbors)', type: 'number', min: 1, max: 30, step: 1,
        description: 'Number of nearest neighbors to vote. Lower = more complex boundary.' },
      { name: 'distanceMetric', label: 'Distance Metric', type: 'select',
        options: [{ value: 'euclidean', label: 'Euclidean' }, { value: 'manhattan', label: 'Manhattan' }],
        description: 'Euclidean = straight-line distance. Manhattan = city-block distance.' },
    ],
  },

  {
    id: 'randomForestClass',
    title: 'Random Forest',
    taskType: 'classification',
    class: RandomForestClassificationSimulation,
    metricKeys: ['loss', 'accuracy', 'recall', 'precision', 'f1'],
    info: {
      tagline: 'Bagging ensemble of random trees',
      description: 'Builds many decision trees on bootstrap samples, each node splitting on a randomly chosen feature. Final prediction = majority vote across all trees.',
      insights: [
        'More trees = more stable predictions (law of large numbers)',
        'Random feature selection reduces tree correlation — key to ensemble benefit',
        'Each Run step adds one new tree to the forest',
      ],
    },
    defaultParams: {
      datasetType: 'spiral', nPoints: 150, noiseLevel: 0.08, seed: 42,
      nTrees: 20, maxDepth: 4, minLeafSize: 3,
    },
    dataParamControls: classDataParams(CLASS_DATASETS, 'spiral'),
    paramControls: [
      { name: 'nTrees', label: 'Trees (Epochs)', type: 'number', min: 1, max: 60, step: 1,
        description: 'Total trees to grow. Each Run step adds one tree.' },
      { name: 'maxDepth', label: 'Max Tree Depth', type: 'number', min: 1, max: 10, step: 1,
        description: 'Max depth of each individual tree. Deeper = more complex.' },
      { name: 'minLeafSize', label: 'Min Leaf Size', type: 'number', min: 2, max: 20, step: 1,
        description: 'Min samples to split a node. Higher = smoother, more pruned trees.' },
    ],
  },

  {
    id: 'svm',
    title: 'SVM',
    taskType: 'classification',
    class: SVMClassificationSimulation,
    metricKeys: ['loss', 'accuracy', 'recall', 'precision', 'f1'],
    info: {
      tagline: 'Maximum-margin hyperplane classifier',
      description: 'Finds the decision boundary that maximizes the margin between classes. Soft-margin SVM allows some misclassifications controlled by C. Yellow-ringed points are support vectors.',
      insights: [
        'High C = harder margin, fits training data tightly (risk of overfit)',
        'Low C = soft margin, more regularization',
        'Poly-2 kernel adds x\u00b2, y\u00b2, xy features enabling curved boundaries',
      ],
    },
    defaultParams: {
      datasetType: 'linear', nPoints: 120, noiseLevel: 0.08, seed: 42,
      C: 1.0, kernel: 'linear', learningRate: 0.05, epochs: 300,
    },
    dataParamControls: classDataParams(
      [{ id: 'linear', label: 'Linear' }, { id: 'moons', label: 'Moons' },
       { id: 'circle', label: 'Circle' }, { id: 'xor', label: 'XOR' }, { id: 'diagonal', label: 'Diagonal' }],
      'linear'
    ),
    paramControls: [
      { name: 'C', label: 'C (Regularization)', type: 'number', min: 0.1, max: 10, step: 0.1,
        description: 'Penalty for misclassified points. High C = hard margin.' },
      { name: 'kernel', label: 'Kernel', type: 'select',
        options: [{ value: 'linear', label: 'Linear' }, { value: 'poly2', label: 'Polynomial (deg 2)' }],
        description: 'Linear: straight boundary. Poly-2: curved boundary using degree-2 features.' },
      { name: 'learningRate', label: 'Learning Rate', type: 'number', min: 0.001, max: 0.2, step: 0.005,
        description: 'Gradient descent step size.' },
      { name: 'epochs', label: 'Epochs', type: 'number', min: 50, max: 1000, step: 50,
        description: 'Gradient descent iterations.' },
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
      description: 'Minimizes Mean Squared Error by repeatedly adjusting weights opposite to the loss gradient. Polynomial features allow fitting nonlinear curves.',
      insights: [
        'Higher degree = more flexible fit but risk of overfitting',
        'L2 (Ridge) keeps weights small and curves smoother',
        'If loss diverges, reduce the learning rate',
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

  {
    id: 'knnRegression',
    title: 'KNN Regression',
    taskType: 'regression',
    class: KNNRegressionSimulation,
    metricKeys: ['loss', 'mae', 'rmse', 'mape', 'nmae'],
    info: {
      tagline: 'Lazy instance-based regressor',
      description: 'Predicts the target as the mean of the K nearest training points. No model weights — every prediction scans the full dataset.',
      insights: [
        'Small k = wiggly fit that interpolates closely (overfits)',
        'Large k = smooth fit that averages over noise (underfits)',
        'Works well when the relationship is locally smooth',
      ],
    },
    defaultParams: {
      datasetType: 'sine', nPoints: 100, noiseLevel: 0.2, seed: 42,
      k: 5, distanceMetric: 'euclidean',
    },
    dataParamControls: regDataParams,
    paramControls: [
      { name: 'k', label: 'K (Neighbors)', type: 'number', min: 1, max: 30, step: 1,
        description: 'Number of nearest neighbors. Low k = wiggly, high k = smooth.' },
      { name: 'distanceMetric', label: 'Distance Metric', type: 'select',
        options: [{ value: 'euclidean', label: 'Euclidean' }, { value: 'manhattan', label: 'Manhattan' }],
        description: 'For 1D regression both metrics are equivalent.' },
    ],
  },

  {
    id: 'randomForestReg',
    title: 'Random Forest Reg.',
    taskType: 'regression',
    class: RandomForestRegressionSimulation,
    metricKeys: ['loss', 'mae', 'rmse', 'mape', 'nmae'],
    info: {
      tagline: 'Ensemble of regression trees',
      description: 'Each tree splits training data to minimize MSE. Prediction = mean across all trees. Bootstrap sampling reduces variance without increasing bias.',
      insights: [
        'More trees = lower variance via averaging',
        'Shallow trees = high bias; deep trees = high variance',
        'Naturally handles nonlinear patterns without feature engineering',
      ],
    },
    defaultParams: {
      datasetType: 'sine', nPoints: 100, noiseLevel: 0.2, seed: 42,
      nTrees: 20, maxDepth: 4, minLeafSize: 3,
    },
    dataParamControls: regDataParams,
    paramControls: [
      { name: 'nTrees', label: 'Trees (Epochs)', type: 'number', min: 1, max: 60, step: 1,
        description: 'Total trees to grow. Each Run step adds one tree.' },
      { name: 'maxDepth', label: 'Max Tree Depth', type: 'number', min: 1, max: 10, step: 1,
        description: 'Max depth of each individual tree.' },
      { name: 'minLeafSize', label: 'Min Leaf Size', type: 'number', min: 2, max: 20, step: 1,
        description: 'Min samples to split a node.' },
    ],
  },

  {
    id: 'svr',
    title: 'SVR',
    taskType: 'regression',
    class: SVRSimulation,
    metricKeys: ['loss', 'mae', 'rmse', 'mape', 'nmae'],
    info: {
      tagline: 'Epsilon-insensitive tube regressor',
      description: 'Fits a tube of width 2\u03b5 around the data. Points inside the tube incur zero loss. Only points outside (support vectors, shown in yellow) influence the fit.',
      insights: [
        'Larger \u03b5 = wider tube = fewer support vectors = smoother fit',
        'Higher C = heavier penalty for points outside the tube',
        'Poly-2 kernel enables fitting quadratic and parabolic curves',
      ],
    },
    defaultParams: {
      datasetType: 'sine', nPoints: 100, noiseLevel: 0.2, seed: 42,
      C: 1.0, epsilon: 0.1, kernel: 'linear', learningRate: 0.02, epochs: 300,
    },
    dataParamControls: regDataParams,
    paramControls: [
      { name: 'C', label: 'C (Regularization)', type: 'number', min: 0.1, max: 10, step: 0.1,
        description: 'Penalty for points outside the epsilon tube.' },
      { name: 'epsilon', label: 'Epsilon (\u03b5)', type: 'number', min: 0.01, max: 0.5, step: 0.01,
        description: 'Half-width of the insensitive tube around the regression curve.' },
      { name: 'kernel', label: 'Kernel', type: 'select',
        options: [{ value: 'linear', label: 'Linear' }, { value: 'poly2', label: 'Polynomial (deg 2)' }],
        description: 'Linear: straight line. Poly-2: parabola.' },
      { name: 'learningRate', label: 'Learning Rate', type: 'number', min: 0.001, max: 0.1, step: 0.005,
        description: 'Gradient descent step size.' },
      { name: 'epochs', label: 'Epochs', type: 'number', min: 50, max: 1000, step: 50,
        description: 'Gradient descent iterations.' },
    ],
  },
];
