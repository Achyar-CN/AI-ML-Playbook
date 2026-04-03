// Shared in-memory store for user-imported CSV data.
// Set by UIController on CSV upload; read by BaseSimulation dataset generators.
export const dataStore = {
  points: null, // [{x, y, label}] for classification | [{x, y}] for regression
  type: null,   // 'classification' | 'regression'
  filename: null,
};
