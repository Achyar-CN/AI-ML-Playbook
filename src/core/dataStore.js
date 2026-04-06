// Shared in-memory store for user-imported CSV data.
// Set by UIController on CSV upload; read by BaseSimulation dataset generators.
export const dataStore = {
  points:       null,   // [{x, y, z?, label?}] – processed 2D or 3D representation
  type:         null,   // 'classification' | 'regression'
  filename:     null,

  // ── Feature metadata (populated on CSV upload) ─────────────────
  featureNames: [],     // input column names, e.g. ['age', 'income'] or ['x₁', 'x₂', 'x₃']
  targetName:   null,   // label/target column name, e.g. 'label' or 'y'
  nFeatures:    0,      // number of input feature columns

  // ── Axis labels for canvas visualisation ───────────────────────
  xLabel:       null,   // e.g. 'age', 'PC1 (62%)'  — null = no label
  yLabel:       null,   // e.g. 'income', 'PC2 (28%)'
  zLabel:       null,   // only set when is3D = true

  // ── Dimensionality flags ────────────────────────────────────────
  is3D:         false,  // true when 3 input features (raw 3D scatter shown)
  pcaInfo:      null,   // { varExplained: [0.62, 0.28] } when PCA was applied (>3 features)
};
