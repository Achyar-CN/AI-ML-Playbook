// Shared in-memory store for user-imported CSV data.
// Set by UIController on CSV upload; read by BaseSimulation dataset generators.
export const dataStore = {
  points:       null,   // [{x, y, z?, label?}] – processed 2D or 3D representation
  type:         null,   // 'classification' | 'regression'
  filename:     null,

  // ── Feature metadata (populated on CSV upload) ─────────────────
  featureNames: [],     // ALL input column names (before feature selection)
  targetName:   null,   // label/target column name, e.g. 'label' or 'y'
  nFeatures:    0,      // number of input feature columns in original CSV

  // ── Axis labels for canvas visualisation ───────────────────────
  xLabel:       null,   // e.g. 'age', 'PC1 (62%)'  — null = no label
  yLabel:       null,   // e.g. 'income', 'PC2 (28%)'
  zLabel:       null,   // only set when is3D = true
  wLabel:       null,   // 4th dimension label (regression bubble: target name)

  // ── Dimensionality flags ────────────────────────────────────────
  is3D:         false,  // true when 3D scatter shown
  regFeatures:  0,      // 1 = 2D line | 2 = 3D surface | 3 = 3D bubble chart
  pcaInfo:      null,   // { varExplained: [0.62, 0.28] } when PCA was applied

  // ── Raw data (for feature re-selection after upload) ────────────
  rawRows:          null,   // numeric rows: number[][] (filtered, all columns)
  rawHeader:        null,   // string[] or null (original header)
  selectedFeatIdxs: null,   // number[] — indices into featureNames currently active
  targetRange:      null,   // {min, max} of raw target — for bubble chart color legend
};
