/**
 * Compute the top-2 principal components of a data matrix.
 *
 * @param {number[][]} X  - Array of n samples; each sample is a d-dimensional array.
 * @returns {{
 *   projected:    number[][],   // n × 2 projected coordinates, normalised to [-1,1]
 *   varExplained: number[],     // [pc1_variance_ratio, pc2_variance_ratio]
 *   components:   number[][]    // [pc1_vector, pc2_vector] (length-d each)
 * }}
 */
export function computePCA(X) {
  const n = X.length;
  const d = X[0].length;
  if (d < 2) throw new Error('PCA requires at least 2 features.');

  // 1. Mean-centre each feature
  const mean = new Array(d).fill(0);
  X.forEach(row => row.forEach((v, j) => { mean[j] += v / n; }));
  const Xc = X.map(row => row.map((v, j) => v - mean[j]));

  // 2. Covariance matrix (d × d)
  const cov = Array.from({ length: d }, () => new Array(d).fill(0));
  Xc.forEach(row => {
    for (let i = 0; i < d; i++)
      for (let j = 0; j < d; j++)
        cov[i][j] += row[i] * row[j] / Math.max(n - 1, 1);
  });

  const totalVar = cov.reduce((s, _, i) => s + cov[i][i], 0) || 1;

  // Helper: matrix × vector
  function matVec(M, v) {
    return M.map(row => row.reduce((s, x, j) => s + x * v[j], 0));
  }
  function normalise(v) {
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
    return v.map(x => x / norm);
  }

  // 3. Power iteration to find the dominant eigenvector of M
  function topEigen(M) {
    let v = normalise(new Array(d).fill(1));
    for (let iter = 0; iter < 300; iter++) {
      const w = normalise(matVec(M, v));
      if (v.every((x, k) => Math.abs(x - w[k]) < 1e-10)) break;
      v = w;
    }
    const eigenval = matVec(M, v).reduce((s, x, i) => s + x * v[i], 0);
    return { vec: v, val: Math.max(0, eigenval) };
  }

  // PC1
  const pc1 = topEigen(cov);

  // Deflate covariance: cov₂ = cov − λ₁·v₁·v₁ᵀ
  const cov2 = cov.map((row, i) =>
    row.map((val, j) => val - pc1.val * pc1.vec[i] * pc1.vec[j])
  );
  // PC2
  const pc2 = topEigen(cov2);

  const varExplained = [pc1.val / totalVar, pc2.val / totalVar];

  // 4. Project data onto the top-2 PCs
  const raw = Xc.map(row => [
    row.reduce((s, x, i) => s + x * pc1.vec[i], 0),
    row.reduce((s, x, i) => s + x * pc2.vec[i], 0),
  ]);

  // 5. Normalise each PC axis to [-1, 1] for model compatibility
  function norm1D(vals) {
    const mn = Math.min(...vals), mx = Math.max(...vals);
    if (mx === mn) return vals.map(() => 0);
    return vals.map(v => 2 * (v - mn) / (mx - mn) - 1);
  }
  const ax = norm1D(raw.map(r => r[0]));
  const ay = norm1D(raw.map(r => r[1]));
  const projected = ax.map((v, i) => [v, ay[i]]);

  return { projected, varExplained, components: [pc1.vec, pc2.vec] };
}
