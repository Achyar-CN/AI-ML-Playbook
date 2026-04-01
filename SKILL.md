# AI/ML Playground Skill Guide

## Tujuan

Menengahkan konsep AI/ML secara visual interaktif dan modular. Proyek ini harus memungkinkan kontributor menambah simulasi baru tanpa mengubah core engine.

## Pendekatan Modular

1. `src/core/` : engine generik, manager, UI controller.
2. `src/simulations/` : setiap konsep ada folder masing-masing sebagai modul.
3. `src/config/simulations.js` : daftar modul tersedia / metadata.
4. `src/main.js` : entry point aplikasi.

## Konvensi Naming

- Modul: `src/simulations/{id}/index.js`
- Kelas simulasi: `{CamelCase}Simulation` extends `BaseSimulation`
- ID simulasi unik: `kebab-case`.
- Parameter default: `defaultParams` di config.

## API Simulasi (BaseSimulation)

Method wajib:
- `setup()` : inisialisasi dataset + state.
- `step()` : satu iterasi per-frame.
- `render()` : menggambar visual di canvas.
- `reset()` : mengatur ulang ke state awal (default memanggil setup).

Optional:
- `serializeState()` / `deserializeState(vars)` untuk URL sharing.

## Menambah Simulasi Baru

1. Buat folder baru:
   - `src/simulations/{id}/index.js`
2. Import `BaseSimulation`:
   - `import { BaseSimulation } from '../baseSimulation.js';`
3. Buat kelas extends `BaseSimulation`.
4. Daftarkan di `src/config/simulations.js`.
5. Tambahkan penjelasan di `README.md` atau `docs/simulations.md`.
6. Tes interaktif di UI:
   - pilih modul di dropdown
   - start/pause/reset

## URL dan Shareable State

- Siapkan adapter state ke hash URL seperti `#sim=perceptron&lr=0.1`.
- Pada load, baca `window.location.hash`.
- Saat parameter berubah, update hash.

## Deployment

1. `npm install`
2. `npm run build`
3. `npm run deploy`

File penting:
- `index.html`
- `package.json` (`scripts`, `homepage`)
- `docs/deployment.md`

## Roadmap Fitur

- v1: Perceptron + visual decision boundary, kontrol learning rate, epochs.
- v1.1: NN 2D, activation select, boundary + loss.
- v2: k-Means, PCA 2D, SVM, interactive dataset.
- v3: Pluggable simulator eksternal, i18n, web worker API.

## Best Practices

- Pisah logic matematika dari rendering.
- `requestAnimationFrame` untuk animasi agar UI tidak freeze.
- `Math.seedrandom` (opsional) untuk deterministik.
- Unit test modul: `vitest/jest` untuk `SimulationManager` dan `BaseSimulation`.
- Gunakan ESLint + Prettier.

## Checklist QA

- [ ] Aplikasi load tanpa error.
- [ ] Simulasi default berjalan.
- [ ] Start/Pause/Reset berfungsi.
- [ ] Parameter bervariasi bisa diaplikasikan.
- [ ] URL state bisa dimuat ulang dengan state sama.
- [ ] Build & deploy GitHub Pages berhasil.
