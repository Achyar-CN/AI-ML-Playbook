# Deployment GitHub Pages

1. Pastikan repo sudah berada di GitHub.
2. Install dependencies:
   - `npm install`
3. Build project:
   - `npm run build`
4. Deploy default ke gh-pages:
   - `npm run deploy`

Konfigurasi (opsional):
- Pastikan `package.json` memiliki `homepage` jika perlu.
- Untuk GitHub Action, buat `.github/workflows/deploy.yml` untuk otomatis deploy setiap push `main`.
