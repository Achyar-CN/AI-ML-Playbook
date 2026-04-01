import { App } from './app.js';

const app = new App({
  root: document.getElementById('simulation-root'),
  sidebar: document.getElementById('sidebar'),
});

app.init();
