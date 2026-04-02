import { App } from './app.js';

const app = new App({
  root: document.getElementById('simulation-root'),
  controlsPanel: document.getElementById('controls'),
});

app.init();
