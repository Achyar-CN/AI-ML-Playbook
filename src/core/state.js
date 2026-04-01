export class StateManager {
  constructor() {
    this.current = this._parseHash(window.location.hash);
    window.addEventListener('hashchange', () => {
      this.current = this._parseHash(window.location.hash);
      if (this.onUpdate) {
        this.onUpdate(this.current);
      }
    });
  }

  _parseHash(hash) {
    const raw = hash.startsWith('#') ? hash.slice(1) : hash;
    const params = new URLSearchParams(raw);
    const result = {};

    for (const [key, value] of params.entries()) {
      if (!Number.isNaN(Number(value)) && value.trim() !== '') {
        result[key] = Number(value);
      } else {
        result[key] = value;
      }
    }

    return result;
  }

  get(key, fallback) {
    if (Object.prototype.hasOwnProperty.call(this.current, key)) {
      return this.current[key];
    }
    return fallback;
  }

  setState(values) {
    this.current = { ...this.current, ...values };
    const params = new URLSearchParams();
    Object.entries(this.current).forEach(([k, v]) => {
      if (v === undefined || v === null) return;
      params.set(k, String(v));
    });
    window.location.hash = params.toString();
  }
}
