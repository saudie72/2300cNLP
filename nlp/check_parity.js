// Confirms the browser classifier in index.html matches scikit-learn's predict_proba.
// Usage: node nlp/check_parity.js   (run python3 nlp/train.py first)
const fs = require('fs'), path = require('path'), vm = require('vm');
const root = path.join(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const src = html.match(/<script>\s*\/\* Energy check-in[\s\S]*?<\/script>/)[0].replace(/<\/?script>/g, '');
const ctx = {document: {getElementById: () => null}, Math};
vm.runInNewContext(src + '\nthis.EC = EnergyCheckin;', ctx);
const model = JSON.parse(fs.readFileSync(path.join(root, 'assets/energy-model.json'), 'utf8'));
const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, 'parity_fixture.json'), 'utf8'));
let worst = 0;
for (const {text, proba} of fixture) {
  const p = ctx.EC.predict(model, text).proba;
  proba.forEach((v, i) => { worst = Math.max(worst, Math.abs(v - p[i])); });
}
console.log(`${fixture.length} messages, max probability difference ${worst.toExponential(2)}`);
process.exit(worst < 1e-3 ? 0 : 1);
