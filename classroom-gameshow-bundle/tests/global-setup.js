// Build the product and the test build before the tests run. build.js skips the work
// when nothing in src/ has changed since the last build.
const { execFileSync } = require('child_process');
const path = require('path');
module.exports = () => {
  execFileSync(process.execPath, [path.join(__dirname, '..', 'build.js')], { stdio: 'inherit' });
  // every edition for sale, and its test build (tests/editions.spec.js)
  execFileSync(process.execPath, [path.join(__dirname, '..', 'build.js'), '--editions'], { stdio: 'ignore' });
};
