// Playwright runs the built single file straight from disk (file://), like a teacher double-clicking it.
const { defineConfig } = require('@playwright/test');
module.exports = defineConfig({
  testDir: './tests',
  globalSetup: require.resolve('./tests/global-setup.js'),   // builds first, skipped if src/ is unchanged
  timeout: 240000,
  workers: 2,
  reporter: [['list']],
  outputDir: 'test-results',
  use: {
    browserName: 'chromium',
    // gameplay tests use a small screen: rules and scoring don't depend on size, and fewer pixels
    // render faster. The screenshot sweep sets the three real sizes itself.
    viewport: { width: 960, height: 600 },
    acceptDownloads: true,
    actionTimeout: 15000,
    launchOptions: { args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--autoplay-policy=no-user-gesture-required'] }
  }
});
