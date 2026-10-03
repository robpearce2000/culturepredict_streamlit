// Playwright runs the built single file straight from disk (file://), like a teacher double-clicking it.
const { defineConfig } = require('@playwright/test');
module.exports = defineConfig({
  testDir: './tests',
  timeout: 240000,
  workers: 2,
  reporter: [['list']],
  outputDir: 'test-results',
  use: {
    browserName: 'chromium',
    viewport: { width: 1366, height: 768 },
    acceptDownloads: true,
    actionTimeout: 15000,
    launchOptions: { args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--autoplay-policy=no-user-gesture-required'] }
  }
});
