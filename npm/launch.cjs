const { spawn, spawnSync } = require('node:child_process');
const path = require('node:path');
const { constants } = require('node:os');

module.exports = function launch(moduleName) {
  const candidates = [['python3'], ['python']];
  if (process.platform === 'win32') {
    candidates.unshift(['py', '-3']);
  }
  const check = 'import sys; sys.exit(sys.version_info < (3, 11))';
  for (const [command, ...options] of candidates) {
    const probe = spawnSync(command, [...options, '-I', '-c', check], { stdio: 'ignore' });
    if (probe.status !== 0) continue;

    const bootstrap = path.join(__dirname, 'bootstrap.py');
    const child = spawn(command, [...options, '-I', bootstrap, moduleName, ...process.argv.slice(2)], {
      stdio: 'inherit',
    });
    const terminate = () => child.kill('SIGTERM');
    process.on('SIGTERM', terminate);
    child.on('close', () => process.off('SIGTERM', terminate));
    child.on('error', (error) => {
      console.error(`Could not run DEM: ${error.message}`);
      process.exitCode = 1;
    });
    child.on('exit', (code, signal) => {
      process.exitCode = signal ? 128 + constants.signals[signal] : code;
    });
    return;
  }

  console.error('DEM requires Python 3.11 or newer. Install it and make it available on PATH.');
  process.exitCode = 1;
};
