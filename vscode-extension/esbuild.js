const esbuild = require('esbuild');

const watch = process.argv.includes('--watch');

const ctx = esbuild.context({
  entryPoints: ['./src/extension.ts'],
  bundle: true,
  outfile: './dist/extension.js',
  external: ['vscode'],
  format: 'cjs',
  platform: 'node',
  sourcemap: true,
  minify: !watch,
}).then(async (context) => {
  if (watch) {
    await context.watch();
    console.log('Watching for changes...');
  } else {
    await context.rebuild();
    await context.dispose();
    console.log('Build complete!');
  }
}).catch(() => process.exit(1));
