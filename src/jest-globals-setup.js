/**
 * Jest global bindings shim for the `default-esm` preset.
 *
 * Jest 30's ESM preset exposes `jest`, `expect`, `describe`, `it`, `beforeEach`
 * and `afterEach` as *named exports* of `@jest/globals` only. The preset no
 * longer injects them onto `globalThis`, so any `.test.[mt]s` file that uses
 * the classic CJS-style globals fails with `ReferenceError: jest is not
 * defined` under `--experimental-vm-modules`. This setup step runs after the
 * module registry is ready and before the first test, and mirrors `@jest/
 * globals` onto `globalThis` so every legacy-style test file picks them up
 * without needing per-file imports.
 */
import { createRequire } from 'node:module';
import {
  afterAll as _afterAll,
  afterEach as _afterEach,
  beforeAll as _beforeAll,
  beforeEach as _beforeEach,
  describe as _describe,
  expect as _expect,
  it as _it,
  jest as _jest,
  test as _test,
} from '@jest/globals';

globalThis.jest = _jest;
globalThis.afterAll = _afterAll;
globalThis.afterEach = _afterEach;
globalThis.beforeAll = _beforeAll;
globalThis.beforeEach = _beforeEach;
globalThis.describe = _describe;
globalThis.expect = _expect;
globalThis.it = _it;
globalThis.test = _test;

// ESM does not ship `__dirname` / `require`; mirror the CJS globals that the
// existing test bodies use. Each test file's own `import.meta.url` would
// technically win if a duplicate exists, but 1 shared require is fine because
// all test files run against the same physical `node_modules`.
const _require = createRequire(import.meta.url);
globalThis.require = _require;
// The engine test files resolve sources as `path.resolve(__dirname, '../js/...')`,
// i.e. they expect `__dirname` to be the `src/__tests__` directory.
globalThis.__dirname = _require('node:url').fileURLToPath(
  new URL('./__tests__/', import.meta.url)
);
globalThis.__filename = _require('node:url').fileURLToPath(import.meta.url);

// The v4 engine is GPU-only: `createParticleSystem` throws unless the TSL
// material + compute-pipeline factory is registered. Register the real
// factory (no renderer → no compute-capability probe) so every suite runs
// against the same module instance as the sources.
import { enableWebGPU } from './webgpu.js';
enableWebGPU();
