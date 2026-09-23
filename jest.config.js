export default {
  preset: 'ts-jest/presets/default-esm',
  testEnvironment: 'node',
  extensionsToTreatAsEsm: ['.ts'],
  transform: {
    '^.+\\.m?[tj]sx?$': 'babel-jest',
  },
  // Jest 30's `default-esm` preset does not inject `jest`/`describe`/`it`/…
  // onto `globalThis`, and ESM lacks `__dirname`/`require`. This setup file
  // mirrors `@jest/globals` + the CJS globals the legacy test bodies use.
  setupFiles: ['<rootDir>/src/jest-globals-setup.js'],
  moduleNameMapper: {
    '^(.+)\\.js$': '$1',
    '^@cyberluke/three-particles$': '<rootDir>/src/index.ts',
  },
  transformIgnorePatterns: ['node_modules/(?!three-noise|three|@newkrok/three-utils)'],
  roots: ['<rootDir>/src'],
  moduleFileExtensions: ['ts', 'js'],
  testMatch: ['**/__tests__/**/*.test.ts'],
  collectCoverage: true,
  coverageReporters: ['text', 'lcov'],
  maxWorkers: '50%',
};
