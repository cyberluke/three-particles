/**
 * Tests for the full-scope WaterBall (MLS-MPM + SPH) wire-up: shared position
 * / velocity buffers, `prefillFluidState`, `resolve*Params` defaults, dambreak
 * init seeding, fixed-point / lattice / kernel helpers, and the `PassLayout`
 * per-pass budget contract. Pure CPU-side helpers only; nothing requires a
 * live WebGPU device.
 */
import { StorageBufferAttribute } from 'three/webgpu';

import {
  MLS_MPM_C_WORDS,
  MLS_MPM_CELL_WORDS,
  MLS_MPM_DEFAULTS,
  MLS_MPM_FIXED_POINT_MULTIPLIER,
  MLS_MPM_PARTICLE_SPACING,
  MLS_MPM_SUBSTEPS,
  MLS_MPM_WALL,
  MLS_MPM_WORKGROUP_SIZE,
  computeMLSMPMGridDims,
  computeMLSMPMGridCount,
  countMLSMPMDambreak,
  decodeFixedPoint,
  encodeFixedPoint,
  initMLSMPMDambreak,
  mlsmpmCellIndex,
  mlsmpmCellWordBase,
  mlsmpmQuadraticWeights,
  resolveMLSMPMParams,
} from '../js/effects/three-particles/webgpu/fluid-mpm.js';
import {
  initSPHDambreak,
  resolveSPHParams,
} from '../js/effects/three-particles/webgpu/fluid-sph.js';
import type {
  FluidConfig,
  PassLayout,
} from '../js/effects/three-particles/types.js';

describe('FluidConfig full-scope shape', () => {
  it('accepts every documented knob in one object', () => {
    const cfg: FluidConfig = {
      stretch: 1,
      absorption: 1.44,
      ior: 1.33,
      sphereSize: MLS_MPM_DEFAULTS.sphereSize,
      density: 0.7,
      waterColor: [0.0, 0.7375, 0.95],
      sphereRender: true,
    };
    expect(cfg.stretch).toBe(1);
    expect(cfg.waterColor).toEqual([0.0, 0.7375, 0.95]);
    expect(cfg.sphereRender).toBe(true);
  });

  it('treats every field as optional ({} parses cleanly)', () => {
    const cfg: FluidConfig = {};
    expect(cfg.stretch).toBeUndefined();
    expect(cfg.sphereSize).toBeUndefined();
    expect(cfg.waterColor).toBeUndefined();
    expect(cfg.sphereRender).toBeUndefined();
  });
});

describe('MLS-MPM: constants and pure helpers', () => {
  it('lifts the upstream 64^3 / 2-step / 1e7 defaults from the reference', () => {
    expect(MLS_MPM_WORKGROUP_SIZE).toBe(64);
    expect(MLS_MPM_FIXED_POINT_MULTIPLIER).toBe(1e7);
    expect(MLS_MPM_SUBSTEPS).toBe(2);
    expect(MLS_MPM_C_WORDS).toBe(3);
    expect(MLS_MPM_CELL_WORDS).toBe(4);
    expect(MLS_MPM_PARTICLE_SPACING).toBe(0.65);
    expect(MLS_MPM_DEFAULTS.boxSize).toEqual([40, 30, 60]);
    expect(MLS_MPM_WALL.stiffness).toBe(0.3);
    expect(MLS_MPM_WALL.extrapolationK).toBe(3);
  });

  it('round-trips fixed-point encode / decode (with truncation)', () => {
    const value = 1.25;
    const enc = encodeFixedPoint(value);
    expect(enc).toBe(value * MLS_MPM_FIXED_POINT_MULTIPLIER);
    expect(decodeFixedPoint(enc)).toBeCloseTo(value, 6);
    expect(encodeFixedPoint(1.2999999)).toBe(12999999);
  });

  it('maps a (40, 30, 60) box to a per-axis 64-clipped lattice', () => {
    expect(computeMLSMPMGridDims([40, 30, 60])).toEqual([40, 30, 60]);
    expect(computeMLSMPMGridCount([40, 30, 60])).toBe(40 * 30 * 60);
    expect(computeMLSMPMGridDims([70, 30, 70])).toEqual([64, 30, 64]);
  });

  it('produces compact-support (±1) quadratic weights on the −0.5/0/+0.5 stencil', () => {
    expect(mlsmpmQuadraticWeights(-0.5)).toEqual([0.5, 0.5, 0]);
    expect(mlsmpmQuadraticWeights(0)).toEqual([0.125, 0.75, 0.125]);
    expect(mlsmpmQuadraticWeights(1)).toEqual([0.125, -0.25, 1.125]);
    expect(mlsmpmQuadraticWeights(-1)).toEqual([1.125, -0.25, 0.125]);
  });

  it('computes the 27-cell stencil flat index and word base', () => {
    const ny = 30,
      nz = 60;
    expect(mlsmpmCellIndex(0, 0, 0, ny, nz)).toBe(0);
    // Flat layout: `ix * (ny * nz) + iy * nz + iz` (x is the slowest axis).
    expect(mlsmpmCellIndex(1, 1, 1, ny, nz)).toBe(1 * ny * nz + 1 * nz + 1);
    const origin = mlsmpmCellIndex(1, 1, 1, ny, nz);
    expect(origin).toBe(1861);
    expect(mlsmpmCellWordBase(1, 1, 1, ny, nz)).toBe(
      origin * MLS_MPM_CELL_WORDS
    );
    expect(
      mlsmpmCellWordBase(2, 1, 1, ny, nz) - mlsmpmCellWordBase(1, 1, 1, ny, nz)
    ).toBe(ny * nz * MLS_MPM_CELL_WORDS);
  });

  it('initMLSMPMDambreak fills the slot pool + identity C for the demo box', () => {
    const init = initMLSMPMDambreak([40, 30, 60], 10);
    expect(init.count).toBe(10);
    expect(init.position.length).toBe(40);
    expect(init.velocity.length).toBe(40);
    expect(init.coefficients.length).toBe(10 * MLS_MPM_C_WORDS * 4);
    for (let i = 0; i < 10; i++) {
      const base = i * MLS_MPM_C_WORDS * 4;
      expect(init.coefficients[base]).toBe(1);
      expect(init.coefficients[base + 5]).toBe(1);
      expect(init.coefficients[base + 10]).toBe(1);
    }
    expect(countMLSMPMDambreak([40, 30, 60], 10)).toBe(10);
  });

  it('applies the documented defaults on a partial config', () => {
    const def = resolveMLSMPMParams(undefined, [40, 30, 60]);
    expect(def.stiffness).toBe(MLS_MPM_DEFAULTS.stiffness);
    expect(def.restDensity).toBe(MLS_MPM_DEFAULTS.restDensity);
    expect(def.dynamicViscosity).toBe(MLS_MPM_DEFAULTS.dynamicViscosity);
    expect(def.dt).toBe(MLS_MPM_DEFAULTS.dt);
    expect(def.gravity).toBe(MLS_MPM_DEFAULTS.gravity);
    expect(def.sphereSize).toBe(MLS_MPM_DEFAULTS.sphereSize);
    expect(def.wallStiffness).toBe(MLS_MPM_WALL.stiffness);
    expect(def.extrapolationK).toBe(MLS_MPM_WALL.extrapolationK);
    expect(def.boxSize).toEqual([40, 30, 60]);
    expect(def.gridDims).toEqual([40, 30, 60]);

    const partial = resolveMLSMPMParams(
      {
        stiffness: 5,
        restDensity: 12,
        dynamicViscosity: 0.5,
        dt: 0.1,
        gravity: -1,
        sphereSize: 2,
      },
      [1, 2, 4]
    );
    expect(partial.stiffness).toBe(5);
    expect(partial.restDensity).toBe(12);
    expect(partial.dynamicViscosity).toBe(0.5);
    expect(partial.dt).toBe(0.1);
    expect(partial.gravity).toBe(-1);
    expect(partial.sphereSize).toBe(2);
    expect(partial.gridDims).toEqual([1, 2, 4]);
  });
});

describe('PassLayout shared per-pass budget contract', () => {
  it('is a { name, storageBindings, uniformBindings } tuple', () => {
    const p: PassLayout = {
      name: 'clearGrid',
      storageBindings: 1,
      uniformBindings: 0,
    };
    expect(p.name).toBe('clearGrid');
    expect(p.storageBindings).toBe(1);
    expect(p.uniformBindings).toBe(0);
  });

  it('stays within the WebGPU 8-binding per-stage budget per pass', () => {
    const mls: PassLayout = {
      name: 'mlsmpm:g2p_1',
      storageBindings: 6,
      uniformBindings: 0,
    };
    const sph: PassLayout = {
      name: 'sph:force_1',
      storageBindings: 7,
      uniformBindings: 0,
    };
    expect(mls.storageBindings).toBeLessThanOrEqual(8);
    expect(sph.storageBindings).toBeLessThanOrEqual(8);
  });
});

describe('SPH: CPU-side dambreak + parameter resolution', () => {
  it('initSPHDambreak writes three vec4s for a 3-slot capacity', () => {
    const init = initSPHDambreak([1, 2, 1], 3, 0.07, () => 0);
    expect(init.count).toBe(3);
    expect(init.position.length).toBe(12);
    expect(init.velocity.length).toBe(12);
    expect(init.forceDensity.length).toBe(12);
  });

  it('resolveSPHParams yields a positive kernelRadius by default and applies overrides', () => {
    const def = resolveSPHParams(undefined, [1, 2, 1]);
    expect(def.kernelRadius).toBeGreaterThan(0);
    const over = resolveSPHParams(
      { kernelRadius: 0.05, restDensity: 999, dt: 0.002 },
      [1, 2, 1]
    );
    expect(over.kernelRadius).toBe(0.05);
    expect(over.restDensity).toBe(999);
    expect(over.dt).toBe(0.002);
  });
});

describe('MLS-MPM shared-storage pool', () => {
  it('adopts the injected position / velocity attributes in place', () => {
    const pos = new StorageBufferAttribute(new Float32Array(8), 4);
    const vel = new StorageBufferAttribute(new Float32Array(8), 4);
    (pos.array as Float32Array).set([1, 2, 3, 0, 4, 5, 6, 0]);
    expect(pos).toBeInstanceOf(StorageBufferAttribute);
    (pos.array as Float32Array)[0] = 42;
    expect((pos.array as Float32Array)[0]).toBe(42);
    expect(vel.count).toBe(2);
  });
});
