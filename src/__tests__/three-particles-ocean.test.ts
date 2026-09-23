/**
 * Integration coverage for the ocean-style fluid family (port of
 * `matsuoka-601/webgpu-ocean`): solver dispatch order, per-pass binding
 * budgets, dambreak initialisation, pass-chain wiring and renderer-type enums.
 */
import { StorageBufferAttribute } from 'three/webgpu';
import { RendererType, resolveWebGPUEffectiveRendererType } from '../index.js';
import {
  MLS_MPM_SUBSTEPS,
  MLS_MPM_DEFAULTS,
  encodeFixedPoint,
  decodeFixedPoint,
  computeMLSMPMGridCount,
  computeMLSMPMGridDims,
  mlsmpmQuadraticWeights,
  mlsmpmCellWordBase,
  initMLSMPMDambreak,
  countMLSMPMDambreak,
  resolveMLSMPMParams,
  createMLSMPMPipeline,
} from '../js/effects/three-particles/webgpu/fluid-mpm.js';
import {
  SPH_SUBSTEPS,
  SPH_DEFAULTS,
  SPH_DEFAULT_KERNEL_RADIUS,
  computeSPHGridCount,
  computeSPHOffset,
  computeSPHScanBlocks,
  computeSPHScanInnerSteps,
  sphKernelPowers,
  sphDensityKernelScale,
  initSPHDambreak,
  countSPHDambreak,
  resolveSPHParams,
  createSPHPipeline,
} from '../js/effects/three-particles/webgpu/fluid-sph.js';
import {
  GAUSSIAN_WEIGHTS,
  BILATERAL_GRID_LEN,
  DEPTH_LEVEL_RADII,
  DEPTH_LEVEL_MIPS,
  bilinearWeight,
  beerLambert,
  fresnelCoefficient,
  buildFluidScreenSpacePasses,
} from '../js/effects/three-particles/webgpu/tsl-fluid-screen-space-material.js';
import { createFluidSimPipeline } from '../js/effects/three-particles/webgpu/tsl-materials.js';

type Layout = {
  name: string;
  storageBindings: number;
  uniformBindings: number;
};

const MAX_STORAGE_BINDINGS = 8;

const makeShared = (capacity: number) => ({
  position: new StorageBufferAttribute(
    new Float32Array(capacity * 4),
    4
  ) as unknown as { array: Float32Array },
  velocity: new StorageBufferAttribute(
    new Float32Array(capacity * 4),
    4
  ) as unknown as { array: Float32Array },
});

describe('MLS-MPM solver (mls-mpm/)', () => {
  const params = resolveMLSMPMParams(undefined, MLS_MPM_DEFAULTS.boxSize);
  const pipeline = createMLSMPMPipeline(2164, params);

  it('dispatches 5 kernels per sub-step in reference order', () => {
    expect(MLS_MPM_SUBSTEPS).toBe(2);
    expect(pipeline.computeNodes).toHaveLength(MLS_MPM_SUBSTEPS * 5);
    expect(pipeline.passNames.slice(0, 5)).toEqual([
      'clearGrid_1',
      'p2g1_1',
      'p2g2_1',
      'updateGrid_1',
      'g2p_1',
    ]);
    expect(pipeline.passNames[9]).toBe('g2p_2');
  });

  it('keeps every pass inside the portable storage-binding budget', () => {
    const layouts = pipeline.passLayouts as Layout[];
    expect(layouts).toHaveLength(pipeline.computeNodes.length);
    layouts.forEach((layout) => {
      expect(layout.storageBindings).toBeLessThanOrEqual(MAX_STORAGE_BINDINGS);
    });
    // position + velocity + C + cells.
    expect(layouts[1].storageBindings).toBe(4);
  });

  it('sizes the 64^3 lattice from the box', () => {
    expect(computeMLSMPMGridDims(MLS_MPM_DEFAULTS.boxSize)).toEqual([
      40, 30, 60,
    ]);
    expect(computeMLSMPMGridCount(MLS_MPM_DEFAULTS.boxSize)).toBe(72000);
    expect(pipeline.gridCount).toBe(72000);
    expect(pipeline.numParticles).toBe(2164);
  });

  it('fills the dambreak lattice with the reference margins', () => {
    const state = initMLSMPMDambreak(
      MLS_MPM_DEFAULTS.boxSize,
      2164,
      0.65,
      () => 0
    );
    expect(state.count).toBe(
      countMLSMPMDambreak(MLS_MPM_DEFAULTS.boxSize, 2164)
    );
    expect(state.count).toBe(2164);
    // First lattice point is (3, 0, 3) without jitter; C is the identity.
    expect(Array.from(state.position.slice(0, 4))).toEqual([3, 0, 3, 0]);
    expect(Array.from(state.coefficients.slice(0, 12))).toEqual([
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
    ]);
  });

  it('matches the reference weights, cell indexing and fixed-point pair', () => {
    const [wm1, w0, wp1] = mlsmpmQuadraticWeights(0);
    expect(wm1).toBeCloseTo(0.125, 6);
    expect(w0).toBeCloseTo(0.75, 6);
    expect(wp1).toBeCloseTo(0.125, 6);
    const weights = mlsmpmQuadraticWeights(0);
    expect(weights[0] + weights[1] + weights[2]).toBeCloseTo(1, 6);
    expect(mlsmpmCellWordBase(1, 2, 3, 30, 60)).toBe(7692);
    expect(encodeFixedPoint(-2.5)).toBe(-25000000);
    expect(decodeFixedPoint(encodeFixedPoint(-2.5))).toBeCloseTo(-2.5, 6);
    expect(decodeFixedPoint(encodeFixedPoint(1 / 3))).toBeCloseTo(1 / 3, 6);
  });
});

describe('SPH solver (sph/)', () => {
  const params = resolveSPHParams(undefined, SPH_DEFAULTS.halfBoxSize);
  const pipeline = createSPHPipeline(148, params);

  it('dispatches the neighbour-search chain twice', () => {
    expect(SPH_SUBSTEPS).toBe(2);
    // gridClear, gridBuild, 3 scan passes, reorder(x2), density,
    // reorder pair again, force, integrate = 12 per sub-step.
    expect(pipeline.computeNodes).toHaveLength(SPH_SUBSTEPS * 12);
    expect(pipeline.passNames.slice(0, 8)).toEqual([
      'gridClear_1',
      'gridBuild_1',
      'scanPartials',
      'scanBlocks',
      'scanApply',
      'reorderPosition_1',
      'reorderForce_1',
      'density_1',
    ]);
    expect(pipeline.passNames[8]).toBe('reorderPositionB_1');
    expect(pipeline.passNames[10]).toBe('force_1');
    expect(pipeline.passNames[11]).toBe('integrate_1');
  });

  it('reports the real per-pass binding subsets', () => {
    const layouts = pipeline.passLayouts as Layout[];
    layouts.forEach((layout) => {
      expect(layout.storageBindings).toBeLessThanOrEqual(MAX_STORAGE_BINDINGS);
    });
    const byName = new Map(layouts.map((l) => [l.name, l.storageBindings]));
    expect(byName.get('gridBuild_1')).toBe(3);
    expect(byName.get('scanBlocks')).toBe(3);
    expect(byName.get('density_1')).toBe(4);
    expect(byName.get('integrate_1')).toBe(3);
  });

  it('derives the lattice, offset and scan geometry from the kernel radius', () => {
    expect(SPH_DEFAULT_KERNEL_RADIUS).toBe(0.07);
    expect(computeSPHGridCount()).toBe(62 * 62 * 62);
    expect(pipeline.gridCount).toBe(62 * 62 * 62);
    expect(computeSPHOffset()).toBeCloseTo(0.14, 6);
    expect(computeSPHScanBlocks(computeSPHGridCount() + 1)).toBe(3724);
    expect(computeSPHScanInnerSteps(3724)).toBe(59);
  });

  it('uses the reference smoothing-kernel constants', () => {
    const powers = sphKernelPowers();
    expect(sphDensityKernelScale(powers)).toBeCloseTo(
      315 / (64 * Math.PI * Math.pow(0.07, 9)),
      3
    );
  });

  it('fills the SPH dambreak with the reference lattice', () => {
    const state = initSPHDambreak(SPH_DEFAULTS.halfBoxSize, 148, 0.07, () => 0);
    expect(state.count).toBe(148);
    expect(state.count).toBe(countSPHDambreak(SPH_DEFAULTS.halfBoxSize, 148));
    expect(state.position[0]).toBeCloseTo(-0.95, 6);
    expect(state.position[1]).toBeCloseTo(-1.9, 6);
    expect(state.position[2]).toBeCloseTo(-0.95, 6);
    expect(pipeline.numParticles).toBe(148);
  });
});

describe('screen-space fluid pass chain (render/*.wgsl)', () => {
  it('exposes the reference filter tables', () => {
    expect(GAUSSIAN_WEIGHTS.length).toBe(5);
    expect(GAUSSIAN_WEIGHTS.reduce((a, b) => a + b, 0)).toBeCloseTo(1.07, 6);
    expect(BILATERAL_GRID_LEN).toBe(6);
    expect(DEPTH_LEVEL_RADII).toEqual([1, 2, 4, 8, 8]);
    expect(DEPTH_LEVEL_MIPS).toEqual([0, 1, 2, 3, 3]);
  });

  it('computes the bilateral / Beer-Lambert / Fresnel scalars', () => {
    expect(bilinearWeight(0)).toBe(1);
    expect(bilinearWeight(1)).toBe(0);
    expect(bilinearWeight(1.5)).toBe(0);
    const rgb = beerLambert(0.7, [0.0, 0.7375, 0.95]);
    // `exp( -k * (1 - waterColor) )`: the red channel (tint 0) is absorbed
    // most, so transmittance grows towards blue (matches `fluid.wgsl`).
    expect(rgb[0]).toBeCloseTo(Math.exp(-0.7), 6);
    expect(rgb[1]).toBeGreaterThan(rgb[0]);
    expect(rgb[2]).toBeGreaterThan(rgb[1]);
    expect(rgb[2]).toBeCloseTo(Math.exp(-0.7 * 0.05), 6);
    // Schlick with the `max(F0, ...)` guard.
    expect(fresnelCoefficient(1, 0.02)).toBeCloseTo(0.02, 6);
    expect(fresnelCoefficient(0, 0.02)).toBeCloseTo(1, 6);
    expect(fresnelCoefficient(0.5, 0.02)).toBeGreaterThan(0.02);
  });

  it('builds the chain and exposes camera-bindable pass nodes', () => {
    const chain = buildFluidScreenSpacePasses();
    expect(chain.material).toBeDefined();
    // depth + 4 bilateral levels + thickness + gaussian x + gaussian y.
    expect(chain.passNodes).toHaveLength(8);
    // Camera is late-bound by the host (`bindFluidPassCameras` in world.ts),
    // so every pass node starts without one.
    chain.passNodes.forEach((node) => expect(node.camera).toBeFalsy());
  });

  it('uses the direct sphere material when sphereRender is enabled', () => {
    const chain = buildFluidScreenSpacePasses({ sphereRender: true });
    expect(chain.material).toBeDefined();
    expect(chain.passNodes).toHaveLength(0);
  });
});

describe('createFluidSimPipeline (shared-pool wiring)', () => {
  it('seeds the shared buffers and merges the solver kernels (MLS-MPM)', () => {
    const shared = makeShared(2164);
    const pipeline = createFluidSimPipeline('MLS-MPM', shared, 2164, {
      renderer: { mlsMpm: { boxSize: [40, 30, 60], boxWidthRatio: 2 } },
    } as never);
    expect(pipeline.numParticles).toBe(2164);
    expect(pipeline.computeNodes.length).toBe(10);
    expect(pipeline.buffers.coefficients).toBeDefined();
    expect(pipeline.buffers.cells).toBeDefined();
    expect(pipeline.uniforms.boxWidthRatio.value).toBe(2);
    // Dambreak seeding is visible through the shared attribute. Every axis is
    // jittered by `2 * random()`, so the first point lies inside the margins.
    expect(shared.position.array[1]).toBeGreaterThanOrEqual(0);
    expect(shared.position.array[1]).toBeLessThanOrEqual(2);
    expect(shared.position.array[0]).toBeGreaterThanOrEqual(3);
    expect(shared.position.array[0]).toBeLessThan(10);
  });

  it('seeds the shared buffers and merges the solver kernels (SPH)', () => {
    const shared = makeShared(148);
    const pipeline = createFluidSimPipeline('SPH', shared, 148, {
      renderer: { sph: { halfBoxSize: [1, 2, 1], boxWidthRatio: 1 } },
    } as never);
    expect(pipeline.numParticles).toBe(148);
    expect(pipeline.computeNodes.length).toBe(24);
    expect(pipeline.buffers.prefixSums).toBeDefined();
    expect(pipeline.buffers.blockOffsets).toBeDefined();
    expect(pipeline.uniforms.boxWidthRatio.value).toBe(1);
    // First lattice row is `y = -0.95 * 2` plus the small `0.001 * random()` jitter.
    expect(shared.position.array[1]).toBeCloseTo(-1.9, 2);
  });
});

describe('renderer types', () => {
  it('routes both ocean solvers through RendererType.FLUID', () => {
    // The solver is selected by `renderer.fluid.solver`, the renderer type is
    // the shared `FLUID` entry (see `FluidConfig.solver`).
    expect(RendererType.FLUID).toBe('FLUID');
    expect(resolveWebGPUEffectiveRendererType('FLUID')).toBe(
      RendererType.FLUID
    );
    const mls = createFluidSimPipeline('MLS-MPM', makeShared(2164), 2164, {
      renderer: {},
    } as never);
    const sph = createFluidSimPipeline('SPH', makeShared(148), 148, {
      renderer: {},
    } as never);
    expect(mls.numParticles).toBe(2164);
    expect(sph.numParticles).toBe(148);
    // Distinct kernel counts: MLS-MPM 5 per sub-step, SPH 12 per sub-step.
    expect(mls.computeNodes.length).toBe(MLS_MPM_SUBSTEPS * 5);
    expect(sph.computeNodes.length).toBe(SPH_SUBSTEPS * 12);
  });
});
