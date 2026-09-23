/**
 * Tests for the new `RendererType.FLUID` metaball particle renderer:
 * InstancedBufferGeometry contract, extra `instanceVelocity` attribute, and
 * the `renderer.fluid` config knobs.
 */
import * as THREE from 'three';
import { RendererType } from '../js/effects/three-particles/three-particles-enums.js';
import {
  createParticleSystem,
  registerTSLMaterialFactory,
  resolveWebGPUEffectiveRendererType,
} from '../js/effects/three-particles/three-particles.js';
import type { ParticleSystem } from '../js/effects/three-particles/types.js';

// The engine is GPU-only in v4, so `createParticleSystem` requires the TSL
// material factory to be registered before any system is constructed.
const stubMaterial = (): THREE.Material => new THREE.MeshBasicMaterial();

const stubFactory = {
  createTSLParticleMaterial: (): THREE.Material => stubMaterial(),
  createTSLTrailMaterial: (): THREE.Material => stubMaterial(),
  createComputePipeline: (maxParticles: number) => ({
    computeNode: null,
    emitNode: null,
    simNode: null,
    computeNodes: [] as unknown[],
    passNames: ['emit', 'simulate'],
    shapeUniforms: { shapeKind: { value: 0 } } as Record<
      string,
      { value: number }
    >,
    passLayouts: [],
    allocatorCount: maxParticles + 1,
    uniforms: { seed: { value: 1 } } as Record<string, { value: unknown }>,
    buffers: {
      position: new THREE.InstancedBufferAttribute(
        new Float32Array(maxParticles * 4),
        4
      ),
      velocity: new THREE.InstancedBufferAttribute(
        new Float32Array(maxParticles * 4),
        4
      ),
      color: new THREE.InstancedBufferAttribute(
        new Float32Array(maxParticles * 4),
        4
      ),
      particleState: new THREE.InstancedBufferAttribute(
        new Float32Array(maxParticles * 4),
        4
      ),
      startValues: new THREE.InstancedBufferAttribute(
        new Float32Array(maxParticles * 4),
        4
      ),
    } as Record<string, THREE.InstancedBufferAttribute>,
    trailMeta: null,
  }),
  encodeForceFieldsForGPU: () => new Float32Array(0),
  encodeCollisionPlanesForGPU: () => new Float32Array(0),
  createSubEmitterFifoAttribute: (capacity: number) => ({
    counter: { array: new Uint32Array(2) },
    payload: { array: new Float32Array(2 * 6 * capacity) },
    capacity,
    windowSize: capacity,
    trigger: 1 as const,
  }),
  createSubEmitterInitUpdate: () => ({
    commandBuildNode: null,
    childInitNode: null,
    counterClearNode: null,
    passLayouts: [],
    passName: 'n',
    counterClearPassName: 'n',
    uniforms: {},
  }),
  createTrailRibbonUpdate: () => ({
    ribbonNode: null,
    passLayouts: [],
    uniforms: {},
    buffers: {},
  }),
  encodeShapeEmitParams: () => ({}),
};

let registered = false;
const ensureRegistered = (): void => {
  if (registered) return;
  registerTSLMaterialFactory(stubFactory as never);
  registered = true;
};

const createFluidSystem = (
  config: Record<string, unknown> = {},
  startTime = 1000
): { ps: ParticleSystem; geom: THREE.InstancedBufferGeometry } => {
  ensureRegistered();
  const ps = createParticleSystem(
    {
      maxParticles: 20,
      duration: 5,
      looping: true,
      startLifetime: 2,
      startSpeed: 1,
      startSize: 1,
      startOpacity: 1,
      emission: { rateOverTime: 10, rateOverDistance: 0 },
      renderer: {
        blending: THREE.NormalBlending,
        transparent: true,
        depthTest: true,
        depthWrite: false,
        rendererType: RendererType.FLUID,
        fluid: { stretch: 1.5 },
      },
      ...config,
    } as never,
    startTime
  );
  return {
    ps,
    geom: (ps.instance as THREE.Mesh).geometry as THREE.InstancedBufferGeometry,
  };
};

describe('resolveWebGPUEffectiveRendererType / FLUID', () => {
  it('passes FLUID through 1:1', () => {
    expect(resolveWebGPUEffectiveRendererType(RendererType.FLUID)).toBe(
      RendererType.FLUID
    );
  });
});

describe('FLUID metaball particle renderer', () => {
  it('builds a THREE.Mesh with InstancedBufferGeometry', () => {
    const { ps, geom } = createFluidSystem();
    expect(ps.instance).toBeInstanceOf(THREE.Mesh);
    expect(geom).toBeInstanceOf(THREE.InstancedBufferGeometry);
    expect(geom.instanceCount).toBe(20);
    ps.dispose();
  });

  it('binds the velocity instance attribute only for FLUID', () => {
    const fluid = createFluidSystem();
    const fluidGeom = fluid.geom;
    expect(fluidGeom.getAttribute('instanceVelocity')).toBeInstanceOf(
      THREE.InstancedBufferAttribute
    );
    // 20 particles × 1 vec4 entry each.
    expect(fluidGeom.getAttribute('instanceVelocity').count).toBe(20);
    fluid.ps.dispose();

    ensureRegistered();
    const instanced = createParticleSystem(
      {
        maxParticles: 10,
        duration: 5,
        looping: true,
        startLifetime: 1,
        startSpeed: 1,
        startSize: 1,
        startOpacity: 1,
        emission: { rateOverTime: 5, rateOverDistance: 0 },
        renderer: {
          blending: THREE.NormalBlending,
          transparent: true,
          depthTest: true,
          depthWrite: false,
          rendererType: RendererType.INSTANCED,
        },
      } as never,
      1000
    );
    const instGeom = (instanced.instance as THREE.Mesh)
      .geometry as THREE.InstancedBufferGeometry;
    expect(instGeom.getAttribute('instanceVelocity')).toBeUndefined();
    instanced.dispose();
  });

  it('propagates fluid.{stretch,absorption,ior} into sharedUniforms', () => {
    const { ps } = createFluidSystem({
      renderer: {
        blending: THREE.NormalBlending,
        transparent: true,
        depthTest: true,
        depthWrite: false,
        rendererType: RendererType.FLUID,
        fluid: { stretch: 2.25, absorption: 0.9, ior: 1.4 },
      },
    });
    const gpuDebug = (
      ps as unknown as {
        gpuDebug?: Record<string, unknown>;
      }
    ).gpuDebug;
    expect(gpuDebug).toBeDefined();
    expect(gpuDebug!.effectiveRendererType).toBe('FLUID');
    expect(gpuDebug!.requestedRendererType).toBe('FLUID');
    ps.dispose();
  });

  it('falls back to documented defaults when fluid is omitted', () => {
    const { ps } = createFluidSystem({
      renderer: {
        blending: THREE.NormalBlending,
        transparent: true,
        depthTest: true,
        depthWrite: false,
        rendererType: RendererType.FLUID,
      },
    });
    const gpuDebug = (
      ps as unknown as {
        gpuDebug?: Record<string, unknown>;
      }
    ).gpuDebug;
    expect(gpuDebug!.effectiveRendererType).toBe('FLUID');
    ps.dispose();
  });
});
