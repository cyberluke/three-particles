import * as THREE from 'three';
import {
  Shape,
  SimulationSpace,
  RendererType,
} from '../js/effects/three-particles/three-particles-enums.js';
import {
  createParticleSystem,
  updateParticleSystems,
  getDefaultParticleSystemConfig,
  blendingMap,
} from '../js/effects/three-particles/three-particles.js';
import type { ParticleSystem } from '../js/effects/three-particles/types.js';

/**
 * GPU-only (4.x) contract tests for `createParticleSystem`.
 *
 * The CPU owns only scalar uniforms; the per-frame emission count is read
 * from `gpuDebug.lastEmitCount()`, and the geometry attributes are the
 * compute-owned vec4 storage pools (`position`, `color`, `particleState`,
 * `startValues`).
 */
const lastEmit = (ps: ParticleSystem): number =>
  (
    ps as unknown as {
      gpuDebug: { lastEmitCount(): number };
    }
  ).gpuDebug.lastEmitCount();

const getAttributes = (ps: ParticleSystem) =>
  (ps.instance as THREE.Points).geometry.attributes;

const createTestSystem = (
  config: Record<string, unknown> = {},
  startTime = 1000
) => {
  const ps = createParticleSystem(
    {
      maxParticles: 50,
      duration: 5,
      looping: true,
      startLifetime: 2,
      startSpeed: 1,
      startSize: 1,
      startOpacity: 1,
      startRotation: 0,
      emission: { rateOverTime: 10, rateOverDistance: 0 },
      ...config,
    } as never,
    startTime
  );

  const step = (timeOffsetMs: number, deltaMs: number = 16) => {
    ps.update({
      now: startTime + timeOffsetMs,
      delta: deltaMs / 1000,
      elapsed: timeOffsetMs / 1000,
    });
  };

  return { ps, step, startTime };
};

// ─── createParticleSystem ────────────────────────────────────────────────────

describe('createParticleSystem', () => {
  it('should create a particle system with default config', () => {
    const { ps } = createTestSystem();
    expect(ps.instance).toBeInstanceOf(THREE.Points);
    expect(ps.update).toBeDefined();
    expect(ps.dispose).toBeDefined();
    expect(ps.updateConfig).toBeDefined();
    ps.dispose();
  });

  it('should create a particle system with custom config', () => {
    const { ps } = createTestSystem({ maxParticles: 200 });
    const dbg = (
      ps as unknown as { gpuDebug: { maxParticles: number } }
    ).gpuDebug;
    expect(dbg.maxParticles).toBe(200);
    ps.dispose();
  });

  it('should create geometry with the GPU storage contract attributes', () => {
    const { ps } = createTestSystem({ maxParticles: 10 });
    const attrs = getAttributes(ps);
    for (const name of [
      'position',
      'color',
      'particleState',
      'startValues',
    ]) {
      expect(attrs[name]).toBeDefined();
      expect(attrs[name].itemSize).toBe(4);
      expect(attrs[name].count).toBe(10);
    }
    // Legacy CPU-era attributes are gone.
    expect(attrs.isActive).toBeUndefined();
    expect(attrs.lifetime).toBeUndefined();
    ps.dispose();
  });

  it('should set the draw range to maxParticles', () => {
    const { ps } = createTestSystem({ maxParticles: 12 });
    const geom = (ps.instance as THREE.Points).geometry;
    expect(geom.drawRange.count).toBe(12);
    ps.dispose();
  });

  it('should accept externalNow parameter', () => {
    const ps = createParticleSystem({ maxParticles: 5 }, 1234);
    expect(ps.instance).toBeDefined();
    ps.dispose();
  });

  it('getDefaultParticleSystemConfig returns a fresh object', () => {
    const a = getDefaultParticleSystemConfig();
    const b = getDefaultParticleSystemConfig();
    expect(a).not.toBe(b);
    expect(a.maxParticles).toBe(
      (getDefaultParticleSystemConfig() as { maxParticles: number })
        .maxParticles
    );
  });
});

// ─── Shape configurations ────────────────────────────────────────────────────

describe('Shape configurations', () => {
  const cases: Array<[Shape, number]> = [
    [Shape.SPHERE, 0],
    [Shape.CONE, 1],
    [Shape.CIRCLE, 2],
    [Shape.RECTANGLE, 3],
    [Shape.BOX, 4],
  ];

  it.each(cases)(
    'should create system with %s shape (gpu kind %i)',
    (shape, kind) => {
      const { ps } = createTestSystem({
        shape: { shape },
      });
      const snap = (
        ps as unknown as {
          gpuDebug: { snapshot(): { shape: { gpuShapeKind: number } } };
        }
      ).gpuDebug.snapshot();
      expect(snap.shape.gpuShapeKind).toBe(kind);
      ps.dispose();
    }
  );

  it('should create system with BOX shape (SHELL)', () => {
    const { ps } = createTestSystem({
      shape: { shape: Shape.BOX, box: { emitFrom: 'SHELL' } },
    });
    const snap = (
      ps as unknown as {
        gpuDebug: { snapshot(): { shape: { gpuShapeKind: number } } };
      }
    ).gpuDebug.snapshot();
    expect(snap.shape.gpuShapeKind).toBe(4);
    ps.dispose();
  });
});

// ─── Particle lifecycle ──────────────────────────────────────────────────────

describe('Particle lifecycle', () => {
  it('should emit particles over time', () => {
    const { ps, step } = createTestSystem({
      emission: { rateOverTime: 100 },
    });

    step(100);
    expect(lastEmit(ps)).toBe(10);

    ps.dispose();
  });

  it('should not emit particles when paused', () => {
    const { ps, step } = createTestSystem({
      emission: { rateOverTime: 100 },
    });

    step(100);
    expect(lastEmit(ps)).toBeGreaterThan(0);

    ps.pauseEmitter();
    step(200, 100);
    expect(lastEmit(ps)).toBe(0);

    ps.dispose();
  });

  it('should resume emitting after resumeEmitter', () => {
    const { ps, step } = createTestSystem({
      emission: { rateOverTime: 100 },
    });

    ps.pauseEmitter();
    step(100);
    expect(lastEmit(ps)).toBe(0);

    ps.resumeEmitter();
    step(200, 100);
    // The paused frame does not advance lastEmissionTime, so the resumed
    // frame covers 200 ms of accumulated time → 20 particles.
    expect(lastEmit(ps)).toBe(20);

    ps.dispose();
  });

  it('should stop emitting in non-looping mode after duration', () => {
    const { ps, step } = createTestSystem({
      looping: false,
      duration: 1,
      emission: { rateOverTime: 100 },
    });

    step(100);
    expect(lastEmit(ps)).toBeGreaterThan(0);

    step(1500, 1400);
    expect(lastEmit(ps)).toBe(0);

    ps.dispose();
  });

  it('should keep emitting across loops when looping is true', () => {
    const { ps, step } = createTestSystem({
      looping: true,
      duration: 1,
      emission: { rateOverTime: 100 },
    });

    step(1500, 1400);
    expect(lastEmit(ps)).toBeGreaterThan(0);

    ps.dispose();
  });
});

// ─── Global update loop ──────────────────────────────────────────────────────

describe('updateParticleSystems', () => {
  it('should update all created particle systems', () => {
    const startTime = 1000;
    const ps1 = createParticleSystem(
      { maxParticles: 5, duration: 5, looping: true, emission: { rateOverTime: 100 } },
      startTime
    );
    const ps2 = createParticleSystem(
      { maxParticles: 5, duration: 5, looping: true, emission: { rateOverTime: 100 } },
      startTime
    );

    updateParticleSystems({
      now: startTime + 100,
      delta: 0.1,
      elapsed: 0.1,
    });

    expect(lastEmit(ps1)).toBe(5);
    expect(lastEmit(ps2)).toBe(5);

    ps1.dispose();
    ps2.dispose();
  });
});

// ─── Renderer types ──────────────────────────────────────────────────────────

describe('renderer types', () => {
  it('creates Points for POINTS', () => {
    const { ps } = createTestSystem({
      renderer: { rendererType: RendererType.POINTS },
    });
    expect(ps.instance).toBeInstanceOf(THREE.Points);
    ps.dispose();
  });

  it('creates Mesh for INSTANCED', () => {
    const { ps } = createTestSystem({
      renderer: { rendererType: RendererType.INSTANCED },
    });
    expect(ps.instance).toBeInstanceOf(THREE.Mesh);
    ps.dispose();
  });

  it('exposes requested/effective renderer types in gpuDebug', () => {
    const { ps } = createTestSystem();
    const dbg = (
      ps as unknown as {
        gpuDebug: {
          requestedRendererType: string;
          effectiveRendererType: string;
        };
      }
    ).gpuDebug;
    expect(dbg.requestedRendererType).toBe('POINTS');
    expect(dbg.effectiveRendererType).toBe('POINTS');
    ps.dispose();
  });
});

// ─── Simulation space ────────────────────────────────────────────────────────

describe('simulation space', () => {
  it('defaults to LOCAL', () => {
    const { ps } = createTestSystem();
    const snap = (
      ps as unknown as {
        gpuDebug: { snapshot(): { simulationSpace: string } };
      }
    ).gpuDebug.snapshot();
    expect(snap.simulationSpace).toBe(SimulationSpace.LOCAL);
    ps.dispose();
  });

  it('WORLD space pins matrixWorld to identity', () => {
    const { ps } = createTestSystem({
      simulationSpace: SimulationSpace.WORLD,
    });
    expect(ps.instance.matrixWorldAutoUpdate).toBe(false);
    expect(ps.instance.matrixWorld.elements).toEqual(
      new THREE.Matrix4().elements
    );
    ps.dispose();
  });
});

// ─── Blending map ────────────────────────────────────────────────────────────

describe('blendingMap', () => {
  it('is exported with the standard entries', () => {
    expect(blendingMap).toBeDefined();
    expect(typeof blendingMap).toBe('object');
  });
});
