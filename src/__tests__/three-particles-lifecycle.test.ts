import * as THREE from 'three';
import {
  createParticleSystem,
  updateParticleSystems,
} from '../js/effects/three-particles/three-particles.js';
import type { ParticleSystem } from '../js/effects/three-particles/types.js';

/**
 * GPU-only (4.x) lifecycle contract: the CPU advances the emission scalars;
 * `gpuDebug.lastEmitCount()` reports the per-frame dispatch count. The legacy
 * `onUpdate` / `onComplete` props are no-op sentinels in the GPU build.
 */
const lastEmit = (ps: ParticleSystem): number =>
  (
    ps as unknown as {
      gpuDebug: { lastEmitCount(): number };
    }
  ).gpuDebug.lastEmitCount();

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

// ─── Update callbacks ────────────────────────────────────────────────────────

describe('update callbacks (GPU-only sentinels)', () => {
  it('exposes the deprecated scalar sentinels', () => {
    const { ps } = createTestSystem();
    expect(typeof ps.getActiveParticleCount).toBe('function');
    expect(ps.getActiveParticleCount!()).toBe(-1);
    expect(ps.gpuDebug).toBeDefined();
    expect(ps.gpuDebug.lastEmitCount()).toBe(0);
    ps.dispose();
  });
});

// ─── Emission lifecycle ──────────────────────────────────────────────────────

describe('Emission lifecycle', () => {
  it('should emit particles over time', () => {
    const { ps, step } = createTestSystem({
      emission: { rateOverTime: 100 },
    });

    step(100);
    expect(lastEmit(ps)).toBe(10);

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

  it('should keep emitting across loops when looping', () => {
    const { ps, step } = createTestSystem({
      looping: true,
      duration: 1,
      emission: { rateOverTime: 100 },
    });

    step(1500, 1400);
    expect(lastEmit(ps)).toBeGreaterThan(0);

    ps.dispose();
  });

  it('should not emit when paused and emit again after resume', () => {
    const { ps, step } = createTestSystem({
      emission: { rateOverTime: 100 },
    });

    step(100);
    expect(lastEmit(ps)).toBeGreaterThan(0);

    ps.pauseEmitter();
    step(200, 100);
    expect(lastEmit(ps)).toBe(0);

    ps.resumeEmitter();
    step(300, 100);
    expect(lastEmit(ps)).toBeGreaterThan(0);

    ps.dispose();
  });
});

// ─── Global update ───────────────────────────────────────────────────────────

describe('updateParticleSystems', () => {
  it('should update all created particle systems', () => {
    const startTime = 1000;
    const a = createParticleSystem(
      {
        maxParticles: 10,
        duration: 5,
        looping: true,
        emission: { rateOverTime: 100 },
      },
      startTime
    );
    const b = createParticleSystem(
      {
        maxParticles: 10,
        duration: 5,
        looping: true,
        emission: { rateOverTime: 100 },
      },
      startTime
    );

    updateParticleSystems({
      now: startTime + 100,
      delta: 0.1,
      elapsed: 0.1,
    });

    expect(lastEmit(a)).toBe(10);
    expect(lastEmit(b)).toBe(10);

    a.dispose();
    b.dispose();
  });

  it('stops updating disposed systems', () => {
    const startTime = 1000;
    const a = createParticleSystem(
      {
        maxParticles: 10,
        duration: 5,
        looping: true,
        emission: { rateOverTime: 100 },
      },
      startTime
    );

    updateParticleSystems({
      now: startTime + 100,
      delta: 0.1,
      elapsed: 0.1,
    });
    expect(lastEmit(a)).toBeGreaterThan(0);

    a.dispose();

    expect(() =>
      updateParticleSystems({
        now: startTime + 200,
        delta: 0.1,
        elapsed: 0.2,
      })
    ).not.toThrow();
  });
});

// ─── Texture sheet animation ─────────────────────────────────────────────────

describe('texture sheet animation', () => {
  it('sets up the sheet fields from config', () => {
    const { ps, step } = createTestSystem({
      textureSheetAnimation: {
        tiles: { x: 4, y: 4 },
        fps: 24,
        startFrame: { min: 0, max: 15 },
        timeMode: 'FPS',
      },
    });

    step(100);
    expect(lastEmit(ps)).toBeGreaterThan(0);
    ps.dispose();
  });
});

// ─── Dispose ─────────────────────────────────────────────────────────────────

describe('dispose', () => {
  it('removes the instance from its parent', () => {
    const scene = new THREE.Group();
    const { ps } = createTestSystem();
    scene.add(ps.instance);
    expect(scene.children.length).toBe(1);
    ps.dispose();
    expect(scene.children.length).toBe(0);
  });

  it('is idempotent', () => {
    const { ps } = createTestSystem();
    ps.dispose();
    expect(() => ps.dispose()).not.toThrow();
  });
});
