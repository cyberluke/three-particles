/**
 * Trail edge-case tests for the GPU-only (4.x) engine.
 *
 * The trail history ring + ribbon are built by the compute kernels; the CPU
 * only writes scalars. These tests pin creation + per-frame dispatch for the
 * previously uncovered branches (vertical tangents, twist prevention, ribbon
 * chaining, early-return guard).
 */

import {
  createParticleSystem,
  registerTSLMaterialFactory,
} from '../js/effects/three-particles/three-particles.js';
import { enableWebGPU } from '../webgpu.js';
import type { ParticleSystem } from '../js/effects/three-particles/types.js';

// ─── Helpers ──────────────────────────────────────────────────────────────────

function createTrailSystem(
  overrides: Record<string, unknown> = {},
  startTime = 1000
): ParticleSystem {
  return createParticleSystem(
    {
      maxParticles: 5,
      duration: 10,
      looping: true,
      startLifetime: 5,
      emission: { rateOverTime: 100 },
      startSpeed: 0,
      renderer: {
        rendererType: 'TRAIL',
        trail: {
          length: 8,
          dieWithParticle: true,
          minVertexDistance: 0.01,
          twistPrevention: true,
          ...(overrides.trail as Record<string, unknown>),
        },
      },
      ...overrides,
    } as never,
    startTime
  );
}

const stepFrames = (ps: ParticleSystem, frames: number, startTime = 1000) => {
  for (let t = 1; t <= frames; t++) {
    ps.update({
      now: startTime + t * 16,
      delta: 0.016,
      elapsed: (t * 16) / 1000,
    });
  }
};

afterEach(() => {
  enableWebGPU();
});

// ─── Vertical velocity tangent alignment ────────────────────────────────────

describe('trail — vertical tangent alignment', () => {
  it('handles nearly vertical particle velocity without error', () => {
    const ps = createTrailSystem({
      startSpeed: 50,
      shape: { shape: 'CONE', cone: { angle: 0, radius: 0 } },
    });

    expect(() => stepFrames(ps, 10)).not.toThrow();
    expect(ps.instance).toBeDefined();
    ps.dispose();
  });
});

// ─── Twist prevention ────────────────────────────────────────────────────────

describe('trail — twist prevention', () => {
  it('does not throw when twist prevention flips ribbon normals', () => {
    const ps = createTrailSystem({
      startSpeed: 10,
      gravity: 20,
      trail: {
        length: 12,
        dieWithParticle: true,
        minVertexDistance: 0.005,
        twistPrevention: true,
      },
    });

    expect(() => stepFrames(ps, 30)).not.toThrow();
    ps.dispose();
  });

  it('works with twist prevention disabled', () => {
    const ps = createTrailSystem({
      startSpeed: 10,
      gravity: 20,
      trail: {
        length: 8,
        dieWithParticle: true,
        minVertexDistance: 0.01,
        twistPrevention: false,
      },
    });

    expect(() => stepFrames(ps, 20)).not.toThrow();
    ps.dispose();
  });
});

// ─── Ribbon mode ─────────────────────────────────────────────────────────────

describe('trail — ribbon mode', () => {
  it('builds the trail-history + ribbon passes for a connected ribbon', () => {
    const ps = createTrailSystem({
      startSpeed: 5,
      trail: {
        length: 10,
        dieWithParticle: true,
        minVertexDistance: 0.01,
        ribbon: true,
        twistPrevention: true,
      },
    });

    expect(() => stepFrames(ps, 40)).not.toThrow();
    const names = (
      ps as unknown as { gpuDebug: { passNames: string[] } }
    ).gpuDebug.passNames;
    expect(names).toContain('trail-history');
    ps.dispose();
  });
});

// ─── Non-trail system early return ──────────────────────────────────────────

describe('trail — early return guard', () => {
  it('non-trail system does not crash during update', () => {
    const ps = createParticleSystem(
      {
        maxParticles: 5,
        duration: 5,
        looping: true,
        emission: { rateOverTime: 10 },
        renderer: { rendererType: 'POINTS' },
      },
      1000
    );

    expect(() => stepFrames(ps, 5)).not.toThrow();
    const names = (
      ps as unknown as { gpuDebug: { passNames: string[] } }
    ).gpuDebug.passNames;
    expect(names).not.toContain('trail-history');
    ps.dispose();
  });
});
