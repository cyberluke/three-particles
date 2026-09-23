/**
 * TSL material for the FLUID metaball particle renderer.
 *
 * Each particle is a velocity-stretched, camera-facing quad whose fragment
 * reconstructs a hemispherical (metaball) normal and applies a simple
 * water-like response: a single directional key light, Beer-Lambert
 * absorption through the sphere thickness, and a Fresnel-mixed reflection.
 *
 * Consumes the same per-instance attribute contract as the other GPU
 * instanced materials, plus a fifth attribute for velocity stretch:
 *   - `instanceOffset`        vec3  particle world position
 *   - `instanceColor`         vec4  packed per-particle RGBA
 *   - `instanceParticleState` vec4  lifetime, size, rotation, startFrame
 *   - `instanceStartValues`   vec4  startLifetime, startSize, opacity, colorR
 *   - `instanceVelocity`      vec4  xyz velocity (with w padding)
 *
 * Reuses shared helpers from `./tsl-shared.js` (soft-particle fade, sprite-
 * sheet UVs, background discard).
 */
import {
  Fn,
  attribute,
  vec2,
  vec3,
  vec4,
  float,
  uniform,
  modelViewMatrix,
  positionLocal,
  cameraProjectionMatrix,
  texture,
  Discard,
  If,
  dot,
  max,
  min,
  pow,
  sqrt,
  exp2,
  varyingProperty,
  type ShaderNodeObject,
  type Node,
} from 'three/tsl';
import { MeshBasicNodeMaterial } from 'three/webgpu';

import {
  POINT_SIZE_SCALE,
  ALPHA_DISCARD_THRESHOLD,
} from '../three-particles-constants.js';
import {
  type SharedUniforms,
  createParticleUniforms,
  computeFrameIndex,
  computeSpriteSheetUV,
  computeSoftParticleFade,
  applyBackgroundDiscard,
} from './tsl-shared.js';

import type * as THREE from 'three';

/**
 * Builds a TSL {@link MeshBasicNodeMaterial} for the FLUID renderer class.
 *
 * @param sharedUniforms - Live uniform values shared with the particle system.
 * @param rendererConfig - Blending / depth / transparency settings.
 * @param gpuCompute - `true` in the WebGPU compute path (packed storage
 *   buffers); `false` selects named per-attribute lookups.
 * @param stretchFactor - Longitudinal velocity-stretch multiplier. `1` is
 *   the physical value; higher values smear fast particles further.
 * @param absorption - Beer-Lambert absorption coefficient (t = 2·nz).
 * @param ior - Index of refraction for the Schlick Fresnel term.
 */
export function createFluidTSLMaterial(
  sharedUniforms: SharedUniforms,
  rendererConfig: {
    transparent: boolean;
    blending: THREE.Blending;
    depthTest: boolean;
    depthWrite: boolean;
  },
  gpuCompute = false,
  stretchFactor = 1,
  absorption = 1.44,
  ior = 1.33
): MeshBasicNodeMaterial {
  const u = createParticleUniforms(sharedUniforms);

  // Back-patch `viewportHeight` so external writes propagate into TSL.
  const uViewportHeight = uniform(
    typeof (sharedUniforms.viewportHeight as { value?: number })?.value ===
      'number'
      ? (sharedUniforms.viewportHeight as { value: number }).value
      : 1.0
  );
  sharedUniforms.viewportHeight = uViewportHeight as unknown as {
    value: unknown;
  };

  // Longitudinal stretch multiplier on the quad's tangent axis.
  const uStretch = uniform(float(stretchFactor));
  // Beer-Lambert absorption coefficient (via exp2, so multiply by log2e).
  const uAbsorbK = uniform(float(absorption * 1.442695));
  // Fresnel reflectance at normal incidence (Schlick, from IOR).
  const f0 = Math.pow(ior - 1, 2) / Math.pow(ior + 1, 2);
  const uF0 = uniform(float(f0));

  // ── Per-instance attributes ────────────────────────────────────────────────
  const aInstanceOffset = attribute('instanceOffset');
  const aColor = attribute('instanceColor');
  const aVelocity = attribute('instanceVelocity');
  const aParticleState = gpuCompute ? attribute('instanceParticleState') : null;
  const aStartValues = gpuCompute ? attribute('instanceStartValues') : null;
  const aSize = gpuCompute ? null : attribute('instanceSize');
  const aLifetime = gpuCompute ? null : attribute('instanceLifetime');
  const aStartLifetime = gpuCompute ? null : attribute('instanceStartLifetime');
  const aStartFrame = gpuCompute ? null : attribute('instanceStartFrame');

  // ── Varyings ───────────────────────────────────────────────────────────────
  const vColor = varyingProperty('vec4', 'vColor');
  const vLifetime = varyingProperty('float', 'vLifetime');
  const vStartLifetime = varyingProperty('float', 'vStartLifetime');
  const vStartFrame = varyingProperty('float', 'vStartFrame');
  const vUv = varyingProperty('vec2', 'vUv');
  const vVelXY = varyingProperty('vec2', 'vVelXY');
  const vVelZ = varyingProperty('float', 'vVelZ');
  const vViewZ = varyingProperty('float', 'vViewZ');

  // ── Vertex stage ───────────────────────────────────────────────────────────
  const vertexNode = Fn((): ShaderNodeObject<Node> => {
    // Dead-particle early-out (see the POINTS / INSTANCED material for why
    // the w-component is `-1` rather than `0`).
    const clipPos = vec4(0.0, 0.0, 0.0, -1.0).toVar();

    If(aColor.w.greaterThan(0.0), () => {
      vColor.assign(aColor.toVar());
      if (gpuCompute) {
        vLifetime.assign(aParticleState!.x);
        vStartLifetime.assign(aStartValues!.x);
        vStartFrame.assign(aParticleState!.w);
      } else {
        vLifetime.assign(aLifetime!);
        vStartLifetime.assign(aStartLifetime!);
        vStartFrame.assign(aStartFrame!);
      }

      // Quad UV: the base quad spans ±0.5; remap to 0..1 and flip Y so the
      // top-left of the sprite-sheet lands on the top-left of the billboard.
      vUv.assign(
        vec2(positionLocal.x.add(0.5), float(0.5).sub(positionLocal.y))
      );

      // Bring world position + velocity into view space.
      const mvPos = modelViewMatrix.mul(vec4(aInstanceOffset.xyz, 1.0)).toVar();
      const mvVel = modelViewMatrix.mul(vec4(aVelocity.xyz, 0.0)).xyz;
      vVelXY.assign(vec2(mvVel.x, mvVel.y));
      vVelZ.assign(mvVel.z);

      // Pixel-accurate billboard size identical to the POINTS / INSTANCED path.
      const dist = sqrt(
        mvPos.x.mul(mvPos.x).add(mvPos.y.mul(mvPos.y)).add(mvPos.z.mul(mvPos.z))
      );
      const sizeVal = gpuCompute ? aParticleState!.y : aSize!;
      const pointSizePx = sizeVal.mul(POINT_SIZE_SCALE).div(dist);
      const projY = cameraProjectionMatrix.element(1).element(1);
      const halfExtent = pointSizePx
        .mul(mvPos.z.negate())
        .div(projY.mul(uViewportHeight).mul(0.5));

      // Orthonormal view-space tangent basis from the projected velocity.
      // Falls back to (X, Y) when the 2-D velocity is (near-)zero.
      const vlen = sqrt(mvVel.x.mul(mvVel.x).add(mvVel.y.mul(mvVel.y)));
      const hasVel = vlen.greaterThan(0.0001);
      const invVlen = float(1.0).div(hasVel.select(vlen, float(1.0)));
      const tx = hasVel.select(mvVel.x.mul(invVlen), float(1.0));
      const ty = hasVel.select(mvVel.y.mul(invVlen), float(0.0));

      // Longitudinal stretch = 1 + min( |v| × stretchFactor, 3 ).
      const stretch = float(1.0).add(min(vlen.mul(uStretch), 3.0));

      // Rotate-and-scale the quad in view space so its long axis follows the
      // projected velocity direction. Major = x-axis × stretch, minor = y × 1.
      const ox = positionLocal.x.mul(halfExtent).mul(stretch);
      const oy = positionLocal.y.mul(halfExtent);
      mvPos.x.addAssign(tx.mul(ox).add(ty.mul(oy).negate()));
      mvPos.y.addAssign(tx.mul(oy).add(ty.mul(ox)));

      vViewZ.assign(mvPos.z.negate());

      clipPos.assign(cameraProjectionMatrix.mul(mvPos));
    });

    return clipPos;
  })();

  // ── Fragment stage ─────────────────────────────────────────────────────────
  const fragmentColor = Fn((): ShaderNodeObject<Node> => {
    // Metaball disc: discard outside the unit circle and lift the
    // hemispherical normal `nz = sqrt(1 - r²)`.
    const p = vUv.mul(2.0).sub(vec2(1.0, 1.0));
    const r2 = p.x.mul(p.x).add(p.y.mul(p.y));
    If(r2.greaterThan(1.0), () => {
      Discard();
    });

    const nz = sqrt(float(1.0).sub(r2));
    const N = vec3(p.x, p.y, nz);

    // Speed from the view-space velocity captured at the vertex stage.
    const speed = sqrt(
      vVelXY.x.mul(vVelXY.x).add(vVelXY.y.mul(vVelXY.y)).add(vVelZ.mul(vVelZ))
    );
    const speedBoost = float(1.0).add(min(speed.mul(0.15), 0.5));

    // Sprite-sheet texture sample (may resolve to a 1×1 dummy in the no-map
    // case, matching the shared uniform default).
    const frameIndex = computeFrameIndex({
      vLifetime,
      vStartLifetime,
      vStartFrame,
      uFps: u.uFps,
      uUseFPSForFrameIndex: u.uUseFPSForFrameIndex,
      uTiles: u.uTiles,
    });
    const uvPoint = computeSpriteSheetUV({
      baseUV: vUv,
      frameIndex,
      uTiles: u.uTiles,
    });
    const texColor = texture(u.uMap, uvPoint);

    // Base color = per-particle color × texture sample.
    const base = vColor.mul(texColor);

    // Directional key light (from camera direction, +Z in view space).
    const NdotL = max(dot(N, vec3(0.0, 0.0, 1.0)), float(0.0));
    const diffuse = float(0.5).add(float(0.5).mul(NdotL));

    // Beer-Lambert absorption through the sphere thickness (t = 2·nz).
    const thickness = nz.mul(2.0);
    // Pre-scaled by log2e so exp2 alone reproduces `e^(-absorption * t)`.
    const absorb = exp2(thickness.mul(uAbsorbK).negate());

    // Fresnel mix (Schlick). uF0 is provided by the JS host at build time.
    const oneMinusNz = float(1.0).sub(nz);
    const fresnel = uF0.add(float(1.0).sub(uF0).mul(pow(oneMinusNz, 5.0)));

    // Two-term water-like shade: refracted/diffused body + reflected rim.
    const refr = base.rgb.mul(absorb).mul(diffuse).mul(speedBoost);
    const reflColor = base.rgb.add(vec3(0.08, 0.08, 0.1));
    const mixedColor = refr
      .mul(float(1.0).sub(fresnel))
      .add(reflColor.mul(fresnel));

    // Alpha uses the same Beer-Lambert thickness term so thick spheres
    // appear denser than thin rims.
    const outColor = vec4(
      mixedColor,
      vColor.w.mul(float(1.0).sub(exp2(thickness.mul(uAbsorbK).negate())))
    );

    // Soft-particle depth fade (shared helper).
    const softFade = computeSoftParticleFade({
      viewZ: vViewZ,
      uSoftEnabled: u.uSoftEnabled,
      uSoftIntensity: u.uSoftIntensity,
      uSceneDepthTex: u.uSceneDepthTex,
      uCameraNearFar: u.uCameraNearFar,
    });
    outColor.assign(vec4(outColor.xyz, outColor.w.mul(softFade)));

    // Background color discard + final alpha threshold.
    applyBackgroundDiscard({
      texColor: outColor,
      uDiscardBg: u.uDiscardBg,
      uBgColor: u.uBgColor,
      uBgTolerance: u.uBgTolerance,
    });
    Discard(outColor.w.lessThan(ALPHA_DISCARD_THRESHOLD));

    return outColor;
  })();

  // ── Material assembly ──────────────────────────────────────────────────────
  const material = new MeshBasicNodeMaterial();
  material.transparent = rendererConfig.transparent;
  material.blending = rendererConfig.blending;
  material.depthTest = rendererConfig.depthTest;
  material.depthWrite = rendererConfig.depthWrite;
  material.toneMapped = false;
  material.fog = false;
  material.vertexNode = vertexNode;
  material.colorNode = fragmentColor;
  return material;
}
