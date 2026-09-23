/**
 * TSL material for the electric arc ribbon (one draw call, §13–§16).
 *
 * Vertex stage: reads previous/current/next `vec4` centerline samples from
 * the compute storage buffer (static packed attribute `position =
 * (arcIndex, side, prevIndex, nextIndex)`) and expands the vertex along the
 * camera-facing perpendicular produced by the shared `billboardPerp` helper
 * (identical frame math to the particle trail ribbon, §14).
 *
 * Fragment stage: one analytic HDR profile per ribbon
 * (`white core + colored sheath + wide halo`), multiplied by the per-sample
 * storage brightness which already carries the global epoch flicker (§15).
 *
 * @module
 */
import { AdditiveBlending, DoubleSide, Vector3 } from 'three';
import {
  Fn,
  attribute,
  cameraPosition,
  vec3,
  vec4,
  float,
  normalize,
  length,
  exp,
  mix,
  storage,
  varyingProperty,
  uniform,
  Discard,
  type ShaderNodeObject,
  type Node,
} from 'three/tsl';
import { MeshBasicNodeMaterial, StorageBufferAttribute } from 'three/webgpu';
import { billboardPerp } from '../../three-particles/webgpu/tsl-shared.js';

export type ElectricArcRibbonParams = {
  /** linear rgb of the plasma (shared Vector3, mutated on live updates). */
  coreColor: Vector3;
  arcColor: Vector3;
  /** full ribbon half-span in world units (§16, scaled by per-sample width). */
  halfWidth: number;
  /** HDR core scale. */
  intensity: number;
  /** sheath scale. */
  glowIntensity: number;
  /** halo scale. */
  haloIntensity: number;
  /**
   * Which analytic layers this ribbon draws (rebuilt glow, §15/§18):
   * `core` = hot white filament only; `sheath` = colored mid skirt;
   * `halo` = wide soft bloom outer layer.
   */
  layers: 'core' | 'sheath' | 'halo';
  /** Cross-section shape: 0 = gaussian exp, 1 = sharp triangle tent. */
  profileMode: 0 | 1;
};

export type ElectricArcRibbonMaterialHandles = {
  material: MeshBasicNodeMaterial;
  halfWidth: ShaderNodeObject<Node>;
  intensity: ShaderNodeObject<Node>;
  glowIntensity: ShaderNodeObject<Node>;
  haloIntensity: ShaderNodeObject<Node>;
  profileMode: ShaderNodeObject<Node>;
};

/** analytic widths: gaussian exp(-k·d²) ↔ triangle 1-|d|·√(0.55k). */
const profileAt = (
  d2: ShaderNodeObject<Node>,
  dAbs: ShaderNodeObject<Node>,
  uProfile: ShaderNodeObject<Node>,
  k: number
): ShaderNodeObject<Node> => {
  const g = exp(d2.mul(float(-k)));
  const t = float(1)
    .sub(dAbs.mul(Math.sqrt(0.55 * k)))
    .max(float(0));
  return mix(g, t, uProfile);
};

export function createElectricArcRibbonMaterial(
  arcBuffer: StorageBufferAttribute,
  widthBuffer: StorageBufferAttribute,
  totalSamples: number,
  params: ElectricArcRibbonParams
): ElectricArcRibbonMaterialHandles {
  const uCore = uniform(params.coreColor);
  const uArc = uniform(params.arcColor);
  const uHalf = uniform(float(params.halfWidth));
  const uIntensity = uniform(float(params.intensity));
  const uGlow = uniform(float(params.glowIntensity));
  const uHalo = uniform(float(params.haloIntensity));
  const uProfile = uniform(float(params.profileMode));

  const aPacked = attribute('position', 'vec4');
  const sArc = storage(arcBuffer, 'vec4', totalSamples);
  const sW = storage(widthBuffer, 'float', totalSamples);

  const vAcross = varyingProperty('float', 'vAcross');
  const vBright = varyingProperty('float', 'vBright');

  // ── Vertex: camera-facing expansion through the shared billboard frame ───
  const positionNode = Fn((): ShaderNodeObject<Node> => {
    const iF = aPacked.x;
    const side = aPacked.y;
    const pI = aPacked.z;
    const nI = aPacked.w;

    const cur = sArc.element(iF.toUint());
    const prev = sArc.element(pI.toUint());
    const next = sArc.element(nI.toUint());

    vAcross.assign(side);
    vBright.assign(cur.w);

    const rawTan = next.sub(prev);
    const tanLen = length(rawTan);
    const tangent = normalize(
      tanLen
        .lessThan(float(0.0001))
        .select(vec3(float(0), float(1), float(0)), rawTan)
    );

    const curPos = vec3(cur.x, cur.y, cur.z);
    const viewDir = normalize(cameraPosition.sub(curPos));
    const perp = billboardPerp({ tangent, viewDir });

    // variable width: per-sample factor from the compute pass
    const wid = sW.element(iF.toUint());

    return curPos.add(perp.mul(side).mul(uHalf.mul(wid)));
  })();

  // ── Fragment: analytic HDR filament layers (§15) ─────────────────────────
  const colorNode = Fn((): ShaderNodeObject<Node> => {
    const dAbs = vAcross.abs(); // 0 at filament center, 1 at ribbon edges
    const d2 = dAbs.mul(dAbs);

    // gaussian ↔ triangle tent blend selected by the uniform profile mode
    const c = profileAt(d2, dAbs, uProfile, 70);
    const i = profileAt(d2, dAbs, uProfile, 8);
    const h = profileAt(d2, dAbs, uProfile, 2);

    const isCore = params.layers === 'core';
    const isSheath = params.layers === 'sheath';

    // single analytic layer per ribbon draw
    const k = isCore
      ? c.mul(uIntensity)
      : isSheath
        ? i.mul(uGlow)
        : h.mul(uHalo);
    const kk = k.mul(vBright);
    const rgb0 = isCore || isSheath ? uCore : uArc;

    const r = rgb0.x.mul(kk);
    const g = rgb0.y.mul(kk);
    const b = rgb0.z.mul(kk);

    const alpha = kk.mul(float(0.5)).min(float(1));
    Discard(alpha.lessThan(float(0.001)));

    return vec4(r, g, b, alpha);
  })();

  const material = new MeshBasicNodeMaterial();
  material.transparent = true;
  material.blending = AdditiveBlending;
  material.depthTest = true;
  material.depthWrite = false;
  material.toneMapped = false;
  material.fog = false;
  material.side = DoubleSide;
  material.positionNode = positionNode;
  material.colorNode = colorNode;

  return {
    material,
    halfWidth: uHalf,
    intensity: uIntensity,
    glowIntensity: uGlow,
    haloIntensity: uHalo,
    profileMode: uProfile,
  };
}
