/**
 * Endpoint contact plasma helpers (shared by both backends).
 *
 * The CPU fallback renders each endpoint as a `THREE.Sprite` with an
 * internally generated radial `THREE.DataTexture` (no external PNG asset).
 * The GPU path uses a procedural TSL billboard
 * (`webgpu/tsl-electric-contact-material.ts`) and only needs the numeric
 * profile helpers.
 *
 * @module
 */
import * as THREE from 'three';
import type { ArcProfileKind } from './electric-arc-types.js';

/** 1D gaussian-ish profile: exp(-d*d*k) at d in [0,1]. */
export const radialProfile = (d: number, k: number): number =>
  Math.exp(-d * d * k);

/**
 * Linear tent profile (sharp, no soft skirt): `max(0, 1 - |d|·√(0.55k))`.
 * The slope is chosen so the 1/e half-width matches the gaussian twin.
 */
export const triangleProfile = (d: number, k: number): number => {
  const slope = Math.sqrt(0.55 * k);
  const v = 1 - Math.abs(d) * slope;
  return v > 0 ? v : 0;
};

/** Profile value for the active cross-section shape. */
export const profileValue = (
  d: number,
  k: number,
  kind: ArcProfileKind = 'gaussian'
): number =>
  kind === 'triangle' ? triangleProfile(d, k) : radialProfile(d, k);

/** Fill an RGBA8 1D texture buffer with the layered 1D profile (§15). */
const fillProfileTexture = (
  data: Uint8Array,
  size: number,
  k: number,
  kind: ArcProfileKind
): void => {
  const innerK = k * 0.16;
  for (let i = 0; i < size; i++) {
    const d = (i / (size - 1)) * 2 - 1; // -1..1 across
    const core = profileValue(d, k, kind);
    const inner = profileValue(d, innerK, kind);
    // baked additive-sum profile: bright hot center, softer outer skirt
    const a = Math.min(1, core + inner * 0.35);
    const o = i * 4;
    data[o] = 255;
    data[o + 1] = 255;
    data[o + 2] = 255;
    data[o + 3] = Math.max(0, Math.min(255, Math.round(a * 255)));
  }
};

/** RGBA8 N×1 gradient texture across the ribbon width (§15 profile). */
export function createProfileTexture(
  size: number,
  k: number,
  kind: ArcProfileKind = 'gaussian'
): THREE.DataTexture {
  const data = new Uint8Array(size * 4);
  fillProfileTexture(data, size, k, kind);
  const tex = new THREE.DataTexture(data, size, 1, THREE.RGBAFormat);
  tex.needsUpdate = true;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  return tex;
}

/** Re-encode an existing profile texture in place (live `profile` switch). */
export function writeProfileTexture(
  tex: THREE.DataTexture,
  size: number,
  k: number,
  kind: ArcProfileKind
): void {
  const img = tex.image as { data: Uint8Array };
  if (!img?.data) return;
  fillProfileTexture(img.data, size, k, kind);
  tex.needsUpdate = true;
}

/** RGBA8 radial sprite texture for endpoint contact glow. */
export function createContactTexture(size = 64): THREE.DataTexture {
  const data = new Uint8Array(size * size * 4);
  const half = (size - 1) / 2;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = (x - half) / half;
      const dy = (y - half) / half;
      const r2 = dx * dx + dy * dy;
      const a =
        r2 > 1 ? 0 : Math.min(1, Math.exp(-r2 * 6) + Math.exp(-r2 * 1.7) * 0.4);
      const o = (y * size + x) * 4;
      data[o] = 255;
      data[o + 1] = 255;
      data[o + 2] = 255;
      data[o + 3] = Math.round(a * 255);
    }
  }
  const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  tex.needsUpdate = true;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  return tex;
}

/** Two contact sprites (at `start` / `end`). Additive, HDR-scaled colors. */
export function createContactSprites(
  radius: number,
  intensity: number,
  color: number,
  texture: THREE.Texture
): {
  group: THREE.Group;
  start: THREE.Sprite;
  end: THREE.Sprite;
  materials: THREE.SpriteMaterial[];
  baseScale: number;
} {
  const baseColor = new THREE.Color(color);
  const scale = Math.max(radius * 2, 0.01);
  const mk = (): THREE.Sprite => {
    const mat = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      toneMapped: false,
    });
    mat.color.setRGB(
      baseColor.r * intensity,
      baseColor.g * intensity,
      baseColor.b * intensity
    );
    const s = new THREE.Sprite(mat);
    s.scale.set(scale, scale, 1);
    return s;
  };
  const start = mk();
  const end = mk();
  const group = new THREE.Group();
  group.add(start);
  group.add(end);
  return {
    group,
    start,
    end,
    materials: [
      start.material as THREE.SpriteMaterial,
      end.material as THREE.SpriteMaterial,
    ],
    baseScale: scale,
  };
}

/** Apply a global flicker factor to contact sprites (§20). */
export function updateContactSprites(
  start: THREE.Sprite,
  end: THREE.Sprite,
  sx: number,
  sy: number,
  sz: number,
  ex: number,
  ey: number,
  ez: number,
  flicker: number,
  baseScale: number
): void {
  const s = baseScale * (0.96 + 0.04 * flicker);
  start.scale.set(s, s, 1);
  end.scale.set(s, s, 1);
  start.position.set(sx, sy, sz);
  end.position.set(ex, ey, ez);
}
