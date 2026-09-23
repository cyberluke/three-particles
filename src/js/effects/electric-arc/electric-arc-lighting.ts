/**
 * Local scene illumination owned by the Electric Arc (§21).
 *
 * Three `THREE.PointLight`s at start / midpoint / end driven by the same
 * discharge flicker as the main filament. Bloom alone does not illuminate
 * surrounding geometry, so the arc owns these lights. Scene-wide lighting
 * is never touched.
 *
 * @module
 */
import * as THREE from 'three';
import type { NormalizedElectricArcConfig } from './electric-arc-types.js';

export type ArcLighting = {
  group: THREE.Group;
  /** start, midpoint, end */
  lights: [THREE.PointLight, THREE.PointLight, THREE.PointLight];
  update: (start: THREE.Vector3, end: THREE.Vector3, flicker: number) => void;
  dispose: () => void;
};

export function createArcLighting(
  cfg: NormalizedElectricArcConfig
): ArcLighting | null {
  if (!cfg.lighting.enabled) return null;

  const color = new THREE.Color(cfg.color);
  const makeLight = (intensity: number): THREE.PointLight => {
    const l = new THREE.PointLight(
      0xffffff,
      intensity,
      cfg.lighting.distance,
      cfg.lighting.decay
    );
    l.color.copy(color);
    return l;
  };

  const startL = makeLight(cfg.lighting.endpointIntensity);
  const midL = makeLight(cfg.lighting.midpointIntensity);
  const endL = makeLight(cfg.lighting.endpointIntensity);
  const group = new THREE.Group();
  group.add(startL, midL, endL);

  let disposed = false;
  return {
    group,
    lights: [startL, midL, endL],
    update: (start, end, flicker) => {
      if (disposed) return;
      startL.position.copy(start);
      endL.position.copy(end);
      midL.position.set(
        (start.x + end.x) * 0.5,
        (start.y + end.y) * 0.5,
        (start.z + end.z) * 0.5
      );
      const base = Math.max(cfg.intensity, 0.001);
      const micro = 0.9 + 0.1 * flicker;
      startL.intensity = cfg.lighting.endpointIntensity * micro;
      endL.intensity = cfg.lighting.endpointIntensity * micro;
      midL.intensity =
        ((cfg.lighting.midpointIntensity * micro) / (base > 1 ? 1 : 1)) * 1;
      midL.intensity = cfg.lighting.midpointIntensity * micro;
    },
    dispose: () => {
      if (disposed) return;
      disposed = true;
      startL.dispose();
      midL.dispose();
      endL.dispose();
    },
  };
}
