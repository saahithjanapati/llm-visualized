import * as THREE from 'three';
import { RotatingStarField } from '../components/RotatingStarField.js';
import { appState } from '../state/appState.js';

/**
 * Initialise the rotating sky stars around the GPT tower. Calculates approximate
 * scene bounds to distribute stars above the tower and registers an update
 * callback so they slowly orbit around the scene.
 *
 * @param {import('../engine/LayerPipeline.js').LayerPipeline} pipeline
 * @returns {RotatingStarField|null}
 */
export function initRotatingStars(pipeline) {
    const scene = pipeline?.engine?.scene;
    if (!scene) return null;

    // Remove any existing instance before recalculating bounds so we don't
    // include the previous star distribution in the new measurements.
    appState.setRotatingStarField(null);

    const bounds = new THREE.Box3().setFromObject(scene);
    const size = bounds.getSize(new THREE.Vector3());
    const centre = bounds.getCenter(new THREE.Vector3());

    const baseY = Number.isFinite(bounds.min.y) ? bounds.min.y : centre.y - size.y / 2;
    const horizontalExtent = Math.max(size.x, size.z) * 0.5;
    const safeInnerRadius = Math.max(1200, horizontalExtent + 600);
    const safeOuterRadius = safeInnerRadius + Math.max(horizontalExtent, 4500);

    const verticalStart = Math.max(900, size.y * 0.35);
    const verticalSpan = Math.max(size.y * 0.9, 7000);

    const starField = new RotatingStarField(scene, {
        center: centre,
        baseY,
        minRadius: safeInnerRadius,
        maxRadius: safeOuterRadius,
        minY: baseY + verticalStart,
        maxY: baseY + verticalStart + verticalSpan,
        starCount: 1400,
        rotationSpeed: 0.09,
        starSize: 200
    });

    let removeUpdater = null;
    if (typeof pipeline?.engine?.addGlobalUpdater === 'function') {
        const updateFn = (dt) => starField.update(dt);
        removeUpdater = pipeline.engine.addGlobalUpdater(updateFn);
    }

    appState.setRotatingStarField(starField, removeUpdater);
    return starField;
}

export default initRotatingStars;
