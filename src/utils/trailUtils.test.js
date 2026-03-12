import { describe, expect, it } from 'vitest';
import * as THREE from 'three';

import {
    StraightLineTrail,
    buildMergedLineSegmentsFromSegments,
    mergeTrailsIntoLineSegments
} from './trailUtils.js';

function createVerticalTrail(scene, {
    start = new THREE.Vector3(0, 0, 0),
    end = new THREE.Vector3(0, 10, 0)
} = {}) {
    const trail = new StraightLineTrail(scene);
    trail.start(start);
    trail.update(end);
    return trail;
}

function getMergedTrails(root) {
    const merged = [];
    root.traverse((obj) => {
        if (obj?.userData?.trailMerged) {
            merged.push(obj);
        }
    });
    return merged;
}

function getLiveTrails(root) {
    const live = [];
    root.traverse((obj) => {
        if (obj?.userData?.trailRef) {
            live.push(obj);
        }
    });
    return live;
}

describe('trailUtils merged trail dedupe', () => {
    it('skips re-freezing identical live trail segments already merged into the scene', () => {
        const scene = new THREE.Scene();

        const firstTrail = createVerticalTrail(scene);
        const firstMerged = mergeTrailsIntoLineSegments([firstTrail], scene);
        expect(firstMerged).toBeTruthy();
        expect(getMergedTrails(scene)).toHaveLength(1);
        expect(getLiveTrails(scene)).toHaveLength(0);

        const duplicateTrail = createVerticalTrail(scene);
        const duplicateMerged = mergeTrailsIntoLineSegments([duplicateTrail], scene);

        expect(duplicateMerged).toBeNull();
        expect(getMergedTrails(scene)).toHaveLength(1);
        expect(getLiveTrails(scene)).toHaveLength(0);
    });

    it('skips building duplicate static segments when identical merged geometry already exists', () => {
        const scene = new THREE.Scene();
        const segment = Float32Array.from([
            0, 0, 0,
            0, 10, 0
        ]);

        const firstMerged = buildMergedLineSegmentsFromSegments([segment], scene);
        expect(firstMerged).toBeTruthy();
        expect(getMergedTrails(scene)).toHaveLength(1);

        const duplicateMerged = buildMergedLineSegmentsFromSegments([segment], scene);

        expect(duplicateMerged).toBeNull();
        expect(getMergedTrails(scene)).toHaveLength(1);
    });
});
