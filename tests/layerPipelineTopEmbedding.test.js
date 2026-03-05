import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import {
    activateLayerNormColor,
    calculateTopEmbeddingTargets,
    findTopLayerNormInfo
} from '../src/engine/layerPipelineTopEmbedding.js';

function makeIdentityRoot() {
    return {
        worldToLocal(vec) {
            return vec;
        }
    };
}

describe('layerPipelineTopEmbedding.calculateTopEmbeddingTargets', () => {
    it('falls back to formula path when scene object is unavailable', () => {
        const lastLayer = {
            root: makeIdentityRoot(),
            mlpDown: { group: { position: { y: 200 } } }
        };
        const result = calculateTopEmbeddingTargets({
            engineScene: null,
            lastLayer,
            mlpDownHeight: 100,
            embedHeight: 300,
            embedInset: 5,
            topEmbedGap: 50,
            topEmbedAdjust: 0,
            maxRiseFraction: 0.5
        });
        expect(Number.isFinite(result.targetYLocal)).toBe(true);
        expect(Number.isFinite(result.exitYLocal)).toBe(true);
        expect(result.exitYLocal).toBeLessThanOrEqual(result.targetYLocal + 150);
    });

    it('uses scene object path when top vocabulary embedding is present', () => {
        const topEmbedObj = {
            userData: { label: 'Vocabulary Embedding (Top)' },
            getWorldPosition(out) {
                out.set(0, 1000, 0);
            }
        };
        const scene = {
            traverse(fn) {
                fn(topEmbedObj);
            }
        };
        const lastLayer = {
            root: makeIdentityRoot(),
            mlpDown: { group: { position: { y: 200 } } }
        };
        const result = calculateTopEmbeddingTargets({
            engineScene: scene,
            lastLayer,
            mlpDownHeight: 100,
            embedHeight: 300,
            embedInset: 5,
            topEmbedGap: 50,
            topEmbedAdjust: 0,
            maxRiseFraction: 1
        });
        expect(result.targetYLocal).toBe(855);
        expect(result.exitYLocal).toBe(1145);
    });
});

describe('layerPipelineTopEmbedding.findTopLayerNormInfo', () => {
    it('returns null when top LayerNorm object is missing', () => {
        const result = findTopLayerNormInfo({
            engineScene: { traverse() {} },
            lastLayer: { root: makeIdentityRoot() },
            lnHeight: 10
        });
        expect(result).toBeNull();
    });

    it('returns position info when top LayerNorm object is found', () => {
        const topLn = {
            userData: { label: 'LayerNorm (Top)' },
            getWorldPosition(out) {
                out.set(0, 300, 0);
            }
        };
        const result = findTopLayerNormInfo({
            engineScene: { traverse(fn) { fn(topLn); } },
            lastLayer: { root: makeIdentityRoot() },
            lnHeight: 40
        });
        expect(result?.lnTopGroup).toBe(topLn);
        expect(result?.lnCenterY).toBe(300);
        expect(result?.lnBottomY).toBe(280);
    });
});

describe('layerPipelineTopEmbedding.activateLayerNormColor', () => {
    it('updates mesh materials to bright active state', () => {
        const material = new THREE.MeshStandardMaterial({
            color: 0x111111,
            emissive: 0x000000,
            emissiveIntensity: 0
        });
        const mesh = { isMesh: true, material };
        const group = {
            traverse(fn) {
                fn(mesh);
            }
        };
        activateLayerNormColor(group, {
            emissiveIntensity: 0.5,
            scaleEmissiveIntensity: (value) => value * 2
        });
        expect(material.color.getHexString()).toBe('ffffff');
        expect(material.emissive.getHexString()).toBe('ffffff');
        expect(material.emissiveIntensity).toBe(1.0);
        expect(material.transparent).toBe(false);
        expect(material.opacity).toBe(1.0);
    });
});
