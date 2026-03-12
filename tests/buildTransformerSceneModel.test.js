import { describe, expect, it } from 'vitest';
import { buildSceneLayout } from '../src/view2d/layout/buildSceneLayout.js';
import {
    resolveSemanticTargetBounds,
    resolveSemanticTargetFocusPath
} from '../src/view2d/layout/resolveSemanticTargetBounds.js';
import { buildTransformerSceneModel } from '../src/view2d/model/buildTransformerSceneModel.js';
import { flattenSceneNodes } from '../src/view2d/schema/sceneTypes.js';
import { D_HEAD, D_MODEL } from '../src/ui/selectionPanelConstants.js';

function createActivationSource(tokenCount = 4) {
    const promptTokens = Array.from({ length: tokenCount }, (_, index) => index);
    const tokenDisplayStrings = promptTokens.map((index) => `tok_${index}`);

    const buildVector = (length, seed = 1, scale = 0.1) => (
        Array.from({ length }, (_, index) => (((index % 17) - 8) * scale * seed))
    );

    return {
        meta: {
            prompt_tokens: promptTokens,
            token_display_strings: tokenDisplayStrings
        },
        getTokenCount() {
            return tokenCount;
        },
        getTokenString(tokenIndex) {
            return tokenDisplayStrings[tokenIndex] || `tok_${tokenIndex}`;
        },
        getEmbedding(kind, tokenIndex, targetLength = D_MODEL) {
            const scale = kind === 'position' ? 0.08 : (kind === 'sum' ? 0.12 : 0.1);
            return buildVector(targetLength, tokenIndex + 1, scale);
        },
        getLayerIncoming(layerIndex, tokenIndex, targetLength = D_MODEL) {
            return buildVector(targetLength, (layerIndex + 1) * (tokenIndex + 1), 0.015);
        },
        getLayerLn1(layerIndex, stage, tokenIndex, targetLength = D_MODEL) {
            return buildVector(targetLength, (layerIndex + 1) + tokenIndex + 1, 0.018);
        },
        getLayerLn2(layerIndex, stage, tokenIndex, targetLength = D_MODEL) {
            return buildVector(targetLength, (layerIndex + 2) + tokenIndex + 1, 0.02);
        },
        getLayerQKVVector(layerIndex, kind, headIndex, tokenIndex, targetLength = D_HEAD) {
            const scale = kind === 'q' ? 0.08 : 0.1;
            return buildVector(targetLength, (layerIndex + 1) * (headIndex + 1) * (tokenIndex + 1), scale);
        },
        getAttentionWeightedSum(layerIndex, headIndex, tokenIndex, targetLength = D_HEAD) {
            return buildVector(targetLength, (layerIndex + 1) * (headIndex + 1) * (tokenIndex + 1), 0.09);
        },
        getAttentionOutputProjection(layerIndex, tokenIndex, targetLength = D_MODEL) {
            return buildVector(targetLength, (layerIndex + 1) * (tokenIndex + 2), 0.022);
        },
        getPostAttentionResidual(layerIndex, tokenIndex, targetLength = D_MODEL) {
            return buildVector(targetLength, (layerIndex + 2) * (tokenIndex + 1), 0.024);
        },
        getMlpUp(layerIndex, tokenIndex, targetLength = D_MODEL * 4) {
            return buildVector(targetLength, (layerIndex + 3) * (tokenIndex + 1), 0.014);
        },
        getMlpActivation(layerIndex, tokenIndex, targetLength = D_MODEL * 4) {
            return buildVector(targetLength, (layerIndex + 4) * (tokenIndex + 1), 0.016);
        },
        getMlpDown(layerIndex, tokenIndex, targetLength = D_MODEL) {
            return buildVector(targetLength, (layerIndex + 5) * (tokenIndex + 1), 0.021);
        },
        getPostMlpResidual(layerIndex, tokenIndex, targetLength = D_MODEL) {
            return buildVector(targetLength, (layerIndex + 6) * (tokenIndex + 1), 0.026);
        },
        getFinalLayerNorm(stage, tokenIndex, targetLength = D_MODEL) {
            return buildVector(targetLength, tokenIndex + 1, 0.03);
        },
        getLogitsForToken(tokenIndex, limit = 8) {
            return Array.from({ length: limit }, (_, index) => ({
                token: `cand_${index}`,
                prob: Math.max(0, 0.9 - (index * 0.08))
            }));
        }
    };
}

describe('buildTransformerSceneModel', () => {
    it('builds a full transformer overview scene with per-layer MHSA head targets', () => {
        const scene = buildTransformerSceneModel({
            activationSource: createActivationSource(4)
        });

        expect(scene).not.toBeNull();
        expect(scene.semantic.componentKind).toBe('transformer');
        expect(scene.metadata.layerCount).toBe(12);
        expect(scene.metadata.tokenCount).toBe(4);

        const nodes = flattenSceneNodes(scene);
        expect(nodes.filter((node) => node.role === 'layer')).toHaveLength(12);
        expect(nodes.filter((node) => node.role === 'head')).toHaveLength(12 * 12);
        expect(nodes.some((node) => (
            node.role === 'module'
            && node.semantic?.componentKind === 'mhsa'
            && node.semantic?.layerIndex === 6
        ))).toBe(true);
        expect(nodes.some((node) => (
            node.role === 'module'
            && node.semantic?.componentKind === 'logits'
        ))).toBe(true);
    });

    it('resolves semantic bounds and focus paths for layer and head targets after layout', () => {
        const scene = buildTransformerSceneModel({
            activationSource: createActivationSource(5)
        });
        const layout = buildSceneLayout(scene);

        const mhsaModuleBounds = resolveSemanticTargetBounds(layout.registry, {
            componentKind: 'mhsa',
            layerIndex: 6,
            role: 'module'
        });
        const headBounds = resolveSemanticTargetBounds(layout.registry, {
            componentKind: 'mhsa',
            layerIndex: 6,
            headIndex: 6,
            role: 'head'
        });
        const focusPath = resolveSemanticTargetFocusPath(layout.registry, {
            componentKind: 'mhsa',
            layerIndex: 6,
            headIndex: 6,
            role: 'head'
        });

        expect(mhsaModuleBounds?.width).toBeGreaterThan(0);
        expect(mhsaModuleBounds?.height).toBeGreaterThan(0);
        expect(headBounds?.width).toBeGreaterThan(0);
        expect(headBounds?.height).toBeGreaterThan(0);
        expect(mhsaModuleBounds.width).toBeGreaterThan(headBounds.width);
        expect(focusPath.map((entry) => entry.role)).toEqual(expect.arrayContaining([
            'layer',
            'module',
            'head'
        ]));
    });
});
