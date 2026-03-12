import { describe, expect, it } from 'vitest';
import { buildSceneLayout } from '../src/view2d/layout/buildSceneLayout.js';
import {
    resolveSemanticTargetBounds,
    resolveSemanticTargetFocusPath
} from '../src/view2d/layout/resolveSemanticTargetBounds.js';
import { buildTransformerSceneModel } from '../src/view2d/model/buildTransformerSceneModel.js';
import {
    flattenSceneNodes,
    VIEW2D_MATRIX_PRESENTATIONS
} from '../src/view2d/schema/sceneTypes.js';
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
        expect(nodes.some((node) => node.role === 'head-card' && node.visual?.styleKey === 'mhsa.head')).toBe(true);
        expect(nodes.some((node) => node.role === 'add-circle' && node.visual?.styleKey === 'residual.add')).toBe(true);
        expect(nodes.some((node) => (
            node.role === 'residual-add-operator'
            && node.visual?.styleKey === 'residual.add-symbol'
        ))).toBe(true);
    });

    it('renders incoming and post-attention residual states as compact value summaries', () => {
        const tokenCount = 4;
        const scene = buildTransformerSceneModel({
            activationSource: createActivationSource(tokenCount)
        });
        const nodes = flattenSceneNodes(scene);

        const incomingResidual = nodes.find((node) => (
            node.semantic?.componentKind === 'residual'
            && node.semantic?.layerIndex === 0
            && node.semantic?.stage === 'incoming'
            && node.role === 'module-card'
        ));
        const postAttentionResidual = nodes.find((node) => (
            node.semantic?.componentKind === 'residual'
            && node.semantic?.layerIndex === 0
            && node.semantic?.stage === 'post-attn-residual'
            && node.role === 'module-card'
        ));

        expect(incomingResidual?.presentation).toBe(VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS);
        expect(postAttentionResidual?.presentation).toBe(VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS);
        expect(incomingResidual?.rowItems).toHaveLength(tokenCount);
        expect(postAttentionResidual?.rowItems).toHaveLength(tokenCount);
        expect(incomingResidual?.rowItems?.[0]?.gradientCss).toContain('linear-gradient(');
        expect(postAttentionResidual?.rowItems?.[0]?.gradientCss).toContain('linear-gradient(');
        expect(postAttentionResidual?.rowItems?.[0]?.gradientCss).not.toBe(incomingResidual?.rowItems?.[0]?.gradientCss);

        const layout = buildSceneLayout(scene);
        const incomingEntry = layout.registry.getNodeEntries().find((entry) => (
            entry.semantic?.componentKind === 'residual'
            && entry.semantic?.layerIndex === 0
            && entry.semantic?.stage === 'incoming'
            && entry.role === 'module-card'
        ));

        expect(incomingEntry?.contentBounds?.height).toBeGreaterThan(36);
        expect(incomingEntry?.contentBounds?.height).toBeLessThan(60);
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
        expect(mhsaModuleBounds.height).toBeGreaterThan(headBounds.height);
        expect(focusPath.map((entry) => entry.role)).toEqual(expect.arrayContaining([
            'layer',
            'module',
            'head'
        ]));
    });

    it('routes the post-attention residual state directly into LN2', () => {
        const scene = buildTransformerSceneModel({
            activationSource: createActivationSource(4)
        });
        const layout = buildSceneLayout(scene);

        const residualStateEntry = layout.registry.getNodeEntries().find((entry) => (
            entry.semantic?.componentKind === 'residual'
            && entry.semantic?.layerIndex === 0
            && entry.semantic?.stage === 'post-attn-residual'
            && entry.role === 'module-card'
        ));
        const connectorEntry = layout.registry.getConnectorEntries().find((entry) => entry.role === 'connector-layer-0-add1-ln2');

        expect(residualStateEntry).toBeTruthy();
        expect(connectorEntry).toBeTruthy();
        expect(connectorEntry.pathPoints).toHaveLength(2);
        expect(connectorEntry.pathPoints[0].x).toBeCloseTo(connectorEntry.pathPoints[1].x, 5);
    });

    it('returns output projection directly under the first residual add node', () => {
        const scene = buildTransformerSceneModel({
            activationSource: createActivationSource(4)
        });
        const layout = buildSceneLayout(scene);

        const connectorEntry = layout.registry.getConnectorEntries().find((entry) => entry.role === 'connector-layer-0-outproj-add1');

        expect(connectorEntry).toBeTruthy();
        expect(connectorEntry.pathPoints).toHaveLength(3);
        expect(connectorEntry.pathPoints[0].y).toBeCloseTo(connectorEntry.pathPoints[1].y, 5);
        expect(connectorEntry.pathPoints[1].x).toBeCloseTo(connectorEntry.pathPoints[2].x, 5);
    });

    it('returns MLP to the second residual add node with a right-then-up elbow', () => {
        const scene = buildTransformerSceneModel({
            activationSource: createActivationSource(4)
        });
        const layout = buildSceneLayout(scene);

        const connectorEntry = layout.registry.getConnectorEntries().find((entry) => entry.role === 'connector-layer-0-mlp-add2');

        expect(connectorEntry).toBeTruthy();
        expect(connectorEntry.pathPoints).toHaveLength(3);
        expect(connectorEntry.pathPoints[0].y).toBeCloseTo(connectorEntry.pathPoints[1].y, 5);
        expect(connectorEntry.pathPoints[1].x).toBeCloseTo(connectorEntry.pathPoints[2].x, 5);
    });

    it('keeps LN1, Head 1, output projection, LN2, and MLP on the same lower-row centerline', () => {
        const scene = buildTransformerSceneModel({
            activationSource: createActivationSource(4)
        });
        const layout = buildSceneLayout(scene);

        const ln1Entry = layout.registry.getNodeEntries().find((entry) => (
            entry.semantic?.componentKind === 'layer-norm'
            && entry.semantic?.layerIndex === 0
            && entry.semantic?.stage === 'ln1'
            && entry.role === 'module-card'
        ));
        const head1Entry = layout.registry.getNodeEntries().find((entry) => (
            entry.semantic?.componentKind === 'mhsa'
            && entry.semantic?.layerIndex === 0
            && entry.semantic?.headIndex === 0
            && entry.role === 'head-card'
        ));
        const outProjEntry = layout.registry.getNodeEntries().find((entry) => (
            entry.semantic?.componentKind === 'output-projection'
            && entry.semantic?.layerIndex === 0
            && entry.role === 'projection-weight'
        ));
        const ln2Entry = layout.registry.getNodeEntries().find((entry) => (
            entry.semantic?.componentKind === 'layer-norm'
            && entry.semantic?.layerIndex === 0
            && entry.semantic?.stage === 'ln2'
            && entry.role === 'module-card'
        ));
        const mlpEntry = layout.registry.getNodeEntries().find((entry) => (
            entry.semantic?.componentKind === 'mlp'
            && entry.semantic?.layerIndex === 0
            && entry.semantic?.stage === 'mlp'
            && entry.role === 'module-card'
        ));

        const ln1CenterY = ln1Entry.contentBounds.y + (ln1Entry.contentBounds.height / 2);
        const head1CenterY = head1Entry.contentBounds.y + (head1Entry.contentBounds.height / 2);
        const outProjCenterY = outProjEntry.contentBounds.y + (outProjEntry.contentBounds.height / 2);
        const ln2CenterY = ln2Entry.contentBounds.y + (ln2Entry.contentBounds.height / 2);
        const mlpCenterY = mlpEntry.contentBounds.y + (mlpEntry.contentBounds.height / 2);

        expect(ln1CenterY).toBeCloseTo(head1CenterY, 5);
        expect(ln1CenterY).toBeCloseTo(outProjCenterY, 5);
        expect(ln2CenterY).toBeCloseTo(mlpCenterY, 5);
        expect(ln1CenterY).toBeCloseTo(ln2CenterY, 5);
    });

});
