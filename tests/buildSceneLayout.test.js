import { describe, expect, it } from 'vitest';
import { buildMhsaSceneModel } from '../src/view2d/model/buildMhsaSceneModel.js';
import { buildSceneLayout } from '../src/view2d/layout/buildSceneLayout.js';

function createActivationSource(tokenCount = 5) {
    return {
        meta: {
            prompt_tokens: Array.from({ length: tokenCount }, (_, index) => index),
            token_display_strings: Array.from({ length: tokenCount }, (_, index) => `tok_${index}`)
        },
        getLayerLn1(layerIndex, mode, tokenIndex) {
            if (mode !== 'shift') return null;
            return Array.from({ length: 768 }, (_, index) => ((index % 17) - 8) * (tokenIndex + 1) * 0.02);
        },
        getLayerQKVVector(layerIndex, kind, headIndex, tokenIndex, width) {
            const scale = kind === 'q' ? 0.08 : (kind === 'k' ? 0.11 : 0.16);
            return Array.from({ length: width }, (_, index) => ((index % 9) - 4) * scale * (tokenIndex + 1));
        },
        getLayerQKVScalar(layerIndex, kind, headIndex, tokenIndex) {
            return (tokenIndex + 1) * (kind === 'q' ? 0.12 : (kind === 'k' ? 0.14 : 0.18));
        },
        getAttentionScoresRow(layerIndex, stage, headIndex, tokenIndex) {
            return Array.from({ length: tokenCount }, (_, columnIndex) => {
                if (columnIndex > tokenIndex) return stage === 'pre' ? -1000 : 0;
                return stage === 'pre'
                    ? ((tokenIndex + 1) * (columnIndex + 2)) / 6
                    : 1 / (tokenIndex + 1);
            });
        },
        getAttentionWeightedSum(layerIndex, headIndex, tokenIndex, width) {
            return Array.from({ length: width }, (_, index) => ((index % 13) - 6) * 0.07 * (tokenIndex + 1));
        }
    };
}

describe('buildSceneLayout', () => {
    it('places the projection stack and attention stage in deterministic world bounds', () => {
        const scene = buildMhsaSceneModel({
            activationSource: createActivationSource(5),
            layerIndex: 3,
            headIndex: 2
        });
        const layout = buildSceneLayout(scene);

        expect(layout).not.toBeNull();
        expect(layout.sceneBounds.width).toBeGreaterThan(0);
        expect(layout.sceneBounds.height).toBeGreaterThan(0);

        const projectionStackEntry = layout.registry.getNodeEntry(
            scene.nodes.find((node) => node.role === 'projection-stack')?.id
        );
        const attentionStageEntry = layout.registry.getNodeEntry(
            scene.nodes.find((node) => node.role === 'attention-stage')?.id
        );
        const qConnectorEntry = layout.registry.getConnectorEntry(
            scene.nodes
                .find((node) => node.role === 'connector-layer')
                ?.children?.find((node) => node.role === 'connector-q')
                ?.id
        );

        expect(attentionStageEntry.bounds.x).toBeGreaterThan(projectionStackEntry.bounds.x + projectionStackEntry.bounds.width);
        expect(qConnectorEntry?.pathPoints?.length).toBeGreaterThanOrEqual(2);
        qConnectorEntry?.pathPoints?.forEach((point) => {
            expect(Number.isFinite(point?.x)).toBe(true);
            expect(Number.isFinite(point?.y)).toBe(true);
        });
    });

    it('grows matrix bounds and connector spans as token counts increase', () => {
        const smallLayout = buildSceneLayout(buildMhsaSceneModel({
            activationSource: createActivationSource(4),
            layerIndex: 3,
            headIndex: 2
        }));
        const largeLayout = buildSceneLayout(buildMhsaSceneModel({
            activationSource: createActivationSource(8),
            layerIndex: 3,
            headIndex: 2
        }));

        const findPostBounds = (layout) => {
            const postEntry = layout.registry.getNodeEntries().find((entry) => entry.role === 'attention-post');
            return postEntry?.contentBounds;
        };
        const findVConnector = (layout) => layout.registry.getConnectorEntries().find((entry) => entry.role === 'connector-v');

        expect(findPostBounds(largeLayout).height).toBeGreaterThan(findPostBounds(smallLayout).height);
        expect(findPostBounds(largeLayout).width).toBeGreaterThan(findPostBounds(smallLayout).width);
        expect(findVConnector(largeLayout).bounds.width).toBeGreaterThan(findVConnector(smallLayout).bounds.width);
    });
});
