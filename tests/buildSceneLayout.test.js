import { describe, expect, it } from 'vitest';
import { buildMhsaSceneModel } from '../src/view2d/model/buildMhsaSceneModel.js';
import { buildSceneLayout } from '../src/view2d/layout/buildSceneLayout.js';
import { buildTransformerSceneModel } from '../src/view2d/model/buildTransformerSceneModel.js';
import {
    fitView2dText,
    resetView2dTextMeasurementCache
} from '../src/view2d/textMeasurement.js';

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

    it('resolves the top-most head card entry for a world-space point', () => {
        const transformerScene = buildTransformerSceneModel({
            activationSource: createActivationSource(4),
            layerCount: 1
        });
        const layout = buildSceneLayout(transformerScene);
        const headEntry = layout.registry.getNodeEntries().find((entry) => (
            entry.semantic?.componentKind === 'mhsa'
            && entry.semantic?.layerIndex === 0
            && entry.semantic?.headIndex === 0
            && entry.role === 'head-card'
        ));
        const pickedEntry = layout.registry.resolveNodeEntryAtPoint(
            headEntry.contentBounds.x + (headEntry.contentBounds.width / 2),
            headEntry.contentBounds.y + (headEntry.contentBounds.height / 2)
        );

        expect(pickedEntry?.nodeId).toBe(headEntry?.nodeId);
        expect(pickedEntry?.semantic).toMatchObject({
            componentKind: 'mhsa',
            layerIndex: 0,
            headIndex: 0
        });
    });

    it('fits long head labels inside their text-fit bounds with measured text widths', () => {
        const originalDocument = globalThis.document;
        const measureContext = {
            font: '',
            measureText(text) {
                const fontSizeMatch = this.font.match(/([0-9.]+)px/);
                const fontSize = Number.parseFloat(fontSizeMatch?.[1] || '12');
                const width = text.length * fontSize * 0.72;
                return {
                    width,
                    actualBoundingBoxLeft: 1,
                    actualBoundingBoxRight: Math.max(1, width - 1),
                    actualBoundingBoxAscent: fontSize * 0.82,
                    actualBoundingBoxDescent: fontSize * 0.22
                };
            }
        };

        globalThis.document = {
            createElement(tagName) {
                if (tagName !== 'canvas') return null;
                return {
                    getContext(kind) {
                        return kind === '2d' ? measureContext : null;
                    }
                };
            }
        };
        resetView2dTextMeasurementCache();

        try {
            const fitted = fitView2dText('Attention Head 12', {
                baseFontSize: 12,
                maxWidth: 116
            });

            expect(fitted.fontSize).toBeLessThan(12);
            expect(fitted.width).toBeLessThanOrEqual(116);
            expect(fitted.textWidth + (fitted.paddingX * 2)).toBeLessThanOrEqual(116);

            const transformerScene = buildTransformerSceneModel({
                activationSource: createActivationSource(4),
                layerCount: 1
            });
            const layout = buildSceneLayout(transformerScene);
            const headLabelEntry = layout.registry.getNodeEntries().find((entry) => (
                entry.role === 'head-label'
                && entry.semantic?.componentKind === 'mhsa'
                && entry.semantic?.layerIndex === 0
                && entry.semantic?.headIndex === 11
            ));

            expect(headLabelEntry?.layoutData?.fontSize).toBeLessThan(12);
            expect(headLabelEntry?.contentBounds?.width).toBeLessThanOrEqual(headLabelEntry?.layoutData?.maxWidth);
        } finally {
            globalThis.document = originalDocument;
            resetView2dTextMeasurementCache();
        }
    });
});
