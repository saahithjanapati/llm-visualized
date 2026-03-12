import { describe, expect, it } from 'vitest';
import { D_HEAD, D_MODEL } from '../src/ui/selectionPanelConstants.js';
import { buildSceneLayout } from '../src/view2d/layout/buildSceneLayout.js';
import { buildMhsaSceneModel } from '../src/view2d/model/buildMhsaSceneModel.js';
import { flattenSceneNodes } from '../src/view2d/schema/sceneTypes.js';
import { VIEW2D_STYLE_KEYS } from '../src/view2d/theme/visualTokens.js';

function createActivationSource(tokenCount = 3) {
    const promptTokens = Array.from({ length: tokenCount }, (_, index) => index);
    const tokenDisplayStrings = promptTokens.map((index) => `tok_${index}`);

    return {
        meta: {
            prompt_tokens: promptTokens,
            token_display_strings: tokenDisplayStrings
        },
        getLayerLn1(layerIndex, mode, tokenIndex) {
            if (mode !== 'shift') return null;
            return Array.from({ length: D_MODEL }, (_, index) => {
                const centered = (index % 9) - 4;
                return (tokenIndex + 1) * centered * 0.0625;
            });
        },
        getLayerQKVVector(layerIndex, kind, headIndex, tokenIndex, width) {
            const scale = kind === 'q' ? 0.125 : (kind === 'k' ? 0.175 : 0.225);
            return Array.from({ length: width }, (_, index) => {
                const centered = (index % 11) - 5;
                return (tokenIndex + 1) * centered * scale;
            });
        },
        getLayerQKVScalar(layerIndex, kind, headIndex, tokenIndex) {
            const scale = kind === 'q' ? 0.2 : (kind === 'k' ? 0.28 : 0.34);
            return (tokenIndex + 1) * scale;
        },
        getAttentionScoresRow(layerIndex, stage, headIndex, tokenIndex) {
            return Array.from({ length: tokenCount }, (_, columnIndex) => {
                if (columnIndex > tokenIndex) {
                    return stage === 'pre' ? -1000 : 0;
                }
                if (stage === 'pre') {
                    return ((tokenIndex + 1) * (columnIndex + 1)) / 5;
                }
                return 1 / (tokenIndex + 1);
            });
        },
        getAttentionWeightedSum(layerIndex, headIndex, tokenIndex, width) {
            return Array.from({ length: width }, (_, index) => {
                const centered = (index % 13) - 6;
                return (tokenIndex + 1) * centered * 0.08;
            });
        }
    };
}

describe('buildMhsaSceneModel', () => {
    it('adapts MHSA preview data into generic matrix, operator, and connector nodes', () => {
        const scene = buildMhsaSceneModel({
            activationSource: createActivationSource(3),
            layerIndex: 2,
            headIndex: 4,
            sampleStep: 256
        });

        expect(scene).not.toBeNull();
        expect(scene.semantic.componentKind).toBe('mhsa');
        expect(scene.metadata.rowCount).toBe(3);

        const nodes = flattenSceneNodes(scene);
        const sourceAnchorNode = nodes.find((node) => node.role === 'projection-source-anchor');
        const projectionSidecarNode = nodes.find((node) => node.role === 'projection-sidecar');
        const projectionStackNode = nodes.find((node) => node.role === 'projection-stack');
        const qInputNode = nodes.find((node) => node.role === 'x-ln-copy' && node.semantic?.branchKey === 'q');
        const qMultiplyNode = nodes.find((node) => node.role === 'projection-multiply' && node.semantic?.stage === 'projection-q');
        const qWeightNode = nodes.find((node) => node.role === 'projection-weight' && node.metadata?.kind === 'q');
        const qBiasNode = nodes.find((node) => node.role === 'projection-bias' && node.metadata?.kind === 'q');
        const qOutputNode = nodes.find((node) => node.role === 'projection-output' && node.metadata?.kind === 'q');
        const attentionQueryNode = nodes.find((node) => node.role === 'attention-query-source');
        const attentionTransposeNode = nodes.find((node) => node.role === 'attention-key-transpose');
        const postNode = nodes.find((node) => node.role === 'attention-post');
        const headOutputNode = nodes.find((node) => node.role === 'attention-head-output');
        const connectorNodes = nodes.filter((node) => node.kind === 'connector');
        const connectorRoles = nodes
            .filter((node) => node.kind === 'connector')
            .map((node) => node.role);

        expect(qInputNode?.rowItems?.[0]?.semantic).toMatchObject({
            componentKind: 'residual',
            layerIndex: 2,
            headIndex: 4,
            stage: 'ln1.shift',
            branchKey: 'q',
            tokenIndex: 0
        });
        expect(qInputNode?.metadata?.caption?.position).toBe('bottom');
        expect(qInputNode?.metadata?.caption?.scaleWithNode).toBe(true);
        expect(qInputNode?.metadata?.compactRows?.compactWidth).toBe(84);
        expect(qInputNode?.metadata?.compactRows?.rowHeight).toBe(7);
        expect(qMultiplyNode?.metadata?.fontScale).toBe(0.82);
        expect(sourceAnchorNode?.metadata?.hidden).toBe(true);
        expect(sourceAnchorNode?.metadata?.card?.cornerRadius).toBe(0);
        expect(projectionSidecarNode?.metadata?.gapOverride).toBe(30);
        expect(projectionStackNode?.metadata?.gapOverride).toBe(120);
        expect(qWeightNode?.visual?.styleKey).toBe(VIEW2D_STYLE_KEYS.MHSA_Q);
        expect(qWeightNode?.visual?.background).toMatch(/^linear-gradient\(/);
        expect(qWeightNode?.metadata?.caption?.renderMode).toBe('dom-katex');
        expect(qWeightNode?.metadata?.caption?.position).toBe('bottom');
        expect(qWeightNode?.metadata?.caption?.scaleWithNode).toBe(true);
        expect(qWeightNode?.metadata?.caption?.minScreenHeightPx).toBe(28);
        expect(qWeightNode?.metadata?.caption?.labelScale).toBe(0.42);
        expect(qWeightNode?.metadata?.caption?.dimensionsScale).toBe(0.5);
        expect(qWeightNode?.metadata?.card?.cornerRadius).toBe(10);
        expect(qBiasNode?.dimensions).toEqual({ rows: 1, cols: D_HEAD });
        expect(qBiasNode?.visual?.styleKey).toBe(VIEW2D_STYLE_KEYS.MHSA_Q);
        expect(qBiasNode?.rowItems).toHaveLength(1);
        expect(qBiasNode?.rowItems?.[0]?.gradientCss).toMatch(/gradient|#/);
        expect(qBiasNode?.metadata?.caption?.renderMode).toBe('dom-katex');
        expect(qBiasNode?.metadata?.caption?.position).toBe('bottom');
        expect(qBiasNode?.metadata?.caption?.dimensionsTex).toBe(`(1, ${D_HEAD})`);
        expect(qBiasNode?.metadata?.caption?.dimensionsText).toBe(`(1, ${D_HEAD})`);
        expect(qBiasNode?.metadata?.caption?.minScreenHeightPx).toBe(12);
        expect(qBiasNode?.metadata?.caption?.scaleWithNode).toBe(true);
        expect(qBiasNode?.metadata?.caption?.labelScale).toBe(0.42);
        expect(qBiasNode?.metadata?.caption?.dimensionsScale).toBe(0.5);
        expect(qBiasNode?.metadata?.compactRows?.rowHeight).toBe(14);
        expect(qBiasNode?.metadata?.compactRows?.compactWidth).toBe(52);
        expect(qBiasNode?.metadata?.card?.cornerRadius).toBe(5);
        expect(qOutputNode?.dimensions).toEqual({ rows: 3, cols: D_HEAD });
        expect(qOutputNode?.rowItems?.[0]?.semantic).toMatchObject({
            componentKind: 'mhsa',
            layerIndex: 2,
            headIndex: 4,
            stage: 'qkv.q',
            branchKey: 'q',
            tokenIndex: 0
        });
        expect(qOutputNode?.metadata?.caption?.renderMode).toBe('dom-katex');
        expect(qOutputNode?.metadata?.caption?.position).toBe('bottom');
        expect(qOutputNode?.metadata?.caption?.dimensionsText).toBe(`(3, ${D_HEAD})`);
        expect(qOutputNode?.metadata?.compactRows?.compactWidth).toBe(66);
        expect(qOutputNode?.metadata?.compactRows?.rowHeight).toBe(7);
        expect(attentionQueryNode?.dimensions).toEqual({ rows: 3, cols: D_HEAD });
        expect(attentionQueryNode?.metadata?.caption?.position).toBe('bottom');
        expect(attentionQueryNode?.metadata?.caption?.dimensionsText).toBe(`(3, ${D_HEAD})`);
        expect(attentionQueryNode?.metadata?.compactRows?.compactWidth).toBe(66);
        expect(attentionQueryNode?.metadata?.compactRows?.rowHeight).toBe(7);
        expect(attentionTransposeNode?.dimensions).toEqual({ rows: D_HEAD, cols: 3 });
        expect(attentionTransposeNode?.metadata?.caption?.position).toBe('bottom');
        expect(attentionTransposeNode?.metadata?.caption?.dimensionsText).toBe(`(${D_HEAD}, 3)`);
        expect(attentionTransposeNode?.metadata?.columnStrip?.colGap).toBe(0);
        expect(attentionTransposeNode?.metadata?.columnStrip?.colHeight).toBe(66);
        expect(attentionTransposeNode?.metadata?.card?.cornerRadius).toBe(8);
        expect(attentionTransposeNode?.columnItems?.[0]?.semantic).toMatchObject({
            componentKind: 'mhsa',
            layerIndex: 2,
            headIndex: 4,
            stage: 'qkv.k',
            branchKey: 'k',
            tokenIndex: 0,
            colIndex: 0
        });
        expect(postNode?.dimensions).toEqual({ rows: 3, cols: 3 });
        expect(postNode?.rowItems).toHaveLength(3);
        expect(postNode?.rowItems[2]?.cells).toHaveLength(3);
        expect(headOutputNode?.dimensions).toEqual({ rows: 3, cols: D_HEAD });
        expect(connectorNodes.every((node) => node.visual?.stroke === 'rgba(255, 255, 255, 0.84)')).toBe(true);
        expect(connectorRoles).toEqual(expect.arrayContaining([
            'connector-xln-q',
            'connector-xln-k',
            'connector-xln-v',
            'connector-q',
            'connector-k',
            'connector-pre',
            'connector-post',
            'connector-v'
        ]));
    });

    it('scales matrix dimensions and layout metadata as token counts grow', () => {
        const smallScene = buildMhsaSceneModel({
            activationSource: createActivationSource(3),
            layerIndex: 1,
            headIndex: 1,
            sampleStep: 256
        });
        const largeScene = buildMhsaSceneModel({
            activationSource: createActivationSource(7),
            layerIndex: 1,
            headIndex: 1,
            sampleStep: 256
        });

        expect(largeScene.metadata.rowCount).toBe(7);
        expect(largeScene.metadata.layoutMetrics.connectorGaps.default)
            .toBeGreaterThan(smallScene.metadata.layoutMetrics.connectorGaps.default);

        const largeNodes = flattenSceneNodes(largeScene);
        const largeQOutputNode = largeNodes.find((node) => node.role === 'projection-output' && node.metadata?.kind === 'q');
        const largePostNode = largeNodes.find((node) => node.role === 'attention-post');

        expect(largeQOutputNode?.dimensions).toEqual({ rows: 7, cols: D_HEAD });
        expect(largePostNode?.dimensions).toEqual({ rows: 7, cols: 7 });
        expect(largePostNode?.rowItems).toHaveLength(7);
        expect(largePostNode?.rowItems[6]?.cells).toHaveLength(7);
    });

    it('sizes projection weights relative to the incoming X_ln strip', () => {
        const scene = buildMhsaSceneModel({
            activationSource: createActivationSource(5),
            layerIndex: 3,
            headIndex: 2,
            sampleStep: 256
        });
        const layout = buildSceneLayout(scene);
        const nodes = flattenSceneNodes(scene);
        const qInputNode = nodes.find((node) => node.role === 'x-ln-copy' && node.semantic?.branchKey === 'q');
        const qMultiplyNode = nodes.find((node) => node.role === 'projection-multiply' && node.semantic?.stage === 'projection-q');
        const qWeightNode = nodes.find((node) => node.role === 'projection-weight' && node.metadata?.kind === 'q');
        const qBiasNode = nodes.find((node) => node.role === 'projection-bias' && node.metadata?.kind === 'q');
        const qOutputNode = nodes.find((node) => node.role === 'projection-output' && node.metadata?.kind === 'q');
        const attentionQueryNode = nodes.find((node) => node.role === 'attention-query-source');
        const attentionTransposeNode = nodes.find((node) => node.role === 'attention-key-transpose');
        const qInputEntry = layout.registry.getNodeEntry(qInputNode?.id || '');
        const qMultiplyEntry = layout.registry.getNodeEntry(qMultiplyNode?.id || '');
        const qWeightEntry = layout.registry.getNodeEntry(qWeightNode?.id || '');
        const qBiasEntry = layout.registry.getNodeEntry(qBiasNode?.id || '');
        const qOutputEntry = layout.registry.getNodeEntry(qOutputNode?.id || '');
        const attentionQueryEntry = layout.registry.getNodeEntry(attentionQueryNode?.id || '');
        const attentionTransposeEntry = layout.registry.getNodeEntry(attentionTransposeNode?.id || '');

        expect(qInputEntry?.contentBounds?.width).toBeGreaterThan(0);
        expect(qMultiplyEntry?.layoutData?.fontSize).toBeLessThan(layout.config.component.operatorFontSize);
        expect(qWeightEntry?.contentBounds?.height).toBe(qInputEntry?.contentBounds?.width);
        expect(qWeightEntry?.contentBounds?.height).toBeGreaterThan(qWeightEntry?.contentBounds?.width);
        expect(qBiasEntry?.contentBounds?.width).toBeGreaterThan(0);
        expect(qBiasEntry?.contentBounds?.width).toBeLessThan(qInputEntry?.contentBounds?.width);
        expect(qBiasEntry?.contentBounds?.width).toBe(52);
        expect(qBiasEntry?.contentBounds?.height).toBe(14);
        expect(qOutputEntry?.contentBounds?.width).toBeGreaterThan(0);
        expect(qOutputEntry?.contentBounds?.width).toBeLessThan(qInputEntry?.contentBounds?.width);
        expect(qOutputEntry?.contentBounds?.width).toBe(66);
        expect(qOutputEntry?.contentBounds?.height).toBe(qInputEntry?.contentBounds?.height);
        expect(attentionQueryEntry?.contentBounds?.width).toBe(66);
        expect(attentionQueryEntry?.contentBounds?.height).toBe(qOutputEntry?.contentBounds?.height);
        expect(attentionTransposeEntry?.contentBounds?.height).toBe(66);
        expect(attentionTransposeEntry?.contentBounds?.width).toBeLessThan(attentionTransposeEntry?.contentBounds?.height);
        expect(attentionTransposeEntry?.contentBounds?.width).toBe(35);
    });

    it('starts the ingress stem at the left source anchor and stops arrowheads short of the X_ln strips', () => {
        const scene = buildMhsaSceneModel({
            activationSource: createActivationSource(5),
            layerIndex: 3,
            headIndex: 2,
            sampleStep: 256
        });
        const layout = buildSceneLayout(scene);
        const nodes = flattenSceneNodes(scene);
        const sourceAnchorNode = nodes.find((node) => node.role === 'projection-source-anchor');
        const qInputNode = nodes.find((node) => node.role === 'x-ln-copy' && node.semantic?.branchKey === 'q');
        const qIngressConnectorNode = nodes.find((node) => node.role === 'connector-xln-q');
        const sourceAnchorEntry = layout.registry.getNodeEntry(sourceAnchorNode?.id || '');
        const qInputEntry = layout.registry.getNodeEntry(qInputNode?.id || '');
        const qIngressConnectorEntry = layout.registry.getConnectorEntry(qIngressConnectorNode?.id || '');
        const connectorSourcePoint = qIngressConnectorEntry?.pathPoints?.[0];
        const connectorTargetPoint = qIngressConnectorEntry?.pathPoints?.[qIngressConnectorEntry.pathPoints.length - 1];

        expect(qIngressConnectorNode?.gap).toBe(0);
        expect(qIngressConnectorNode?.sourceGap).toBe(0);
        expect(qIngressConnectorNode?.targetGap).toBe(8);
        expect(connectorSourcePoint).toEqual(sourceAnchorEntry?.anchors?.left);
        expect(connectorTargetPoint).toEqual({
            x: (qInputEntry?.anchors?.left?.x || 0) - 8,
            y: qInputEntry?.anchors?.left?.y || 0
        });
        expect(qInputEntry?.contentBounds?.x).toBeGreaterThan(sourceAnchorEntry?.bounds?.x || 0);
    });
});
