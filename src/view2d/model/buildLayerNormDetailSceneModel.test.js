import { describe, expect, it } from 'vitest';

import { D_MODEL } from '../../ui/selectionPanelConstants.js';
import {
    createMhsaDetailSceneIndex,
    resolveMhsaDetailHoverState
} from '../mhsaDetailInteraction.js';
import { buildSceneLayout } from '../layout/buildSceneLayout.js';
import {
    flattenSceneNodes,
    VIEW2D_ANCHOR_SIDES,
    VIEW2D_NODE_KINDS
} from '../schema/sceneTypes.js';
import { resolveMhsaDetailFixedTextSizing } from '../shared/mhsaDetailFixedLabelSizing.js';
import { buildLayerNormDetailSceneModel } from './buildLayerNormDetailSceneModel.js';

function createVector(seed = 0, length = D_MODEL) {
    return Array.from({ length }, (_, index) => Number((seed + (index * 0.01)).toFixed(4)));
}

function createMockActivationSource() {
    return {
        getLayerIncoming(_layerIndex = 0, tokenIndex = 0, targetLength = D_MODEL) {
            return createVector(0.1 + tokenIndex, targetLength);
        },
        getPostAttentionResidual(_layerIndex = 0, tokenIndex = 0, targetLength = D_MODEL) {
            return createVector(0.2 + tokenIndex, targetLength);
        },
        getPostMlpResidual(_layerIndex = 0, tokenIndex = 0, targetLength = D_MODEL) {
            return createVector(0.3 + tokenIndex, targetLength);
        },
        getLayerLn1(_layerIndex = 0, stage = 'norm', tokenIndex = 0, targetLength = D_MODEL) {
            const stageOffsets = {
                norm: 0.4,
                scale: 0.5,
                shift: 0.6
            };
            return createVector((stageOffsets[stage] || 0.4) + tokenIndex, targetLength);
        },
        getLayerLn2(_layerIndex = 0, stage = 'norm', tokenIndex = 0, targetLength = D_MODEL) {
            const stageOffsets = {
                norm: 0.7,
                scale: 0.8,
                shift: 0.9
            };
            return createVector((stageOffsets[stage] || 0.7) + tokenIndex, targetLength);
        },
        getFinalLayerNorm(stage = 'norm', tokenIndex = 0, targetLength = D_MODEL) {
            const stageOffsets = {
                norm: 1.0,
                scale: 1.1,
                shift: 1.2
            };
            return createVector((stageOffsets[stage] || 1.0) + tokenIndex, targetLength);
        }
    };
}

describe('buildLayerNormDetailSceneModel', () => {
    function buildScene(target = {
        layerNormKind: 'ln1',
        layerIndex: 3
    }) {
        return buildLayerNormDetailSceneModel({
            activationSource: createMockActivationSource(),
            layerNormDetailTarget: target,
            tokenRefs: [
                { rowIndex: 0, tokenIndex: 0, tokenLabel: 'Token A' },
                { rowIndex: 1, tokenIndex: 1, tokenLabel: 'Token B' }
            ],
            layerCount: 12
        });
    }

    it('builds the layer norm detail flow from x through normalization, scale, shift, and output', () => {
        const scene = buildScene();
        const nodes = flattenSceneNodes(scene);

        const matrixNodes = nodes.filter((node) => node?.kind === VIEW2D_NODE_KINDS.MATRIX);
        const connectorNodes = nodes.filter((node) => node?.kind === VIEW2D_NODE_KINDS.CONNECTOR);
        const operatorNodes = nodes.filter((node) => node?.kind === VIEW2D_NODE_KINDS.OPERATOR);
        const inputNode = nodes.find((node) => node?.role === 'layer-norm-input') || null;
        const normalizedNode = nodes.find((node) => node?.role === 'layer-norm-normalized') || null;
        const normalizedCopyNode = nodes.find((node) => node?.role === 'layer-norm-normalized-copy') || null;
        const scaleNode = nodes.find((node) => node?.role === 'layer-norm-scale') || null;
        const scaledNode = nodes.find((node) => node?.role === 'layer-norm-scaled') || null;
        const scaledCopyNode = nodes.find((node) => node?.role === 'layer-norm-scaled-copy') || null;
        const shiftNode = nodes.find((node) => node?.role === 'layer-norm-shift') || null;
        const outputNode = nodes.find((node) => node?.role === 'layer-norm-output') || null;
        const normalizationEquationNode = nodes.find((node) => node?.role === 'layer-norm-normalization-equation') || null;
        const hadamardNode = nodes.find((node) => node?.role === 'layer-norm-hadamard') || null;
        const scaleEqualsNode = nodes.find((node) => node?.role === 'layer-norm-scale-equals') || null;
        const shiftPlusNode = nodes.find((node) => node?.role === 'layer-norm-shift-plus') || null;
        const shiftEqualsNode = nodes.find((node) => node?.role === 'layer-norm-shift-equals') || null;
        const incomingSpacerNode = nodes.find((node) => node?.role === 'incoming-arrow-spacer') || null;
        const outgoingSpacerNode = nodes.find((node) => node?.role === 'outgoing-arrow-spacer') || null;
        const normalizationBridgeNode = nodes.find((node) => node?.role === 'layer-norm-normalization-bridge') || null;
        const inputConnectorNode = nodes.find((node) => node?.role === 'connector-layer-norm-input') || null;
        const normalizationConnectorNode = nodes.find((node) => node?.role === 'connector-layer-norm-normalization') || null;
        const normalizedCopyConnectorNode = nodes.find((node) => node?.role === 'connector-layer-norm-copy-normalized') || null;
        const scaledCopyConnectorNode = nodes.find((node) => node?.role === 'connector-layer-norm-copy-scaled') || null;
        const outputConnectorNode = nodes.find((node) => node?.role === 'connector-layer-norm-output') || null;
        const fixedTextSizing = resolveMhsaDetailFixedTextSizing(scene, 1280);

        expect(scene?.metadata?.visualContract).toBe('selection-panel-layer-norm-v1');
        expect(fixedTextSizing?.captionLabelScreenFontPx).toBe(17.5);
        expect(fixedTextSizing?.textScreenFontPx).toBe(19);
        expect(matrixNodes).toHaveLength(13);
        expect(connectorNodes).toHaveLength(5);
        expect(operatorNodes).toHaveLength(4);
        expect(inputNode?.label?.tex).toBe('x');
        expect(inputNode?.metadata?.caption?.minScreenHeightPx).toBe(1);
        expect(normalizedNode?.label?.tex).toBe('\\hat{x}');
        expect(normalizedCopyNode?.label?.tex).toBe('\\hat{x}');
        expect(scaleNode?.label?.tex).toBe('\\gamma');
        expect(scaleNode?.dimensions).toEqual({
            rows: 1,
            cols: D_MODEL
        });
        expect(scaledNode?.label?.tex).toBe('\\gamma \\odot \\hat{x}');
        expect(scaledCopyNode?.label?.tex).toBe('\\gamma \\odot \\hat{x}');
        expect(shiftNode?.label?.tex).toBe('\\beta');
        expect(outputNode?.label?.tex).toBe('x_{\\ln}');
        expect(normalizationEquationNode?.tex).toBe('\\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}}');
        expect(normalizationEquationNode?.metadata?.fontScale).toBeGreaterThan(1.15);
        expect(normalizationBridgeNode?.metadata?.gapOverride).toBeGreaterThan(10);
        expect(hadamardNode?.text).toBe('⊙');
        expect(scaleEqualsNode?.text).toBe('=');
        expect(shiftPlusNode?.text).toBe('+');
        expect(shiftEqualsNode?.text).toBe('=');
        expect(inputNode?.rowItems).toHaveLength(2);
        expect(normalizedNode?.rowItems).toHaveLength(2);
        expect(scaleNode?.rowItems).toHaveLength(1);
        expect(scaledNode?.rowItems).toHaveLength(2);
        expect(shiftNode?.rowItems).toHaveLength(1);
        expect(outputNode?.rowItems).toHaveLength(2);
        expect(inputNode?.rowItems?.[1]?.semantic).toMatchObject({
            componentKind: 'residual',
            layerIndex: 3,
            stage: 'layer.incoming',
            role: 'layer-norm-input-row',
            rowIndex: 1,
            tokenIndex: 1
        });
        expect(normalizedNode?.rowItems?.[1]?.semantic).toMatchObject({
            componentKind: 'layer-norm',
            layerIndex: 3,
            stage: 'ln1.norm',
            role: 'layer-norm-normalized-row',
            rowIndex: 1,
            tokenIndex: 1
        });
        expect(scaleNode?.rowItems?.[0]?.semantic).toMatchObject({
            componentKind: 'layer-norm',
            layerIndex: 3,
            stage: 'ln1.param.scale',
            role: 'layer-norm-scale-row',
            rowIndex: 0
        });
        expect(shiftNode?.rowItems?.[0]?.semantic).toMatchObject({
            componentKind: 'layer-norm',
            layerIndex: 3,
            stage: 'ln1.param.shift',
            role: 'layer-norm-shift-row',
            rowIndex: 0
        });
        expect(outputNode?.rowItems?.[1]?.semantic).toMatchObject({
            componentKind: 'residual',
            layerIndex: 3,
            stage: 'ln1.shift',
            role: 'layer-norm-output-row',
            rowIndex: 1,
            tokenIndex: 1
        });
        expect(scaleNode?.rowItems?.[0]?.gradientCss).toContain('linear-gradient(');
        expect(shiftNode?.rowItems?.[0]?.gradientCss).toContain('linear-gradient(');
        expect(outputNode?.rowItems?.[1]?.gradientCss).toContain('linear-gradient(');
        expect(inputConnectorNode?.source).toMatchObject({
            nodeId: incomingSpacerNode?.id,
            anchor: 'left'
        });
        expect(inputConnectorNode?.target).toMatchObject({
            nodeId: inputNode?.id,
            anchor: 'left'
        });
        expect(normalizationConnectorNode?.source).toMatchObject({
            nodeId: inputNode?.id,
            anchor: 'right'
        });
        expect(normalizationConnectorNode?.target).toMatchObject({
            nodeId: normalizedNode?.id,
            anchor: 'left'
        });
        expect(normalizedCopyConnectorNode?.source).toMatchObject({
            nodeId: normalizedNode?.id,
            anchor: 'right'
        });
        expect(normalizedCopyConnectorNode?.target).toMatchObject({
            nodeId: normalizedCopyNode?.id,
            anchor: 'left'
        });
        expect(scaledCopyConnectorNode?.source).toMatchObject({
            nodeId: scaledNode?.id,
            anchor: 'right'
        });
        expect(scaledCopyConnectorNode?.target).toMatchObject({
            nodeId: scaledCopyNode?.id,
            anchor: 'left'
        });
        expect(outputConnectorNode?.source).toMatchObject({
            nodeId: outputNode?.id,
            anchor: 'right'
        });
        expect(outputConnectorNode?.target).toMatchObject({
            nodeId: outgoingSpacerNode?.id,
            anchor: 'right'
        });

        const layout = buildSceneLayout(scene);
        const inputEntry = layout?.registry?.getNodeEntry(inputNode?.id);
        const normalizedEntry = layout?.registry?.getNodeEntry(normalizedNode?.id);
        const normalizedCopyEntry = layout?.registry?.getNodeEntry(normalizedCopyNode?.id);
        const scaleEntry = layout?.registry?.getNodeEntry(scaleNode?.id);
        const scaledEntry = layout?.registry?.getNodeEntry(scaledNode?.id);
        const scaledCopyEntry = layout?.registry?.getNodeEntry(scaledCopyNode?.id);
        const shiftEntry = layout?.registry?.getNodeEntry(shiftNode?.id);
        const outputEntry = layout?.registry?.getNodeEntry(outputNode?.id);
        const equationEntry = layout?.registry?.getNodeEntry(normalizationEquationNode?.id);
        const normalizationConnectorEntry = layout?.registry?.getConnectorEntry(normalizationConnectorNode?.id);

        expect(inputEntry?.contentBounds).toBeTruthy();
        expect(normalizedEntry?.contentBounds?.x).toBeGreaterThan(
            (inputEntry?.contentBounds?.x ?? 0) + (inputEntry?.contentBounds?.width ?? 0)
        );
        expect(normalizedCopyEntry?.contentBounds?.x).toBeGreaterThan(
            (normalizedEntry?.contentBounds?.x ?? 0) + (normalizedEntry?.contentBounds?.width ?? 0)
        );
        expect(scaleEntry?.contentBounds?.x).toBeGreaterThan(
            (normalizedCopyEntry?.contentBounds?.x ?? 0) + (normalizedCopyEntry?.contentBounds?.width ?? 0)
        );
        expect(scaledEntry?.contentBounds?.x).toBeGreaterThan(
            (scaleEntry?.contentBounds?.x ?? 0) + (scaleEntry?.contentBounds?.width ?? 0)
        );
        expect(scaledCopyEntry?.contentBounds?.x).toBeGreaterThan(
            (scaledEntry?.contentBounds?.x ?? 0) + (scaledEntry?.contentBounds?.width ?? 0)
        );
        expect(shiftEntry?.contentBounds?.x).toBeGreaterThan(
            (scaledCopyEntry?.contentBounds?.x ?? 0) + (scaledCopyEntry?.contentBounds?.width ?? 0)
        );
        expect(outputEntry?.contentBounds?.x).toBeGreaterThan(
            (shiftEntry?.contentBounds?.x ?? 0) + (shiftEntry?.contentBounds?.width ?? 0)
        );
        expect(equationEntry?.bounds?.y).toBeLessThan(
            inputEntry?.contentBounds?.y ?? Number.POSITIVE_INFINITY
        );
        expect(normalizationConnectorEntry?.pathPoints?.length).toBeGreaterThan(1);
        expect(
            Math.abs(
                (inputEntry?.anchors?.[VIEW2D_ANCHOR_SIDES.CENTER]?.y ?? 0)
                - (normalizedEntry?.anchors?.[VIEW2D_ANCHOR_SIDES.CENTER]?.y ?? 0)
            )
        ).toBeLessThan(0.5);
    });

    it('supports final layer norm detail scenes without requiring an explicit layer index', () => {
        const scene = buildScene({
            layerNormKind: 'final'
        });
        const nodes = flattenSceneNodes(scene);
        const outputNode = nodes.find((node) => node?.role === 'layer-norm-output') || null;
        const inputNode = nodes.find((node) => node?.role === 'layer-norm-input') || null;

        expect(scene?.metadata?.layerNormKind).toBe('final');
        expect(scene?.metadata?.layerIndex).toBe(11);
        expect(inputNode?.semantic?.layerIndex).toBe(11);
        expect(outputNode?.label?.tex).toBe('x_{\\mathrm{final}}');
        expect(outputNode?.rowItems?.[0]?.semantic).toMatchObject({
            componentKind: 'layer-norm',
            layerIndex: 11,
            stage: 'final_ln.shift',
            role: 'layer-norm-output-row',
            rowIndex: 0,
            tokenIndex: 0
        });
    });

    it('reuses deep-detail hover behavior for layer norm token rows', () => {
        const scene = buildScene();
        const nodes = flattenSceneNodes(scene);
        const inputNode = nodes.find((node) => node?.role === 'layer-norm-input') || null;
        const normalizedNode = nodes.find((node) => node?.role === 'layer-norm-normalized') || null;
        const normalizedCopyNode = nodes.find((node) => node?.role === 'layer-norm-normalized-copy') || null;
        const scaleNode = nodes.find((node) => node?.role === 'layer-norm-scale') || null;
        const scaledNode = nodes.find((node) => node?.role === 'layer-norm-scaled') || null;
        const scaledCopyNode = nodes.find((node) => node?.role === 'layer-norm-scaled-copy') || null;
        const shiftNode = nodes.find((node) => node?.role === 'layer-norm-shift') || null;
        const outputNode = nodes.find((node) => node?.role === 'layer-norm-output') || null;
        const index = createMhsaDetailSceneIndex(scene);
        const hoverState = resolveMhsaDetailHoverState(index, {
            node: inputNode,
            rowHit: {
                rowIndex: 1,
                rowItem: inputNode?.rowItems?.[1]
            }
        });

        expect(hoverState?.label).toBe('LayerNorm 1 Input Vector');
        expect(hoverState?.info?.activationData?.stage).toBe('ln1.input');
        expect(hoverState?.info?.activationData?.sourceStage).toBe('layer.incoming');
        expect(hoverState?.info?.activationData?.tokenIndex).toBe(1);
        expect(hoverState?.info?.activationData?.tokenLabel).toBe('Token B');
        expect(hoverState?.focusState?.activeNodeIds).toContain(inputNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(normalizedNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(normalizedCopyNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(scaleNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(scaledNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(scaledCopyNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(shiftNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(outputNode?.id);
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: inputNode?.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: normalizedNode?.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: normalizedCopyNode?.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: scaledNode?.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: scaledCopyNode?.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: outputNode?.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: scaleNode?.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: shiftNode?.id,
            rowIndex: 0
        });
    });

    it('keeps layer norm parameter cards hoverable with layer-norm-specific tooltips', () => {
        const scene = buildScene();
        const nodes = flattenSceneNodes(scene);
        const inputNode = nodes.find((node) => node?.role === 'layer-norm-input') || null;
        const scaleNode = nodes.find((node) => node?.role === 'layer-norm-scale') || null;
        const outputNode = nodes.find((node) => node?.role === 'layer-norm-output') || null;
        const index = createMhsaDetailSceneIndex(scene);
        const hoverState = resolveMhsaDetailHoverState(index, {
            node: scaleNode,
            rowHit: {
                rowIndex: 0,
                rowItem: scaleNode?.rowItems?.[0]
            }
        });

        expect(hoverState?.label).toBe('LayerNorm 1 Scale');
        expect(hoverState?.info?.activationData?.stage).toBe('ln1.param.scale');
        expect(hoverState?.info?.activationData?.layerNormKind).toBe('ln1');
        expect(hoverState?.focusState?.activeNodeIds).toContain(scaleNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(inputNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(outputNode?.id);
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: scaleNode?.id,
            rowIndex: 0
        });
    });

    it('uses a final-layer-specific hover label for the top layer norm output', () => {
        const scene = buildScene({
            layerNormKind: 'final'
        });
        const nodes = flattenSceneNodes(scene);
        const outputNode = nodes.find((node) => node?.role === 'layer-norm-output') || null;
        const index = createMhsaDetailSceneIndex(scene);
        const hoverState = resolveMhsaDetailHoverState(index, {
            node: outputNode,
            rowHit: {
                rowIndex: 0,
                rowItem: outputNode?.rowItems?.[0]
            }
        });

        expect(hoverState?.label).toBe('LayerNorm (Top) Output Vector');
        expect(hoverState?.info?.activationData?.stage).toBe('final_ln.output');
        expect(hoverState?.info?.activationData?.sourceStage).toBe('final_ln.shift');
        expect(hoverState?.info?.activationData?.layerNormKind).toBe('final');
    });
});
