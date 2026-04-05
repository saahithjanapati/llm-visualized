import { describe, expect, it } from 'vitest';

import { D_MODEL, RESIDUAL_COLOR_CLAMP } from '../../ui/selectionPanelConstants.js';
import { mapValueToColor } from '../../utils/colors.js';
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

function expectGradientToUseActiveColorStops(gradientCss = '') {
    const matches = String(gradientCss || '').match(/#([0-9a-f]{6})/gi) || [];
    expect(matches.length).toBeGreaterThan(0);
    const hasColoredStop = matches.some((token) => {
        const hex = token.slice(1).toLowerCase();
        return hex.slice(0, 2) !== hex.slice(2, 4)
            || hex.slice(2, 4) !== hex.slice(4, 6);
    });
    expect(hasColoredStop).toBe(true);
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
        const outputPlusNode = nodes.find((node) => node?.role === 'layer-norm-output-plus') || null;
        const outputEqualsNode = nodes.find((node) => node?.role === 'layer-norm-output-equals') || null;
        const incomingSpacerNode = nodes.find((node) => node?.role === 'incoming-arrow-spacer') || null;
        const normalizedCopyDropSpacerNode = nodes.find((node) => node?.role === 'layer-norm-copy-drop-spacer') || null;
        const outgoingSpacerNode = nodes.find((node) => node?.role === 'outgoing-arrow-spacer') || null;
        const inputConnectorNode = nodes.find((node) => node?.role === 'connector-layer-norm-input') || null;
        const normalizationConnectorNode = nodes.find((node) => node?.role === 'connector-layer-norm-normalization') || null;
        const normalizedCopyConnectorNode = nodes.find((node) => node?.role === 'connector-layer-norm-copy-normalized') || null;
        const scaledCopyConnectorNode = nodes.find((node) => node?.role === 'connector-layer-norm-copy-scaled') || null;
        const outputConnectorNode = nodes.find((node) => node?.role === 'connector-layer-norm-output') || null;
        const fixedTextSizing = resolveMhsaDetailFixedTextSizing(scene, 1280);

        expect(scene?.metadata?.visualContract).toBe('selection-panel-layer-norm-v1');
        expect(fixedTextSizing?.captionLabelScreenFontPx).toBeNull();
        expect(fixedTextSizing?.textScreenFontPx).toBeNull();
        expect(fixedTextSizing?.operatorBehavior).toBe('scene-relative');
        expect(matrixNodes).toHaveLength(12);
        expect(connectorNodes).toHaveLength(5);
        expect(operatorNodes).toHaveLength(4);
        expect(inputNode?.label?.tex).toBe('x');
        expect(inputNode?.metadata?.caption?.minScreenHeightPx).toBe(1);
        expect(normalizedNode?.label?.tex).toBe('\\hat{x}');
        expect(normalizedCopyNode?.label?.tex).toBe('\\hat{x}');
        expect(scaleNode?.label?.tex).toBe('\\gamma');
        expect(scaleNode?.metadata?.caption?.labelScale).toBeGreaterThan(1.4);
        expect(scaleNode?.metadata?.card?.cornerRadius).toBe(5);
        expect(scaleNode?.dimensions).toEqual({
            rows: 1,
            cols: D_MODEL
        });
        expect(scaledNode?.label?.tex).toBe('\\gamma \\odot \\hat{x}');
        expect(scaledCopyNode?.label?.tex).toBe('\\gamma \\odot \\hat{x}');
        expect(shiftNode?.label?.tex).toBe('\\beta');
        expect(shiftNode?.metadata?.caption?.labelScale).toBeGreaterThan(1.4);
        expect(shiftNode?.metadata?.card?.cornerRadius).toBe(5);
        expect(inputNode?.metadata?.card?.cornerRadius).toBe(10);
        expect(outputNode?.label?.tex).toBe('x_{\\ln}');
        expect(normalizationEquationNode?.tex).toBe('\\hat{x} = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}}');
        expect(normalizationEquationNode?.metadata?.fontScale).toBeGreaterThan(1.3);
        expect(normalizationEquationNode?.layout?.offsetY).toBeLessThan(-10);
        expect(hadamardNode?.text).toBe('⊙');
        expect(hadamardNode?.tex).toBe('\\odot');
        expect(scaleEqualsNode?.text).toBe('=');
        expect(outputPlusNode?.text).toBe('+');
        expect(outputEqualsNode?.text).toBe('=');
        expect(normalizedCopyDropSpacerNode?.metadata?.hidden).toBe(true);
        expect(normalizedCopyDropSpacerNode?.metadata?.card?.height).toBeGreaterThan(44);
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
            stage: 'ln1.output',
            role: 'layer-norm-output-row',
            rowIndex: 1,
            tokenIndex: 1
        });
        expect(scaleNode?.rowItems?.[0]?.gradientCss).toContain('linear-gradient(');
        expectGradientToUseActiveColorStops(scaleNode?.rowItems?.[0]?.gradientCss);
        expect(shiftNode?.rowItems?.[0]?.gradientCss).toContain('linear-gradient(');
        expectGradientToUseActiveColorStops(shiftNode?.rowItems?.[0]?.gradientCss);
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
            anchor: 'bottom'
        });
        expect(normalizedCopyConnectorNode?.metadata?.sourceAnchorMode).toBe('caption-bottom');
        expect(normalizedCopyConnectorNode?.target).toMatchObject({
            nodeId: normalizedCopyNode?.id,
            anchor: 'top'
        });
        expect(scaledCopyConnectorNode?.source).toMatchObject({
            nodeId: scaledNode?.id,
            anchor: 'bottom'
        });
        expect(scaledCopyConnectorNode?.metadata?.sourceAnchorMode).toBe('caption-bottom');
        expect(scaledCopyConnectorNode?.target).toMatchObject({
            nodeId: scaledCopyNode?.id,
            anchor: 'top'
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
        expect(
            Math.abs(
                (normalizedCopyEntry?.anchors?.[VIEW2D_ANCHOR_SIDES.CENTER]?.x ?? 0)
                - (normalizedEntry?.anchors?.[VIEW2D_ANCHOR_SIDES.CENTER]?.x ?? 0)
            )
        ).toBeLessThan(0.5);
        expect(normalizedCopyEntry?.contentBounds?.y).toBeGreaterThan(
            (normalizedEntry?.contentBounds?.y ?? 0) + (normalizedEntry?.contentBounds?.height ?? 0)
        );
        expect(
            (normalizedCopyEntry?.contentBounds?.y ?? 0)
            - (
                (normalizedEntry?.contentBounds?.y ?? 0)
                + (normalizedEntry?.contentBounds?.height ?? 0)
            )
        ).toBeGreaterThan(48);
        expect(scaleEntry?.contentBounds?.x).toBeGreaterThan(
            (normalizedCopyEntry?.contentBounds?.x ?? 0) + (normalizedCopyEntry?.contentBounds?.width ?? 0)
        );
        expect(scaledEntry?.contentBounds?.x).toBeGreaterThan(
            (scaleEntry?.contentBounds?.x ?? 0) + (scaleEntry?.contentBounds?.width ?? 0)
        );
        expect(
            Math.abs(
                (scaledCopyEntry?.anchors?.[VIEW2D_ANCHOR_SIDES.CENTER]?.x ?? 0)
                - (scaledEntry?.anchors?.[VIEW2D_ANCHOR_SIDES.CENTER]?.x ?? 0)
            )
        ).toBeLessThan(0.5);
        expect(scaledCopyEntry?.contentBounds?.y).toBeGreaterThan(
            (scaledEntry?.contentBounds?.y ?? 0) + (scaledEntry?.contentBounds?.height ?? 0)
        );
        expect(
            (scaledCopyEntry?.contentBounds?.y ?? 0)
            - (
                (scaledEntry?.contentBounds?.y ?? 0)
                + (scaledEntry?.contentBounds?.height ?? 0)
            )
        ).toBeGreaterThan(24);
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
        expect((layout?.sceneBounds?.height ?? 0) / Math.max(1, layout?.sceneBounds?.width ?? 1)).toBeGreaterThan(0.12);
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
            stage: 'final_ln.output',
            role: 'layer-norm-output-row',
            rowIndex: 0,
            tokenIndex: 0
        });
    });

    it('uses the activation-source base vector length for layer norm activation rows', () => {
        const incomingValues = Array.from({ length: 12 }, (_, index) => Number((0.1 + (index * 0.1)).toFixed(4)));
        const postAttentionValues = Array.from({ length: 12 }, (_, index) => Number((0.2 + (index * 0.1)).toFixed(4)));
        const postMlpValues = Array.from({ length: 12 }, (_, index) => Number((0.3 + (index * 0.1)).toFixed(4)));
        const ln1NormValues = Array.from({ length: 12 }, (_, index) => Number((0.4 + (index * 0.1)).toFixed(4)));
        const ln1ScaleValues = Array.from({ length: 12 }, (_, index) => Number((0.5 + (index * 0.1)).toFixed(4)));
        const ln1ShiftValues = Array.from({ length: 12 }, (_, index) => Number((0.6 + (index * 0.1)).toFixed(4)));
        const ln2NormValues = Array.from({ length: 12 }, (_, index) => Number((0.7 + (index * 0.1)).toFixed(4)));
        const ln2ScaleValues = Array.from({ length: 12 }, (_, index) => Number((0.8 + (index * 0.1)).toFixed(4)));
        const ln2ShiftValues = Array.from({ length: 12 }, (_, index) => Number((0.9 + (index * 0.1)).toFixed(4)));
        const requestedLengths = [];
        const buildExpandedValues = (seed = 0, length = D_MODEL) => (
            Array.from({ length }, (_, index) => Number((seed + (index * 0.001)).toFixed(4)))
        );
        const activationSource = {
            getBaseVectorLength() {
                return 12;
            },
            getLayerIncoming(_layerIndex = 0, _tokenIndex = 0, targetLength = D_MODEL) {
                requestedLengths.push(['incoming', targetLength]);
                return targetLength === 12 ? incomingValues : buildExpandedValues(10, targetLength);
            },
            getPostAttentionResidual(_layerIndex = 0, _tokenIndex = 0, targetLength = D_MODEL) {
                requestedLengths.push(['post-attention', targetLength]);
                return targetLength === 12 ? postAttentionValues : buildExpandedValues(20, targetLength);
            },
            getPostMlpResidual(_layerIndex = 0, _tokenIndex = 0, targetLength = D_MODEL) {
                requestedLengths.push(['post-mlp', targetLength]);
                return targetLength === 12 ? postMlpValues : buildExpandedValues(30, targetLength);
            },
            getLayerLn1(_layerIndex = 0, stage = 'norm', _tokenIndex = 0, targetLength = D_MODEL) {
                requestedLengths.push([`ln1.${stage}`, targetLength]);
                if (targetLength !== 12) return buildExpandedValues(40, targetLength);
                if (stage === 'scale') return ln1ScaleValues;
                if (stage === 'shift') return ln1ShiftValues;
                return ln1NormValues;
            },
            getLayerLn2(_layerIndex = 0, stage = 'norm', _tokenIndex = 0, targetLength = D_MODEL) {
                requestedLengths.push([`ln2.${stage}`, targetLength]);
                if (targetLength !== 12) return buildExpandedValues(50, targetLength);
                if (stage === 'scale') return ln2ScaleValues;
                if (stage === 'shift') return ln2ShiftValues;
                return ln2NormValues;
            },
            getFinalLayerNorm(stage = 'norm', _tokenIndex = 0, targetLength = D_MODEL) {
                requestedLengths.push([`final.${stage}`, targetLength]);
                return targetLength === 12 ? postMlpValues : buildExpandedValues(60, targetLength);
            }
        };

        const ln1Scene = buildLayerNormDetailSceneModel({
            activationSource,
            layerNormDetailTarget: {
                layerNormKind: 'ln1',
                layerIndex: 3
            },
            tokenRefs: [
                { rowIndex: 0, tokenIndex: 0, tokenLabel: 'Token A' }
            ],
            layerCount: 12
        });
        const ln2Scene = buildLayerNormDetailSceneModel({
            activationSource,
            layerNormDetailTarget: {
                layerNormKind: 'ln2',
                layerIndex: 3
            },
            tokenRefs: [
                { rowIndex: 0, tokenIndex: 0, tokenLabel: 'Token A' }
            ],
            layerCount: 12
        });

        const ln1Nodes = flattenSceneNodes(ln1Scene);
        const ln2Nodes = flattenSceneNodes(ln2Scene);
        const ln1InputNode = ln1Nodes.find((node) => node?.role === 'layer-norm-input') || null;
        const ln2InputNode = ln2Nodes.find((node) => node?.role === 'layer-norm-input') || null;

        expect(ln1InputNode?.rowItems?.[0]?.rawValues).toEqual(incomingValues);
        expect(ln2InputNode?.rowItems?.[0]?.rawValues).toEqual(postAttentionValues);
        expect(requestedLengths).toContainEqual(['incoming', 12]);
        expect(requestedLengths).toContainEqual(['post-attention', 12]);
        expect(requestedLengths).not.toContainEqual(['incoming', D_MODEL]);
        expect(requestedLengths).not.toContainEqual(['post-attention', D_MODEL]);
        expect(requestedLengths).toContainEqual(['ln1.norm', 12]);
        expect(requestedLengths).toContainEqual(['ln2.norm', 12]);
    });

    it('marks single-token layer norm vector captions to skip single-row caption assist', () => {
        const scene = buildLayerNormDetailSceneModel({
            activationSource: createMockActivationSource(),
            layerNormDetailTarget: {
                layerNormKind: 'ln1',
                layerIndex: 3
            },
            tokenRefs: [
                { rowIndex: 0, tokenIndex: 0, tokenLabel: 'Token A' }
            ],
            layerCount: 12
        });
        const nodes = flattenSceneNodes(scene);
        const inputNode = nodes.find((node) => node?.role === 'layer-norm-input') || null;
        const normalizedNode = nodes.find((node) => node?.role === 'layer-norm-normalized') || null;
        const scaleNode = nodes.find((node) => node?.role === 'layer-norm-scale') || null;
        const outputNode = nodes.find((node) => node?.role === 'layer-norm-output') || null;

        expect(inputNode?.metadata?.caption?.disableSingleRowAssist).toBe(true);
        expect(normalizedNode?.metadata?.caption?.disableSingleRowAssist).toBe(true);
        expect(scaleNode?.metadata?.caption?.disableSingleRowAssist).toBe(true);
        expect(outputNode?.metadata?.caption?.disableSingleRowAssist).toBe(true);
    });

    it('uses the residual-stream palette for layer norm parameter rows', () => {
        const scene = buildScene();
        const nodes = flattenSceneNodes(scene);
        const scaleNode = nodes.find((node) => node?.role === 'layer-norm-scale') || null;
        const shiftNode = nodes.find((node) => node?.role === 'layer-norm-shift') || null;
        const scaleValues = scaleNode?.rowItems?.[0]?.rawValues || [];
        const shiftValues = shiftNode?.rowItems?.[0]?.rawValues || [];
        const scaleExpectedStart = `#${mapValueToColor(
            scaleValues[0],
            { clampMax: RESIDUAL_COLOR_CLAMP }
        ).getHexString()}`;
        const shiftExpectedStart = `#${mapValueToColor(
            shiftValues[0],
            { clampMax: RESIDUAL_COLOR_CLAMP }
        ).getHexString()}`;

        expect(scaleNode?.rowItems?.[0]?.gradientCss).toContain(scaleExpectedStart);
        expect(shiftNode?.rowItems?.[0]?.gradientCss).toContain(shiftExpectedStart);
    });

    it('limits layer norm input hover focus to the immediate normalization step', () => {
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
        const normalizationConnectorNode = nodes.find((node) => node?.role === 'connector-layer-norm-normalization') || null;
        const index = createMhsaDetailSceneIndex(scene);
        const hoverState = resolveMhsaDetailHoverState(index, {
            node: inputNode,
            rowHit: {
                rowIndex: 1,
                rowItem: inputNode?.rowItems?.[1]
            }
        });

        expect(hoverState?.label).toBe('Residual Stream Vector');
        expect(hoverState?.info?.activationData?.stage).toBe('layer.incoming');
        expect(hoverState?.info?.activationData?.sourceStage).toBe('layer.incoming');
        expect(hoverState?.info?.activationData?.tokenIndex).toBe(1);
        expect(hoverState?.info?.activationData?.tokenLabel).toBe('Token B');
        expect(hoverState?.info?.activationData?.values).toEqual(inputNode?.rowItems?.[1]?.rawValues);
        expect(hoverState?.focusState?.activeNodeIds).toContain(inputNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(normalizedNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(normalizedCopyNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(scaleNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(scaledNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(scaledCopyNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(shiftNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(outputNode?.id);
        expect(hoverState?.focusState?.activeConnectorIds).toEqual([normalizationConnectorNode?.id]);
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: inputNode?.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: normalizedNode?.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toHaveLength(2);
    });

    it('limits layer norm normalized hover focus to the direct input vector', () => {
        const scene = buildScene();
        const nodes = flattenSceneNodes(scene);
        const inputNode = nodes.find((node) => node?.role === 'layer-norm-input') || null;
        const normalizedNode = nodes.find((node) => node?.role === 'layer-norm-normalized') || null;
        const normalizedCopyNode = nodes.find((node) => node?.role === 'layer-norm-normalized-copy') || null;
        const scaleNode = nodes.find((node) => node?.role === 'layer-norm-scale') || null;
        const normalizationConnectorNode = nodes.find((node) => node?.role === 'connector-layer-norm-normalization') || null;
        const index = createMhsaDetailSceneIndex(scene);
        const hoverState = resolveMhsaDetailHoverState(index, {
            node: normalizedNode,
            rowHit: {
                rowIndex: 1,
                rowItem: normalizedNode?.rowItems?.[1]
            }
        });

        expect(hoverState?.label).toBe('LayerNorm 1 Normalized Vector');
        expect(hoverState?.info?.activationData?.stage).toBe('ln1.norm');
        expect(hoverState?.focusState?.activeNodeIds).toContain(normalizedNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(inputNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(normalizedCopyNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(scaleNode?.id);
        expect(hoverState?.focusState?.activeConnectorIds).toEqual([normalizationConnectorNode?.id]);
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: normalizedNode?.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: inputNode?.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toHaveLength(2);
    });

    it('keeps layer norm parameter hovers on the direct operand only', () => {
        const scene = buildScene();
        const nodes = flattenSceneNodes(scene);
        const normalizedCopyNode = nodes.find((node) => node?.role === 'layer-norm-normalized-copy') || null;
        const scaleNode = nodes.find((node) => node?.role === 'layer-norm-scale') || null;
        const scaledNode = nodes.find((node) => node?.role === 'layer-norm-scaled') || null;
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
        expect(hoverState?.focusState?.activeNodeIds).toContain(normalizedCopyNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(scaledNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(outputNode?.id);
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: scaleNode?.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toHaveLength(1);
    });

    it('keeps layer norm output hover focus on the direct additive inputs', () => {
        const scene = buildScene();
        const nodes = flattenSceneNodes(scene);
        const inputNode = nodes.find((node) => node?.role === 'layer-norm-input') || null;
        const scaleNode = nodes.find((node) => node?.role === 'layer-norm-scale') || null;
        const scaledCopyNode = nodes.find((node) => node?.role === 'layer-norm-scaled-copy') || null;
        const shiftNode = nodes.find((node) => node?.role === 'layer-norm-shift') || null;
        const outputNode = nodes.find((node) => node?.role === 'layer-norm-output') || null;
        const index = createMhsaDetailSceneIndex(scene);
        const hoverState = resolveMhsaDetailHoverState(index, {
            node: outputNode,
            rowHit: {
                rowIndex: 1,
                rowItem: outputNode?.rowItems?.[1]
            }
        });

        expect(hoverState?.label).toBe('Post LayerNorm 1 Residual Vector');
        expect(hoverState?.info?.activationData?.stage).toBe('ln1.output');
        expect(hoverState?.focusState?.activeNodeIds).toContain(outputNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(scaledCopyNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(shiftNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(inputNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(scaleNode?.id);
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: outputNode?.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: scaledCopyNode?.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: shiftNode?.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toHaveLength(3);
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
