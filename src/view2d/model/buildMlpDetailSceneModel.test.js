import { describe, expect, it } from 'vitest';

import { D_MODEL, FINAL_MLP_COLOR } from '../../ui/selectionPanelConstants.js';
import {
    MLP_ACTIVATION_TOOLTIP_LABEL,
    MLP_DOWN_BIAS_TOOLTIP_LABEL,
    MLP_DOWN_TOOLTIP_LABEL
} from '../../utils/mlpLabels.js';
import { buildSceneLayout } from '../layout/buildSceneLayout.js';
import { flattenSceneNodes, VIEW2D_NODE_KINDS } from '../schema/sceneTypes.js';
import { createMhsaDetailSceneIndex, resolveMhsaDetailHoverState } from '../mhsaDetailInteraction.js';
import { buildMlpDetailSceneModel } from './buildMlpDetailSceneModel.js';

function createVector(seed = 0, length = D_MODEL) {
    return Array.from({ length }, (_, index) => Number((seed + (index * 0.01)).toFixed(4)));
}

function toKatexColorHex(hex = 0xFFFFFF) {
    return `#${Math.max(0, Math.min(0xFFFFFF, Math.floor(
        Number.isFinite(hex) ? hex : 0xFFFFFF
    ))).toString(16).padStart(6, '0')}`;
}

function createMockActivationSource() {
    return {
        getLayerLn2(_layerIndex = 0, _stage = 'shift', tokenIndex = 0, targetLength = D_MODEL) {
            return createVector(0.25 + tokenIndex, targetLength);
        },
        getMlpUp(_layerIndex = 0, tokenIndex = 0, targetLength = D_MODEL * 4) {
            return createVector(0.5 + tokenIndex, targetLength);
        },
        getMlpActivation(_layerIndex = 0, tokenIndex = 0, targetLength = D_MODEL * 4) {
            return createVector(0.75 + tokenIndex, targetLength);
        },
        getMlpDown(_layerIndex = 0, tokenIndex = 0, targetLength = D_MODEL) {
            return createVector(1.0 + tokenIndex, targetLength);
        }
    };
}

describe('buildMlpDetailSceneModel', () => {
    function buildScene(tokenCount = 2) {
        return buildMlpDetailSceneModel({
            activationSource: createMockActivationSource(),
            mlpDetailTarget: {
                layerIndex: 4
            },
            tokenRefs: Array.from({ length: tokenCount }, (_, rowIndex) => ({
                rowIndex,
                tokenIndex: rowIndex,
                tokenLabel: `Token ${String.fromCharCode(65 + (rowIndex % 26))}${rowIndex >= 26 ? rowIndex : ''}`.trim()
            }))
        });
    }

    it('builds the MLP up-projection detail scene around the incoming lower-case x_ln residual matrix', () => {
        const scene = buildScene();

        const nodes = flattenSceneNodes(scene);
        const matrixNodes = nodes.filter((node) => node?.kind === VIEW2D_NODE_KINDS.MATRIX);
        const connectorNodes = nodes.filter((node) => node?.kind === VIEW2D_NODE_KINDS.CONNECTOR);
        const operatorNodes = nodes.filter((node) => node?.kind === VIEW2D_NODE_KINDS.OPERATOR);
        const textNodes = nodes.filter((node) => node?.kind === VIEW2D_NODE_KINDS.TEXT);
        const inputNode = nodes.find((node) => node?.role === 'projection-source-xln') || null;
        const weightNode = nodes.find((node) => node?.role === 'mlp-up-weight') || null;
        const biasNode = nodes.find((node) => node?.role === 'mlp-up-bias') || null;
        const outputNode = nodes.find((node) => node?.role === 'mlp-up-output') || null;
        const outputCopyNode = nodes.find((node) => node?.role === 'mlp-up-output-copy') || null;
        const activationNode = nodes.find((node) => node?.role === 'mlp-activation-output') || null;
        const activationCopyNode = nodes.find((node) => node?.role === 'mlp-activation-output-copy') || null;
        const downWeightNode = nodes.find((node) => node?.role === 'mlp-down-weight') || null;
        const downBiasNode = nodes.find((node) => node?.role === 'mlp-down-bias') || null;
        const downOutputNode = nodes.find((node) => node?.role === 'mlp-down-output') || null;
        const incomingSpacerNode = nodes.find((node) => node?.role === 'incoming-arrow-spacer') || null;
        const outgoingSpacerNode = nodes.find((node) => node?.role === 'outgoing-arrow-spacer') || null;
        const inputConnectorNode = nodes.find((node) => node?.role === 'connector-mlp-input') || null;
        const geluConnectorNode = nodes.find((node) => node?.role === 'connector-mlp-gelu-input') || null;
        const downConnectorNode = nodes.find((node) => node?.role === 'connector-mlp-down-input') || null;
        const downOutgoingConnectorNode = nodes.find((node) => node?.role === 'connector-mlp-down-output-outgoing') || null;
        const multiplyNode = nodes.find((node) => node?.role === 'mlp-up-multiply') || null;
        const plusNode = nodes.find((node) => node?.role === 'mlp-up-plus') || null;
        const equalsNode = nodes.find((node) => node?.role === 'mlp-up-equals') || null;
        const geluLabelNode = nodes.find((node) => node?.role === 'mlp-gelu-label') || null;
        const geluOpenNode = nodes.find((node) => node?.role === 'mlp-gelu-open') || null;
        const geluCloseNode = nodes.find((node) => node?.role === 'mlp-gelu-close') || null;
        const geluEqualsNode = nodes.find((node) => node?.role === 'mlp-gelu-equals') || null;
        const downMultiplyNode = nodes.find((node) => node?.role === 'mlp-down-multiply') || null;
        const downPlusNode = nodes.find((node) => node?.role === 'mlp-down-plus') || null;
        const downEqualsNode = nodes.find((node) => node?.role === 'mlp-down-equals') || null;

        expect(matrixNodes).toHaveLength(12);
        expect(connectorNodes).toHaveLength(4);
        expect(operatorNodes).toHaveLength(9);
        expect(textNodes).toHaveLength(1);
        expect(inputNode?.role).toBe('projection-source-xln');
        expect(inputNode?.label?.tex).toBe('x_{\\ln}');
        expect(inputNode?.label?.text).toBe('x_ln');
        expect(inputNode?.metadata?.disableEdgeOrnament).toBe(true);
        expect(inputNode?.semantic).toMatchObject({
            componentKind: 'mlp',
            layerIndex: 4,
            stage: 'mlp-input',
            role: 'projection-source-xln'
        });
        expect(inputNode?.metadata?.caption?.position).toBe('bottom');
        expect(inputNode?.metadata?.compactRows?.rowGap).toBe(0);
        expect(inputNode?.metadata?.compactRows?.paddingY).toBe(0);
        expect(inputNode?.rowItems).toHaveLength(2);
        expect(inputNode?.rowItems?.every((rowItem) => (
            rowItem?.semantic?.componentKind === 'residual'
            && rowItem?.semantic?.stage === 'ln2.shift'
            && rowItem?.semantic?.role === 'x-ln-row'
        ))).toBe(true);
        expect(weightNode?.label?.tex).toBe('W_{\\mathrm{up}}');
        expect(weightNode?.label?.text).toBe('W_up');
        expect(weightNode?.dimensions).toMatchObject({
            rows: D_MODEL,
            cols: D_MODEL * 4
        });
        expect(weightNode?.metadata?.card?.height).toBe(inputNode?.metadata?.compactRows?.compactWidth);
        expect(weightNode?.metadata?.card?.width).toBe(outputNode?.metadata?.compactRows?.compactWidth);
        expect(weightNode?.metadata?.caption?.dimensionsText).toBe('(768, 3072)');
        expect(biasNode?.label?.tex).toBe('b_{\\mathrm{up}}');
        expect(biasNode?.label?.text).toBe('b_up');
        expect(biasNode?.dimensions).toMatchObject({
            rows: 1,
            cols: D_MODEL * 4
        });
        expect(biasNode?.metadata?.caption?.dimensionsText).toBe('(1, 3072)');
        expect(biasNode?.rowItems).toHaveLength(1);
        expect(biasNode?.rowItems?.[0]?.semantic).toMatchObject({
            componentKind: 'mlp',
            layerIndex: 4,
            stage: 'mlp.up.bias',
            role: 'mlp-up-bias-row',
            rowIndex: 0
        });
        expect(biasNode?.rowItems?.[0]?.gradientCss).toContain('linear-gradient(');
        expect(biasNode?.metadata?.compactRows?.rowHeight).toBeGreaterThan(inputNode?.metadata?.compactRows?.rowHeight || 0);
        expect(outputNode?.label?.tex).toBe('a');
        expect(outputNode?.label?.text).toBe('a');
        expect(outputNode?.dimensions).toMatchObject({
            rows: 2,
            cols: D_MODEL * 4
        });
        expect(outputNode?.metadata?.caption?.dimensionsText).toBe('(2, 3072)');
        expect(outputNode?.rowItems).toHaveLength(2);
        expect(outputNode?.rowItems?.every((rowItem) => (
            rowItem?.semantic?.componentKind === 'mlp'
            && rowItem?.semantic?.stage === 'mlp-up'
            && rowItem?.semantic?.role === 'mlp-up-row'
        ))).toBe(true);
        expect(outputCopyNode?.label?.text).toBe('a');
        expect(outputCopyNode?.dimensions).toMatchObject({
            rows: 2,
            cols: D_MODEL * 4
        });
        expect(outputCopyNode?.metadata?.caption?.dimensionsText).toBe('(2, 3072)');
        expect(outputCopyNode?.rowItems?.every((rowItem) => (
            rowItem?.semantic?.componentKind === 'mlp'
            && rowItem?.semantic?.stage === 'mlp-up'
            && rowItem?.semantic?.role === 'mlp-up-copy-row'
        ))).toBe(true);
        expect(activationNode?.label?.tex).toBe('z');
        expect(activationNode?.label?.text).toBe('z');
        expect(activationNode?.dimensions).toMatchObject({
            rows: 2,
            cols: D_MODEL * 4
        });
        expect(activationNode?.metadata?.caption?.dimensionsText).toBe('(2, 3072)');
        expect(activationNode?.rowItems?.every((rowItem) => (
            rowItem?.semantic?.componentKind === 'mlp'
            && rowItem?.semantic?.stage === 'mlp.activation'
            && rowItem?.semantic?.role === 'mlp-activation-row'
        ))).toBe(true);
        expect(activationCopyNode?.label?.tex).toBe('z');
        expect(activationCopyNode?.label?.text).toBe('z');
        expect(activationCopyNode?.dimensions).toMatchObject({
            rows: 2,
            cols: D_MODEL * 4
        });
        expect(activationCopyNode?.metadata?.caption?.dimensionsText).toBe('(2, 3072)');
        expect(activationCopyNode?.rowItems?.every((rowItem) => (
            rowItem?.semantic?.componentKind === 'mlp'
            && rowItem?.semantic?.stage === 'mlp.activation'
            && rowItem?.semantic?.role === 'mlp-activation-copy-row'
        ))).toBe(true);
        expect(downWeightNode?.label?.tex).toBe('W_{\\mathrm{down}}');
        expect(downWeightNode?.label?.text).toBe('W_down');
        expect(downWeightNode?.dimensions).toMatchObject({
            rows: D_MODEL * 4,
            cols: D_MODEL
        });
        expect(downWeightNode?.metadata?.card?.height).toBe(activationCopyNode?.metadata?.compactRows?.compactWidth);
        expect(downWeightNode?.metadata?.card?.width).toBe(downOutputNode?.metadata?.compactRows?.compactWidth);
        expect(downWeightNode?.metadata?.caption?.dimensionsText).toBe('(3072, 768)');
        expect(downBiasNode?.label?.tex).toBe('b_{\\mathrm{down}}');
        expect(downBiasNode?.label?.text).toBe('b_down');
        expect(downBiasNode?.dimensions).toMatchObject({
            rows: 1,
            cols: D_MODEL
        });
        expect(downBiasNode?.metadata?.caption?.dimensionsText).toBe('(1, 768)');
        expect(downBiasNode?.rowItems?.[0]?.semantic).toMatchObject({
            componentKind: 'mlp',
            layerIndex: 4,
            stage: 'mlp.down.bias',
            role: 'mlp-down-bias-row',
            rowIndex: 0
        });
        expect(downOutputNode?.label?.tex).toBe(
            `\\textcolor{${toKatexColorHex(FINAL_MLP_COLOR)}}{\\mathrm{MLP}}(x_{\\ln})`
        );
        expect(downOutputNode?.label?.text).toBe('MLP(x_ln)');
        expect(downOutputNode?.dimensions).toMatchObject({
            rows: 2,
            cols: D_MODEL
        });
        expect(downOutputNode?.metadata?.caption?.dimensionsText).toBe('(2, 768)');
        expect(downOutputNode?.metadata?.disableEdgeOrnament).toBe(true);
        expect(downOutputNode?.rowItems?.every((rowItem) => (
            rowItem?.semantic?.componentKind === 'mlp'
            && rowItem?.semantic?.stage === 'mlp.down'
            && rowItem?.semantic?.role === 'mlp-down-row'
        ))).toBe(true);
        expect(multiplyNode?.text).toBe('×');
        expect(plusNode?.text).toBe('+');
        expect(equalsNode?.text).toBe('=');
        expect(geluLabelNode?.tex).toBe('\\mathrm{GELU}');
        expect(geluLabelNode?.text).toBe('GELU');
        expect(geluLabelNode?.metadata?.renderMode).toBe('dom-katex');
        expect(geluOpenNode?.text).toBe('(');
        expect(geluCloseNode?.text).toBe(')');
        expect(geluEqualsNode?.text).toBe('=');
        expect(downMultiplyNode?.text).toBe('×');
        expect(downPlusNode?.text).toBe('+');
        expect(downEqualsNode?.text).toBe('=');
        expect(inputConnectorNode?.source).toMatchObject({
            nodeId: incomingSpacerNode?.id,
            anchor: 'left'
        });
        expect(inputConnectorNode?.target).toMatchObject({
            nodeId: inputNode?.id,
            anchor: 'left'
        });
        expect(inputConnectorNode?.targetGap).toBeGreaterThan(0);
        expect(geluConnectorNode?.source).toMatchObject({
            nodeId: outputNode?.id,
            anchor: 'bottom'
        });
        expect(geluConnectorNode?.target).toMatchObject({
            nodeId: outputCopyNode?.id,
            anchor: 'top'
        });
        expect(geluConnectorNode?.metadata?.sourceAnchorMode).toBe('caption-bottom');
        expect(geluConnectorNode?.targetGap).toBeGreaterThan(0);
        expect(geluConnectorNode?.route).toBe('vertical');
        expect(downConnectorNode?.source).toMatchObject({
            nodeId: activationNode?.id,
            anchor: 'right'
        });
        expect(downConnectorNode?.target).toMatchObject({
            nodeId: activationCopyNode?.id,
            anchor: 'left'
        });
        expect(downConnectorNode?.targetGap).toBeGreaterThan(0);
        expect(downConnectorNode?.route).toBe('horizontal');
        expect(downOutgoingConnectorNode?.source).toMatchObject({
            nodeId: downOutputNode?.id,
            anchor: 'right'
        });
        expect(downOutgoingConnectorNode?.target).toMatchObject({
            nodeId: outgoingSpacerNode?.id,
            anchor: 'right'
        });
        expect(downOutgoingConnectorNode?.sourceGap).toBeGreaterThan(0);
        expect(downOutgoingConnectorNode?.route).toBe('horizontal');

        const layout = buildSceneLayout(scene);
        const inputEntry = layout?.registry?.getNodeEntry(inputNode?.id);
        const weightEntry = layout?.registry?.getNodeEntry(weightNode?.id);
        const biasEntry = layout?.registry?.getNodeEntry(biasNode?.id);
        const outputEntry = layout?.registry?.getNodeEntry(outputNode?.id);
        const outputCopyEntry = layout?.registry?.getNodeEntry(outputCopyNode?.id);
        const activationEntry = layout?.registry?.getNodeEntry(activationNode?.id);
        const activationCopyEntry = layout?.registry?.getNodeEntry(activationCopyNode?.id);
        const downWeightEntry = layout?.registry?.getNodeEntry(downWeightNode?.id);
        const downBiasEntry = layout?.registry?.getNodeEntry(downBiasNode?.id);
        const downOutputEntry = layout?.registry?.getNodeEntry(downOutputNode?.id);
        const downOutgoingConnectorEntry = layout?.registry?.getConnectorEntry(downOutgoingConnectorNode?.id);
        expect(inputEntry?.contentBounds).toBeTruthy();
        expect(Math.abs(
            ((weightEntry?.contentBounds?.y || 0) + ((weightEntry?.contentBounds?.height || 0) / 2))
            - ((inputEntry?.contentBounds?.y || 0) + ((inputEntry?.contentBounds?.height || 0) / 2))
        )).toBeLessThanOrEqual(1);
        expect(Math.abs(
            ((biasEntry?.contentBounds?.y || 0) + ((biasEntry?.contentBounds?.height || 0) / 2))
            - ((inputEntry?.contentBounds?.y || 0) + ((inputEntry?.contentBounds?.height || 0) / 2))
        )).toBeLessThanOrEqual(1);
        expect(Math.abs(
            ((outputCopyEntry?.contentBounds?.x || 0) + ((outputCopyEntry?.contentBounds?.width || 0) / 2))
            - ((outputEntry?.contentBounds?.x || 0) + ((outputEntry?.contentBounds?.width || 0) / 2))
        )).toBeLessThanOrEqual(1);
        expect((outputCopyEntry?.contentBounds?.y || 0)).toBeGreaterThan(
            (outputEntry?.contentBounds?.y || 0) + (outputEntry?.contentBounds?.height || 0)
        );
        expect((activationEntry?.contentBounds?.x || 0)).toBeGreaterThan(
            (outputCopyEntry?.contentBounds?.x || 0) + (outputCopyEntry?.contentBounds?.width || 0)
        );
        expect((activationCopyEntry?.contentBounds?.x || 0)).toBeGreaterThan(
            (activationEntry?.contentBounds?.x || 0) + (activationEntry?.contentBounds?.width || 0)
        );
        expect((activationCopyEntry?.contentBounds?.x || 0)).toBeGreaterThanOrEqual(
            (activationEntry?.contentBounds?.x || 0) + (activationEntry?.contentBounds?.width || 0) + 39
        );
        expect(Math.abs(
            ((activationCopyEntry?.contentBounds?.y || 0) + ((activationCopyEntry?.contentBounds?.height || 0) / 2))
            - ((activationEntry?.contentBounds?.y || 0) + ((activationEntry?.contentBounds?.height || 0) / 2))
        )).toBeLessThanOrEqual(1);
        expect(Math.abs(
            ((downWeightEntry?.contentBounds?.y || 0) + ((downWeightEntry?.contentBounds?.height || 0) / 2))
            - ((activationCopyEntry?.contentBounds?.y || 0) + ((activationCopyEntry?.contentBounds?.height || 0) / 2))
        )).toBeLessThanOrEqual(1);
        expect(Math.abs(
            ((downBiasEntry?.contentBounds?.y || 0) + ((downBiasEntry?.contentBounds?.height || 0) / 2))
            - ((activationCopyEntry?.contentBounds?.y || 0) + ((activationCopyEntry?.contentBounds?.height || 0) / 2))
        )).toBeLessThanOrEqual(1);
        expect((downOutputEntry?.contentBounds?.x || 0)).toBeGreaterThan(
            (downBiasEntry?.contentBounds?.x || 0) + (downBiasEntry?.contentBounds?.width || 0)
        );
        expect(downOutgoingConnectorEntry?.pathPoints?.length).toBeGreaterThanOrEqual(2);
        expect(downOutgoingConnectorEntry?.pathPoints?.[0]?.x || 0).toBeGreaterThan(
            (downOutputEntry?.contentBounds?.x || 0) + (downOutputEntry?.contentBounds?.width || 0)
        );
    });

    it('adapts dense token windows with denser rows and larger GELU spacing while preserving dimension ordering', () => {
        const baseScene = buildScene(2);
        const denseScene = buildScene(12);
        const baseNodes = flattenSceneNodes(baseScene);
        const denseNodes = flattenSceneNodes(denseScene);
        const baseInputNode = baseNodes.find((node) => node?.role === 'projection-source-xln') || null;
        const denseInputNode = denseNodes.find((node) => node?.role === 'projection-source-xln') || null;
        const baseOutputNode = baseNodes.find((node) => node?.role === 'mlp-up-output') || null;
        const denseOutputNode = denseNodes.find((node) => node?.role === 'mlp-up-output') || null;
        const baseOutputCopyNode = baseNodes.find((node) => node?.role === 'mlp-up-output-copy') || null;
        const denseOutputCopyNode = denseNodes.find((node) => node?.role === 'mlp-up-output-copy') || null;
        const baseActivationNode = baseNodes.find((node) => node?.role === 'mlp-activation-output') || null;
        const denseActivationNode = denseNodes.find((node) => node?.role === 'mlp-activation-output') || null;
        const baseUpWeightNode = baseNodes.find((node) => node?.role === 'mlp-up-weight') || null;
        const denseUpWeightNode = denseNodes.find((node) => node?.role === 'mlp-up-weight') || null;
        const baseDownWeightNode = baseNodes.find((node) => node?.role === 'mlp-down-weight') || null;
        const denseDownWeightNode = denseNodes.find((node) => node?.role === 'mlp-down-weight') || null;

        expect(Number(denseScene?.metadata?.layoutMetrics?.extraRows) || 0).toBeGreaterThan(0);
        expect((denseInputNode?.metadata?.compactRows?.rowHeight || 0)).toBeLessThan(
            (baseInputNode?.metadata?.compactRows?.rowHeight || 0)
        );
        expect((denseOutputNode?.metadata?.compactRows?.rowHeight || 0)).toBeLessThan(
            (baseOutputNode?.metadata?.compactRows?.rowHeight || 0)
        );
        expect((denseActivationNode?.metadata?.compactRows?.rowHeight || 0)).toBeLessThan(
            (baseActivationNode?.metadata?.compactRows?.rowHeight || 0)
        );
        expect((baseOutputNode?.metadata?.compactRows?.compactWidth || 0)).toBeGreaterThan(
            (baseInputNode?.metadata?.compactRows?.compactWidth || 0)
        );
        expect((denseOutputNode?.metadata?.compactRows?.compactWidth || 0)).toBeGreaterThan(
            (denseInputNode?.metadata?.compactRows?.compactWidth || 0)
        );
        expect(baseUpWeightNode?.metadata?.card?.height).toBe(baseInputNode?.metadata?.compactRows?.compactWidth);
        expect(baseUpWeightNode?.metadata?.card?.width).toBe(baseOutputNode?.metadata?.compactRows?.compactWidth);
        expect(denseUpWeightNode?.metadata?.card?.height).toBe(denseInputNode?.metadata?.compactRows?.compactWidth);
        expect(denseUpWeightNode?.metadata?.card?.width).toBe(denseOutputNode?.metadata?.compactRows?.compactWidth);
        expect(baseDownWeightNode?.metadata?.card?.height).toBe(baseActivationNode?.metadata?.compactRows?.compactWidth);
        expect(baseDownWeightNode?.metadata?.card?.width).toBe(baseInputNode?.metadata?.compactRows?.compactWidth);
        expect(denseDownWeightNode?.metadata?.card?.height).toBe(denseActivationNode?.metadata?.compactRows?.compactWidth);
        expect(denseDownWeightNode?.metadata?.card?.width).toBe(denseInputNode?.metadata?.compactRows?.compactWidth);

        const baseLayout = buildSceneLayout(baseScene);
        const denseLayout = buildSceneLayout(denseScene);
        const baseOutputEntry = baseLayout?.registry?.getNodeEntry(baseOutputNode?.id);
        const denseOutputEntry = denseLayout?.registry?.getNodeEntry(denseOutputNode?.id);
        const baseOutputCopyEntry = baseLayout?.registry?.getNodeEntry(baseOutputCopyNode?.id);
        const denseOutputCopyEntry = denseLayout?.registry?.getNodeEntry(denseOutputCopyNode?.id);
        const baseGap = (baseOutputCopyEntry?.contentBounds?.y || 0)
            - ((baseOutputEntry?.contentBounds?.y || 0) + (baseOutputEntry?.contentBounds?.height || 0));
        const denseGap = (denseOutputCopyEntry?.contentBounds?.y || 0)
            - ((denseOutputEntry?.contentBounds?.y || 0) + (denseOutputEntry?.contentBounds?.height || 0));
        expect(denseGap).toBeGreaterThan(baseGap);
    });

    it('reuses the detail-scene row hover behavior for ln2 x_ln rows', () => {
        const scene = buildScene();
        const nodes = flattenSceneNodes(scene);
        const inputNode = nodes.find((node) => node?.role === 'projection-source-xln') || null;
        const outputNode = nodes.find((node) => node?.role === 'mlp-up-output') || null;
        const outputCopyNode = nodes.find((node) => node?.role === 'mlp-up-output-copy') || null;
        const activationNode = nodes.find((node) => node?.role === 'mlp-activation-output') || null;
        const activationCopyNode = nodes.find((node) => node?.role === 'mlp-activation-output-copy') || null;
        const downOutputNode = nodes.find((node) => node?.role === 'mlp-down-output') || null;
        const index = createMhsaDetailSceneIndex(scene);
        const hoverState = resolveMhsaDetailHoverState(index, {
            node: inputNode,
            rowHit: {
                rowIndex: 1,
                rowItem: inputNode?.rowItems?.[1]
            }
        });

        expect(hoverState?.label).toBe('Post LayerNorm 2 Residual Vector');
        expect(hoverState?.info?.activationData?.stage).toBe('ln2.shift');
        expect(hoverState?.info?.activationData?.tokenIndex).toBe(1);
        expect(hoverState?.info?.activationData?.tokenLabel).toBe('Token B');
        expect(hoverState?.focusState?.activeNodeIds).toContain(inputNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(outputNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(outputCopyNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(activationNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(activationCopyNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(downOutputNode?.id);
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: inputNode?.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: outputNode?.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: outputCopyNode?.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: activationNode?.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: activationCopyNode?.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: downOutputNode?.id,
            rowIndex: 1
        });
    });

    it('reuses the 3D MLP-up tooltip payload for output rows', () => {
        const scene = buildScene();
        const nodes = flattenSceneNodes(scene);
        const inputNode = nodes.find((node) => node?.role === 'projection-source-xln') || null;
        const biasNode = nodes.find((node) => node?.role === 'mlp-up-bias') || null;
        const outputNode = nodes.find((node) => node?.role === 'mlp-up-output') || null;
        const outputCopyNode = nodes.find((node) => node?.role === 'mlp-up-output-copy') || null;
        const weightNode = nodes.find((node) => node?.role === 'mlp-up-weight') || null;
        const activationNode = nodes.find((node) => node?.role === 'mlp-activation-output') || null;
        const activationCopyNode = nodes.find((node) => node?.role === 'mlp-activation-output-copy') || null;
        const index = createMhsaDetailSceneIndex(scene);
        const hoverState = resolveMhsaDetailHoverState(index, {
            node: outputNode,
            rowHit: {
                rowIndex: 0,
                rowItem: outputNode?.rowItems?.[0]
            }
        });

        expect(hoverState?.label).toBe('MLP Up Projection');
        expect(hoverState?.info?.activationData?.label).toBe('MLP Up Projection');
        expect(hoverState?.info?.activationData?.stage).toBe('mlp.up');
        expect(hoverState?.info?.activationData?.tokenIndex).toBe(0);
        expect(hoverState?.info?.activationData?.tokenLabel).toBe('Token A');
        expect(hoverState?.focusState?.activeNodeIds).toContain(outputNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(outputCopyNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(inputNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(weightNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(biasNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(activationNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(activationCopyNode?.id);
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: outputNode?.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: outputCopyNode?.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: inputNode?.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: activationNode?.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: activationCopyNode?.id,
            rowIndex: 0
        });
    });

    it('keeps the copied GELU input matrix hoverable with the same tooltip payload', () => {
        const scene = buildScene();
        const nodes = flattenSceneNodes(scene);
        const outputNode = nodes.find((node) => node?.role === 'mlp-up-output') || null;
        const outputCopyNode = nodes.find((node) => node?.role === 'mlp-up-output-copy') || null;
        const index = createMhsaDetailSceneIndex(scene);
        const hoverState = resolveMhsaDetailHoverState(index, {
            node: outputCopyNode,
            rowHit: {
                rowIndex: 1,
                rowItem: outputCopyNode?.rowItems?.[1]
            }
        });

        expect(hoverState?.label).toBe('MLP Up Projection');
        expect(hoverState?.info?.activationData?.stage).toBe('mlp.up');
        expect(hoverState?.info?.activationData?.tokenIndex).toBe(1);
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: outputCopyNode?.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: outputNode?.id,
            rowIndex: 1
        });
    });

    it('reuses the 3D MLP activation tooltip payload for post-GELU rows', () => {
        const scene = buildScene();
        const nodes = flattenSceneNodes(scene);
        const inputNode = nodes.find((node) => node?.role === 'projection-source-xln') || null;
        const outputNode = nodes.find((node) => node?.role === 'mlp-up-output') || null;
        const outputCopyNode = nodes.find((node) => node?.role === 'mlp-up-output-copy') || null;
        const activationNode = nodes.find((node) => node?.role === 'mlp-activation-output') || null;
        const activationCopyNode = nodes.find((node) => node?.role === 'mlp-activation-output-copy') || null;
        const downWeightNode = nodes.find((node) => node?.role === 'mlp-down-weight') || null;
        const downBiasNode = nodes.find((node) => node?.role === 'mlp-down-bias') || null;
        const downOutputNode = nodes.find((node) => node?.role === 'mlp-down-output') || null;
        const index = createMhsaDetailSceneIndex(scene);
        const hoverState = resolveMhsaDetailHoverState(index, {
            node: activationNode,
            rowHit: {
                rowIndex: 0,
                rowItem: activationNode?.rowItems?.[0]
            }
        });

        expect(hoverState?.label).toBe(MLP_ACTIVATION_TOOLTIP_LABEL);
        expect(hoverState?.info?.activationData?.label).toBe(MLP_ACTIVATION_TOOLTIP_LABEL);
        expect(hoverState?.info?.activationData?.stage).toBe('mlp.activation');
        expect(hoverState?.info?.activationData?.tokenIndex).toBe(0);
        expect(hoverState?.focusState?.activeNodeIds).toContain(inputNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(outputNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(outputCopyNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(activationNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(activationCopyNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(downWeightNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(downBiasNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(downOutputNode?.id);
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: inputNode?.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: activationNode?.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: activationCopyNode?.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: downOutputNode?.id,
            rowIndex: 0
        });
    });

    it('keeps the copied post-GELU matrix hoverable with the same tooltip payload', () => {
        const scene = buildScene();
        const nodes = flattenSceneNodes(scene);
        const inputNode = nodes.find((node) => node?.role === 'projection-source-xln') || null;
        const activationNode = nodes.find((node) => node?.role === 'mlp-activation-output') || null;
        const activationCopyNode = nodes.find((node) => node?.role === 'mlp-activation-output-copy') || null;
        const downOutputNode = nodes.find((node) => node?.role === 'mlp-down-output') || null;
        const index = createMhsaDetailSceneIndex(scene);
        const hoverState = resolveMhsaDetailHoverState(index, {
            node: activationCopyNode,
            rowHit: {
                rowIndex: 1,
                rowItem: activationCopyNode?.rowItems?.[1]
            }
        });

        expect(hoverState?.label).toBe(MLP_ACTIVATION_TOOLTIP_LABEL);
        expect(hoverState?.info?.activationData?.stage).toBe('mlp.activation');
        expect(hoverState?.info?.activationData?.tokenIndex).toBe(1);
        expect(hoverState?.focusState?.activeNodeIds).toContain(inputNode?.id);
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: inputNode?.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: activationNode?.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: activationCopyNode?.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: downOutputNode?.id,
            rowIndex: 1
        });
    });

    it('preserves token metadata on MLP-down output row hover payloads', () => {
        const scene = buildScene();
        const nodes = flattenSceneNodes(scene);
        const inputNode = nodes.find((node) => node?.role === 'projection-source-xln') || null;
        const outputNode = nodes.find((node) => node?.role === 'mlp-up-output') || null;
        const outputCopyNode = nodes.find((node) => node?.role === 'mlp-up-output-copy') || null;
        const activationNode = nodes.find((node) => node?.role === 'mlp-activation-output') || null;
        const activationCopyNode = nodes.find((node) => node?.role === 'mlp-activation-output-copy') || null;
        const downWeightNode = nodes.find((node) => node?.role === 'mlp-down-weight') || null;
        const downBiasNode = nodes.find((node) => node?.role === 'mlp-down-bias') || null;
        const downOutputNode = nodes.find((node) => node?.role === 'mlp-down-output') || null;
        const index = createMhsaDetailSceneIndex(scene);
        const hoverState = resolveMhsaDetailHoverState(index, {
            node: downOutputNode,
            rowHit: {
                rowIndex: 0,
                rowItem: downOutputNode?.rowItems?.[0]
            }
        });

        expect(hoverState?.label).toBe('MLP Down Projection');
        expect(hoverState?.info?.activationData?.label).toBe('MLP Down Projection');
        expect(hoverState?.info?.activationData?.stage).toBe('mlp.down');
        expect(hoverState?.info?.suppressTokenChip).toBeUndefined();
        expect(hoverState?.info?.activationData?.suppressTokenChip).toBeUndefined();
        expect(hoverState?.info?.activationData?.tokenIndex).toBe(0);
        expect(hoverState?.focusState?.activeNodeIds).toContain(inputNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(outputNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(outputCopyNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(activationNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(activationCopyNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(downWeightNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(downBiasNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(downOutputNode?.id);
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: inputNode?.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: outputNode?.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: outputCopyNode?.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: activationNode?.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: activationCopyNode?.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: downOutputNode?.id,
            rowIndex: 0
        });
    });

    it('reuses the 3D W_up tooltip payload for the MLP weight matrix card', () => {
        const scene = buildScene();
        const nodes = flattenSceneNodes(scene);
        const inputNode = nodes.find((node) => node?.role === 'projection-source-xln') || null;
        const outputNode = nodes.find((node) => node?.role === 'mlp-up-output') || null;
        const outputCopyNode = nodes.find((node) => node?.role === 'mlp-up-output-copy') || null;
        const weightNode = nodes.find((node) => node?.role === 'mlp-up-weight') || null;
        const biasNode = nodes.find((node) => node?.role === 'mlp-up-bias') || null;
        const index = createMhsaDetailSceneIndex(scene);
        const hoverState = resolveMhsaDetailHoverState(index, {
            node: weightNode
        });

        expect(hoverState?.label).toBe('MLP Up Weight Matrix');
        expect(hoverState?.info?.activationData?.label).toBe('MLP Up Weight Matrix');
        expect(hoverState?.info?.activationData?.stage).toBe('mlp.up');
        expect(hoverState?.info?.activationData?.layerIndex).toBe(4);
        expect(hoverState?.focusState?.activeNodeIds).toContain(weightNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(inputNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(biasNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(outputNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(outputCopyNode?.id);
    });

    it('shows a 2D-only tooltip for the MLP up bias vector', () => {
        const scene = buildScene();
        const nodes = flattenSceneNodes(scene);
        const inputNode = nodes.find((node) => node?.role === 'projection-source-xln') || null;
        const weightNode = nodes.find((node) => node?.role === 'mlp-up-weight') || null;
        const biasNode = nodes.find((node) => node?.role === 'mlp-up-bias') || null;
        const outputNode = nodes.find((node) => node?.role === 'mlp-up-output') || null;
        const outputCopyNode = nodes.find((node) => node?.role === 'mlp-up-output-copy') || null;
        const index = createMhsaDetailSceneIndex(scene);
        const hoverState = resolveMhsaDetailHoverState(index, {
            node: biasNode,
            rowHit: {
                rowIndex: 0,
                rowItem: biasNode?.rowItems?.[0]
            }
        });

        expect(hoverState?.label).toBe('Bias Vector for MLP Up Matrix');
        expect(hoverState?.info?.activationData?.label).toBe('Bias Vector for MLP Up Matrix');
        expect(hoverState?.info?.activationData?.stage).toBe('mlp.up.bias');
        expect(hoverState?.info?.activationData?.layerIndex).toBe(4);
        expect(hoverState?.focusState?.activeNodeIds).toContain(biasNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(inputNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(weightNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(outputNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(outputCopyNode?.id);
    });

    it('reuses the 3D W_down tooltip payload for the MLP down weight matrix card', () => {
        const scene = buildScene();
        const nodes = flattenSceneNodes(scene);
        const activationNode = nodes.find((node) => node?.role === 'mlp-activation-output') || null;
        const activationCopyNode = nodes.find((node) => node?.role === 'mlp-activation-output-copy') || null;
        const weightNode = nodes.find((node) => node?.role === 'mlp-down-weight') || null;
        const biasNode = nodes.find((node) => node?.role === 'mlp-down-bias') || null;
        const outputNode = nodes.find((node) => node?.role === 'mlp-down-output') || null;
        const index = createMhsaDetailSceneIndex(scene);
        const hoverState = resolveMhsaDetailHoverState(index, {
            node: weightNode
        });

        expect(hoverState?.label).toBe(MLP_DOWN_TOOLTIP_LABEL);
        expect(hoverState?.info?.activationData?.label).toBe(MLP_DOWN_TOOLTIP_LABEL);
        expect(hoverState?.info?.activationData?.stage).toBe('mlp.down');
        expect(hoverState?.info?.activationData?.layerIndex).toBe(4);
        expect(hoverState?.focusState?.activeNodeIds).toContain(weightNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(activationNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(activationCopyNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(biasNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(outputNode?.id);
    });

    it('shows a 2D-only tooltip for the MLP down bias vector', () => {
        const scene = buildScene();
        const nodes = flattenSceneNodes(scene);
        const activationNode = nodes.find((node) => node?.role === 'mlp-activation-output') || null;
        const activationCopyNode = nodes.find((node) => node?.role === 'mlp-activation-output-copy') || null;
        const weightNode = nodes.find((node) => node?.role === 'mlp-down-weight') || null;
        const biasNode = nodes.find((node) => node?.role === 'mlp-down-bias') || null;
        const outputNode = nodes.find((node) => node?.role === 'mlp-down-output') || null;
        const index = createMhsaDetailSceneIndex(scene);
        const hoverState = resolveMhsaDetailHoverState(index, {
            node: biasNode,
            rowHit: {
                rowIndex: 0,
                rowItem: biasNode?.rowItems?.[0]
            }
        });

        expect(hoverState?.label).toBe(MLP_DOWN_BIAS_TOOLTIP_LABEL);
        expect(hoverState?.info?.activationData?.label).toBe(MLP_DOWN_BIAS_TOOLTIP_LABEL);
        expect(hoverState?.info?.activationData?.stage).toBe('mlp.down.bias');
        expect(hoverState?.info?.activationData?.layerIndex).toBe(4);
        expect(hoverState?.focusState?.activeNodeIds).toContain(biasNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(activationNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(activationCopyNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(weightNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(outputNode?.id);
    });
});
