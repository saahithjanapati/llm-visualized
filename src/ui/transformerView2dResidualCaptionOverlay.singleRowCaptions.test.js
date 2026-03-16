// @vitest-environment jsdom

import { describe, expect, it } from 'vitest';

import { D_HEAD, D_MODEL } from './selectionPanelConstants.js';
import { createTransformerView2dResidualCaptionOverlay } from './transformerView2dResidualCaptionOverlay.js';
import { buildSceneLayout } from '../view2d/layout/buildSceneLayout.js';
import { buildLayerNormDetailSceneModel } from '../view2d/model/buildLayerNormDetailSceneModel.js';
import { buildMlpDetailSceneModel } from '../view2d/model/buildMlpDetailSceneModel.js';
import { buildOutputProjectionDetailSceneModel } from '../view2d/model/buildOutputProjectionDetailSceneModel.js';
import { flattenSceneNodes } from '../view2d/schema/sceneTypes.js';

function createResidualValues(seed = 0, length = D_MODEL) {
    return Array.from({ length }, (_, index) => Number((seed + (index * 0.01)).toFixed(4)));
}

function queryCaptionItem(nodeId = '') {
    return document.querySelector(`[data-node-id="${nodeId}"]`);
}

function createCaptionTestCanvas({
    width = 1400,
    height = 900
} = {}) {
    const parent = document.createElement('div');
    document.body.appendChild(parent);
    const canvas = document.createElement('canvas');
    parent.appendChild(canvas);
    Object.defineProperties(canvas, {
        clientWidth: { configurable: true, value: width },
        clientHeight: { configurable: true, value: height },
        offsetLeft: { configurable: true, value: 0 },
        offsetTop: { configurable: true, value: 0 }
    });
    canvas.width = width;
    canvas.height = height;
    return {
        parent,
        canvas
    };
}

function buildMlpCaptionFixtures(tokenCount = 1) {
    const tokenRefs = Array.from({ length: tokenCount }, (_, rowIndex) => ({
        rowIndex,
        tokenIndex: rowIndex,
        tokenLabel: `Token ${rowIndex + 1}`
    }));
    const scene = buildMlpDetailSceneModel({
        activationSource: {
            getLayerLn2(_layerIndex = 0, _kind = 'shift', tokenIndex = 0, targetLength = D_MODEL) {
                return createResidualValues(tokenIndex * 0.1, targetLength);
            }
        },
        mlpDetailTarget: {
            layerIndex: 2
        },
        tokenRefs
    });
    const layout = buildSceneLayout(scene);
    const nodes = flattenSceneNodes(scene);
    const inputNode = nodes.find((node) => node?.role === 'projection-source-xln') || null;
    const upBiasNode = nodes.find((node) => node?.role === 'mlp-up-bias') || null;
    const downBiasNode = nodes.find((node) => node?.role === 'mlp-down-bias') || null;
    const surface = createCaptionTestCanvas({});
    const overlay = createTransformerView2dResidualCaptionOverlay({
        documentRef: document,
        parent: surface.parent
    });
    return {
        scene,
        layout,
        inputNode,
        upBiasNode,
        downBiasNode,
        canvas: surface.canvas,
        overlay,
        cleanup() {
            overlay.destroy();
            surface.parent.remove();
        }
    };
}

function buildOutputProjectionCaptionFixtures(tokenCount = 1) {
    const tokenRefs = Array.from({ length: tokenCount }, (_, rowIndex) => ({
        rowIndex,
        tokenIndex: rowIndex,
        tokenLabel: `Token ${rowIndex + 1}`
    }));
    const scene = buildOutputProjectionDetailSceneModel({
        activationSource: {
            getAttentionWeightedSum(_layerIndex = 0, headIndex = 0, tokenIndex = 0, targetLength = D_HEAD) {
                return Array.from({ length: targetLength }, (_, index) => Number(
                    ((headIndex * 0.15) + (tokenIndex * 0.07) + (index * 0.01)).toFixed(4)
                ));
            },
            getAttentionOutputProjection(_layerIndex = 0, tokenIndex = 0, targetLength = D_MODEL) {
                return Array.from({ length: targetLength }, (_, index) => Number(
                    ((tokenIndex * 0.05) + (index * 0.004)).toFixed(4)
                ));
            }
        },
        outputProjectionDetailTarget: {
            layerIndex: 2
        },
        tokenRefs
    });
    const layout = buildSceneLayout(scene);
    const nodes = flattenSceneNodes(scene);
    const headMatrixNode = nodes.find((node) => (
        node?.role === 'head-output-matrix'
        && Number(node?.semantic?.headIndex) === 0
    )) || null;
    const projectionBiasNode = nodes.find((node) => node?.role === 'projection-bias') || null;
    const surface = createCaptionTestCanvas({});
    const overlay = createTransformerView2dResidualCaptionOverlay({
        documentRef: document,
        parent: surface.parent
    });
    return {
        scene,
        layout,
        headMatrixNode,
        projectionBiasNode,
        canvas: surface.canvas,
        overlay,
        cleanup() {
            overlay.destroy();
            surface.parent.remove();
        }
    };
}

function buildLayerNormCaptionFixtures(tokenCount = 1) {
    const tokenRefs = Array.from({ length: tokenCount }, (_, rowIndex) => ({
        rowIndex,
        tokenIndex: rowIndex,
        tokenLabel: `Token ${rowIndex + 1}`
    }));
    const scene = buildLayerNormDetailSceneModel({
        activationSource: {
            getLayerIncoming(_layerIndex = 0, tokenIndex = 0, targetLength = D_MODEL) {
                return createResidualValues(tokenIndex * 0.1, targetLength);
            },
            getLayerLn1(_layerIndex = 0, stage = 'norm', tokenIndex = 0, targetLength = D_MODEL) {
                const stageSeed = stage === 'scale' ? 0.2 : (stage === 'shift' ? 0.3 : 0.1);
                return createResidualValues(stageSeed + (tokenIndex * 0.1), targetLength);
            }
        },
        layerNormDetailTarget: {
            layerNormKind: 'ln1',
            layerIndex: 2
        },
        tokenRefs,
        layerCount: 12
    });
    const layout = buildSceneLayout(scene);
    const nodes = flattenSceneNodes(scene);
    const scaleNode = nodes.find((node) => node?.role === 'layer-norm-scale') || null;
    const shiftNode = nodes.find((node) => node?.role === 'layer-norm-shift') || null;
    const surface = createCaptionTestCanvas({});
    const overlay = createTransformerView2dResidualCaptionOverlay({
        documentRef: document,
        parent: surface.parent
    });
    return {
        scene,
        layout,
        scaleNode,
        shiftNode,
        canvas: surface.canvas,
        overlay,
        cleanup() {
            overlay.destroy();
            surface.parent.remove();
        }
    };
}

function syncOverlayAroundNode({
    overlay = null,
    scene = null,
    layout = null,
    canvas = null,
    nodeId = '',
    scale = 0.34,
    targetX = 72,
    targetY = 60
} = {}) {
    const entry = layout?.registry?.getNodeEntry(nodeId);
    const offsetX = targetX - ((Number(entry?.contentBounds?.x) || 0) * scale);
    const offsetY = targetY - ((Number(entry?.contentBounds?.y) || 0) * scale);
    overlay?.sync({
        scene,
        layout,
        canvas,
        projectBounds: (bounds) => ({
            x: (bounds.x * scale) + offsetX,
            y: (bounds.y * scale) + offsetY,
            width: bounds.width * scale,
            height: bounds.height * scale
        }),
        visible: true,
        enabled: true
    });
}

describe('transformerView2dResidualCaptionOverlay single-row vector captions', () => {
    it('keeps the single-row MLP x_ln caption legible at overview-like scales', () => {
        const fixtures = buildMlpCaptionFixtures(1);

        try {
            syncOverlayAroundNode({
                overlay: fixtures.overlay,
                scene: fixtures.scene,
                layout: fixtures.layout,
                canvas: fixtures.canvas,
                nodeId: fixtures.inputNode?.id || '',
                scale: 0.34
            });

            const captionItem = queryCaptionItem(fixtures.inputNode?.id || '');
            const labelSize = Number.parseFloat(
                captionItem?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            const dimensionsSize = Number.parseFloat(
                captionItem?.style.getPropertyValue('--detail-transformer-view2d-caption-dimensions-size') || '0'
            );

            expect(captionItem).toBeTruthy();
            expect(captionItem?.hidden).toBe(false);
            expect(labelSize).toBeGreaterThanOrEqual(12);
            expect(dimensionsSize).toBeGreaterThanOrEqual(10.5);
        } finally {
            fixtures.cleanup();
        }
    });

    it('keeps the single-row output-projection head caption legible at overview-like scales', () => {
        const fixtures = buildOutputProjectionCaptionFixtures(1);

        try {
            syncOverlayAroundNode({
                overlay: fixtures.overlay,
                scene: fixtures.scene,
                layout: fixtures.layout,
                canvas: fixtures.canvas,
                nodeId: fixtures.headMatrixNode?.id || '',
                scale: 0.08,
                targetX: 56,
                targetY: 52
            });

            const captionItem = queryCaptionItem(fixtures.headMatrixNode?.id || '');
            const labelSize = Number.parseFloat(
                captionItem?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            const dimensionsSize = Number.parseFloat(
                captionItem?.style.getPropertyValue('--detail-transformer-view2d-caption-dimensions-size') || '0'
            );

            expect(captionItem).toBeTruthy();
            expect(captionItem?.hidden).toBe(false);
            expect(labelSize).toBeGreaterThanOrEqual(12);
            expect(dimensionsSize).toBeGreaterThanOrEqual(10.5);
        } finally {
            fixtures.cleanup();
        }
    });

    it('lets MLP and output-projection bias terms shrink with scene zoom', () => {
        const outputProjectionFixtures = buildOutputProjectionCaptionFixtures(1);
        const mlpFixtures = buildMlpCaptionFixtures(1);

        try {
            syncOverlayAroundNode({
                overlay: outputProjectionFixtures.overlay,
                scene: outputProjectionFixtures.scene,
                layout: outputProjectionFixtures.layout,
                canvas: outputProjectionFixtures.canvas,
                nodeId: outputProjectionFixtures.projectionBiasNode?.id || '',
                scale: 0.08,
                targetX: 56,
                targetY: 52
            });
            const outputProjectionBiasZoomedOutSize = Number.parseFloat(
                queryCaptionItem(outputProjectionFixtures.projectionBiasNode?.id || '')?.style
                    .getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            syncOverlayAroundNode({
                overlay: outputProjectionFixtures.overlay,
                scene: outputProjectionFixtures.scene,
                layout: outputProjectionFixtures.layout,
                canvas: outputProjectionFixtures.canvas,
                nodeId: outputProjectionFixtures.projectionBiasNode?.id || '',
                scale: 0.2,
                targetX: 56,
                targetY: 52
            });
            const outputProjectionBiasZoomedInSize = Number.parseFloat(
                queryCaptionItem(outputProjectionFixtures.projectionBiasNode?.id || '')?.style
                    .getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );

            syncOverlayAroundNode({
                overlay: mlpFixtures.overlay,
                scene: mlpFixtures.scene,
                layout: mlpFixtures.layout,
                canvas: mlpFixtures.canvas,
                nodeId: mlpFixtures.upBiasNode?.id || '',
                scale: 0.34
            });
            const mlpUpBiasZoomedOutSize = Number.parseFloat(
                queryCaptionItem(mlpFixtures.upBiasNode?.id || '')?.style
                    .getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            syncOverlayAroundNode({
                overlay: mlpFixtures.overlay,
                scene: mlpFixtures.scene,
                layout: mlpFixtures.layout,
                canvas: mlpFixtures.canvas,
                nodeId: mlpFixtures.downBiasNode?.id || '',
                scale: 0.34
            });
            const mlpDownBiasZoomedOutSize = Number.parseFloat(
                queryCaptionItem(mlpFixtures.downBiasNode?.id || '')?.style
                    .getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            syncOverlayAroundNode({
                overlay: mlpFixtures.overlay,
                scene: mlpFixtures.scene,
                layout: mlpFixtures.layout,
                canvas: mlpFixtures.canvas,
                nodeId: mlpFixtures.upBiasNode?.id || '',
                scale: 1
            });
            const mlpUpBiasZoomedInSize = Number.parseFloat(
                queryCaptionItem(mlpFixtures.upBiasNode?.id || '')?.style
                    .getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            syncOverlayAroundNode({
                overlay: mlpFixtures.overlay,
                scene: mlpFixtures.scene,
                layout: mlpFixtures.layout,
                canvas: mlpFixtures.canvas,
                nodeId: mlpFixtures.downBiasNode?.id || '',
                scale: 1
            });
            const mlpDownBiasZoomedInSize = Number.parseFloat(
                queryCaptionItem(mlpFixtures.downBiasNode?.id || '')?.style
                    .getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );

            expect(outputProjectionBiasZoomedOutSize).toBeGreaterThan(0);
            expect(outputProjectionBiasZoomedInSize).toBeGreaterThan(outputProjectionBiasZoomedOutSize);
            expect(mlpUpBiasZoomedOutSize).toBeGreaterThan(0);
            expect(mlpUpBiasZoomedInSize).toBeGreaterThan(mlpUpBiasZoomedOutSize);
            expect(mlpDownBiasZoomedOutSize).toBeGreaterThan(0);
            expect(mlpDownBiasZoomedInSize).toBeGreaterThan(mlpDownBiasZoomedOutSize);
        } finally {
            outputProjectionFixtures.cleanup();
            mlpFixtures.cleanup();
        }
    });

    it('lets layer-norm gamma and beta terms shrink with scene zoom', () => {
        const fixtures = buildLayerNormCaptionFixtures(1);

        try {
            syncOverlayAroundNode({
                overlay: fixtures.overlay,
                scene: fixtures.scene,
                layout: fixtures.layout,
                canvas: fixtures.canvas,
                nodeId: fixtures.scaleNode?.id || '',
                scale: 0.34
            });
            const scaleLabelZoomedOutSize = Number.parseFloat(
                queryCaptionItem(fixtures.scaleNode?.id || '')?.style
                    .getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            syncOverlayAroundNode({
                overlay: fixtures.overlay,
                scene: fixtures.scene,
                layout: fixtures.layout,
                canvas: fixtures.canvas,
                nodeId: fixtures.shiftNode?.id || '',
                scale: 0.34
            });
            const shiftLabelZoomedOutSize = Number.parseFloat(
                queryCaptionItem(fixtures.shiftNode?.id || '')?.style
                    .getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            syncOverlayAroundNode({
                overlay: fixtures.overlay,
                scene: fixtures.scene,
                layout: fixtures.layout,
                canvas: fixtures.canvas,
                nodeId: fixtures.scaleNode?.id || '',
                scale: 1
            });
            const scaleLabelZoomedInSize = Number.parseFloat(
                queryCaptionItem(fixtures.scaleNode?.id || '')?.style
                    .getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            syncOverlayAroundNode({
                overlay: fixtures.overlay,
                scene: fixtures.scene,
                layout: fixtures.layout,
                canvas: fixtures.canvas,
                nodeId: fixtures.shiftNode?.id || '',
                scale: 1
            });
            const shiftLabelZoomedInSize = Number.parseFloat(
                queryCaptionItem(fixtures.shiftNode?.id || '')?.style
                    .getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );

            expect(scaleLabelZoomedOutSize).toBeGreaterThan(0);
            expect(scaleLabelZoomedInSize).toBeGreaterThan(scaleLabelZoomedOutSize);
            expect(shiftLabelZoomedOutSize).toBeGreaterThan(0);
            expect(shiftLabelZoomedInSize).toBeGreaterThan(shiftLabelZoomedOutSize);
        } finally {
            fixtures.cleanup();
        }
    });

    it('applies extra overlay role scaling to output-projection, MLP, and layer-norm parameter labels', () => {
        const outputProjectionFixtures = buildOutputProjectionCaptionFixtures(1);
        const mlpFixtures = buildMlpCaptionFixtures(1);
        const layerNormFixtures = buildLayerNormCaptionFixtures(1);

        try {
            syncOverlayAroundNode({
                overlay: outputProjectionFixtures.overlay,
                scene: outputProjectionFixtures.scene,
                layout: outputProjectionFixtures.layout,
                canvas: outputProjectionFixtures.canvas,
                nodeId: outputProjectionFixtures.projectionBiasNode?.id || '',
                scale: 1
            });
            syncOverlayAroundNode({
                overlay: mlpFixtures.overlay,
                scene: mlpFixtures.scene,
                layout: mlpFixtures.layout,
                canvas: mlpFixtures.canvas,
                nodeId: mlpFixtures.upBiasNode?.id || '',
                scale: 1
            });
            syncOverlayAroundNode({
                overlay: layerNormFixtures.overlay,
                scene: layerNormFixtures.scene,
                layout: layerNormFixtures.layout,
                canvas: layerNormFixtures.canvas,
                nodeId: layerNormFixtures.scaleNode?.id || '',
                scale: 1
            });

            const outputProjectionBiasItem = queryCaptionItem(outputProjectionFixtures.projectionBiasNode?.id || '');
            const mlpUpBiasItem = queryCaptionItem(mlpFixtures.upBiasNode?.id || '');
            const mlpDownBiasItem = queryCaptionItem(mlpFixtures.downBiasNode?.id || '');
            const layerNormScaleItem = queryCaptionItem(layerNormFixtures.scaleNode?.id || '');
            const layerNormShiftItem = queryCaptionItem(layerNormFixtures.shiftNode?.id || '');

            const outputProjectionBiasRoleScale = Number.parseFloat(
                outputProjectionBiasItem?.style.getPropertyValue('--detail-transformer-view2d-caption-label-role-scale') || '0'
            );
            const outputProjectionBiasDimensionsRoleScale = Number.parseFloat(
                outputProjectionBiasItem?.style.getPropertyValue('--detail-transformer-view2d-caption-dimensions-role-scale') || '0'
            );
            const mlpUpBiasRoleScale = Number.parseFloat(
                mlpUpBiasItem?.style.getPropertyValue('--detail-transformer-view2d-caption-label-role-scale') || '0'
            );
            const mlpDownBiasRoleScale = Number.parseFloat(
                mlpDownBiasItem?.style.getPropertyValue('--detail-transformer-view2d-caption-label-role-scale') || '0'
            );
            const mlpBiasDimensionsRoleScale = Number.parseFloat(
                mlpUpBiasItem?.style.getPropertyValue('--detail-transformer-view2d-caption-dimensions-role-scale') || '0'
            );
            const outputProjectionBiasSubscriptScale = Number.parseFloat(
                outputProjectionBiasItem?.style.getPropertyValue('--detail-transformer-view2d-caption-katex-subscript-scale') || '0'
            );
            const mlpBiasSubscriptScale = Number.parseFloat(
                mlpUpBiasItem?.style.getPropertyValue('--detail-transformer-view2d-caption-katex-subscript-scale') || '0'
            );
            const layerNormScaleRoleScale = Number.parseFloat(
                layerNormScaleItem?.style.getPropertyValue('--detail-transformer-view2d-caption-label-role-scale') || '0'
            );
            const layerNormScaleDimensionsRoleScale = Number.parseFloat(
                layerNormScaleItem?.style.getPropertyValue('--detail-transformer-view2d-caption-dimensions-role-scale') || '0'
            );
            const layerNormShiftRoleScale = Number.parseFloat(
                layerNormShiftItem?.style.getPropertyValue('--detail-transformer-view2d-caption-label-role-scale') || '0'
            );
            const layerNormShiftDimensionsRoleScale = Number.parseFloat(
                layerNormShiftItem?.style.getPropertyValue('--detail-transformer-view2d-caption-dimensions-role-scale') || '0'
            );
            const outputProjectionBiasCaptionScale = Number(
                outputProjectionFixtures.projectionBiasNode?.metadata?.caption?.labelScale || 0
            );
            const mlpBiasCaptionScale = Number(
                mlpFixtures.upBiasNode?.metadata?.caption?.labelScale || 0
            );
            const layerNormCaptionScale = Number(
                layerNormFixtures.scaleNode?.metadata?.caption?.labelScale || 0
            );

            expect(outputProjectionBiasRoleScale).toBeGreaterThan(outputProjectionBiasCaptionScale);
            expect(outputProjectionBiasDimensionsRoleScale).toBeGreaterThan(1);
            expect(mlpUpBiasRoleScale).toBeGreaterThan(mlpBiasCaptionScale);
            expect(mlpDownBiasRoleScale).toBe(mlpUpBiasRoleScale);
            expect(mlpBiasDimensionsRoleScale).toBeGreaterThan(1);
            expect(outputProjectionBiasSubscriptScale).toBeLessThan(0.8);
            expect(mlpBiasSubscriptScale).toBeLessThan(0.8);
            expect(layerNormScaleRoleScale).toBeGreaterThan(layerNormCaptionScale);
            expect(layerNormShiftRoleScale).toBe(layerNormScaleRoleScale);
            expect(layerNormScaleDimensionsRoleScale).toBeGreaterThan(1);
            expect(layerNormShiftDimensionsRoleScale).toBe(layerNormScaleDimensionsRoleScale);
        } finally {
            outputProjectionFixtures.cleanup();
            mlpFixtures.cleanup();
            layerNormFixtures.cleanup();
        }
    });
});
