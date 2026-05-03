import { describe, expect, it } from 'vitest';

import {
    createAnchorRef,
    createConnectorNode,
    createGroupNode,
    createMatrixNode,
    createOperatorNode,
    createSceneModel,
    createTextNode,
    flattenSceneNodes,
    VIEW2D_ANCHOR_SIDES,
    VIEW2D_CONNECTOR_ROUTES,
    VIEW2D_LAYOUT_DIRECTIONS,
    VIEW2D_MATRIX_PRESENTATIONS,
    VIEW2D_MATRIX_SHAPES
} from '../../schema/sceneTypes.js';
import { buildSceneLayout } from '../../layout/buildSceneLayout.js';
import { D_HEAD, D_MODEL } from '../../../ui/selectionPanelConstants.js';
import {
    createMhsaDetailSceneIndex,
    resolveMhsaDetailHoverState
} from '../../mhsaDetailInteraction.js';
import { buildMlpDetailSceneModel } from '../../model/buildMlpDetailSceneModel.js';
import { buildMhsaSceneModel } from '../../model/buildMhsaSceneModel.js';
import { VIEW2D_VECTOR_STRIP_VARIANT } from '../../shared/vectorStrip.js';
import { VIEW2D_STYLE_KEYS } from '../../theme/visualTokens.js';
import { CanvasSceneRenderer } from './CanvasSceneRenderer.js';

function createMockContext() {
    const stateStack = [];
    let currentPath = [];
    return {
        operations: [],
        font: '',
        fillStyle: '',
        strokeStyle: '',
        lineWidth: 0,
        globalAlpha: 1,
        globalCompositeOperation: 'source-over',
        filter: 'none',
        shadowBlur: 0,
        shadowColor: 'transparent',
        textAlign: 'left',
        textBaseline: 'alphabetic',
        currentScaleX: 1,
        currentScaleY: 1,
        currentTranslateX: 0,
        currentTranslateY: 0,
        save() {
            stateStack.push({
                globalAlpha: this.globalAlpha,
                fillStyle: this.fillStyle,
                strokeStyle: this.strokeStyle,
                lineWidth: this.lineWidth,
                globalCompositeOperation: this.globalCompositeOperation,
                filter: this.filter,
                shadowBlur: this.shadowBlur,
                shadowColor: this.shadowColor,
                textAlign: this.textAlign,
                textBaseline: this.textBaseline,
                font: this.font,
                currentScaleX: this.currentScaleX,
                currentScaleY: this.currentScaleY,
                currentTranslateX: this.currentTranslateX,
                currentTranslateY: this.currentTranslateY
            });
        },
        restore() {
            const state = stateStack.pop();
            if (!state) return;
            this.globalAlpha = state.globalAlpha;
            this.fillStyle = state.fillStyle;
            this.strokeStyle = state.strokeStyle;
            this.lineWidth = state.lineWidth;
            this.globalCompositeOperation = state.globalCompositeOperation;
            this.filter = state.filter;
            this.shadowBlur = state.shadowBlur;
            this.shadowColor = state.shadowColor;
            this.textAlign = state.textAlign;
            this.textBaseline = state.textBaseline;
            this.font = state.font;
            this.currentScaleX = state.currentScaleX;
            this.currentScaleY = state.currentScaleY;
            this.currentTranslateX = state.currentTranslateX;
            this.currentTranslateY = state.currentTranslateY;
        },
        setTransform(a = 1, _b = 0, _c = 0, d = 1, e = 0, f = 0) {
            this.currentScaleX = Number.isFinite(a) ? a : 1;
            this.currentScaleY = Number.isFinite(d) ? d : 1;
            this.currentTranslateX = Number.isFinite(e) ? e : 0;
            this.currentTranslateY = Number.isFinite(f) ? f : 0;
        },
        clearRect() {},
        fillRect(x, y, width, height) {
            this.operations.push({
                type: 'fillRect',
                x,
                y,
                width,
                height,
                fillStyle: this.fillStyle,
                globalAlpha: this.globalAlpha,
                filter: this.filter
            });
        },
        translate() {},
        scale(x = 1, y = 1) {
            this.currentScaleX *= Number.isFinite(x) ? x : 1;
            this.currentScaleY *= Number.isFinite(y) ? y : 1;
        },
        beginPath() {
            currentPath = [];
        },
        rect(x, y, width, height) {
            currentPath.push({ type: 'rect', x, y, width, height });
        },
        moveTo(x, y) {
            currentPath.push({ type: 'moveTo', x, y });
        },
        lineTo(x, y) {
            currentPath.push({ type: 'lineTo', x, y });
        },
        arcTo(x1, y1, x2, y2, radius) {
            currentPath.push({ type: 'arcTo', x1, y1, x2, y2, radius });
        },
        quadraticCurveTo(cpx, cpy, x, y) {
            currentPath.push({ type: 'quadraticCurveTo', cpx, cpy, x, y });
        },
        closePath() {
            currentPath.push({ type: 'closePath' });
        },
        fill() {
            this.operations.push({
                type: 'fill',
                globalAlpha: this.globalAlpha,
                filter: this.filter,
                fillStyle: this.fillStyle,
                transformScaleX: this.currentScaleX,
                transformScaleY: this.currentScaleY,
                path: currentPath.map((entry) => ({ ...entry }))
            });
        },
        stroke() {
            this.operations.push({
                type: 'stroke',
                globalAlpha: this.globalAlpha,
                filter: this.filter,
                strokeStyle: this.strokeStyle,
                lineWidth: this.lineWidth,
                transformScaleX: this.currentScaleX,
                transformScaleY: this.currentScaleY,
                effectiveLineWidth: this.lineWidth * this.currentScaleY,
                path: currentPath.map((entry) => ({ ...entry }))
            });
        },
        strokeRect(x, y, width, height) {
            this.operations.push({
                type: 'strokeRect',
                x,
                y,
                width,
                height,
                strokeStyle: this.strokeStyle,
                lineWidth: this.lineWidth,
                globalAlpha: this.globalAlpha,
                filter: this.filter
            });
        },
        clip() {},
        fillText(text, x, y) {
            const fontPx = parseFontPx(this.font);
            this.operations.push({
                type: 'fillText',
                text,
                x,
                y,
                font: this.font,
                effectiveFontPx: fontPx * this.currentScaleY
            });
        },
        measureText(text) {
            const fontSizeMatch = String(this.font || '').match(/([0-9.]+)px/);
            const fontSize = Number.parseFloat(fontSizeMatch?.[1] || '12');
            return {
                width: Math.max(1, String(text || ' ').length) * fontSize * 0.6,
                actualBoundingBoxLeft: 0,
                actualBoundingBoxRight: Math.max(1, String(text || ' ').length) * fontSize * 0.6,
                actualBoundingBoxAscent: fontSize * 0.8,
                actualBoundingBoxDescent: fontSize * 0.3
            };
        },
        createLinearGradient() {
            return {
                stops: [],
                addColorStop(offset, color) {
                    this.stops.push({ offset, color });
                }
            };
        },
        createRadialGradient() {
            return {
                stops: [],
                addColorStop(offset, color) {
                    this.stops.push({ offset, color });
                }
            };
        },
        drawImage(source, x = 0, y = 0) {
            this.operations.push({
                type: 'drawImage',
                source,
                x,
                y,
                transformScaleX: this.currentScaleX,
                transformScaleY: this.currentScaleY,
                transformTranslateX: this.currentTranslateX,
                transformTranslateY: this.currentTranslateY
            });
        }
    };
}

function createMockCanvas(ctx, width = 400, height = 240) {
    return {
        width,
        height,
        clientWidth: width,
        clientHeight: height,
        getContext(kind) {
            return kind === '2d' ? ctx : null;
        },
        getBoundingClientRect() {
            return {
                width,
                height,
                left: 0,
                top: 0,
                right: width,
                bottom: height
            };
        }
    };
}

function parseFontPx(font = '') {
    return Number.parseFloat(String(font).match(/([0-9.]+)px/)?.[1] || '0');
}

function createMhsaDetailVectorValues(seed = 0) {
    return Array.from({ length: D_HEAD }, (_, index) => Number((seed + (index * 0.01)).toFixed(4)));
}

function createMhsaDetailTokenLabels(count = 2) {
    return Array.from({ length: count }, (_, index) => `Token ${String.fromCharCode(65 + index)}`);
}

function createMhsaDetailBaseRows(tokenLabels = []) {
    return tokenLabels.map((tokenLabel, rowIndex) => ({
        rowIndex,
        tokenIndex: rowIndex,
        tokenLabel,
        rawValues: createMhsaDetailVectorValues(rowIndex),
        gradientCss: `rgba(${120 + (rowIndex * 16)}, 220, 255, 0.9)`
    }));
}

function createMhsaDetailProjectionOutputRows(label = 'Q', tokenLabels = []) {
    return tokenLabels.map((tokenLabel, rowIndex) => ({
        rowIndex,
        tokenIndex: rowIndex,
        tokenLabel,
        rawValue: Number((rowIndex + 0.25).toFixed(3)),
        rawValues: createMhsaDetailVectorValues(rowIndex + 1),
        gradientCss: `rgba(${180 - (rowIndex * 20)}, ${140 + (rowIndex * 24)}, 255, 0.88)`,
        title: `${tokenLabel}: ${label} vector`
    }));
}

function createMhsaDetailGridRows(fillCss = 'rgba(255, 255, 255, 0.28)', tokenLabels = [], {
    stageKey = 'pre'
} = {}) {
    return tokenLabels.map((tokenLabel, rowIndex) => ({
        rowIndex,
        tokenIndex: rowIndex,
        tokenLabel,
        cells: tokenLabels.map((colLabel, colIndex) => {
            const isMasked = colIndex > rowIndex;
            const preScore = Number(((rowIndex + 1) * (colIndex + 1) * 0.125).toFixed(3));
            const postScore = Number((((rowIndex + 1) + (colIndex + 1)) * 0.05).toFixed(3));
            const maskValue = isMasked ? Number.NEGATIVE_INFINITY : 0;
            return {
                rowIndex,
                colIndex,
                rowTokenLabel: tokenLabel,
                colTokenLabel: colLabel,
                queryTokenIndex: rowIndex,
                keyTokenIndex: colIndex,
                queryTokenLabel: tokenLabel,
                keyTokenLabel: colLabel,
                preScore,
                postScore: isMasked ? 0 : postScore,
                maskValue,
                rawValue: stageKey === 'mask'
                    ? maskValue
                    : (stageKey === 'post' ? (isMasked ? null : postScore) : preScore),
                fillCss,
                isMasked,
                isEmpty: stageKey === 'post' ? isMasked : false,
                title: `${tokenLabel} -> ${colLabel}`
            };
        }),
        hasAnyValue: true
    }));
}

function createMhsaDetailPreviewData(tokenCount = 2) {
    const tokenLabels = createMhsaDetailTokenLabels(tokenCount);
    const rows = createMhsaDetailBaseRows(tokenLabels);
    const queryOutputRows = createMhsaDetailProjectionOutputRows('Q', tokenLabels);
    const keyOutputRows = createMhsaDetailProjectionOutputRows('K', tokenLabels);
    const valueOutputRows = createMhsaDetailProjectionOutputRows('V', tokenLabels);
    const createProjection = (kind, outputLabelTex, outputRows) => ({
        kind,
        weightLabelTex: `W_${kind.toLowerCase()}`,
        biasLabelTex: `b_${kind.toLowerCase()}`,
        outputLabelTex,
        weightRowCount: D_MODEL,
        weightColumnCount: D_HEAD,
        biasValue: 0.15,
        biasVectorGradientCss: 'rgba(255, 255, 255, 0.2)',
        outputRowCount: outputRows.length,
        outputColumnCount: D_HEAD,
        outputRows
    });
    const attentionGridRows = createMhsaDetailGridRows('rgba(255, 255, 255, 0.28)', tokenLabels, {
        stageKey: 'pre'
    });
    const maskGridRows = createMhsaDetailGridRows('rgba(0, 0, 0, 0.94)', tokenLabels, {
        stageKey: 'mask'
    });
    const postGridRows = createMhsaDetailGridRows('rgba(160, 220, 255, 0.34)', tokenLabels, {
        stageKey: 'post'
    });

    return {
        rowCount: rows.length,
        columnCount: D_MODEL,
        bandCount: 12,
        sampleStep: 64,
        rows,
        projections: [
            createProjection('Q', 'Q', queryOutputRows),
            createProjection('K', 'K', keyOutputRows),
            createProjection('V', 'V', valueOutputRows)
        ],
        attentionScoreStage: {
            queryLabelTex: 'Q',
            queryRowCount: queryOutputRows.length,
            queryColumnCount: D_HEAD,
            queryRows: queryOutputRows,
            transposeLabelTex: 'K^{\\mathsf{T}}',
            transposeRowCount: D_HEAD,
            transposeColumnCount: keyOutputRows.length,
            transposeColumns: keyOutputRows.map((rowData) => ({
                colIndex: rowData.rowIndex,
                tokenIndex: rowData.tokenIndex,
                rawValue: rowData.rawValue,
                rawValues: rowData.rawValues,
                fillCss: rowData.gradientCss,
                tokenLabel: rowData.tokenLabel
            })),
            scaleLabelTex: '\\sqrt{d_{\\mathrm{head}}}',
            outputLabelTex: 'A_{\\mathrm{pre}}',
            outputRowCount: attentionGridRows.length,
            outputColumnCount: attentionGridRows.length,
            outputRows: attentionGridRows,
            maskLabelTex: 'M_{\\mathrm{causal}}',
            maskRows: maskGridRows,
            softmaxLabelTex: '\\mathrm{softmax}',
            postLabelTex: 'A_{\\mathrm{post}}',
            postRowCount: postGridRows.length,
            postColumnCount: postGridRows.length,
            postRows: postGridRows,
            valueLabelTex: 'V',
            valueRowCount: valueOutputRows.length,
            valueColumnCount: D_HEAD,
            valueRows: valueOutputRows,
            headOutputLabelTex: 'H_i',
            headOutputRowCount: valueOutputRows.length,
            headOutputColumnCount: D_HEAD,
            headOutputRows: valueOutputRows
        }
    };
}

function resolveOpsInsideBounds(operations = [], bounds = null) {
    if (!bounds) return [];
    const minX = Number(bounds.x) || 0;
    const minY = Number(bounds.y) || 0;
    const maxX = minX + (Number(bounds.width) || 0);
    const maxY = minY + (Number(bounds.height) || 0);
    const containsRectCenter = (x = 0, y = 0, width = 0, height = 0) => {
        const centerX = x + (width / 2);
        const centerY = y + (height / 2);
        return centerX >= minX && centerX <= maxX && centerY >= minY && centerY <= maxY;
    };
    return operations.filter((entry) => {
        if (entry.type === 'fillRect') {
            return containsRectCenter(
                Number(entry.x) || 0,
                Number(entry.y) || 0,
                Number(entry.width) || 0,
                Number(entry.height) || 0
            );
        }
        return false;
    });
}

function findHorizontalArrowShaft(operations = [], bounds = null, {
    direction = 'right',
    yTolerance = 1.05
} = {}) {
    if (!bounds) return null;
    const expectedCenterY = bounds.y + (bounds.height * 0.5);
    const isLeft = String(direction || '').trim().toLowerCase() === 'left';
    return operations.find((operation) => {
        if (operation?.type !== 'stroke' || !Array.isArray(operation.path) || operation.path.length !== 2) {
            return false;
        }
        const [move, line] = operation.path;
        if (move?.type !== 'moveTo' || line?.type !== 'lineTo') {
            return false;
        }
        const isMatchingDirection = isLeft
            ? (move.x < line.x && line.x <= (bounds.x + 0.51))
            : (move.x >= ((bounds.x + bounds.width) - 0.51) && line.x > move.x);
        return isMatchingDirection
            && Math.abs(move.y - expectedCenterY) <= yTolerance
            && Math.abs(line.y - expectedCenterY) <= yTolerance;
    }) || null;
}

function findHorizontalArrowHeadFill(operations = [], bounds = null, {
    direction = 'right',
    yTolerance = 1.05
} = {}) {
    if (!bounds) return null;
    const expectedCenterY = bounds.y + (bounds.height * 0.5);
    const isLeft = String(direction || '').trim().toLowerCase() === 'left';
    return operations.find((operation) => {
        if (operation?.type !== 'fill' || !Array.isArray(operation.path) || operation.path.length < 3) {
            return false;
        }
        const [tip, baseA, baseB] = operation.path;
        if (
            tip?.type !== 'moveTo'
            || baseA?.type !== 'lineTo'
            || baseB?.type !== 'lineTo'
        ) {
            return false;
        }
        const baseMidY = (baseA.y + baseB.y) * 0.5;
        const isMatchingDirection = isLeft
            ? tip.x <= (bounds.x + 0.51)
            : tip.x >= ((bounds.x + bounds.width) - 0.51);
        return isMatchingDirection
            && Math.abs(tip.y - expectedCenterY) <= yTolerance
            && Math.abs(baseMidY - expectedCenterY) <= yTolerance;
    }) || null;
}

function measureHorizontalArrowScreenMetrics(operations = [], bounds = null, options = {}) {
    const shaft = findHorizontalArrowShaft(operations, bounds, options);
    const head = findHorizontalArrowHeadFill(operations, bounds, options);
    expect(shaft).toBeTruthy();
    expect(head).toBeTruthy();

    const [shaftStart, shaftEnd] = shaft.path;
    const [tip, baseA, baseB] = head.path;
    const shaftScale = Number(shaft.transformScaleX) || 1;
    const headScaleX = Number(head.transformScaleX) || 1;
    const headScaleY = Number(head.transformScaleY) || 1;
    const baseMidX = (baseA.x + baseB.x) * 0.5;
    const baseMidY = (baseA.y + baseB.y) * 0.5;
    const effectiveHeadLength = Math.hypot(
        (tip.x - baseMidX) * headScaleX,
        (tip.y - baseMidY) * headScaleY
    );
    const effectiveHeadSpan = Math.hypot(
        (baseA.x - baseB.x) * headScaleX,
        (baseA.y - baseB.y) * headScaleY
    );

    return {
        effectiveLength: Math.abs((shaftEnd.x - shaftStart.x) * shaftScale),
        effectiveLineWidth: Number(shaft.effectiveLineWidth) || 0,
        effectiveHeadLength,
        effectiveHeadSpan,
        tipX: Number(tip.x) || 0
    };
}

function findConnectorStrokeByPath(operations = [], pathPoints = [], tolerance = 1.25) {
    if (!Array.isArray(pathPoints) || pathPoints.length < 2) return null;
    const startPoint = pathPoints[0];
    const endPoint = pathPoints[pathPoints.length - 1];
    return operations.find((operation) => {
        if (operation?.type !== 'stroke' || !Array.isArray(operation.path) || operation.path.length < 2) {
            return false;
        }
        const move = operation.path[0];
        const line = operation.path[operation.path.length - 1];
        if (move?.type !== 'moveTo' || line?.type !== 'lineTo') {
            return false;
        }
        return Math.abs(move.x - startPoint.x) <= tolerance
            && Math.abs(move.y - startPoint.y) <= tolerance
            && Math.abs(line.x - endPoint.x) <= tolerance
            && Math.abs(line.y - endPoint.y) <= tolerance;
    }) || null;
}

function findConnectorArrowHeadFillByPath(operations = [], pathPoints = [], tolerance = 1.25) {
    if (!Array.isArray(pathPoints) || pathPoints.length < 2) return null;
    const headPoint = pathPoints[pathPoints.length - 1];
    return operations.find((operation) => {
        if (operation?.type !== 'fill' || !Array.isArray(operation.path) || operation.path.length < 3) {
            return false;
        }
        const [tip, baseA, baseB] = operation.path;
        if (
            tip?.type !== 'moveTo'
            || baseA?.type !== 'lineTo'
            || baseB?.type !== 'lineTo'
        ) {
            return false;
        }
        const baseMidX = (baseA.x + baseB.x) * 0.5;
        return Math.abs(tip.x - headPoint.x) <= tolerance
            && Math.abs(tip.y - headPoint.y) <= tolerance
            && Math.abs(baseMidX - headPoint.x) <= 32;
    }) || null;
}

function measureConnectorArrowScreenMetrics(operations = [], pathPoints = []) {
    const shaft = findConnectorStrokeByPath(operations, pathPoints);
    const head = findConnectorArrowHeadFillByPath(operations, pathPoints);
    expect(shaft).toBeTruthy();
    expect(head).toBeTruthy();

    const shaftStart = shaft.path[0];
    const shaftEnd = shaft.path[shaft.path.length - 1];
    const [tip, baseA, baseB] = head.path;
    const shaftScale = Number(shaft.transformScaleX) || 1;
    const headScaleX = Number(head.transformScaleX) || 1;
    const headScaleY = Number(head.transformScaleY) || 1;
    const baseMidX = (baseA.x + baseB.x) * 0.5;
    const baseMidY = (baseA.y + baseB.y) * 0.5;
    const effectiveHeadLength = Math.hypot(
        (tip.x - baseMidX) * headScaleX,
        (tip.y - baseMidY) * headScaleY
    );
    const effectiveHeadSpan = Math.hypot(
        (baseA.x - baseB.x) * headScaleX,
        (baseA.y - baseB.y) * headScaleY
    );

    return {
        effectiveLength: Math.abs((shaftEnd.x - shaftStart.x) * shaftScale),
        effectiveLineWidth: Number(shaft.effectiveLineWidth) || 0,
        effectiveHeadLength,
        effectiveHeadSpan,
        tipX: Number(tip.x) || 0
    };
}

describe('CanvasSceneRenderer', () => {
    it('applies per-node visual opacity to matrix fills', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx, 480, 320);
        const renderer = new CanvasSceneRenderer({ canvas });
        const scene = createSceneModel({
            nodes: [
                createMatrixNode({
                    role: 'faded-card',
                    semantic: {
                        componentKind: 'test',
                        stage: 'opacity',
                        role: 'faded-card'
                    },
                    label: {
                        tex: 'F',
                        text: 'F'
                    },
                    dimensions: {
                        rows: 1,
                        cols: 1
                    },
                    presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
                    shape: VIEW2D_MATRIX_SHAPES.MATRIX,
                    visual: {
                        styleKey: VIEW2D_STYLE_KEYS.RESIDUAL,
                        background: 'rgba(120, 220, 255, 0.9)',
                        stroke: 'rgba(255, 255, 255, 0.9)',
                        disableCardSurfaceEffects: true,
                        opacity: 0.4
                    },
                    metadata: {
                        card: {
                            width: 120,
                            height: 64,
                            cornerRadius: 10
                        }
                    }
                })
            ]
        });
        renderer.setScene(scene);

        expect(renderer.render({
            width: 480,
            height: 320,
            dpr: 1
        })).toBe(true);

        const visibleFillOps = ctx.operations.filter((entry) => (
            entry.type === 'fill'
            && entry.fillStyle !== '#000'
            && entry.fillStyle !== 'rgba(0, 0, 0, 0)'
        ));

        expect(visibleFillOps.some((entry) => Math.abs((entry.globalAlpha || 0) - 0.4) < 0.01)).toBe(true);
    });

    it('scales MHSA detail operator glyphs with the scene while zooming', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });
        const scene = createSceneModel({
            nodes: [
                createOperatorNode({
                    role: 'attention-equals',
                    semantic: {
                        componentKind: 'mhsa',
                        stage: 'attention',
                        role: 'attention-equals',
                        operatorKey: 'equals'
                    },
                    text: '=',
                    visual: {
                        styleKey: VIEW2D_STYLE_KEYS.OPERATOR
                    }
                })
            ],
            metadata: {
                visualContract: 'selection-panel-mhsa-v1'
            }
        });
        renderer.setScene(scene);

        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);
        const firstDraw = ctx.operations.find((entry) => entry.type === 'fillText' && entry.text === '=');
        expect(firstDraw).toBeTruthy();
        const firstScreenFontPx = Number(firstDraw?.effectiveFontPx || 0);

        ctx.operations = [];
        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            viewportTransform: {
                scale: 2,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);
        const secondDraw = ctx.operations.find((entry) => entry.type === 'fillText' && entry.text === '=');
        expect(secondDraw).toBeTruthy();
        const secondScreenFontPx = Number(secondDraw?.effectiveFontPx || 0);

        expect(firstScreenFontPx).toBeGreaterThan(0);
        expect(secondScreenFontPx).toBeGreaterThan(firstScreenFontPx);
        expect(secondScreenFontPx / Math.max(1, firstScreenFontPx)).toBeCloseTo(2, 1);
    });

    it('reveals canvas text and captions together at the same zoom threshold', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });
        const scene = createSceneModel({
            nodes: [
                createGroupNode({
                    direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
                    gapKey: 'default',
                    children: [
                        createMatrixNode({
                            role: 'caption-card',
                            semantic: {
                                componentKind: 'test',
                                role: 'caption-card'
                            },
                            label: {
                                text: 'Caption'
                            },
                            dimensions: {
                                rows: 1,
                                cols: 1
                            },
                            presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
                            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
                            visual: {
                                styleKey: VIEW2D_STYLE_KEYS.RESIDUAL
                            },
                            metadata: {
                                card: {
                                    width: 72,
                                    height: 40
                                },
                                caption: {
                                    position: 'top',
                                    minScreenHeightPx: 28
                                }
                            }
                        }),
                        createTextNode({
                            role: 'module-title',
                            semantic: {
                                componentKind: 'test',
                                role: 'module-title'
                            },
                            text: 'Label',
                            visual: {
                                styleKey: VIEW2D_STYLE_KEYS.LABEL
                            }
                        })
                    ]
                })
            ]
        });
        renderer.setScene(scene);

        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            viewportTransform: {
                scale: 0.88,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);

        const renderedTexts = ctx.operations
            .filter((entry) => entry.type === 'fillText')
            .map((entry) => entry.text);

        expect(renderedTexts).toContain('Caption');
        expect(renderedTexts).toContain('Label');
    });

    it('reveals persistent labels earlier than default canvas text labels', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });
        const scene = createSceneModel({
            nodes: [
                createGroupNode({
                    direction: VIEW2D_LAYOUT_DIRECTIONS.VERTICAL,
                    gapKey: 'default',
                    children: [
                        createTextNode({
                            role: 'persistent-label',
                            semantic: {
                                componentKind: 'test',
                                role: 'persistent-label'
                            },
                            text: 'Persistent',
                            visual: {
                                styleKey: VIEW2D_STYLE_KEYS.LABEL
                            },
                            metadata: {
                                persistentMinScreenFontPx: 10
                            }
                        }),
                        createTextNode({
                            role: 'default-label',
                            semantic: {
                                componentKind: 'test',
                                role: 'default-label'
                            },
                            text: 'Default',
                            visual: {
                                styleKey: VIEW2D_STYLE_KEYS.LABEL
                            }
                        })
                    ]
                })
            ]
        });
        renderer.setScene(scene);

        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            viewportTransform: {
                scale: 0.75,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);

        const renderedTexts = ctx.operations
            .filter((entry) => entry.type === 'fillText')
            .map((entry) => entry.text);

        expect(renderedTexts).toContain('Persistent');
        expect(renderedTexts).not.toContain('Default');
    });

    it('keeps connector arrowheads visible during interactive head-detail zoom renders', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });

        const sourceNode = createMatrixNode({
            role: 'source-anchor',
            semantic: {
                componentKind: 'mhsa',
                layerIndex: 0,
                headIndex: 0,
                stage: 'head-detail',
                role: 'source-anchor'
            },
            dimensions: { rows: 1, cols: 1 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.RESIDUAL
            },
            metadata: {
                hidden: true,
                card: {
                    width: 24,
                    height: 24,
                    cornerRadius: 0
                }
            }
        });
        const targetNode = createMatrixNode({
            role: 'target-anchor',
            semantic: {
                componentKind: 'mhsa',
                layerIndex: 0,
                headIndex: 0,
                stage: 'head-detail',
                role: 'target-anchor'
            },
            dimensions: { rows: 1, cols: 1 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.RESIDUAL
            },
            metadata: {
                hidden: true,
                card: {
                    width: 24,
                    height: 24,
                    cornerRadius: 0
                }
            }
        });
        const connectorNode = createConnectorNode({
            role: 'x-ln-copy-connector-0',
            semantic: {
                componentKind: 'mhsa',
                layerIndex: 0,
                headIndex: 0,
                stage: 'head-detail',
                role: 'x-ln-copy-connector',
                copyIndex: 0
            },
            source: createAnchorRef(sourceNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
            target: createAnchorRef(targetNode.id, VIEW2D_ANCHOR_SIDES.LEFT),
            route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL,
                stroke: 'rgba(255, 255, 255, 0.84)'
            },
            metadata: {
                preserveColor: true
            }
        });
        const headDetailScene = createSceneModel({
            nodes: [
                createGroupNode({
                    role: 'head-detail-stage',
                    semantic: {
                        componentKind: 'mhsa',
                        layerIndex: 0,
                        headIndex: 0,
                        stage: 'head-detail',
                        role: 'head-detail-stage'
                    },
                    direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
                    children: [sourceNode, targetNode]
                }),
                createGroupNode({
                    role: 'head-detail-connectors',
                    semantic: {
                        componentKind: 'mhsa',
                        layerIndex: 0,
                        headIndex: 0,
                        stage: 'head-detail',
                        role: 'head-detail-connectors'
                    },
                    direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
                    children: [connectorNode]
                })
            ],
            metadata: {
                visualContract: 'selection-panel-mhsa-v1'
            }
        });
        const scene = createSceneModel({
            nodes: [],
            metadata: {
                visualContract: 'selection-panel-mhsa-v1',
                headDetailTarget: {
                    layerIndex: 0,
                    headIndex: 0
                },
                headDetailScene
            }
        });

        renderer.setScene(scene);

        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            interacting: true,
            headDetailDepthActive: true,
            detailViewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);

        const arrowHeadFill = ctx.operations.find((entry) => (
            entry.type === 'fill'
            && entry.path.filter((step) => step.type === 'lineTo').length >= 2
        ));

        expect(arrowHeadFill).toBeTruthy();
    });

    it('renders curved trapezoid card surfaces for position-embedding overview nodes', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });

        const positionEmbeddingNode = createMatrixNode({
            role: 'position-embedding-card',
            semantic: {
                componentKind: 'embedding',
                stage: 'embedding.position',
                role: 'position-embedding-card'
            },
            dimensions: { rows: 1024, cols: D_MODEL },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.EMBEDDING_POSITION_STREAM
            },
            metadata: {
                card: {
                    width: 156,
                    height: 88,
                    cornerRadius: 18,
                    shape: 'curved-trapezoid',
                    shapeConfig: {
                        leftInset: 12,
                        rightInset: 8,
                        rightHeightRatio: 0.38,
                        leftBulge: 12,
                        rightBulge: 8
                    }
                }
            }
        });

        renderer.setScene(createSceneModel({
            nodes: [positionEmbeddingNode]
        }));

        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);

        const curvedFill = ctx.operations.find((entry) => (
            entry.type === 'fill'
            && entry.path.some((step) => step.type === 'quadraticCurveTo')
        )) || null;

        expect(curvedFill).toBeTruthy();
        expect(
            curvedFill.path.filter((step) => step.type === 'quadraticCurveTo')
        ).toHaveLength(4);
    });

    it('renders mirrored curved trapezoid card surfaces for output unembedding nodes', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });

        const unembeddingNode = createMatrixNode({
            role: 'unembedding',
            semantic: {
                componentKind: 'logits',
                stage: 'unembedding',
                role: 'unembedding'
            },
            dimensions: { rows: D_MODEL, cols: 50257 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.EMBEDDING_TOKEN_STREAM
            },
            metadata: {
                card: {
                    width: 196,
                    height: 144,
                    cornerRadius: 18,
                    shape: 'curved-trapezoid',
                    shapeConfig: {
                        leftInset: 4,
                        rightInset: 0,
                        leftHeightRatio: 0.36,
                        rightHeightRatio: 1,
                        cornerRadius: 14
                    }
                }
            }
        });

        renderer.setScene(createSceneModel({
            nodes: [unembeddingNode]
        }));

        expect(renderer.render({
            width: 420,
            height: 260,
            dpr: 1,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);

        const curvedFill = ctx.operations.find((entry) => (
            entry.type === 'fill'
            && entry.path.some((step) => step.type === 'quadraticCurveTo')
        )) || null;
        const quadratics = curvedFill?.path?.filter((step) => step.type === 'quadraticCurveTo') || [];

        expect(curvedFill).toBeTruthy();
        expect(quadratics).toHaveLength(4);
        expect(quadratics[0]?.cpy || 0).toBeGreaterThan(quadratics[1]?.cpy || 0);
    });

    it('dims the matrix surface during local MHSA detail cell focus without dimming the selected cell', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });

        const matrixNode = createMatrixNode({
            role: 'attention-pre-score',
            semantic: {
                componentKind: 'mhsa',
                layerIndex: 0,
                headIndex: 0,
                stage: 'attention',
                role: 'attention-pre-score'
            },
            dimensions: { rows: 2, cols: 2 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.GRID,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            rowItems: [
                {
                    cells: [
                        { fillCss: 'rgba(255, 255, 255, 0.8)' },
                        { fillCss: 'rgba(160, 160, 160, 0.8)' }
                    ]
                },
                {
                    cells: [
                        { fillCss: 'rgba(120, 120, 120, 0.8)' },
                        { fillCss: 'rgba(80, 80, 80, 0.8)' }
                    ]
                }
            ],
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.MHSA_SCORE
            },
            metadata: {
                grid: {
                    preserveDetail: true
                }
            }
        });

        const scene = createSceneModel({
            nodes: [matrixNode],
            metadata: {
                visualContract: 'selection-panel-mhsa-v1'
            }
        });
        renderer.setScene(scene);

        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            },
            interactionState: {
                detailSceneFocus: {
                    activeNodeIds: [matrixNode.id],
                    cellSelections: [
                        {
                            nodeId: matrixNode.id,
                            rowIndex: 0,
                            colIndex: 0
                        }
                    ]
                }
            }
        })).toBe(true);

        const fillOps = ctx.operations.filter((entry) => entry.type === 'fill');
        const surfaceFill = fillOps.find((entry) => Math.abs(Number(entry.globalAlpha) - 0.24) < 0.001) || null;
        const focusedCellFill = fillOps.find((entry) => Math.abs(Number(entry.globalAlpha) - 1) < 0.001) || null;

        expect(surfaceFill).toBeTruthy();
        expect(focusedCellFill).toBeTruthy();
    });

    it('keeps multiple selected rows bright in compact vector-strip matrices', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });

        const matrixNode = createMatrixNode({
            role: 'projection-source-xln',
            semantic: {
                componentKind: 'mhsa',
                layerIndex: 0,
                headIndex: 0,
                stage: 'ln1.output',
                role: 'projection-source-xln'
            },
            dimensions: { rows: 3, cols: 64 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            rowItems: [
                { label: 'Token A', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token B', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token C', gradientCss: 'rgba(80, 160, 255, 0.96)' }
            ],
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.RESIDUAL
            },
            metadata: {
                compactRows: {
                    compactWidth: 120,
                    rowHeight: 12,
                    rowGap: 4,
                    paddingX: 0,
                    paddingY: 0,
                    variant: VIEW2D_VECTOR_STRIP_VARIANT
                }
            }
        });

        renderer.setScene(createSceneModel({
            nodes: [matrixNode],
            metadata: {
                visualContract: 'selection-panel-mhsa-v1'
            }
        }));

        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            },
            interactionState: {
                detailSceneFocus: {
                    activeNodeIds: [matrixNode.id],
                    rowSelections: [
                        { nodeId: matrixNode.id, rowIndex: 0 },
                        { nodeId: matrixNode.id, rowIndex: 2 }
                    ]
                }
            }
        })).toBe(true);

        const fillRects = ctx.operations.filter((entry) => (
            entry.type === 'fillRect'
            && String(entry.fillStyle || '').includes('rgba(80, 160, 255, 0.96)')
        ));
        expect(fillRects).toHaveLength(3);
        const sortedAlphas = fillRects
            .map((entry) => Number(entry.globalAlpha) || 0)
            .sort((left, right) => right - left);

        expect(sortedAlphas[0]).toBeCloseTo(1, 3);
        expect(sortedAlphas[1]).toBeCloseTo(1, 3);
        expect(sortedAlphas[2]).toBeCloseTo(0.18, 3);
    });

    it('lifts a focused next-cache preview row above the node preview opacity', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });

        const matrixNode = createMatrixNode({
            role: 'projection-cache-next',
            semantic: {
                componentKind: 'mhsa',
                layerIndex: 0,
                headIndex: 0,
                stage: 'kv-cache.v.next',
                role: 'projection-cache-next',
                branchKey: 'v'
            },
            dimensions: { rows: 3, cols: 64 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            rowItems: [
                { label: 'Token A', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token B', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token C', gradientCss: 'rgba(80, 160, 255, 0.96)' }
            ],
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.MHSA_V,
                opacity: 0.4
            },
            metadata: {
                nextCachePreviewNode: true,
                compactRows: {
                    compactWidth: 120,
                    rowHeight: 12,
                    rowGap: 4,
                    paddingX: 0,
                    paddingY: 0,
                    variant: VIEW2D_VECTOR_STRIP_VARIANT
                }
            }
        });

        renderer.setScene(createSceneModel({
            nodes: [matrixNode],
            metadata: {
                visualContract: 'selection-panel-mhsa-v1'
            }
        }));

        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            },
            interactionState: {
                detailSceneFocus: {
                    activeNodeIds: [matrixNode.id],
                    rowSelections: [
                        { nodeId: matrixNode.id, rowIndex: 2 }
                    ]
                }
            }
        })).toBe(true);

        const fillRects = ctx.operations.filter((entry) => (
            entry.type === 'fillRect'
            && String(entry.fillStyle || '').includes('rgba(80, 160, 255, 0.96)')
        ));
        expect(fillRects).toHaveLength(3);
        const sortedAlphas = fillRects
            .map((entry) => Number(entry.globalAlpha) || 0)
            .sort((left, right) => right - left);

        expect(sortedAlphas[0]).toBeCloseTo(1, 3);
        expect(sortedAlphas[1]).toBeCloseTo(0.072, 3);
        expect(sortedAlphas[2]).toBeCloseTo(0.072, 3);
    });

    it('focuses a specific compact-row vector-strip band when cell selections target it', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });

        const matrixNode = createMatrixNode({
            role: 'concat-output-matrix',
            semantic: {
                componentKind: 'output-projection',
                layerIndex: 0,
                stage: 'concatenate',
                role: 'concat-output-matrix'
            },
            dimensions: { rows: 3, cols: 256 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            rowItems: [
                { label: 'Token A', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token B', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token C', gradientCss: 'rgba(80, 160, 255, 0.96)' }
            ],
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.MHSA_HEAD_OUTPUT
            },
            metadata: {
                interactiveBandHit: true,
                compactRows: {
                    compactWidth: 120,
                    rowHeight: 12,
                    rowGap: 4,
                    paddingX: 0,
                    paddingY: 0,
                    variant: VIEW2D_VECTOR_STRIP_VARIANT,
                    bandCount: 4,
                    bandSeparatorOpacity: 0.22,
                    hoverStrokeColor: 'rgba(255,255,255,0.10)'
                }
            }
        });

        renderer.setScene(createSceneModel({
            nodes: [matrixNode],
            metadata: {
                visualContract: 'selection-panel-output-projection-v1'
            }
        }));

        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            },
            interactionState: {
                detailSceneFocus: {
                    activeNodeIds: [matrixNode.id],
                    cellSelections: [
                        { nodeId: matrixNode.id, rowIndex: 1, colIndex: 2 }
                    ]
                }
            }
        })).toBe(true);

        const entry = renderer.layout?.registry?.getNodeEntry(matrixNode.id);
        expect(entry?.contentBounds).toBeTruthy();
        expect(entry?.layoutData).toBeTruthy();

        const bandWidth = entry.layoutData.compactWidth / 4;
        const expectedX = entry.contentBounds.x + entry.layoutData.innerPaddingX + (bandWidth * 2) + 0.5;
        const expectedY = entry.contentBounds.y + entry.layoutData.innerPaddingY + (entry.layoutData.rowHeight + entry.layoutData.rowGap) + 0.5;
        const strokeRect = ctx.operations.find((operation) => (
            operation.type === 'strokeRect'
            && Math.abs(operation.x - expectedX) < 1.25
            && Math.abs(operation.y - expectedY) < 1.25
        ));

        expect(strokeRect).toBeTruthy();
        expect(strokeRect?.width).toBeCloseTo(Math.max(0, bandWidth - 1), 2);
    });

    it('keeps preserve-detail grid cells visible during interactive hover fast paths', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });

        const matrixNode = createMatrixNode({
            role: 'attention-post',
            semantic: {
                componentKind: 'mhsa',
                layerIndex: 0,
                headIndex: 0,
                stage: 'attention',
                role: 'attention-post'
            },
            dimensions: { rows: 3, cols: 3 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.GRID,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            rowItems: [
                {
                    cells: [
                        { fillCss: 'rgba(255, 255, 255, 0.8)' },
                        { fillCss: 'rgba(220, 220, 220, 0.8)' },
                        { fillCss: 'rgba(190, 190, 190, 0.8)' }
                    ]
                },
                {
                    cells: [
                        { fillCss: 'rgba(160, 160, 160, 0.8)' },
                        { fillCss: 'rgba(130, 130, 130, 0.8)' },
                        { fillCss: 'rgba(100, 100, 100, 0.8)' }
                    ]
                },
                {
                    cells: [
                        { fillCss: 'rgba(90, 90, 90, 0.8)' },
                        { fillCss: 'rgba(70, 70, 70, 0.8)' },
                        { fillCss: 'rgba(50, 50, 50, 0.8)' }
                    ]
                }
            ],
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.MHSA_SCORE
            },
            metadata: {
                grid: {
                    preserveDetail: true
                }
            }
        });

        renderer.setScene(createSceneModel({
            nodes: [matrixNode],
            metadata: {
                visualContract: 'selection-panel-mhsa-v1'
            }
        }));

        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            interacting: true,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            },
            interactionState: {
                detailSceneFocus: {
                    activeNodeIds: [matrixNode.id],
                    cellSelections: [
                        {
                            nodeId: matrixNode.id,
                            rowIndex: 1,
                            colIndex: 1
                        }
                    ]
                }
            }
        })).toBe(true);

        const fillOps = ctx.operations.filter((entry) => entry.type === 'fill');
        const focusedCellStroke = ctx.operations.find((entry) => (
            entry.type === 'stroke'
            && entry.strokeStyle === 'rgba(255, 255, 255, 0.88)'
        )) || null;

        expect(fillOps.length).toBeGreaterThan(2);
        expect(focusedCellStroke).toBeTruthy();
        expect(Number(focusedCellStroke?.lineWidth) || 0).toBeLessThanOrEqual(0.93);
    });

    it('applies focus alpha to card surface effects and edge strokes for inactive matrices', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });

        const cardNode = createMatrixNode({
            role: 'output-projection-card',
            semantic: {
                componentKind: 'mhsa',
                layerIndex: 0,
                headIndex: 0,
                stage: 'projection-q',
                role: 'output-projection-card'
            },
            dimensions: { rows: 1, cols: 1 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.OUTPUT_PROJECTION
            },
            metadata: {
                card: {
                    width: 72,
                    height: 72,
                    cornerRadius: 12
                }
            }
        });

        const scene = createSceneModel({
            nodes: [cardNode],
            metadata: {
                visualContract: 'selection-panel-mhsa-v1'
            }
        });
        renderer.setScene(scene);

        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            },
            interactionState: {
                detailSceneFocus: {
                    activeNodeIds: ['different-node']
                }
            }
        })).toBe(true);

        const nonBackgroundOps = ctx.operations.filter((entry) => {
            if (entry.type !== 'fill' && entry.type !== 'stroke') return false;
            return entry.fillStyle !== 'rgba(0, 0, 0, 0)';
        });
        const maxAlpha = Math.max(...nonBackgroundOps.map((entry) => Number(entry.globalAlpha) || 0));
        const hasInactiveFilter = nonBackgroundOps.some((entry) => entry.filter === 'saturate(0.06) brightness(0.62) grayscale(0.78)');

        expect(maxAlpha).toBeLessThanOrEqual(0.181);
        expect(hasInactiveFilter).toBe(true);
    });

    it('applies the inactive dimming filter to inactive compact-row matrices', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });

        const compactRowNode = createMatrixNode({
            role: 'inactive-query-vector',
            semantic: {
                componentKind: 'mhsa',
                layerIndex: 0,
                headIndex: 0,
                stage: 'projection-q',
                role: 'inactive-query-vector'
            },
            dimensions: { rows: 2, cols: 64 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            rowItems: [
                { label: 'Token A', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token B', gradientCss: 'rgba(80, 160, 255, 0.96)' }
            ],
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.MHSA_Q
            },
            metadata: {
                compactRows: {
                    compactWidth: 120,
                    rowHeight: 12,
                    rowGap: 4,
                    paddingX: 0,
                    paddingY: 0
                }
            }
        });

        const scene = createSceneModel({
            nodes: [compactRowNode],
            metadata: {
                visualContract: 'selection-panel-mhsa-v1'
            }
        });
        renderer.setScene(scene);

        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            },
            interactionState: {
                detailSceneFocus: {
                    activeNodeIds: ['different-node']
                }
            }
        })).toBe(true);

        const fillOps = ctx.operations.filter((entry) => (
            (entry.type === 'fill' || entry.type === 'fillRect')
            && entry.fillStyle !== 'rgba(0, 0, 0, 0)'
            && entry.fillStyle !== 'rgb(0, 0, 0)'
            && !(entry.type === 'fillRect' && entry.width === 400 && entry.height === 240)
        ));
        expect(fillOps.length).toBeGreaterThan(0);
        expect(fillOps.every((entry) => entry.filter === 'saturate(0.06) brightness(0.62) grayscale(0.78)')).toBe(true);
    });

    it('renders the scene-backed output-projection detail scene during deep detail mode', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });

        const overviewNode = createMatrixNode({
            role: 'projection-weight',
            semantic: {
                componentKind: 'output-projection',
                layerIndex: 0,
                stage: 'attn-out',
                role: 'projection-weight'
            },
            dimensions: { rows: D_MODEL, cols: D_MODEL },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.OUTPUT_PROJECTION
            },
            metadata: {
                card: {
                    width: 96,
                    height: 72,
                    cornerRadius: 12
                }
            }
        });

        const detailNode = createMatrixNode({
            role: 'head-output-matrix',
            semantic: {
                componentKind: 'output-projection',
                layerIndex: 0,
                headIndex: 0,
                stage: 'head-output',
                role: 'head-output-matrix'
            },
            dimensions: { rows: 2, cols: D_HEAD },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            rowItems: [
                { label: 'Token A', gradientCss: 'rgba(120, 220, 255, 0.9)' },
                { label: 'Token B', gradientCss: 'rgba(120, 220, 255, 0.9)' }
            ],
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.MHSA_HEAD_OUTPUT
            },
            metadata: {
                compactRows: {
                    variant: VIEW2D_VECTOR_STRIP_VARIANT,
                    compactWidth: 104,
                    rowHeight: 7,
                    rowGap: 0,
                    paddingX: 0,
                    paddingY: 0,
                    bandCount: 12,
                    bandSeparatorOpacity: 0,
                    hoverScaleY: 1.16,
                    hoverGlowColor: 'rgba(255,255,255,0.08)',
                    hoverGlowBlur: 12,
                    hoverStrokeColor: 'rgba(255,255,255,0.10)',
                    dimmedRowOpacity: 0.18,
                    hideSurface: true
                },
                card: {
                    cornerRadius: 10
                }
            }
        });

        const outputProjectionDetailScene = createSceneModel({
            nodes: [detailNode],
            metadata: {
                visualContract: 'selection-panel-output-projection-v1'
            }
        });

        renderer.setScene(createSceneModel({
            nodes: [overviewNode],
            metadata: {
                outputProjectionDetailTarget: {
                    layerIndex: 0
                },
                outputProjectionDetailScene
            }
        }));

        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            headDetailDepthActive: true,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            },
            detailViewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);

        expect(renderer.getLastRenderState()?.detailTargetKind).toBe('output-projection');
        expect(renderer.getActiveCaptionSceneState()?.scene).toBe(outputProjectionDetailScene);
    });

    it('renders the scene-backed layer norm detail scene during deep detail mode', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });

        const overviewNode = createMatrixNode({
            role: 'module-card',
            semantic: {
                componentKind: 'layer-norm',
                layerIndex: 0,
                stage: 'ln1',
                role: 'module-card'
            },
            dimensions: { rows: 1, cols: D_MODEL },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.LAYER_NORM
            },
            metadata: {
                card: {
                    width: 96,
                    height: 48,
                    cornerRadius: 999
                }
            }
        });

        const detailNode = createMatrixNode({
            role: 'layer-norm-normalized',
            semantic: {
                componentKind: 'layer-norm',
                layerIndex: 0,
                stage: 'ln1.norm',
                role: 'layer-norm-normalized'
            },
            dimensions: { rows: 2, cols: D_MODEL },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            rowItems: [
                { label: 'Token A', gradientCss: 'rgba(120, 220, 255, 0.9)' },
                { label: 'Token B', gradientCss: 'rgba(120, 220, 255, 0.9)' }
            ],
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.RESIDUAL
            },
            metadata: {
                compactRows: {
                    variant: VIEW2D_VECTOR_STRIP_VARIANT,
                    compactWidth: 104,
                    rowHeight: 7,
                    rowGap: 0,
                    paddingX: 0,
                    paddingY: 0,
                    bandCount: 12,
                    bandSeparatorOpacity: 0,
                    hoverScaleY: 1.16,
                    hoverGlowColor: 'rgba(255,255,255,0.08)',
                    hoverGlowBlur: 12,
                    hoverStrokeColor: 'rgba(255,255,255,0.10)',
                    dimmedRowOpacity: 0.18,
                    hideSurface: true
                },
                card: {
                    cornerRadius: 10
                }
            }
        });

        const layerNormDetailScene = createSceneModel({
            nodes: [detailNode],
            metadata: {
                visualContract: 'selection-panel-layer-norm-v1'
            }
        });

        renderer.setScene(createSceneModel({
            nodes: [overviewNode],
            metadata: {
                layerNormDetailTarget: {
                    layerNormKind: 'ln1',
                    layerIndex: 0
                },
                layerNormDetailScene
            }
        }));

        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            headDetailDepthActive: true,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            },
            detailViewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);

        expect(renderer.getLastRenderState()?.detailTargetKind).toBe('layer-norm');
        expect(renderer.getActiveCaptionSceneState()?.scene).toBe(layerNormDetailScene);
    });

    it('applies stronger opacity dimming to explicitly dimmed nodes', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });

        const compactRowNode = createMatrixNode({
            role: 'inactive-query-vector',
            semantic: {
                componentKind: 'mhsa',
                layerIndex: 0,
                headIndex: 0,
                stage: 'projection-q',
                role: 'inactive-query-vector'
            },
            dimensions: { rows: 2, cols: 64 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            rowItems: [
                { label: 'Token A', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token B', gradientCss: 'rgba(80, 160, 255, 0.96)' }
            ],
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.MHSA_Q
            },
            metadata: {
                compactRows: {
                    compactWidth: 120,
                    rowHeight: 12,
                    rowGap: 4,
                    paddingX: 0,
                    paddingY: 0
                }
            }
        });

        const scene = createSceneModel({
            nodes: [compactRowNode],
            metadata: {
                visualContract: 'selection-panel-mhsa-v1'
            }
        });
        renderer.setScene(scene);

        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            },
            interactionState: {
                detailSceneFocus: {
                    activeNodeIds: ['different-node'],
                    dimNodeIds: [compactRowNode.id]
                }
            }
        })).toBe(true);

        const fillOps = ctx.operations.filter((entry) => (
            (entry.type === 'fill' || entry.type === 'fillRect')
            && entry.fillStyle !== 'rgba(0, 0, 0, 0)'
            && entry.fillStyle !== 'rgb(0, 0, 0)'
            && !(entry.type === 'fillRect' && entry.width === 400 && entry.height === 240)
        ));
        expect(fillOps.length).toBeGreaterThan(0);
        expect(Math.max(...fillOps.map((entry) => Number(entry.globalAlpha) || 0))).toBeLessThanOrEqual(0.081);
    });

    it('skips the inactive dimming filter inside deep MHSA detail renders', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });

        const compactRowNode = createMatrixNode({
            role: 'attention-value-post',
            semantic: {
                componentKind: 'mhsa',
                layerIndex: 0,
                headIndex: 0,
                stage: 'attention',
                role: 'attention-value-post'
            },
            dimensions: { rows: 2, cols: 64 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            rowItems: [
                { label: 'Token A', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token B', gradientCss: 'rgba(80, 160, 255, 0.96)' }
            ],
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.MHSA_V
            },
            metadata: {
                compactRows: {
                    compactWidth: 120,
                    rowHeight: 12,
                    rowGap: 4,
                    paddingX: 0,
                    paddingY: 0
                }
            }
        });

        const detailScene = createSceneModel({
            nodes: [compactRowNode],
            metadata: {
                visualContract: 'selection-panel-mhsa-v1'
            }
        });
        const wrapperScene = createSceneModel({
            nodes: [],
            metadata: {
                visualContract: 'selection-panel-mhsa-v1',
                headDetailTarget: {
                    layerIndex: 0,
                    headIndex: 0
                },
                mhsaHeadDetailScene: detailScene
            }
        });
        renderer.setScene(wrapperScene);

        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            },
            detailViewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            },
            headDetailDepthActive: true,
            interactionState: {
                detailSceneFocus: {
                    activeNodeIds: ['different-node']
                }
            }
        })).toBe(true);

        const fillOps = ctx.operations.filter((entry) => (
            (entry.type === 'fill' || entry.type === 'fillRect')
            && entry.fillStyle !== '#000'
            && entry.fillStyle !== 'rgba(0, 0, 0, 0)'
        ));
        expect(fillOps.length).toBeGreaterThan(0);
        expect(fillOps.every((entry) => entry.filter === 'none')).toBe(true);
    });

    it('dims Q/K setup matrix fills during deep-detail V hover focus', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx, 1280, 720);
        const renderer = new CanvasSceneRenderer({ canvas });

        const detailScene = buildMhsaSceneModel({
            previewData: createMhsaDetailPreviewData(),
            layerIndex: 2,
            headIndex: 1,
            isSmallScreen: false
        });
        const detailNodes = flattenSceneNodes(detailScene);
        const detailIndex = createMhsaDetailSceneIndex(detailScene);
        const valueNode = detailNodes.find((node) => node.role === 'attention-value-post') || null;
        const queryInputNode = detailNodes.find((node) => (
            node.role === 'x-ln-copy'
            && String(node.semantic?.branchKey || '').toLowerCase() === 'q'
        )) || null;
        const keyInputNode = detailNodes.find((node) => (
            node.role === 'x-ln-copy'
            && String(node.semantic?.branchKey || '').toLowerCase() === 'k'
        )) || null;
        const queryBiasNode = detailNodes.find((node) => (
            node.role === 'projection-bias'
            && String(node.metadata?.kind || '').toLowerCase() === 'q'
        )) || null;
        const keyBiasNode = detailNodes.find((node) => (
            node.role === 'projection-bias'
            && String(node.metadata?.kind || '').toLowerCase() === 'k'
        )) || null;
        const queryProjectionNode = detailNodes.find((node) => (
            node.role === 'projection-output'
            && String(node.metadata?.kind || '').toLowerCase() === 'q'
        )) || null;
        const keyTransposeNode = detailNodes.find((node) => node.role === 'attention-key-transpose') || null;
        const hoverState = resolveMhsaDetailHoverState(detailIndex, {
            node: valueNode
        });

        expect(valueNode).toBeTruthy();
        expect(queryInputNode).toBeTruthy();
        expect(keyInputNode).toBeTruthy();
        expect(queryBiasNode).toBeTruthy();
        expect(keyBiasNode).toBeTruthy();
        expect(queryProjectionNode).toBeTruthy();
        expect(keyTransposeNode).toBeTruthy();
        expect(hoverState?.focusState?.dimNodeIds).toContain(queryInputNode?.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(keyInputNode?.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(queryBiasNode?.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(keyBiasNode?.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(queryProjectionNode?.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(keyTransposeNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(queryInputNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(keyInputNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(queryBiasNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(keyBiasNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(queryProjectionNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(keyTransposeNode?.id);

        renderer.setScene(createSceneModel({
            nodes: [],
            metadata: {
                visualContract: 'selection-panel-mhsa-v1',
                headDetailTarget: {
                    layerIndex: 2,
                    headIndex: 1
                },
                mhsaHeadDetailScene: detailScene
            }
        }));

        expect(renderer.render({
            width: 1280,
            height: 720,
            dpr: 1,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            },
            detailViewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            },
            headDetailDepthActive: true,
            interactionState: {
                detailSceneFocus: hoverState?.focusState || null
            }
        })).toBe(true);

        const registry = renderer.headDetailSceneState?.layout?.registry || null;
        const assertNodeIsDimmed = (node) => {
            const entry = registry?.getNodeEntry(node?.id || '') || null;
            const fillOps = resolveOpsInsideBounds(ctx.operations, entry?.contentBounds || entry?.bounds || null);
            const maxAlpha = Math.max(...fillOps.map((operation) => Number(operation.globalAlpha) || 0));
            expect(fillOps.length).toBeGreaterThan(0);
            expect(maxAlpha, node?.id || node?.role || 'node').toBeLessThanOrEqual(0.081);
        };

        assertNodeIsDimmed(queryInputNode);
        assertNodeIsDimmed(keyInputNode);
        assertNodeIsDimmed(queryBiasNode);
        assertNodeIsDimmed(keyBiasNode);
        assertNodeIsDimmed(queryProjectionNode);
        assertNodeIsDimmed(keyTransposeNode);
    });

    it('resolves grid-cell hits directly and ignores pointer positions inside grid gaps', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });

        const gridNode = createMatrixNode({
            role: 'attention-post',
            semantic: {
                componentKind: 'mhsa',
                layerIndex: 0,
                headIndex: 0,
                stage: 'attention',
                role: 'attention-post'
            },
            dimensions: { rows: 2, cols: 3 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.GRID,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            rowItems: [
                {
                    cells: [
                        { fillCss: 'rgba(255, 255, 255, 0.8)' },
                        { fillCss: 'rgba(160, 160, 160, 0.8)' },
                        { fillCss: 'rgba(120, 120, 120, 0.8)' }
                    ]
                },
                {
                    cells: [
                        { fillCss: 'rgba(90, 90, 90, 0.8)' },
                        { fillCss: 'rgba(70, 70, 70, 0.8)' },
                        { fillCss: 'rgba(50, 50, 50, 0.8)' }
                    ]
                }
            ],
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.MHSA_SCORE
            },
            metadata: {
                grid: {
                    preserveDetail: true
                }
            }
        });

        renderer.setScene(createSceneModel({
            nodes: [gridNode],
            metadata: {
                visualContract: 'selection-panel-mhsa-v1'
            }
        }));

        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);

        const entry = renderer.layout?.registry?.getNodeEntry(gridNode.id);
        expect(entry?.contentBounds).toBeTruthy();
        expect(entry?.layoutData).toBeTruthy();

        const contentBounds = entry.contentBounds;
        const layoutData = entry.layoutData;
        const cellX = contentBounds.x + layoutData.innerPaddingX + layoutData.cellSize + layoutData.cellGap + (layoutData.cellSize * 0.5);
        const cellY = contentBounds.y + layoutData.innerPaddingY + (layoutData.cellSize * 0.5);
        const cellHit = renderer.resolveInteractiveHitAtPoint(cellX, cellY);

        expect(cellHit?.cellHit?.rowIndex).toBe(0);
        expect(cellHit?.cellHit?.colIndex).toBe(1);

        const gapX = contentBounds.x + layoutData.innerPaddingX + layoutData.cellSize + (layoutData.cellGap * 0.5);
        const gapHit = renderer.resolveInteractiveHitAtPoint(gapX, cellY);

        expect(gapHit?.cellHit).toBeNull();
    });

    it('resolves column-strip hovers without scanning every column', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });

        const columnNode = createMatrixNode({
            role: 'attention-key-transpose',
            semantic: {
                componentKind: 'mhsa',
                layerIndex: 0,
                headIndex: 0,
                stage: 'attention',
                role: 'attention-key-transpose'
            },
            dimensions: { rows: 4, cols: 3 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.COLUMN_STRIP,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            columnItems: [
                { fillCss: 'rgba(80, 160, 255, 0.92)' },
                { fillCss: 'rgba(80, 160, 255, 0.92)' },
                { fillCss: 'rgba(80, 160, 255, 0.92)' }
            ],
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.MHSA_K
            },
            metadata: {
                columnStrip: {
                    colWidth: 14,
                    colGap: 6,
                    colHeight: 72,
                    paddingX: 0,
                    paddingY: 0
                }
            }
        });

        renderer.setScene(createSceneModel({
            nodes: [columnNode],
            metadata: {
                visualContract: 'selection-panel-mhsa-v1'
            }
        }));

        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);

        const entry = renderer.layout?.registry?.getNodeEntry(columnNode.id);
        expect(entry?.contentBounds).toBeTruthy();
        expect(entry?.layoutData).toBeTruthy();

        const contentBounds = entry.contentBounds;
        const layoutData = entry.layoutData;
        const colX = contentBounds.x + layoutData.innerPaddingX + (2 * (layoutData.colWidth + layoutData.colGap)) + (layoutData.colWidth * 0.5);
        const colY = contentBounds.y + layoutData.innerPaddingY + (layoutData.colHeight * 0.5);
        const colHit = renderer.resolveInteractiveHitAtPoint(colX, colY);

        expect(colHit?.columnHit?.colIndex).toBe(2);

        const gapX = contentBounds.x + layoutData.innerPaddingX + layoutData.colWidth + (layoutData.colGap * 0.5);
        const gapHit = renderer.resolveInteractiveHitAtPoint(gapX, colY);

        expect(gapHit?.columnHit).toBeNull();
    });

    it('keeps vector-strip row hover fallback behavior when the pointer lands in a row gap', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });

        const compactRowNode = createMatrixNode({
            role: 'attention-head-output',
            semantic: {
                componentKind: 'mhsa',
                layerIndex: 0,
                headIndex: 0,
                stage: 'attention',
                role: 'attention-head-output'
            },
            dimensions: { rows: 3, cols: 64 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            rowItems: [
                { label: 'Token A', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token B', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token C', gradientCss: 'rgba(80, 160, 255, 0.96)' }
            ],
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.MHSA_V
            },
            metadata: {
                compactRows: {
                    compactWidth: 120,
                    rowHeight: 12,
                    rowGap: 8,
                    paddingX: 0,
                    paddingY: 0,
                    variant: VIEW2D_VECTOR_STRIP_VARIANT
                }
            }
        });

        renderer.setScene(createSceneModel({
            nodes: [compactRowNode],
            metadata: {
                visualContract: 'selection-panel-mhsa-v1'
            }
        }));

        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);

        const entry = renderer.layout?.registry?.getNodeEntry(compactRowNode.id);
        expect(entry?.contentBounds).toBeTruthy();
        expect(entry?.layoutData).toBeTruthy();

        const contentBounds = entry.contentBounds;
        const layoutData = entry.layoutData;
        const hitX = contentBounds.x + layoutData.innerPaddingX + (layoutData.compactWidth * 0.5);
        const hitY = contentBounds.y + layoutData.innerPaddingY + layoutData.rowHeight + (layoutData.rowGap * 0.75);
        const rowHit = renderer.resolveInteractiveHitAtPoint(hitX, hitY);

        expect(rowHit?.rowHit?.rowIndex).toBe(1);
    });

    it('resolves overview residual row hits from a small screen-space hover target when zoomed out', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });

        const residualNode = createMatrixNode({
            role: 'module-card',
            semantic: {
                componentKind: 'residual',
                layerIndex: 0,
                stage: 'incoming',
                role: 'module-card'
            },
            dimensions: { rows: 3, cols: 768 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            rowItems: [
                { label: 'Token A', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token B', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token C', gradientCss: 'rgba(80, 160, 255, 0.96)' }
            ],
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.RESIDUAL
            },
            metadata: {
                compactRows: {
                    compactWidth: 120,
                    rowHeight: 6,
                    rowGap: 0,
                    paddingX: 0,
                    paddingY: 0,
                    variant: VIEW2D_VECTOR_STRIP_VARIANT
                }
            }
        });

        renderer.setScene(createSceneModel({
            nodes: [residualNode]
        }));

        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            viewportTransform: {
                scale: 0.25,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);

        const entry = renderer.layout?.registry?.getNodeEntry(residualNode.id);
        const renderState = renderer.getLastRenderState();
        expect(entry?.contentBounds).toBeTruthy();
        expect(renderState?.worldScale).toBeGreaterThan(0);

        const worldScale = renderState.worldScale;
        const offsetX = renderState.offsetX || 0;
        const offsetY = renderState.offsetY || 0;
        const screenContentRight = offsetX + ((entry.contentBounds.x + entry.contentBounds.width) * worldScale);
        const screenRowHeight = (entry.layoutData?.rowHeight || 0) * worldScale;
        const screenRowCenterY = offsetY + ((entry.contentBounds.y + (entry.layoutData?.rowHeight || 0) + ((entry.layoutData?.rowHeight || 0) * 0.5)) * worldScale);
        const hoverX = screenContentRight + 3;
        const hoverY = screenRowCenterY;
        const worldX = (hoverX - offsetX) / worldScale;
        const worldY = (hoverY - offsetY) / worldScale;

        expect(screenRowHeight).toBeLessThan(4);
        expect(renderer.resolveInteractiveHitAtPoint(worldX, worldY)).toBeNull();

        const rowHit = renderer.resolveInteractiveHitAtScreenPoint(hoverX, hoverY);

        expect(rowHit?.node?.id).toBe(residualNode.id);
        expect(rowHit?.rowHit?.rowIndex).toBe(1);
    });

    it('prefers overview residual row hits over generic matrix hits at the same screen point', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });

        const residualNode = createMatrixNode({
            id: 'residual-overview-node',
            role: 'module-card',
            semantic: {
                componentKind: 'residual',
                layerIndex: 0,
                stage: 'incoming',
                role: 'module-card'
            },
            dimensions: { rows: 3, cols: 768 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            rowItems: [
                { label: 'Token A', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token B', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token C', gradientCss: 'rgba(80, 160, 255, 0.96)' }
            ],
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.RESIDUAL
            },
            metadata: {
                compactRows: {
                    compactWidth: 120,
                    rowHeight: 6,
                    rowGap: 0,
                    paddingX: 0,
                    paddingY: 0,
                    variant: VIEW2D_VECTOR_STRIP_VARIANT
                }
            }
        });
        const genericNode = createMatrixNode({
            id: 'generic-overlap-node',
            role: 'projection-weight',
            semantic: {
                componentKind: 'projection',
                layerIndex: 0,
                stage: 'qkv.q'
            },
            dimensions: { rows: 1, cols: 1 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX
        });

        renderer.setScene(createSceneModel({
            nodes: [residualNode]
        }));

        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            viewportTransform: {
                scale: 0.25,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);

        const entry = renderer.layout?.registry?.getNodeEntry(residualNode.id);
        const renderState = renderer.getLastRenderState();
        expect(entry?.contentBounds).toBeTruthy();
        expect(renderState?.worldScale).toBeGreaterThan(0);

        const worldScale = renderState.worldScale;
        const offsetX = renderState.offsetX || 0;
        const offsetY = renderState.offsetY || 0;
        const screenContentRight = offsetX + ((entry.contentBounds.x + entry.contentBounds.width) * worldScale);
        const screenRowCenterY = offsetY + ((entry.contentBounds.y + (entry.layoutData?.rowHeight || 0) + ((entry.layoutData?.rowHeight || 0) * 0.5)) * worldScale);
        const hoverX = screenContentRight + 3;
        const hoverY = screenRowCenterY;

        renderer.resolveInteractiveHitAtPoint = () => ({
            entry: {
                nodeId: genericNode.id
            },
            node: genericNode,
            rowHit: null,
            cellHit: null,
            columnHit: null
        });

        const hit = renderer.resolveInteractiveHitAtScreenPoint(hoverX, hoverY);

        expect(hit?.node?.id).toBe(residualNode.id);
        expect(hit?.rowHit?.rowIndex).toBe(1);
    });

    it('resolves overview residual row hits from the full card width, not only the compact strip bounds', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });

        const residualNode = createMatrixNode({
            id: 'residual-card-hover-node',
            role: 'module-card',
            semantic: {
                componentKind: 'residual',
                layerIndex: 0,
                stage: 'incoming',
                role: 'module-card'
            },
            dimensions: { rows: 3, cols: 768 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            rowItems: [
                { label: 'Token A', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token B', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token C', gradientCss: 'rgba(80, 160, 255, 0.96)' }
            ],
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.RESIDUAL
            },
            metadata: {
                compactRows: {
                    compactWidth: 120,
                    rowHeight: 6,
                    rowGap: 0,
                    paddingX: 12,
                    paddingY: 8,
                    variant: VIEW2D_VECTOR_STRIP_VARIANT
                }
            }
        });

        renderer.setScene(createSceneModel({
            nodes: [residualNode]
        }));

        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);

        const entry = renderer.layout?.registry?.getNodeEntry(residualNode.id);
        expect(entry?.bounds).toBeTruthy();
        expect(entry?.contentBounds).toBeTruthy();
        expect(entry?.layoutData).toBeTruthy();

        const contentBounds = entry.contentBounds;
        const layoutData = entry.layoutData;
        const stripRight = contentBounds.x + layoutData.innerPaddingX + layoutData.compactWidth;
        const cardRight = entry.bounds.x + entry.bounds.width;
        expect(cardRight).toBeGreaterThan(stripRight);

        const hoverX = Math.min(cardRight - 1, stripRight + 4);
        const hoverY = contentBounds.y + layoutData.innerPaddingY + layoutData.rowHeight + (layoutData.rowHeight * 0.5);
        const hit = renderer.resolveInteractiveHitAtPoint(hoverX, hoverY);

        expect(hit?.node?.id).toBe(residualNode.id);
        expect(hit?.rowHit?.rowIndex).toBe(1);
    });

    it('resolves zoomed-out residual row hits from the card footprint outside the compact strip width', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });

        const residualNode = createMatrixNode({
            id: 'zoomed-out-residual-card-node',
            role: 'module-card',
            semantic: {
                componentKind: 'residual',
                layerIndex: 0,
                stage: 'incoming',
                role: 'module-card'
            },
            dimensions: { rows: 3, cols: 768 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            rowItems: [
                { label: 'Token A', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token B', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token C', gradientCss: 'rgba(80, 160, 255, 0.96)' }
            ],
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.RESIDUAL
            },
            metadata: {
                compactRows: {
                    compactWidth: 120,
                    rowHeight: 6,
                    rowGap: 0,
                    paddingX: 12,
                    paddingY: 8,
                    variant: VIEW2D_VECTOR_STRIP_VARIANT
                }
            }
        });

        renderer.setScene(createSceneModel({
            nodes: [residualNode]
        }));

        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            viewportTransform: {
                scale: 0.25,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);

        const entry = renderer.layout?.registry?.getNodeEntry(residualNode.id);
        const renderState = renderer.getLastRenderState();
        expect(entry?.bounds).toBeTruthy();
        expect(entry?.contentBounds).toBeTruthy();
        expect(entry?.layoutData).toBeTruthy();
        expect(renderState?.worldScale).toBeGreaterThan(0);

        const worldScale = renderState.worldScale;
        const offsetX = renderState.offsetX || 0;
        const offsetY = renderState.offsetY || 0;
        const screenEntryBounds = {
            x: offsetX + (entry.bounds.x * worldScale),
            y: offsetY + (entry.bounds.y * worldScale),
            width: entry.bounds.width * worldScale,
            height: entry.bounds.height * worldScale
        };
        const screenStripRight = offsetX + ((entry.contentBounds.x + entry.layoutData.innerPaddingX + entry.layoutData.compactWidth) * worldScale);
        const hoverX = Math.min(screenEntryBounds.x + screenEntryBounds.width - 1, screenStripRight + 12);
        const hoverY = offsetY + ((entry.contentBounds.y + entry.layoutData.innerPaddingY + entry.layoutData.rowHeight + (entry.layoutData.rowHeight * 0.5)) * worldScale);

        expect(hoverX).toBeGreaterThan(screenStripRight);

        const hit = renderer.resolveInteractiveHitAtScreenPoint(hoverX, hoverY);

        expect(hit?.node?.id).toBe(residualNode.id);
        expect(hit?.rowHit?.rowIndex).toBe(1);
    });

    it('builds a per-frame overview screen-hit cache for zoomed-out residual row fallback', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });

        const residualNode = createMatrixNode({
            id: 'cached-screen-hit-residual-node',
            role: 'module-card',
            semantic: {
                componentKind: 'residual',
                layerIndex: 0,
                stage: 'incoming',
                role: 'module-card'
            },
            dimensions: { rows: 3, cols: 768 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            rowItems: [
                { label: 'Token A', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token B', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token C', gradientCss: 'rgba(80, 160, 255, 0.96)' }
            ],
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.RESIDUAL
            },
            metadata: {
                compactRows: {
                    compactWidth: 120,
                    rowHeight: 6,
                    rowGap: 0,
                    paddingX: 12,
                    paddingY: 8,
                    variant: VIEW2D_VECTOR_STRIP_VARIANT
                }
            }
        });

        renderer.setScene(createSceneModel({
            nodes: [residualNode]
        }));

        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            viewportTransform: {
                scale: 0.25,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);

        const entry = renderer.layout?.registry?.getNodeEntry(residualNode.id);
        const renderState = renderer.getLastRenderState();
        expect(entry?.bounds).toBeTruthy();
        expect(renderState?.overviewScreenHitCache?.nodes?.length).toBeGreaterThan(0);
        expect(typeof renderState?.overviewScreenHitCache?.resolveTopmostScreenNodeIdAtPoint).toBe('function');

        const worldScale = renderState.worldScale;
        const offsetX = renderState.offsetX || 0;
        const offsetY = renderState.offsetY || 0;
        const screenEntryBounds = {
            x: offsetX + (entry.bounds.x * worldScale),
            y: offsetY + (entry.bounds.y * worldScale),
            width: entry.bounds.width * worldScale,
            height: entry.bounds.height * worldScale
        };
        const hoverX = screenEntryBounds.x + Math.min(12, screenEntryBounds.width * 0.5);
        const hoverY = offsetY + ((entry.contentBounds.y + entry.layoutData.innerPaddingY + entry.layoutData.rowHeight + (entry.layoutData.rowHeight * 0.5)) * worldScale);

        const hit = renderer.resolveInteractiveHitAtScreenPoint(hoverX, hoverY);

        expect(hit?.node?.id).toBe(residualNode.id);
        expect(hit?.rowHit?.rowIndex).toBe(1);
    });

    it('does not resolve residual fallback row hovers through a competing visible overview node', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });

        const residualNode = createMatrixNode({
            id: 'residual-fallback-node',
            role: 'module-card',
            semantic: {
                componentKind: 'residual',
                layerIndex: 0,
                stage: 'incoming',
                role: 'module-card'
            },
            dimensions: { rows: 3, cols: 768 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            rowItems: [
                { label: 'Token A', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token B', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token C', gradientCss: 'rgba(80, 160, 255, 0.96)' }
            ],
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.RESIDUAL
            },
            metadata: {
                compactRows: {
                    compactWidth: 120,
                    rowHeight: 6,
                    rowGap: 0,
                    paddingX: 12,
                    paddingY: 8,
                    variant: VIEW2D_VECTOR_STRIP_VARIANT
                },
                card: {
                    width: 168,
                    height: 64
                }
            }
        });
        const competingNode = createMatrixNode({
            id: 'competing-overview-node',
            role: 'projection-weight',
            semantic: {
                componentKind: 'projection',
                layerIndex: 0,
                stage: 'qkv.q',
                role: 'projection-weight'
            },
            dimensions: { rows: 1, cols: 1 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.MHSA_Q
            },
            metadata: {
                card: {
                    width: 64,
                    height: 64
                }
            }
        });

        const scene = createSceneModel({
            nodes: [residualNode, competingNode]
        });
        renderer.setScene(scene);

        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            viewportTransform: {
                scale: 0.25,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);

        const residualEntry = renderer.layout?.registry?.getNodeEntry(residualNode.id);
        const competingEntry = renderer.layout?.registry?.getNodeEntry(competingNode.id);
        const renderState = renderer.getLastRenderState();
        expect(residualEntry?.bounds).toBeTruthy();
        expect(competingEntry?.bounds).toBeTruthy();
        expect(renderState?.worldScale).toBeGreaterThan(0);

        const worldScale = renderState.worldScale;
        const offsetX = renderState.offsetX || 0;
        const offsetY = renderState.offsetY || 0;
        const screenCompetingBounds = {
            x: offsetX + (competingEntry.bounds.x * worldScale),
            y: offsetY + (competingEntry.bounds.y * worldScale),
            width: competingEntry.bounds.width * worldScale,
            height: competingEntry.bounds.height * worldScale
        };
        const hoverX = screenCompetingBounds.x + (screenCompetingBounds.width * 0.5);
        const hoverY = screenCompetingBounds.y + (screenCompetingBounds.height * 0.5);

        const hit = renderer.resolveInteractiveHitAtScreenPoint(hoverX, hoverY);

        expect(hit?.node?.id).toBe(competingNode.id);
        expect(hit?.rowHit).toBeNull();
    });

    it('keeps residual fallback row hovers when overlapping nodes sit behind the residual strip', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });

        const competingNode = createMatrixNode({
            id: 'background-overview-node',
            role: 'projection-weight',
            semantic: {
                componentKind: 'projection',
                layerIndex: 0,
                stage: 'qkv.q',
                role: 'projection-weight'
            },
            dimensions: { rows: 1, cols: 1 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.MHSA_Q
            }
        });
        const residualNode = createMatrixNode({
            id: 'foreground-residual-node',
            role: 'module-card',
            semantic: {
                componentKind: 'residual',
                layerIndex: 0,
                stage: 'incoming',
                role: 'module-card'
            },
            dimensions: { rows: 3, cols: 768 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            rowItems: [
                { label: 'Token A', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token B', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token C', gradientCss: 'rgba(80, 160, 255, 0.96)' }
            ],
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.RESIDUAL
            },
            metadata: {
                compactRows: {
                    compactWidth: 120,
                    rowHeight: 6,
                    rowGap: 0,
                    paddingX: 12,
                    paddingY: 8,
                    variant: VIEW2D_VECTOR_STRIP_VARIANT
                }
            }
        });

        const competingEntry = {
            bounds: { x: 0, y: 0, width: 96, height: 48 },
            contentBounds: { x: 0, y: 0, width: 96, height: 48 },
            layoutData: {}
        };
        const residualEntry = {
            bounds: { x: 0, y: 0, width: 168, height: 64 },
            contentBounds: { x: 0, y: 0, width: 168, height: 64 },
            layoutData: {
                innerPaddingX: 12,
                innerPaddingY: 8,
                compactWidth: 120,
                rowHeight: 6,
                rowGap: 0
            }
        };

        renderer.drawableNodes = [
            { node: competingNode, entry: competingEntry },
            { node: residualNode, entry: residualEntry }
        ];
        renderer.visibleDrawableNodes = renderer.drawableNodes;
        renderer.lastRenderState = {
            worldScale: 0.25,
            offsetX: 0,
            offsetY: 0
        };
        renderer.resolveInteractiveHitAtPoint = () => ({
            entry: {
                nodeId: competingNode.id
            },
            node: competingNode,
            rowHit: null,
            cellHit: null,
            columnHit: null
        });

        const hoverX = 12;
        const hoverY = (residualEntry.layoutData.innerPaddingY + residualEntry.layoutData.rowHeight + (residualEntry.layoutData.rowHeight * 0.5)) * renderer.lastRenderState.worldScale;
        const hit = renderer.resolveInteractiveHitAtScreenPoint(hoverX, hoverY);

        expect(hit?.node?.id).toBe(residualNode.id);
        expect(hit?.rowHit?.rowIndex).toBe(1);
    });

    it('renders focused compact-row selections with the same stroke highlight used for hovered rows', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });

        const matrixNode = createMatrixNode({
            id: 'focused-row-node',
            role: 'module-card',
            semantic: {
                componentKind: 'residual',
                layerIndex: 0,
                stage: 'incoming',
                role: 'module-card'
            },
            dimensions: { rows: 3, cols: 768 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            rowItems: [
                { label: 'Token A', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token B', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token C', gradientCss: 'rgba(80, 160, 255, 0.96)' }
            ],
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.RESIDUAL
            },
            metadata: {
                compactRows: {
                    compactWidth: 120,
                    rowHeight: 6,
                    rowGap: 0,
                    paddingX: 0,
                    paddingY: 0,
                    variant: VIEW2D_VECTOR_STRIP_VARIANT,
                    hoverScaleY: 1.16,
                    hoverGlowColor: 'rgba(255,255,255,0.08)',
                    hoverGlowBlur: 12,
                    hoverStrokeColor: 'rgba(255,255,255,0.10)'
                }
            }
        });

        renderer.setScene(createSceneModel({
            nodes: [matrixNode]
        }));

        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            },
            interactionState: {
                detailSceneFocus: {
                    activeNodeIds: [matrixNode.id],
                    rowSelections: [
                        { nodeId: matrixNode.id, rowIndex: 1 }
                    ]
                }
            }
        })).toBe(true);

        const entry = renderer.layout?.registry?.getNodeEntry(matrixNode.id);
        expect(entry?.contentBounds).toBeTruthy();
        expect(entry?.layoutData).toBeTruthy();

        const expectedY = entry.contentBounds.y + entry.layoutData.innerPaddingY + entry.layoutData.rowHeight + 0.5;
        const strokeRect = ctx.operations.find((operation) => (
            operation.type === 'strokeRect'
            && Math.abs(operation.y - expectedY) < 1.25
        ));

        expect(strokeRect).toBeTruthy();
    });

    it('resolves band hits for band-interactive compact-row vector strips', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });

        const compactRowNode = createMatrixNode({
            role: 'concat-output-matrix',
            semantic: {
                componentKind: 'output-projection',
                layerIndex: 0,
                stage: 'concatenate',
                role: 'concat-output-matrix'
            },
            dimensions: { rows: 3, cols: 768 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            rowItems: [
                { label: 'Token A', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token B', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token C', gradientCss: 'rgba(80, 160, 255, 0.96)' }
            ],
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.MHSA_HEAD_OUTPUT
            },
            metadata: {
                interactiveBandHit: true,
                compactRows: {
                    compactWidth: 144,
                    rowHeight: 12,
                    rowGap: 4,
                    paddingX: 0,
                    paddingY: 0,
                    variant: VIEW2D_VECTOR_STRIP_VARIANT,
                    bandCount: 12,
                    bandSeparatorOpacity: 0.22
                }
            }
        });

        renderer.setScene(createSceneModel({
            nodes: [compactRowNode],
            metadata: {
                visualContract: 'selection-panel-output-projection-v1'
            }
        }));

        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);

        const entry = renderer.layout?.registry?.getNodeEntry(compactRowNode.id);
        expect(entry?.contentBounds).toBeTruthy();
        expect(entry?.layoutData).toBeTruthy();

        const contentBounds = entry.contentBounds;
        const layoutData = entry.layoutData;
        const bandWidth = layoutData.compactWidth / 12;
        const hitX = contentBounds.x + layoutData.innerPaddingX + (bandWidth * 4.5);
        const hitY = contentBounds.y + layoutData.innerPaddingY + layoutData.rowHeight + layoutData.rowGap + (layoutData.rowHeight * 0.5);
        const bandHit = renderer.resolveInteractiveHitAtPoint(hitX, hitY);

        expect(bandHit?.cellHit?.rowIndex).toBe(1);
        expect(bandHit?.cellHit?.colIndex).toBe(4);
        expect(bandHit?.cellHit?.rowItem?.label).toBe('Token B');
    });

    it('draws a connector-style right-facing continuation arrow with a gap for the H_i matrix', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });

        const headOutputNode = createMatrixNode({
            role: 'attention-head-output',
            semantic: {
                componentKind: 'mhsa',
                layerIndex: 0,
                headIndex: 0,
                stage: 'head-output',
                role: 'attention-head-output'
            },
            dimensions: { rows: 3, cols: 64 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            rowItems: [
                { label: 'Token A', gradientCss: 'rgba(242, 136, 48, 0.96)' },
                { label: 'Token B', gradientCss: 'rgba(242, 136, 48, 0.96)' },
                { label: 'Token C', gradientCss: 'rgba(242, 136, 48, 0.96)' }
            ],
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.MHSA_HEAD_OUTPUT
            },
            metadata: {
                compactRows: {
                    compactWidth: 120,
                    rowHeight: 12,
                    rowGap: 8,
                    paddingX: 0,
                    paddingY: 0,
                    variant: VIEW2D_VECTOR_STRIP_VARIANT
                }
            }
        });

        renderer.setScene(createSceneModel({
            nodes: [headOutputNode],
            metadata: {
                visualContract: 'selection-panel-mhsa-v1'
            }
        }));

        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);

        const entry = renderer.layout?.registry?.getNodeEntry(headOutputNode.id);
        expect(entry?.contentBounds).toBeTruthy();

        const contentBounds = entry.contentBounds;
        const expectedCenterY = contentBounds.y + (contentBounds.height * 0.5);
        const arrowShaft = ctx.operations.find((operation) => {
            if (operation?.type !== 'stroke' || !Array.isArray(operation.path) || operation.path.length !== 2) {
                return false;
            }
            const [move, line] = operation.path;
            return move?.type === 'moveTo'
                && line?.type === 'lineTo'
                && move.x > (contentBounds.x + contentBounds.width)
                && line.x > move.x
                && Math.abs(move.y - expectedCenterY) <= 0.51
                && Math.abs(line.y - expectedCenterY) <= 0.51;
        });

        expect(arrowShaft).toBeTruthy();
        expect(arrowShaft.path[0].x - (contentBounds.x + contentBounds.width)).toBeGreaterThan(0);
    });

    it('keeps the H_i continuation arrow visible at low zoom', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });

        const headOutputNode = createMatrixNode({
            role: 'attention-head-output',
            semantic: {
                componentKind: 'mhsa',
                layerIndex: 0,
                headIndex: 0,
                stage: 'head-output',
                role: 'attention-head-output'
            },
            dimensions: { rows: 3, cols: 64 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            rowItems: [
                { label: 'Token A', gradientCss: 'rgba(242, 136, 48, 0.96)' },
                { label: 'Token B', gradientCss: 'rgba(242, 136, 48, 0.96)' },
                { label: 'Token C', gradientCss: 'rgba(242, 136, 48, 0.96)' }
            ],
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.MHSA_HEAD_OUTPUT
            },
            metadata: {
                compactRows: {
                    compactWidth: 120,
                    rowHeight: 12,
                    rowGap: 8,
                    paddingX: 0,
                    paddingY: 0,
                    variant: VIEW2D_VECTOR_STRIP_VARIANT
                }
            }
        });

        renderer.setScene(createSceneModel({
            nodes: [headOutputNode],
            metadata: {
                visualContract: 'selection-panel-mhsa-v1'
            }
        }));

        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            viewportTransform: {
                scale: 0.12,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);

        const entry = renderer.layout?.registry?.getNodeEntry(headOutputNode.id);
        expect(entry?.contentBounds).toBeTruthy();

        const contentBounds = entry.contentBounds;
        const arrowShaft = ctx.operations.find((operation) => {
            if (operation?.type !== 'stroke' || !Array.isArray(operation.path) || operation.path.length !== 2) {
                return false;
            }
            const [move, line] = operation.path;
            return move?.type === 'moveTo'
                && line?.type === 'lineTo'
                && move.x > (contentBounds.x + contentBounds.width)
                && line.x > move.x;
        });

        expect(arrowShaft).toBeTruthy();
    });

    it('draws a left-side incoming connector-style arrow with a gap for the parent X_ln matrix', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx);
        const renderer = new CanvasSceneRenderer({ canvas });

        const projectionSourceNode = createMatrixNode({
            role: 'projection-source-xln',
            semantic: {
                componentKind: 'mhsa',
                layerIndex: 0,
                headIndex: 0,
                stage: 'projection-source',
                role: 'projection-source-xln'
            },
            dimensions: { rows: 3, cols: 64 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            rowItems: [
                { label: 'Token A', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token B', gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { label: 'Token C', gradientCss: 'rgba(80, 160, 255, 0.96)' }
            ],
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.RESIDUAL
            },
            metadata: {
                compactRows: {
                    compactWidth: 120,
                    rowHeight: 12,
                    rowGap: 8,
                    paddingX: 0,
                    paddingY: 0,
                    variant: VIEW2D_VECTOR_STRIP_VARIANT
                }
            }
        });

        renderer.setScene(createSceneModel({
            nodes: [projectionSourceNode],
            metadata: {
                visualContract: 'selection-panel-mhsa-v1'
            }
        }));

        expect(renderer.render({
            width: 400,
            height: 240,
            dpr: 1,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);

        const entry = renderer.layout?.registry?.getNodeEntry(projectionSourceNode.id);
        expect(entry?.contentBounds).toBeTruthy();

        const contentBounds = entry.contentBounds;
        const expectedCenterY = contentBounds.y + (contentBounds.height * 0.5);
        const arrowShaft = ctx.operations.find((operation) => {
            if (operation?.type !== 'stroke' || !Array.isArray(operation.path) || operation.path.length !== 2) {
                return false;
            }
            const [move, line] = operation.path;
            return move?.type === 'moveTo'
                && line?.type === 'lineTo'
                && move.x < line.x
                && line.x < contentBounds.x
                && Math.abs(move.y - expectedCenterY) <= 0.51
                && Math.abs(line.y - expectedCenterY) <= 0.51;
        });

        expect(arrowShaft).toBeTruthy();
        expect(contentBounds.x - arrowShaft.path[1].x).toBeGreaterThan(0);
        expect(arrowShaft.path[1].x - arrowShaft.path[0].x).toBeGreaterThan(35);
    });

    it('keeps the shared x_ln incoming arrow shaft inside the fitted MHSA detail viewport', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx, 960, 540);
        const renderer = new CanvasSceneRenderer({ canvas });
        const scene = buildMhsaSceneModel({
            previewData: createMhsaDetailPreviewData(),
            layerIndex: 2,
            headIndex: 1,
            isSmallScreen: false
        });

        renderer.setScene(scene);

        expect(renderer.render({
            width: 960,
            height: 540,
            dpr: 1
        })).toBe(true);

        const projectionSourceNode = flattenSceneNodes(scene).find((node) => node.role === 'projection-source-xln') || null;
        const entry = renderer.layout?.registry?.getNodeEntry(projectionSourceNode?.id || '');
        expect(entry?.contentBounds).toBeTruthy();

        const contentBounds = entry.contentBounds;
        const expectedCenterY = contentBounds.y + (contentBounds.height * 0.5);
        const arrowShaft = ctx.operations.find((operation) => {
            if (operation?.type !== 'stroke' || !Array.isArray(operation.path) || operation.path.length !== 2) {
                return false;
            }
            const [move, line] = operation.path;
            return move?.type === 'moveTo'
                && line?.type === 'lineTo'
                && move.x < line.x
                && line.x < contentBounds.x
                && Math.abs(move.y - expectedCenterY) <= 0.51
                && Math.abs(line.y - expectedCenterY) <= 0.51;
        });

        expect(arrowShaft).toBeTruthy();
        expect(Math.min(arrowShaft.path[0].x, arrowShaft.path[1].x)).toBeGreaterThanOrEqual(0);
    });

    it('scales the MLP x_ln incoming edge connector with the scene zoom', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx, 960, 540);
        const renderer = new CanvasSceneRenderer({ canvas });
        const tokenRefs = ['Token A', 'Token B'].map((tokenLabel, rowIndex) => ({
            rowIndex,
            tokenIndex: rowIndex,
            tokenLabel
        }));
        const scene = buildMlpDetailSceneModel({
            activationSource: {
                getLayerLn2(_layerIndex = 0, _kind = 'shift', tokenIndex = 0, targetLength = D_MODEL) {
                    return Array.from({ length: targetLength }, (_, index) => Number(
                        ((tokenIndex * 0.1) + (index * 0.01)).toFixed(4)
                    ));
                }
            },
            mlpDetailTarget: {
                layerIndex: 0
            },
            tokenRefs
        });
        const inputNode = flattenSceneNodes(scene).find((node) => node.role === 'projection-source-xln') || null;

        renderer.setScene(scene);

        const renderAndMeasureArrow = (scale) => {
            ctx.operations.length = 0;
            expect(renderer.render({
                width: 960,
                height: 540,
                dpr: 1,
                viewportTransform: {
                    scale,
                    offsetX: 0,
                    offsetY: 0
                }
            })).toBe(true);

            const entry = renderer.layout?.registry?.getNodeEntry(inputNode?.id || '');
            const contentBounds = entry?.contentBounds || null;
            expect(contentBounds).toBeTruthy();
            return {
                ...measureHorizontalArrowScreenMetrics(ctx.operations, contentBounds, {
                    direction: 'left'
                }),
                contentBounds
            };
        };

        const zoomedOutArrow = renderAndMeasureArrow(0.24);
        const zoomedInArrow = renderAndMeasureArrow(1.16);
        const zoomRatio = 1.16 / 0.24;

        expect(zoomedOutArrow.tipX).toBeLessThan((zoomedOutArrow.contentBounds?.x || 0) - 4);
        expect(zoomedInArrow.tipX).toBeLessThan((zoomedInArrow.contentBounds?.x || 0) - 4);
        expect(zoomedInArrow.effectiveLength / Math.max(0.01, zoomedOutArrow.effectiveLength)).toBeCloseTo(zoomRatio, 0);
        expect(zoomedInArrow.effectiveLineWidth).toBeGreaterThan(zoomedOutArrow.effectiveLineWidth);
        expect(zoomedInArrow.effectiveHeadLength).toBeGreaterThan(zoomedOutArrow.effectiveHeadLength);
        expect(zoomedInArrow.effectiveHeadSpan).toBeGreaterThan(zoomedOutArrow.effectiveHeadSpan);
    });

    it('uses the scene connector layer for the MLP(x_ln) outgoing right-edge arrow', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx, 960, 540);
        const renderer = new CanvasSceneRenderer({ canvas });
        const tokenRefs = ['Token A', 'Token B'].map((tokenLabel, rowIndex) => ({
            rowIndex,
            tokenIndex: rowIndex,
            tokenLabel
        }));
        const scene = buildMlpDetailSceneModel({
            activationSource: {
                getLayerLn2(_layerIndex = 0, _kind = 'shift', tokenIndex = 0, targetLength = D_MODEL) {
                    return Array.from({ length: targetLength }, (_, index) => Number(
                        ((tokenIndex * 0.1) + (index * 0.01)).toFixed(4)
                    ));
                }
            },
            mlpDetailTarget: {
                layerIndex: 0
            },
            tokenRefs
        });
        const nodes = flattenSceneNodes(scene);
        const downOutputNode = nodes.find((node) => node.role === 'mlp-down-output') || null;
        const outgoingSpacerNode = nodes.find((node) => node.role === 'outgoing-arrow-spacer') || null;
        const connectorNode = nodes.find((node) => node.role === 'connector-mlp-down-output-outgoing') || null;

        renderer.setScene(scene);
        expect(renderer.render({
            width: 960,
            height: 540,
            dpr: 1,
            viewportTransform: {
                scale: 0.28,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);

        const connectorEntry = renderer.layout?.registry?.getConnectorEntry(connectorNode?.id || '');
        const outputEntry = renderer.layout?.registry?.getNodeEntry(downOutputNode?.id || '');
        const outputBounds = outputEntry?.contentBounds || null;

        expect(downOutputNode?.metadata?.disableEdgeOrnament).toBe(true);
        expect(connectorNode?.source).toMatchObject({
            nodeId: downOutputNode?.id,
            anchor: 'right'
        });
        expect(connectorNode?.target).toMatchObject({
            nodeId: outgoingSpacerNode?.id,
            anchor: 'right'
        });
        expect(connectorEntry?.pathPoints?.length).toBeGreaterThanOrEqual(2);
        expect(outputBounds).toBeTruthy();
        expect(connectorEntry?.pathPoints?.[0]?.x).toBeGreaterThan(
            (outputBounds?.x || 0) + (outputBounds?.width || 0) + 4
        );
        const lastPathPoint = connectorEntry?.pathPoints?.[connectorEntry.pathPoints.length - 1] || null;
        expect((lastPathPoint?.x || 0)).toBeGreaterThan(connectorEntry?.pathPoints?.[0]?.x || 0);
    });

    it('scales the MLP(x_ln) outgoing right-edge connector with the scene zoom', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx, 960, 540);
        const renderer = new CanvasSceneRenderer({ canvas });
        const tokenRefs = ['Token A', 'Token B'].map((tokenLabel, rowIndex) => ({
            rowIndex,
            tokenIndex: rowIndex,
            tokenLabel
        }));
        const scene = buildMlpDetailSceneModel({
            activationSource: {
                getLayerLn2(_layerIndex = 0, _kind = 'shift', tokenIndex = 0, targetLength = D_MODEL) {
                    return Array.from({ length: targetLength }, (_, index) => Number(
                        ((tokenIndex * 0.1) + (index * 0.01)).toFixed(4)
                    ));
                }
            },
            mlpDetailTarget: {
                layerIndex: 0
            },
            tokenRefs
        });
        const sceneNodes = flattenSceneNodes(scene);
        const outputNode = sceneNodes.find((node) => node.role === 'mlp-down-output') || null;
        const connectorNode = sceneNodes.find((node) => node.role === 'connector-mlp-down-output-outgoing') || null;
        const sceneLayout = buildSceneLayout(scene);
        const sceneConnectorEntry = sceneLayout?.registry?.getConnectorEntry(connectorNode?.id || '');
        const anchorPoint = sceneConnectorEntry?.pathPoints?.[sceneConnectorEntry.pathPoints.length - 1] || null;

        renderer.setScene(scene);

        const renderAndMeasureArrow = (scale) => {
            ctx.operations.length = 0;
            expect(renderer.render({
                width: 960,
                height: 540,
                dpr: 1,
                viewportTransform: {
                    scale,
                    offsetX: 900 - ((anchorPoint?.x || 0) * scale),
                    offsetY: 270 - ((anchorPoint?.y || 0) * scale)
                }
            })).toBe(true);

            const entry = renderer.layout?.registry?.getNodeEntry(outputNode?.id || '');
            const connectorEntry = renderer.layout?.registry?.getConnectorEntry(connectorNode?.id || '');
            const contentBounds = entry?.contentBounds || null;
            expect(contentBounds).toBeTruthy();
            return {
                ...measureConnectorArrowScreenMetrics(ctx.operations, connectorEntry?.pathPoints || []),
                contentBounds,
                connectorEntry
            };
        };

        const zoomedOutArrow = renderAndMeasureArrow(0.24);
        const zoomedInArrow = renderAndMeasureArrow(1.16);
        const zoomRatio = 1.16 / 0.24;

        expect(zoomedOutArrow.tipX).toBeGreaterThan(
            (zoomedOutArrow.contentBounds?.x || 0) + (zoomedOutArrow.contentBounds?.width || 0) + 4
        );
        expect(zoomedInArrow.tipX).toBeGreaterThan(
            (zoomedInArrow.contentBounds?.x || 0) + (zoomedInArrow.contentBounds?.width || 0) + 4
        );
        expect(zoomedInArrow.effectiveLength / Math.max(0.01, zoomedOutArrow.effectiveLength)).toBeCloseTo(zoomRatio, 0);
        expect(zoomedInArrow.effectiveLineWidth).toBeGreaterThan(zoomedOutArrow.effectiveLineWidth);
        expect(zoomedInArrow.effectiveHeadLength).toBeGreaterThan(zoomedOutArrow.effectiveHeadLength);
        expect(zoomedInArrow.effectiveHeadSpan).toBeGreaterThan(zoomedOutArrow.effectiveHeadSpan);
    });

    it('uses the scene connector layer for the MHSA x_ln incoming edge arrow', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx, 960, 540);
        const renderer = new CanvasSceneRenderer({ canvas });
        const scene = buildMhsaSceneModel({
            previewData: createMhsaDetailPreviewData(),
            layerIndex: 2,
            headIndex: 1,
            isSmallScreen: false
        });
        const nodes = flattenSceneNodes(scene);
        const projectionSourceNode = flattenSceneNodes(scene).find((node) => node.role === 'projection-source-xln') || null;
        const incomingSpacerNode = nodes.find((node) => node.role === 'incoming-arrow-spacer') || null;
        const connectorNode = nodes.find((node) => node.role === 'connector-source-xln') || null;

        renderer.setScene(scene);
        expect(renderer.render({
            width: 960,
            height: 540,
            dpr: 1,
            viewportTransform: {
                scale: 0.28,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);

        const connectorEntry = renderer.layout?.registry?.getConnectorEntry(connectorNode?.id || '');
        expect(projectionSourceNode?.metadata?.disableEdgeOrnament).toBe(true);
        expect(connectorNode?.source).toMatchObject({
            nodeId: incomingSpacerNode?.id,
            anchor: 'left'
        });
        expect(connectorNode?.target).toMatchObject({
            nodeId: projectionSourceNode?.id,
            anchor: 'left'
        });
        expect(connectorNode?.targetGap).toBeGreaterThan(0);
        expect(connectorEntry?.pathPoints?.length).toBeGreaterThanOrEqual(2);
        expect(connectorEntry?.pathPoints?.[connectorEntry.pathPoints.length - 1]?.x).toBeLessThan(
            (renderer.layout?.registry?.getNodeEntry(projectionSourceNode?.id || '')?.contentBounds?.x || 0) - 4
        );
    });

    it('uses the scene connector layer for the MHSA H_i continuation edge arrow', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx, 960, 540);
        const renderer = new CanvasSceneRenderer({ canvas });
        const scene = buildMhsaSceneModel({
            previewData: createMhsaDetailPreviewData(),
            layerIndex: 2,
            headIndex: 1,
            isSmallScreen: false
        });
        const nodes = flattenSceneNodes(scene);
        const headOutputNode = flattenSceneNodes(scene).find((node) => node.role === 'attention-head-output') || null;
        const outgoingSpacerNode = nodes.find((node) => node.role === 'outgoing-arrow-spacer') || null;
        const connectorNode = nodes.find((node) => node.role === 'connector-head-output-outgoing') || null;

        renderer.setScene(scene);
        expect(renderer.render({
            width: 960,
            height: 540,
            dpr: 1,
            viewportTransform: {
                scale: 0.28,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);

        const connectorEntry = renderer.layout?.registry?.getConnectorEntry(connectorNode?.id || '');
        expect(headOutputNode?.metadata?.disableEdgeOrnament).toBe(true);
        expect(connectorNode?.source).toMatchObject({
            nodeId: headOutputNode?.id,
            anchor: 'right'
        });
        expect(connectorNode?.sourceGap).toBeGreaterThan(0);
        expect(connectorNode?.target).toMatchObject({
            nodeId: outgoingSpacerNode?.id,
            anchor: 'right'
        });
        expect(connectorEntry?.pathPoints?.length).toBeGreaterThanOrEqual(2);
        expect(connectorEntry?.pathPoints?.[0]?.x).toBeGreaterThan(
            ((renderer.layout?.registry?.getNodeEntry(headOutputNode?.id || '')?.contentBounds?.x || 0)
                + (renderer.layout?.registry?.getNodeEntry(headOutputNode?.id || '')?.contentBounds?.width || 0)
                + 4)
        );
    });

    it('reuses the cached overview frame while the outer scene is being panned or zoomed', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx, 480, 320);
        const renderer = new CanvasSceneRenderer({ canvas });
        const scene = createSceneModel({
            nodes: [
                createMatrixNode({
                    role: 'overview-card',
                    semantic: {
                        componentKind: 'test',
                        stage: 'overview',
                        role: 'overview-card'
                    },
                    label: {
                        tex: 'X',
                        text: 'X'
                    },
                    dimensions: {
                        rows: 4,
                        cols: 4
                    },
                    presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
                    shape: VIEW2D_MATRIX_SHAPES.MATRIX,
                    visual: {
                        styleKey: VIEW2D_STYLE_KEYS.RESIDUAL,
                        disableCardSurfaceEffects: true
                    },
                    metadata: {
                        card: {
                            width: 180,
                            height: 120
                        }
                    }
                }),
                createTextNode({
                    text: 'Residual stream',
                    metadata: {
                        minScreenHeightPx: 0
                    }
                })
            ]
        });

        renderer.setScene(scene);
        expect(renderer.render({
            width: 480,
            height: 320,
            dpr: 1,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);
        renderer.overviewRenderCache = {
            ...renderer.overviewRenderCache,
            surface: renderer.overviewRenderCache.surface || { width: 800, height: 640 },
            ctx: renderer.overviewRenderCache.ctx || {},
            scene,
            dpr: 1,
            pixelWidth: 800,
            pixelHeight: 640,
            viewportPixelWidth: 480,
            viewportPixelHeight: 320,
            logicalWidth: 800,
            logicalHeight: 640,
            worldScale: 1,
            viewportOffsetX: 0,
            viewportOffsetY: 0,
            renderOffsetX: 160,
            renderOffsetY: 160,
            offsetX: 0,
            offsetY: 0
        };

        ctx.operations.length = 0;

        expect(renderer.render({
            width: 480,
            height: 320,
            dpr: 1,
            interacting: true,
            viewportTransform: {
                scale: 1.25,
                offsetX: 12,
                offsetY: 8
            }
        })).toBe(true);

        const drawImageOperation = ctx.operations.find((operation) => operation.type === 'drawImage') || null;
        expect(drawImageOperation).toBeTruthy();
        expect(drawImageOperation?.transformScaleX).toBeCloseTo(1.25, 6);
        expect(drawImageOperation?.transformScaleY).toBeCloseTo(1.25, 6);
        expect(drawImageOperation?.transformTranslateX).toBeCloseTo(-188, 6);
        expect(drawImageOperation?.transformTranslateY).toBeCloseTo(-192, 6);
        expect(ctx.operations.some((operation) => operation.type === 'fillText')).toBe(false);
    });

    it('rerenders the overview when a pan would expose uncovered cache edges', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx, 480, 320);
        const renderer = new CanvasSceneRenderer({ canvas });
        const scene = createSceneModel({
            nodes: [
                createMatrixNode({
                    role: 'overview-card',
                    semantic: {
                        componentKind: 'test',
                        stage: 'overview',
                        role: 'overview-card'
                    },
                    label: {
                        tex: 'X',
                        text: 'X'
                    },
                    dimensions: {
                        rows: 4,
                        cols: 4
                    },
                    presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
                    shape: VIEW2D_MATRIX_SHAPES.MATRIX,
                    visual: {
                        styleKey: VIEW2D_STYLE_KEYS.RESIDUAL,
                        disableCardSurfaceEffects: true
                    },
                    metadata: {
                        card: {
                            width: 180,
                            height: 120
                        }
                    }
                }),
                createTextNode({
                    text: 'Residual stream',
                    metadata: {
                        minScreenHeightPx: 0
                    }
                })
            ]
        });

        renderer.setScene(scene);
        renderer.overviewRenderCache = {
            ...renderer.overviewRenderCache,
            surface: { width: 480, height: 320 },
            ctx: {},
            scene,
            dpr: 1,
            pixelWidth: 480,
            pixelHeight: 320,
            worldScale: 1,
            offsetX: 0,
            offsetY: 0
        };

        expect(renderer.render({
            width: 480,
            height: 320,
            dpr: 1,
            interacting: true,
            viewportTransform: {
                scale: 1,
                offsetX: 140,
                offsetY: 0
            }
        })).toBe(true);

        const drawImageOperation = ctx.operations.find((operation) => operation.type === 'drawImage') || null;
        expect(drawImageOperation).toBeFalsy();
        expect(ctx.operations.some((operation) => (
            operation.type === 'fillText'
            && operation.text === 'Residual stream'
        ))).toBe(true);
    });

    it('reuses the cached overview frame during residual row-hover dimming and redraws only the affected node', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx, 480, 320);
        const renderer = new CanvasSceneRenderer({ canvas });
        const compactRowNode = createMatrixNode({
            id: 'hovered-row-node',
            role: 'overview-row-strip',
            semantic: {
                componentKind: 'test',
                stage: 'overview',
                role: 'overview-row-strip'
            },
            dimensions: { rows: 2, cols: 64 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            rowItems: [
                { gradientCss: 'rgba(80, 160, 255, 0.96)' },
                { gradientCss: 'rgba(80, 160, 255, 0.96)' }
            ],
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.MHSA_Q
            },
            metadata: {
                compactRows: {
                    compactWidth: 120,
                    rowHeight: 12,
                    rowGap: 4,
                    paddingX: 0,
                    paddingY: 0
                }
            }
        });
        const scene = createSceneModel({
            nodes: [
                compactRowNode,
                createTextNode({
                    text: 'Residual stream',
                    metadata: {
                        minScreenHeightPx: 0
                    }
                })
            ]
        });

        renderer.setScene(scene);
        expect(renderer.render({
            width: 480,
            height: 320,
            dpr: 1,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);
        renderer.overviewRenderCache = {
            ...renderer.overviewRenderCache,
            surface: renderer.overviewRenderCache.surface || { width: 480, height: 320 },
            ctx: renderer.overviewRenderCache.ctx || {},
            scene,
            dpr: 1,
            pixelWidth: 480,
            pixelHeight: 320,
            worldScale: 1,
            offsetX: 0,
            offsetY: 0
        };

        ctx.operations.length = 0;

        expect(renderer.render({
            width: 480,
            height: 320,
            dpr: 1,
            interacting: true,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            },
            interactionState: {
                hoveredRow: {
                    nodeId: compactRowNode.id,
                    rowIndex: 1
                },
                hoverDimStrength: 1,
                hoverRowBlend: 1
            }
        })).toBe(true);

        const drawImageIndex = ctx.operations.findIndex((operation) => operation.type === 'drawImage');
        expect(drawImageIndex).toBeGreaterThanOrEqual(0);
        expect(ctx.operations.some((operation) => (
            operation.type === 'fillText'
            && operation.text === 'Residual stream'
        ))).toBe(false);
        expect(ctx.operations.slice(drawImageIndex + 1).some((operation) => (
            operation.type === 'fill'
            || operation.type === 'fillRect'
            || operation.type === 'stroke'
            || operation.type === 'strokeRect'
        ))).toBe(true);
    });

    it('rerenders the overview during zoom interactions instead of scaling the cached frame', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx, 480, 320);
        const renderer = new CanvasSceneRenderer({ canvas });
        const scene = createSceneModel({
            nodes: [
                createMatrixNode({
                    role: 'overview-card',
                    semantic: {
                        componentKind: 'test',
                        stage: 'overview',
                        role: 'overview-card'
                    },
                    label: {
                        tex: 'X',
                        text: 'X'
                    },
                    dimensions: {
                        rows: 4,
                        cols: 4
                    },
                    presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
                    shape: VIEW2D_MATRIX_SHAPES.MATRIX,
                    visual: {
                        styleKey: VIEW2D_STYLE_KEYS.RESIDUAL,
                        disableCardSurfaceEffects: true
                    },
                    metadata: {
                        card: {
                            width: 180,
                            height: 120
                        }
                    }
                }),
                createTextNode({
                    text: 'Residual stream',
                    metadata: {
                        minScreenHeightPx: 0
                    }
                })
            ]
        });

        renderer.setScene(scene);
        expect(renderer.render({
            width: 480,
            height: 320,
            dpr: 1,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);
        renderer.overviewRenderCache = {
            ...renderer.overviewRenderCache,
            surface: renderer.overviewRenderCache.surface || { width: 480, height: 320 },
            ctx: renderer.overviewRenderCache.ctx || {},
            scene,
            dpr: 1,
            pixelWidth: 480,
            pixelHeight: 320,
            worldScale: 1,
            offsetX: 0,
            offsetY: 0
        };

        ctx.operations.length = 0;

        expect(renderer.render({
            width: 480,
            height: 320,
            dpr: 1,
            interacting: true,
            viewportTransform: {
                scale: 1.25,
                offsetX: 12,
                offsetY: 8
            },
            interactionState: {
                interactionKind: 'zoom'
            }
        })).toBe(true);

        const drawImageOperation = ctx.operations.find((operation) => operation.type === 'drawImage') || null;
        expect(drawImageOperation).toBeFalsy();
        expect(ctx.operations.some((operation) => operation.type === 'fillText')).toBe(true);
    });

    it('rerenders the overview during animated viewport transitions instead of scaling the cached frame', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx, 480, 320);
        const renderer = new CanvasSceneRenderer({ canvas });
        const scene = createSceneModel({
            nodes: [
                createMatrixNode({
                    role: 'overview-card',
                    semantic: {
                        componentKind: 'test',
                        stage: 'overview',
                        role: 'overview-card'
                    },
                    label: {
                        tex: 'X',
                        text: 'X'
                    },
                    dimensions: {
                        rows: 4,
                        cols: 4
                    },
                    presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
                    shape: VIEW2D_MATRIX_SHAPES.MATRIX,
                    visual: {
                        styleKey: VIEW2D_STYLE_KEYS.RESIDUAL,
                        disableCardSurfaceEffects: true
                    },
                    metadata: {
                        card: {
                            width: 180,
                            height: 120
                        }
                    }
                }),
                createTextNode({
                    text: 'Residual stream',
                    metadata: {
                        minScreenHeightPx: 0
                    }
                })
            ]
        });

        renderer.setScene(scene);
        expect(renderer.render({
            width: 480,
            height: 320,
            dpr: 1,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);
        renderer.overviewRenderCache = {
            ...renderer.overviewRenderCache,
            surface: renderer.overviewRenderCache.surface || { width: 480, height: 320 },
            ctx: renderer.overviewRenderCache.ctx || {},
            scene,
            dpr: 1,
            pixelWidth: 480,
            pixelHeight: 320,
            worldScale: 1,
            offsetX: 0,
            offsetY: 0
        };

        ctx.operations.length = 0;

        expect(renderer.render({
            width: 480,
            height: 320,
            dpr: 1,
            interacting: true,
            viewportTransform: {
                scale: 1.25,
                offsetX: 12,
                offsetY: 8
            },
            interactionState: {
                viewportAnimationActive: true
            }
        })).toBe(true);

        const drawImageOperation = ctx.operations.find((operation) => operation.type === 'drawImage') || null;
        expect(drawImageOperation).toBeFalsy();
        expect(ctx.operations.some((operation) => operation.type === 'fillText')).toBe(true);
    });

    it('applies overview focus dimming even when the overview cache path is unavailable', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx, 480, 320);
        const renderer = new CanvasSceneRenderer({ canvas });
        const focusedNode = createMatrixNode({
            role: 'overview-card-focused',
            semantic: {
                componentKind: 'test',
                stage: 'focused',
                role: 'overview-card'
            },
            label: {
                tex: 'A',
                text: 'A'
            },
            dimensions: {
                rows: 2,
                cols: 2
            },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.RESIDUAL,
                disableCardSurfaceEffects: true
            },
            metadata: {
                card: {
                    width: 140,
                    height: 96
                }
            }
        });
        const dimmedNode = createMatrixNode({
            role: 'overview-card-dimmed',
            semantic: {
                componentKind: 'test',
                stage: 'dimmed',
                role: 'overview-card'
            },
            label: {
                tex: 'B',
                text: 'B'
            },
            dimensions: {
                rows: 2,
                cols: 2
            },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.RESIDUAL,
                disableCardSurfaceEffects: true
            },
            metadata: {
                card: {
                    width: 140,
                    height: 96
                }
            }
        });
        const scene = createSceneModel({
            nodes: [
                createGroupNode({
                    direction: VIEW2D_LAYOUT_DIRECTIONS.ROW,
                    gap: 40,
                    children: [focusedNode, dimmedNode]
                })
            ]
        });

        renderer.setScene(scene);
        expect(renderer.render({
            width: 480,
            height: 320,
            dpr: 1,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            },
            interactionState: {
                overviewFocusTransition: {
                    currentFocus: {
                        activeNodeIds: [focusedNode.id],
                        inactiveOpacity: 0.18
                    },
                    focusBlend: 1,
                    dimStrength: 1
                }
            }
        })).toBe(true);

        const translucentFills = ctx.operations.filter((operation) => (
            (operation.type === 'fill' || operation.type === 'fillRect')
            && operation.globalAlpha < 0.99
        ));
        expect(translucentFills.length).toBeGreaterThan(0);
        expect(ctx.operations.some((operation) => operation.type === 'drawImage')).toBe(false);
    });

    it('reuses the visible overview index for cached row-hover redraws', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx, 480, 320);
        const renderer = new CanvasSceneRenderer({ canvas });
        const hoveredNode = createMatrixNode({
            role: 'overview-card-hovered',
            semantic: {
                componentKind: 'test',
                stage: 'overview',
                role: 'overview-card'
            },
            label: {
                tex: 'X',
                text: 'X'
            },
            dimensions: {
                rows: 4,
                cols: 4
            },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.RESIDUAL,
                disableCardSurfaceEffects: true
            },
            metadata: {
                card: {
                    width: 180,
                    height: 120
                }
            }
        });
        const targetNode = createMatrixNode({
            role: 'overview-card-target',
            semantic: {
                componentKind: 'test',
                stage: 'overview',
                role: 'overview-card'
            },
            label: {
                tex: 'Y',
                text: 'Y'
            },
            dimensions: {
                rows: 4,
                cols: 4
            },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.RESIDUAL,
                disableCardSurfaceEffects: true
            },
            metadata: {
                card: {
                    width: 180,
                    height: 120
                }
            }
        });
        const connectorNode = createConnectorNode({
            role: 'overview-hover-connector',
            semantic: {
                componentKind: 'test',
                stage: 'overview',
                role: 'overview-hover-connector'
            },
            source: createAnchorRef(hoveredNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
            target: createAnchorRef(targetNode.id, VIEW2D_ANCHOR_SIDES.LEFT),
            route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL,
                stroke: 'rgba(255, 255, 255, 0.84)'
            }
        });
        const scene = createSceneModel({
            nodes: [
                createGroupNode({
                    direction: VIEW2D_LAYOUT_DIRECTIONS.ROW,
                    gap: 40,
                    children: [hoveredNode, targetNode]
                }),
                createGroupNode({
                    direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
                    children: [connectorNode]
                })
            ]
        });

        renderer.setScene(scene);
        expect(renderer.render({
            width: 480,
            height: 320,
            dpr: 1,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);
        expect(renderer.overviewVisibleIndex?.nodesById?.get(hoveredNode.id)?.node?.id).toBe(hoveredNode.id);
        expect(renderer.overviewVisibleIndex?.connectorsByNodeId?.get(hoveredNode.id)?.[0]?.node?.id).toBe(connectorNode.id);

        renderer.drawTransformedOverviewRenderCache = () => true;
        ctx.operations.length = 0;

        expect(renderer.drawCachedOverviewRowHoverState({
            ctx,
            resolution: {
                width: 480,
                height: 320,
                dpr: 1,
                pixelWidth: 480,
                pixelHeight: 320
            },
            config: renderer.metrics || renderer.layout?.config || null,
            worldScale: renderer.lastRenderState?.worldScale || 1,
            detailScale: renderer.lastRenderState?.detailScale || 1,
            offsetX: renderer.lastRenderState?.offsetX || 0,
            offsetY: renderer.lastRenderState?.offsetY || 0,
            visibleDrawableNodes: [],
            visibleConnectors: [],
            overviewVisibleIndex: renderer.overviewVisibleIndex,
            interactionState: {
                hoveredRow: {
                    nodeId: hoveredNode.id,
                    rowIndex: 1
                }
            }
        })).toBe(true);

        expect(ctx.operations.some((operation) => operation.type === 'fillText' && operation.text === 'X')).toBe(true);
        expect(ctx.operations.some((operation) => operation.type === 'stroke')).toBe(true);
    });

    it('renders the 2D scene against a solid black background even when the scene background token is transparent', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx, 480, 320);
        const renderer = new CanvasSceneRenderer({ canvas });
        const scene = createSceneModel({
            nodes: [
                createMatrixNode({
                    role: 'overview-card',
                    semantic: {
                        componentKind: 'test',
                        stage: 'overview',
                        role: 'overview-card'
                    },
                    label: {
                        tex: 'X',
                        text: 'X'
                    },
                    dimensions: {
                        rows: 2,
                        cols: 2
                    },
                    presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
                    shape: VIEW2D_MATRIX_SHAPES.MATRIX,
                    visual: {
                        styleKey: VIEW2D_STYLE_KEYS.RESIDUAL,
                        disableCardSurfaceEffects: true
                    },
                    metadata: {
                        card: {
                            width: 140,
                            height: 96
                        }
                    }
                })
            ]
        });

        renderer.setScene(scene);
        expect(renderer.render({
            width: 480,
            height: 320,
            dpr: 1,
            viewportTransform: {
                scale: 1,
                offsetX: 0,
                offsetY: 0
            }
        })).toBe(true);

        const firstBackgroundFill = ctx.operations.find((operation) => operation.type === 'fillRect') || null;
        expect(firstBackgroundFill?.fillStyle).toBe('rgb(0, 0, 0)');
    });

    it('keeps connector paths unsnapped during interaction renders so zoom motion stays smooth', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx, 480, 320);
        const renderer = new CanvasSceneRenderer({ canvas });
        const sourceNode = createMatrixNode({
            role: 'source',
            semantic: {
                componentKind: 'test',
                stage: 'overview',
                role: 'source'
            },
            dimensions: { rows: 1, cols: 1 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.RESIDUAL
            },
            metadata: {
                hidden: true,
                card: {
                    width: 24,
                    height: 24,
                    cornerRadius: 0
                }
            }
        });
        const targetNode = createMatrixNode({
            role: 'target',
            semantic: {
                componentKind: 'test',
                stage: 'overview',
                role: 'target'
            },
            dimensions: { rows: 1, cols: 1 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.RESIDUAL
            },
            metadata: {
                hidden: true,
                card: {
                    width: 24,
                    height: 24,
                    cornerRadius: 0
                }
            }
        });
        const connectorNode = createConnectorNode({
            role: 'interaction-connector',
            semantic: {
                componentKind: 'test',
                stage: 'overview',
                role: 'interaction-connector'
            },
            source: createAnchorRef(sourceNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
            target: createAnchorRef(targetNode.id, VIEW2D_ANCHOR_SIDES.LEFT),
            route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL,
                stroke: 'rgba(255, 255, 255, 0.84)'
            }
        });
        const scene = createSceneModel({
            nodes: [
                createGroupNode({
                    direction: VIEW2D_LAYOUT_DIRECTIONS.ROW,
                    gap: 37,
                    children: [sourceNode, targetNode]
                }),
                createGroupNode({
                    direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
                    children: [connectorNode]
                })
            ]
        });

        renderer.setScene(scene);
        const connectorEntry = renderer.layout?.registry?.getConnectorEntry(connectorNode.id) || null;
        const rawStartPoint = connectorEntry?.pathPoints?.[0] || null;
        expect(rawStartPoint).toBeTruthy();

        expect(renderer.render({
            width: 480,
            height: 320,
            dpr: 1,
            interacting: true,
            viewportTransform: {
                scale: 1.3,
                offsetX: 18,
                offsetY: 11
            }
        })).toBe(true);

        const connectorStroke = ctx.operations.find((operation) => (
            operation.type === 'stroke'
            && operation.path?.some((segment) => segment.type === 'lineTo')
        )) || null;
        const firstMove = connectorStroke?.path?.find((segment) => segment.type === 'moveTo') || null;
        expect(firstMove?.x).toBeCloseTo(rawStartPoint?.x || 0, 6);
        expect(firstMove?.y).toBeCloseTo(rawStartPoint?.y || 0, 6);
    });

    it('keeps connector paths continuous after zoom interactions settle', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx, 480, 320);
        const renderer = new CanvasSceneRenderer({ canvas });
        const sourceNode = createMatrixNode({
            role: 'source',
            semantic: {
                componentKind: 'test',
                stage: 'overview',
                role: 'source'
            },
            dimensions: { rows: 1, cols: 1 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.RESIDUAL
            },
            metadata: {
                hidden: true,
                card: {
                    width: 24,
                    height: 24,
                    cornerRadius: 0
                }
            }
        });
        const targetNode = createMatrixNode({
            role: 'target',
            semantic: {
                componentKind: 'test',
                stage: 'overview',
                role: 'target'
            },
            dimensions: { rows: 1, cols: 1 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.RESIDUAL
            },
            metadata: {
                hidden: true,
                card: {
                    width: 24,
                    height: 24,
                    cornerRadius: 0
                }
            }
        });
        const connectorNode = createConnectorNode({
            role: 'settled-connector',
            semantic: {
                componentKind: 'test',
                stage: 'overview',
                role: 'settled-connector'
            },
            source: createAnchorRef(sourceNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
            target: createAnchorRef(targetNode.id, VIEW2D_ANCHOR_SIDES.LEFT),
            route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL,
                stroke: 'rgba(255, 255, 255, 0.84)'
            }
        });
        const scene = createSceneModel({
            nodes: [
                createGroupNode({
                    direction: VIEW2D_LAYOUT_DIRECTIONS.ROW,
                    gap: 37,
                    children: [sourceNode, targetNode]
                }),
                createGroupNode({
                    direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
                    children: [connectorNode]
                })
            ]
        });

        renderer.setScene(scene);
        const connectorItem = renderer.connectors.find((item) => item.node?.id === connectorNode.id) || null;
        connectorItem.entry.pathPoints = [
            { x: 10.13, y: 20.27 },
            { x: 72.89, y: 20.27 }
        ];

        expect(renderer.render({
            width: 480,
            height: 320,
            dpr: 1,
            viewportTransform: {
                scale: 1.3,
                offsetX: 18,
                offsetY: 11
            }
        })).toBe(true);

        const connectorStroke = ctx.operations.find((operation) => (
            operation.type === 'stroke'
            && operation.path?.some((segment) => segment.type === 'lineTo')
        )) || null;
        const firstMove = connectorStroke?.path?.find((segment) => segment.type === 'moveTo') || null;
        expect(firstMove?.x).toBeCloseTo(10.13, 6);
        expect(firstMove?.y).toBeCloseTo(20.27, 6);
    });

    it('keeps connector stroke width continuous across zoom thresholds', () => {
        const ctx = createMockContext();
        const canvas = createMockCanvas(ctx, 480, 320);
        const renderer = new CanvasSceneRenderer({ canvas });
        const sourceNode = createMatrixNode({
            role: 'source',
            semantic: {
                componentKind: 'test',
                stage: 'overview',
                role: 'source'
            },
            dimensions: { rows: 1, cols: 1 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.RESIDUAL
            },
            metadata: {
                hidden: true,
                card: {
                    width: 24,
                    height: 24,
                    cornerRadius: 0
                }
            }
        });
        const targetNode = createMatrixNode({
            role: 'target',
            semantic: {
                componentKind: 'test',
                stage: 'overview',
                role: 'target'
            },
            dimensions: { rows: 1, cols: 1 },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.RESIDUAL
            },
            metadata: {
                hidden: true,
                card: {
                    width: 24,
                    height: 24,
                    cornerRadius: 0
                }
            }
        });
        const connectorNode = createConnectorNode({
            role: 'continuous-width-connector',
            semantic: {
                componentKind: 'test',
                stage: 'overview',
                role: 'continuous-width-connector'
            },
            source: createAnchorRef(sourceNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
            target: createAnchorRef(targetNode.id, VIEW2D_ANCHOR_SIDES.LEFT),
            route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL,
                stroke: 'rgba(255, 255, 255, 0.84)'
            }
        });
        const scene = createSceneModel({
            nodes: [
                createGroupNode({
                    direction: VIEW2D_LAYOUT_DIRECTIONS.ROW,
                    gap: 37,
                    children: [sourceNode, targetNode]
                }),
                createGroupNode({
                    direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
                    children: [connectorNode]
                })
            ]
        });

        renderer.setScene(scene);

        const measureConnectorWidth = (scale) => {
            ctx.operations.length = 0;
            expect(renderer.render({
                width: 480,
                height: 320,
                dpr: 1,
                viewportTransform: {
                    scale,
                    offsetX: 18,
                    offsetY: 11
                },
                interactionState: {
                    interactionKind: 'zoom'
                }
            })).toBe(true);
            const connectorStroke = ctx.operations.find((operation) => (
                operation.type === 'stroke'
                && operation.path?.some((segment) => segment.type === 'lineTo')
            )) || null;
            return Number(connectorStroke?.effectiveLineWidth) || 0;
        };

        const belowThresholdWidth = measureConnectorWidth(0.69);
        const aboveThresholdWidth = measureConnectorWidth(0.71);
        expect(Math.abs(aboveThresholdWidth - belowThresholdWidth)).toBeLessThan(0.05);
    });
});
