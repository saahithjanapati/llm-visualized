import * as THREE from 'three';

const DEFAULT_CANVAS_FONT_FAMILY = 'ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial';
const DEFAULT_SECONDARY_TEXT_SCALE = 0.16;
const DEFAULT_TEXT_LINE_GAP_SCALE = 0.06;
const DEFAULT_PRIMARY_TEXT_Y_OFFSET_RATIO = 0.05;
const DEFAULT_SECONDARY_TEXT_BOTTOM_INSET_RATIO = 0.07;
const DEFAULT_SECONDARY_TEXT_COLOR = 0xd7d1c8;
const CAP_OFFSET = 0.05;
const FACE_OFFSET = 0.02;

function getStyleNumber(style, key, fallback) {
    const value = style?.[key];
    return Number.isFinite(value) ? value : fallback;
}

function getStyleColor(style, key, fallback) {
    const value = style?.[key];
    if (value instanceof THREE.Color) return value.clone();
    if (typeof value === 'number' || typeof value === 'string') {
        try {
            return new THREE.Color(value);
        } catch (_) {
            return new THREE.Color(fallback);
        }
    }
    return new THREE.Color(fallback);
}

function createCanvasTextMetrics(ctx, text, fontSize, fontFamily) {
    if (typeof text !== 'string' || !text.trim().length) {
        return { width: 0, height: 0 };
    }
    ctx.font = `600 ${fontSize}px ${fontFamily}`;
    return {
        width: Math.ceil(ctx.measureText(text).width),
        height: Math.ceil(fontSize * 1.12)
    };
}

function createGeometryTextLine(font, text, fontSize, textDepth) {
    if (!font || typeof text !== 'string' || !text.trim().length) return null;
    const textShapes = font.generateShapes(text, fontSize, 2);
    if (!Array.isArray(textShapes) || !textShapes.length) return null;

    const textGeo = new THREE.ExtrudeGeometry(textShapes, {
        depth: textDepth,
        curveSegments: 4,
        bevelEnabled: false
    });
    textGeo.computeBoundingBox();
    textGeo.computeVertexNormals();
    const bounds = textGeo.boundingBox;
    if (!bounds || !Number.isFinite(bounds.max.x) || !Number.isFinite(bounds.min.x)) {
        textGeo.dispose();
        return null;
    }

    const width = Math.max(0, bounds.max.x - bounds.min.x);
    const height = Math.max(0, bounds.max.y - bounds.min.y);
    const centerX = (bounds.min.x + bounds.max.x) / 2;
    const centerY = (bounds.min.y + bounds.max.y) / 2;

    textGeo.translate(-centerX, -centerY, -textDepth / 2);
    textGeo.computeBoundingBox();

    const faceGeo = new THREE.ShapeGeometry(textShapes);
    faceGeo.computeVertexNormals();
    faceGeo.translate(-centerX, -centerY, 0);

    return { textGeo, faceGeo, width, height };
}

function disposeGeometryTextLine(line) {
    if (!line) return;
    if (line.textGeo) line.textGeo.dispose();
    if (line.faceGeo) line.faceGeo.dispose();
}

function buildGeometryTextLineGroup(line, textMat, textCullMat, textDepth) {
    const group = new THREE.Group();
    const textMesh = new THREE.Mesh(line.textGeo, [textCullMat, textMat]);
    group.add(textMesh);

    const frontFace = new THREE.Mesh(line.faceGeo, textMat);
    frontFace.position.z = textDepth / 2 + FACE_OFFSET;

    const backFace = new THREE.Mesh(line.faceGeo, textMat);
    backFace.position.z = -textDepth / 2 - FACE_OFFSET;

    group.add(frontFace, backFace);
    return group;
}

export function buildRoundedRectShape(width, height, radius) {
    const clampedRadius = Math.max(0, Math.min(radius, Math.min(width, height) / 2 - 1));
    const halfW = width / 2;
    const halfH = height / 2;
    const shape = new THREE.Shape();
    shape.moveTo(-halfW + clampedRadius, -halfH);
    shape.lineTo(halfW - clampedRadius, -halfH);
    shape.quadraticCurveTo(halfW, -halfH, halfW, -halfH + clampedRadius);
    shape.lineTo(halfW, halfH - clampedRadius);
    shape.quadraticCurveTo(halfW, halfH, halfW - clampedRadius, halfH);
    shape.lineTo(-halfW + clampedRadius, halfH);
    shape.quadraticCurveTo(-halfW, halfH, -halfW, halfH - clampedRadius);
    shape.lineTo(-halfW, -halfH + clampedRadius);
    shape.quadraticCurveTo(-halfW, -halfH, -halfW + clampedRadius, -halfH);
    shape.closePath();
    return shape;
}

export function createTokenChipMesh({
    labelText = '',
    secondaryText = '',
    style = {},
    font = null,
    canvasFontFamily = DEFAULT_CANVAS_FONT_FAMILY
} = {}) {
    const primaryText = typeof labelText === 'string' ? labelText : '';
    const auxiliaryText = typeof secondaryText === 'string' ? secondaryText : '';
    const hasSecondaryText = auxiliaryText.trim().length > 0;

    const primaryTextSize = getStyleNumber(style, 'textSize', 52);
    const secondaryTextSize = hasSecondaryText
        ? getStyleNumber(style, 'secondaryTextSize', Math.max(12, primaryTextSize * DEFAULT_SECONDARY_TEXT_SCALE))
        : 0;
    const desiredTextDepth = getStyleNumber(style, 'textDepth', 0);
    const chipDepth = getStyleNumber(style, 'depth', desiredTextDepth);
    const textDepth = Number.isFinite(chipDepth) ? chipDepth + CAP_OFFSET * 2 : desiredTextDepth;
    const primaryTextColor = getStyleColor(style, 'textColor', 0xffffff);
    const secondaryTextColor = getStyleColor(style, 'secondaryTextColor', DEFAULT_SECONDARY_TEXT_COLOR);

    let primaryLine = createGeometryTextLine(font, primaryText, primaryTextSize, textDepth);
    let secondaryLine = hasSecondaryText
        ? createGeometryTextLine(font, auxiliaryText, secondaryTextSize, textDepth)
        : null;

    const canUseGeometryText = !!primaryLine && (!hasSecondaryText || !!secondaryLine);

    let contentWidth = 0;
    let contentHeight = 0;
    let primaryTextMat = null;
    let secondaryTextMat = null;
    let textCullMat = null;
    let textGeo = null;
    let textTexture = null;
    let textPlaneMat = null;
    let primaryMetrics = null;
    let secondaryMetrics = null;

    if (canUseGeometryText) {
        contentWidth = Math.max(primaryLine?.width || 0, secondaryLine?.width || 0);
        contentHeight = Math.max(
            0,
            primaryLine?.height || 0,
            hasSecondaryText
                ? Math.max(0, primaryLine?.height || 0) + Math.max(0, secondaryLine?.height || 0)
                : 0
        );
    } else {
        disposeGeometryTextLine(primaryLine);
        disposeGeometryTextLine(secondaryLine);
        primaryLine = null;
        secondaryLine = null;

        const measureCanvas = document.createElement('canvas');
        const measureCtx = measureCanvas.getContext('2d');
        primaryMetrics = createCanvasTextMetrics(measureCtx, primaryText, primaryTextSize, canvasFontFamily);
        secondaryMetrics = hasSecondaryText
            ? createCanvasTextMetrics(measureCtx, auxiliaryText, secondaryTextSize, canvasFontFamily)
            : { width: 0, height: 0 };

        contentWidth = Math.max(primaryMetrics.width, secondaryMetrics.width);
        contentHeight = Math.max(
            primaryMetrics.height,
            hasSecondaryText ? primaryMetrics.height + secondaryMetrics.height : 0
        );
    }

    const chipWidth = Math.max(getStyleNumber(style, 'minWidth', 220), contentWidth + getStyleNumber(style, 'padding', 80));
    const chipHeight = Number.isFinite(style?.height)
        ? style.height
        : Math.max(getStyleNumber(style, 'minHeight', 100), contentHeight + getStyleNumber(style, 'padding', 80));
    const primaryTextYOffset = hasSecondaryText
        ? getStyleNumber(style, 'primaryTextYOffset', Math.max(4, chipHeight * DEFAULT_PRIMARY_TEXT_Y_OFFSET_RATIO))
        : 0;
    const secondaryTextBottomInset = hasSecondaryText
        ? getStyleNumber(
            style,
            'secondaryTextBottomInset',
            Math.max(6, chipHeight * DEFAULT_SECONDARY_TEXT_BOTTOM_INSET_RATIO)
        )
        : 0;
    const textLineGap = hasSecondaryText
        ? getStyleNumber(style, 'textLineGap', Math.max(4, primaryTextSize * DEFAULT_TEXT_LINE_GAP_SCALE))
        : 0;
    const chipRadius = Math.min(getStyleNumber(style, 'cornerRadius', 18), Math.min(chipWidth, chipHeight) / 2 - 1);
    const chipShape = buildRoundedRectShape(chipWidth, chipHeight, chipRadius);

    const chipGeo = new THREE.ExtrudeGeometry(chipShape, {
        depth: chipDepth,
        bevelEnabled: false
    });
    chipGeo.translate(0, 0, -chipDepth / 2);
    chipGeo.computeVertexNormals();

    const chipMat = new THREE.MeshStandardMaterial({
        color: 0xf2e8d5,
        roughness: 0.84,
        metalness: 0.01,
        emissive: 0x000000,
        emissiveIntensity: 0,
        side: THREE.DoubleSide
    });
    const chipMesh = new THREE.Mesh(chipGeo, chipMat);

    const capMat = chipMat.clone();
    capMat.polygonOffset = false;
    capMat.polygonOffsetFactor = 0;
    capMat.polygonOffsetUnits = 0;
    const capGeo = new THREE.ShapeGeometry(chipShape);
    capGeo.computeVertexNormals();
    const frontCap = new THREE.Mesh(capGeo, capMat);
    frontCap.position.z = chipDepth / 2 + CAP_OFFSET;
    const backCap = new THREE.Mesh(capGeo, capMat);
    backCap.position.z = -chipDepth / 2 - CAP_OFFSET;
    backCap.rotation.y = Math.PI;

    const group = new THREE.Group();
    group.add(chipMesh, frontCap, backCap);

    if (canUseGeometryText && primaryLine) {
        primaryTextMat = new THREE.MeshBasicMaterial({
            color: primaryTextColor,
            side: THREE.DoubleSide,
            depthWrite: true,
            depthTest: true,
            toneMapped: false,
            polygonOffset: true,
            polygonOffsetFactor: -0.5,
            polygonOffsetUnits: -0.5
        });
        primaryTextMat.userData = {
            ...(primaryTextMat.userData || {}),
            tokenChipTextRole: 'primary'
        };
        textCullMat = primaryTextMat.clone();
        textCullMat.userData = {
            ...(textCullMat.userData || {}),
            tokenChipTextRole: 'cull'
        };
        textCullMat.colorWrite = false;
        textCullMat.depthWrite = false;
        textCullMat.transparent = true;
        textCullMat.opacity = 0;
        secondaryTextMat = hasSecondaryText
            ? primaryTextMat.clone()
            : null;
        if (secondaryTextMat) {
            secondaryTextMat.color.copy(secondaryTextColor);
            secondaryTextMat.userData = {
                ...(secondaryTextMat.userData || {}),
                tokenChipTextRole: 'secondary',
                tokenChipTextColor: secondaryTextColor.getHex()
            };
        }

        const textGroup = new THREE.Group();
        const primaryGroup = buildGeometryTextLineGroup(primaryLine, primaryTextMat, textCullMat, textDepth);
        primaryGroup.position.y = primaryTextYOffset;
        if (hasSecondaryText && secondaryLine) {
            const secondaryGroup = buildGeometryTextLineGroup(
                secondaryLine,
                secondaryTextMat || primaryTextMat,
                textCullMat,
                textDepth
            );
            secondaryGroup.position.y = -chipHeight / 2 + secondaryTextBottomInset + secondaryLine.height / 2;
            const minPrimaryY = secondaryGroup.position.y + secondaryLine.height / 2 + textLineGap + primaryLine.height / 2;
            if (primaryGroup.position.y < minPrimaryY) {
                primaryGroup.position.y = minPrimaryY;
            }
            textGroup.add(primaryGroup, secondaryGroup);
        } else {
            textGroup.add(primaryGroup);
        }
        group.add(textGroup);
    } else {
        const canvas = document.createElement('canvas');
        const pixelScale = Math.max(2, getStyleNumber(style, 'canvasResolutionScale', 4));
        canvas.width = Math.max(256, Math.ceil(chipWidth * pixelScale));
        canvas.height = Math.max(128, Math.ceil(chipHeight * pixelScale));
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;

        ctx.fillStyle = primaryTextColor.getStyle();
        ctx.font = `600 ${Math.round(primaryTextSize * pixelScale)}px ${canvasFontFamily}`;
        ctx.fillText(primaryText, centerX, centerY - primaryTextYOffset * pixelScale);

        if (hasSecondaryText) {
            const secondaryY = centerY + (chipHeight / 2 - secondaryTextBottomInset - (secondaryMetrics?.height || secondaryTextSize) / 2) * pixelScale;
            ctx.fillStyle = secondaryTextColor.getStyle();
            ctx.font = `600 ${Math.round(secondaryTextSize * pixelScale)}px ${canvasFontFamily}`;
            ctx.fillText(auxiliaryText, centerX, secondaryY);
        }

        textTexture = new THREE.CanvasTexture(canvas);
        textTexture.minFilter = THREE.LinearFilter;
        textTexture.magFilter = THREE.LinearFilter;
        textTexture.needsUpdate = true;

        textGeo = new THREE.PlaneGeometry(chipWidth * 0.9, chipHeight * 0.9);
        textPlaneMat = new THREE.MeshBasicMaterial({
            map: textTexture,
            transparent: true,
            depthWrite: true,
            depthTest: true,
            toneMapped: false,
            polygonOffset: true,
            polygonOffsetFactor: -0.5,
            polygonOffsetUnits: -0.5,
            side: THREE.DoubleSide
        });
        const textMesh = new THREE.Mesh(textGeo, textPlaneMat);
        textMesh.position.z = chipDepth / 2 + getStyleNumber(style, 'textOffset', 0.6);
        group.add(textMesh);
    }

    const scaleFactor = getStyleNumber(style, 'scale', 1);
    if (scaleFactor > 0 && scaleFactor !== 1) {
        group.scale.setScalar(scaleFactor);
    }

    group.userData = group.userData || {};
    group.userData.tokenChipStyle = style && typeof style === 'object'
        ? { ...style }
        : {};
    group.userData.size = {
        width: chipWidth * (scaleFactor > 0 ? scaleFactor : 1),
        height: chipHeight * (scaleFactor > 0 ? scaleFactor : 1)
    };

    return {
        group,
        dispose: () => {
            chipGeo.dispose();
            chipMat.dispose();
            capGeo.dispose();
            capMat.dispose();
            disposeGeometryTextLine(primaryLine);
            disposeGeometryTextLine(secondaryLine);
            if (textGeo) textGeo.dispose();
            if (primaryTextMat) primaryTextMat.dispose();
            if (secondaryTextMat) secondaryTextMat.dispose();
            if (textCullMat) textCullMat.dispose();
            if (textPlaneMat) textPlaneMat.dispose();
            if (textTexture) textTexture.dispose();
        }
    };
}
