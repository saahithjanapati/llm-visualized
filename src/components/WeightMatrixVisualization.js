import * as THREE from 'three';
import { toCreasedNormals } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import {
    QUALITY_PRESET,
    VECTOR_DEPTH_SPACING,
    USE_GLB_MATERIALS,
    USE_INSTANCED_MATRIX_SLICES
} from '../utils/constants.js';
import {
    createSciFiMaterial,
    updateSciFiDimensions,
    updateSciFiMaterialColor,
    updateSciFiMaterialUniforms
} from '../utils/sciFiMaterial.js';

const WM_GEOMETRY_VERSION = 'wm-fp-v10';
const EPSILON = 1e-5;

const __geometryCache = new Map();
const __capFrontCache = new Map();
const __capBackCache = new Map();
const __materialCache = new Map();
const __sliceWallCache = new Map();
const SLIT_SHADER_STABILITY_UNIFORMS = {
    stripeStrength: 0,
    depthAccentStrength: 0,
    scanlineStrength: 0,
    glintStrength: 0,
    noiseStrength: 0
};

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function clamp01(value) {
    return clamp(value, 0, 1);
}

function getAttrComponent(attr, index, component) {
    if (attr.isInterleavedBufferAttribute) {
        if (component === 0) return attr.getX(index);
        if (component === 1) return attr.getY(index);
        if (component === 2) return attr.getZ(index);
        if (component === 3) return attr.getW(index);
    }
    return attr.array[index * attr.itemSize + component];
}

function stripSliceEndCaps(geometry, depth) {
    if (!geometry || !geometry.attributes || !geometry.attributes.position) {
        return geometry;
    }

    const src = geometry.index ? geometry.toNonIndexed() : geometry;
    const shouldDisposeSrc = !!geometry.index;
    const posAttr = src.attributes.position;
    const attrNames = Object.keys(src.attributes);
    const buffers = {};
    const itemSizes = {};
    const normalized = {};
    const constructors = {};

    for (const name of attrNames) {
        const attr = src.attributes[name];
        buffers[name] = [];
        itemSizes[name] = attr.itemSize;
        normalized[name] = attr.normalized;
        constructors[name] = attr.array.constructor;
    }

    const zFront = depth / 2;
    const zBack = -depth / 2;
    const eps = Math.max(1e-3, depth * 0.001);

    for (let i = 0; i < posAttr.count; i += 3) {
        const z0 = posAttr.getZ(i);
        const z1 = posAttr.getZ(i + 1);
        const z2 = posAttr.getZ(i + 2);
        const isFront = Math.abs(z0 - zFront) < eps
            && Math.abs(z1 - zFront) < eps
            && Math.abs(z2 - zFront) < eps;
        const isBack = Math.abs(z0 - zBack) < eps
            && Math.abs(z1 - zBack) < eps
            && Math.abs(z2 - zBack) < eps;

        if (isFront || isBack) {
            continue;
        }

        for (const name of attrNames) {
            const attr = src.attributes[name];
            const itemSize = attr.itemSize;
            for (let v = 0; v < 3; v++) {
                const idx = i + v;
                for (let c = 0; c < itemSize; c++) {
                    buffers[name].push(getAttrComponent(attr, idx, c));
                }
            }
        }
    }

    const out = new THREE.BufferGeometry();
    for (const name of attrNames) {
        const ctor = constructors[name] || Float32Array;
        const array = new ctor(buffers[name]);
        out.setAttribute(name, new THREE.BufferAttribute(array, itemSizes[name], normalized[name]));
    }

    out.computeBoundingBox();
    out.computeBoundingSphere();

    if (shouldDisposeSrc) {
        src.dispose();
    }

    return out;
}

function getCacheKey(
    width,
    height,
    depth,
    topWidthFactor,
    cornerRadius,
    numberOfSlits,
    slitWidth,
    slitDepthFactor,
    slitBottomWidthFactor,
    slitTopWidthFactor
) {
    return [
        width,
        height,
        depth,
        topWidthFactor,
        cornerRadius,
        numberOfSlits,
        slitWidth,
        slitDepthFactor,
        slitBottomWidthFactor,
        slitTopWidthFactor,
        QUALITY_PRESET,
        WM_GEOMETRY_VERSION
    ].join('|');
}

function mapLegacyCacheKeyToCurrent(cacheKey) {
    if (typeof cacheKey !== 'string' || cacheKey.length === 0) return cacheKey;
    if (cacheKey.endsWith(`|${WM_GEOMETRY_VERSION}`)) return cacheKey;
    return `${cacheKey}|${WM_GEOMETRY_VERSION}`;
}

function computeSafeCornerRadius(width, height, topWidthFactor, cornerRadius) {
    const hw = width / 2;
    const hTopW = (width * topWidthFactor) / 2;

    const horizontalDiff = hw - hTopW;
    const maxBottomRadius = horizontalDiff > 0 ? horizontalDiff : hw;
    const maxTopRadius = hTopW;
    const sideLen = Math.hypot(hw - hTopW, height);
    const maxSideRadius = sideLen / 2;

    return Math.max(0, Math.min(cornerRadius, maxBottomRadius, maxTopRadius, maxSideRadius));
}

function createTrapezoidShape(width, height, topWidthFactor, cornerRadius = 0) {
    const shape = new THREE.Shape();

    const hw = width / 2;
    const hTopW = (width * topWidthFactor) / 2;
    const hh = height / 2;
    const cr = computeSafeCornerRadius(width, height, topWidthFactor, cornerRadius);

    if (cr < EPSILON) {
        shape.moveTo(-hw, -hh);
        shape.lineTo(hw, -hh);
        shape.lineTo(hTopW, hh);
        shape.lineTo(-hTopW, hh);
        shape.closePath();
        return shape;
    }

    const blStart = new THREE.Vector2(-hw + cr, -hh);
    const brStart = new THREE.Vector2(hw - cr, -hh);
    const trEnd = new THREE.Vector2(hTopW - cr, hh);
    const tlEnd = new THREE.Vector2(-hTopW + cr, hh);

    const rightSideDir = new THREE.Vector2(hTopW - hw, 2 * hh).normalize();
    const leftSideDir = new THREE.Vector2(-hTopW + hw, 2 * hh).normalize();

    const brSidePt = new THREE.Vector2(hw, -hh).addScaledVector(rightSideDir, cr);
    const trSidePt = new THREE.Vector2(hTopW, hh).addScaledVector(rightSideDir.clone().negate(), cr);
    const tlSidePt = new THREE.Vector2(-hTopW, hh).addScaledVector(leftSideDir.clone().negate(), cr);
    const blSidePt = new THREE.Vector2(-hw, -hh).addScaledVector(leftSideDir, cr);

    shape.moveTo(blStart.x, blStart.y);
    shape.lineTo(brStart.x, brStart.y);
    shape.quadraticCurveTo(hw, -hh, brSidePt.x, brSidePt.y);
    shape.lineTo(trSidePt.x, trSidePt.y);
    shape.quadraticCurveTo(hTopW, hh, trEnd.x, trEnd.y);
    shape.lineTo(tlEnd.x, tlEnd.y);
    shape.quadraticCurveTo(-hTopW, hh, tlSidePt.x, tlSidePt.y);
    shape.lineTo(blSidePt.x, blSidePt.y);
    shape.quadraticCurveTo(-hw, -hh, blStart.x, blStart.y);
    shape.closePath();

    return shape;
}

function createCapGeometry(width, height, topWidthFactor, cornerRadius) {
    const shape = createTrapezoidShape(width, height, topWidthFactor, cornerRadius);
    const capGeometry = new THREE.ShapeGeometry(shape);
    capGeometry.center();
    capGeometry.computeBoundingBox();
    capGeometry.computeBoundingSphere();
    return capGeometry;
}

function buildMergedSlitRanges(depth, numberOfSlits, slitWidth) {
    const slitCount = Math.max(0, Math.floor(numberOfSlits || 0));
    if (slitCount < 1 || slitWidth <= EPSILON || depth <= EPSILON) {
        return [];
    }

    const halfDepth = depth / 2;
    const spacing = depth / (slitCount + 1);
    const minGap = Math.max(1.0, spacing * 0.08);
    const maxSlitWidth = Math.max(EPSILON * 2, spacing - minGap);
    const halfSlit = Math.min(slitWidth, maxSlitWidth) / 2;

    const ranges = [];
    for (let i = 0; i < slitCount; i++) {
        const center = -halfDepth + spacing * (i + 1);
        const z0 = clamp(center - halfSlit, -halfDepth + EPSILON, halfDepth - EPSILON);
        const z1 = clamp(center + halfSlit, -halfDepth + EPSILON, halfDepth - EPSILON);
        if (z1 - z0 > EPSILON) {
            ranges.push({ z0, z1 });
        }
    }

    ranges.sort((a, b) => a.z0 - b.z0);
    const merged = [];
    for (const current of ranges) {
        const prev = merged[merged.length - 1];
        if (!prev || current.z0 > prev.z1 + EPSILON) {
            merged.push({ ...current });
            continue;
        }
        prev.z1 = Math.max(prev.z1, current.z1);
    }

    return merged;
}

function buildDepthBands(depth, slitRanges) {
    const halfDepth = depth / 2;
    const points = [-halfDepth, halfDepth];

    for (const range of slitRanges) {
        points.push(range.z0, range.z1);
    }

    points.sort((a, b) => a - b);

    const unique = [];
    for (const point of points) {
        if (!unique.length || Math.abs(unique[unique.length - 1] - point) > EPSILON) {
            unique.push(point);
        }
    }

    const bands = [];
    for (let i = 0; i < unique.length - 1; i++) {
        const z0 = unique[i];
        const z1 = unique[i + 1];
        if (z1 - z0 <= EPSILON) continue;

        const mid = (z0 + z1) / 2;
        const insideSlit = slitRanges.some(range => mid > range.z0 + EPSILON && mid < range.z1 - EPSILON);
        bands.push({ z0, z1, insideSlit });
    }

    return bands;
}

function widthAtY(y, width, height, topWidthFactor) {
    const hh = height / 2;
    if (hh <= EPSILON) return width;

    const t = clamp01((y + hh) / (2 * hh));
    const topWidth = width * topWidthFactor;
    return THREE.MathUtils.lerp(width, topWidth, t);
}

function slitWidthFactorAtY(y, height, slitBottomWidthFactor, slitTopWidthFactor) {
    const hh = height / 2;
    if (hh <= EPSILON) return slitBottomWidthFactor;

    const t = clamp01((y + hh) / (2 * hh));
    return THREE.MathUtils.lerp(slitBottomWidthFactor, slitTopWidthFactor, t);
}

function resolveHoleHalfWidth(y, width, height, topWidthFactor, slitBottomWidthFactor, slitTopWidthFactor) {
    const outerHalf = Math.max(0, widthAtY(y, width, height, topWidthFactor) / 2);
    const factor = Math.max(0, slitWidthFactorAtY(y, height, slitBottomWidthFactor, slitTopWidthFactor));

    const requestedHoleHalf = outerHalf * factor;
    const requiredWallHalf = Math.max(1.5, outerHalf * 0.08);
    const maxHoleHalf = Math.max(0, outerHalf - requiredWallHalf);

    return Math.min(requestedHoleHalf, maxHoleHalf);
}

class TriangleBuilder {
    constructor() {
        this.positions = [];
        this.normals = [];
    }

    static computeNormal(a, b, c) {
        const abx = b[0] - a[0];
        const aby = b[1] - a[1];
        const abz = b[2] - a[2];

        const acx = c[0] - a[0];
        const acy = c[1] - a[1];
        const acz = c[2] - a[2];

        let nx = aby * acz - abz * acy;
        let ny = abz * acx - abx * acz;
        let nz = abx * acy - aby * acx;

        const len = Math.hypot(nx, ny, nz);
        if (len <= EPSILON) {
            return [0, 0, 0];
        }

        nx /= len;
        ny /= len;
        nz /= len;

        return [nx, ny, nz];
    }

    addTriangle(a, b, c, targetNormal = null) {
        let v1 = b;
        let v2 = c;

        let normal = TriangleBuilder.computeNormal(a, v1, v2);
        if (normal[0] === 0 && normal[1] === 0 && normal[2] === 0) return;

        if (targetNormal) {
            const dot = normal[0] * targetNormal[0] + normal[1] * targetNormal[1] + normal[2] * targetNormal[2];
            if (dot < 0) {
                v1 = c;
                v2 = b;
                normal = TriangleBuilder.computeNormal(a, v1, v2);
                if (normal[0] === 0 && normal[1] === 0 && normal[2] === 0) return;
            }
        }

        this.positions.push(
            a[0], a[1], a[2],
            v1[0], v1[1], v1[2],
            v2[0], v2[1], v2[2]
        );

        this.normals.push(
            normal[0], normal[1], normal[2],
            normal[0], normal[1], normal[2],
            normal[0], normal[1], normal[2]
        );
    }

    addQuad(a, b, c, d, targetNormal = null) {
        this.addTriangle(a, b, c, targetNormal);
        this.addTriangle(a, c, d, targetNormal);
    }

    toGeometry() {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(this.positions, 3));
        geometry.setAttribute('normal', new THREE.Float32BufferAttribute(this.normals, 3));
        geometry.computeBoundingBox();
        geometry.computeBoundingSphere();
        return geometry;
    }
}

function addTopStrip(builder, y, z0, z1, xMin, xMax) {
    if (xMax - xMin <= EPSILON || z1 - z0 <= EPSILON) return;

    builder.addQuad(
        [xMin, y, z0],
        [xMin, y, z1],
        [xMax, y, z1],
        [xMax, y, z0],
        [0, 1, 0]
    );
}

function addBottomStrip(builder, y, z0, z1, xMin, xMax) {
    if (xMax - xMin <= EPSILON || z1 - z0 <= EPSILON) return;

    builder.addQuad(
        [xMin, y, z0],
        [xMax, y, z0],
        [xMax, y, z1],
        [xMin, y, z1],
        [0, -1, 0]
    );
}

function buildSolidTrapezoidGeometry(width, height, depth, topWidthFactor, cornerRadius) {
    const shape = createTrapezoidShape(width, height, topWidthFactor, cornerRadius);
    const geometry = new THREE.ExtrudeGeometry(shape, {
        steps: 1,
        depth,
        bevelEnabled: false
    });

    geometry.center();

    const creased = toCreasedNormals(geometry, Math.PI / 3);
    if (creased !== geometry) {
        geometry.dispose();
    }

    creased.computeBoundingBox();
    creased.computeBoundingSphere();

    return creased;
}

function buildFirstPrinciplesSlitWeightMatrixGeometry({
    width,
    height,
    depth,
    topWidthFactor,
    numberOfSlits,
    slitWidth,
    slitDepthFactor,
    slitBottomWidthFactor,
    slitTopWidthFactor
}) {
    const hh = height / 2;
    const halfDepth = depth / 2;
    const bottomHalfWidth = width / 2;
    const topHalfWidth = (width * topWidthFactor) / 2;

    const slitRanges = buildMergedSlitRanges(depth, numberOfSlits, slitWidth);
    const cutDepth = clamp(height * clamp01(slitDepthFactor), 0, height);

    if (!slitRanges.length || cutDepth <= EPSILON) {
        return buildSolidTrapezoidGeometry(width, height, depth, topWidthFactor, 0);
    }

    const builder = new TriangleBuilder();

    const frontBL = [-bottomHalfWidth, -hh, halfDepth];
    const frontBR = [bottomHalfWidth, -hh, halfDepth];
    const frontTR = [topHalfWidth, hh, halfDepth];
    const frontTL = [-topHalfWidth, hh, halfDepth];
    builder.addQuad(frontBL, frontBR, frontTR, frontTL, [0, 0, 1]);

    const backBL = [-bottomHalfWidth, -hh, -halfDepth];
    const backBR = [bottomHalfWidth, -hh, -halfDepth];
    const backTR = [topHalfWidth, hh, -halfDepth];
    const backTL = [-topHalfWidth, hh, -halfDepth];
    builder.addQuad(backBR, backBL, backTL, backTR, [0, 0, -1]);

    builder.addQuad(
        [bottomHalfWidth, -hh, -halfDepth],
        [topHalfWidth, hh, -halfDepth],
        [topHalfWidth, hh, halfDepth],
        [bottomHalfWidth, -hh, halfDepth],
        [1, 0, 0]
    );

    builder.addQuad(
        [-bottomHalfWidth, -hh, halfDepth],
        [-topHalfWidth, hh, halfDepth],
        [-topHalfWidth, hh, -halfDepth],
        [-bottomHalfWidth, -hh, -halfDepth],
        [-1, 0, 0]
    );

    let hasTopCut = cutDepth > EPSILON;
    let hasBottomCut = hasTopCut && slitDepthFactor < 0.95;

    const topCutHigh = hh;
    let topCutLow = Math.max(-hh, hh - cutDepth);

    const bottomCutLow = -hh;
    const bottomCutHigh = Math.min(hh, -hh + cutDepth);

    if (hasBottomCut && topCutLow <= bottomCutHigh + 1e-3) {
        hasBottomCut = false;
        topCutLow = -hh;
    }

    const topReachesBottom = hasTopCut && topCutLow <= -hh + 1e-3;

    const topHoleHalfAtTop = resolveHoleHalfWidth(
        hh,
        width,
        height,
        topWidthFactor,
        slitBottomWidthFactor,
        slitTopWidthFactor
    );

    const bottomHoleHalfAtBottom = resolveHoleHalfWidth(
        -hh,
        width,
        height,
        topWidthFactor,
        slitBottomWidthFactor,
        slitTopWidthFactor
    );

    const bands = buildDepthBands(depth, slitRanges);

    const shouldOpenTop = hasTopCut && topHoleHalfAtTop > EPSILON;
    for (const band of bands) {
        if (!band.insideSlit || !shouldOpenTop) {
            addTopStrip(builder, hh, band.z0, band.z1, -topHalfWidth, topHalfWidth);
            continue;
        }

        addTopStrip(builder, hh, band.z0, band.z1, -topHalfWidth, -topHoleHalfAtTop);
        addTopStrip(builder, hh, band.z0, band.z1, topHoleHalfAtTop, topHalfWidth);
    }

    const shouldOpenBottom = (hasBottomCut || topReachesBottom) && bottomHoleHalfAtBottom > EPSILON;
    for (const band of bands) {
        if (!band.insideSlit || !shouldOpenBottom) {
            addBottomStrip(builder, -hh, band.z0, band.z1, -bottomHalfWidth, bottomHalfWidth);
            continue;
        }

        addBottomStrip(builder, -hh, band.z0, band.z1, -bottomHalfWidth, -bottomHoleHalfAtBottom);
        addBottomStrip(builder, -hh, band.z0, band.z1, bottomHoleHalfAtBottom, bottomHalfWidth);
    }

    if (hasTopCut && topHoleHalfAtTop > EPSILON) {
        const topHoleHalfAtLow = resolveHoleHalfWidth(
            topCutLow,
            width,
            height,
            topWidthFactor,
            slitBottomWidthFactor,
            slitTopWidthFactor
        );

        const addTopFloor = topCutLow > -hh + 1e-3 && (!hasBottomCut || topCutLow > bottomCutHigh + 1e-3);

        for (const slit of slitRanges) {
            const z0 = slit.z0;
            const z1 = slit.z1;

            builder.addQuad(
                [topHoleHalfAtTop, topCutHigh, z0],
                [topHoleHalfAtLow, topCutLow, z0],
                [topHoleHalfAtLow, topCutLow, z1],
                [topHoleHalfAtTop, topCutHigh, z1],
                [-1, 0, 0]
            );

            builder.addQuad(
                [-topHoleHalfAtTop, topCutHigh, z1],
                [-topHoleHalfAtLow, topCutLow, z1],
                [-topHoleHalfAtLow, topCutLow, z0],
                [-topHoleHalfAtTop, topCutHigh, z0],
                [1, 0, 0]
            );

            if (addTopFloor) {
                builder.addQuad(
                    [-topHoleHalfAtLow, topCutLow, z0],
                    [-topHoleHalfAtLow, topCutLow, z1],
                    [topHoleHalfAtLow, topCutLow, z1],
                    [topHoleHalfAtLow, topCutLow, z0],
                    [0, 1, 0]
                );
            }

            builder.addQuad(
                [-topHoleHalfAtTop, topCutHigh, z0],
                [topHoleHalfAtTop, topCutHigh, z0],
                [topHoleHalfAtLow, topCutLow, z0],
                [-topHoleHalfAtLow, topCutLow, z0],
                [0, 0, 1]
            );

            builder.addQuad(
                [topHoleHalfAtTop, topCutHigh, z1],
                [-topHoleHalfAtTop, topCutHigh, z1],
                [-topHoleHalfAtLow, topCutLow, z1],
                [topHoleHalfAtLow, topCutLow, z1],
                [0, 0, -1]
            );
        }
    }

    if (hasBottomCut && bottomHoleHalfAtBottom > EPSILON) {
        const bottomHoleHalfAtHigh = resolveHoleHalfWidth(
            bottomCutHigh,
            width,
            height,
            topWidthFactor,
            slitBottomWidthFactor,
            slitTopWidthFactor
        );

        const addBottomCeiling = bottomCutHigh < hh - 1e-3 && (!hasTopCut || bottomCutHigh < topCutLow - 1e-3);

        for (const slit of slitRanges) {
            const z0 = slit.z0;
            const z1 = slit.z1;

            builder.addQuad(
                [bottomHoleHalfAtHigh, bottomCutHigh, z0],
                [bottomHoleHalfAtBottom, bottomCutLow, z0],
                [bottomHoleHalfAtBottom, bottomCutLow, z1],
                [bottomHoleHalfAtHigh, bottomCutHigh, z1],
                [-1, 0, 0]
            );

            builder.addQuad(
                [-bottomHoleHalfAtHigh, bottomCutHigh, z1],
                [-bottomHoleHalfAtBottom, bottomCutLow, z1],
                [-bottomHoleHalfAtBottom, bottomCutLow, z0],
                [-bottomHoleHalfAtHigh, bottomCutHigh, z0],
                [1, 0, 0]
            );

            if (addBottomCeiling) {
                builder.addQuad(
                    [-bottomHoleHalfAtHigh, bottomCutHigh, z0],
                    [bottomHoleHalfAtHigh, bottomCutHigh, z0],
                    [bottomHoleHalfAtHigh, bottomCutHigh, z1],
                    [-bottomHoleHalfAtHigh, bottomCutHigh, z1],
                    [0, -1, 0]
                );
            }

            builder.addQuad(
                [-bottomHoleHalfAtHigh, bottomCutHigh, z0],
                [bottomHoleHalfAtHigh, bottomCutHigh, z0],
                [bottomHoleHalfAtBottom, bottomCutLow, z0],
                [-bottomHoleHalfAtBottom, bottomCutLow, z0],
                [0, 0, 1]
            );

            builder.addQuad(
                [bottomHoleHalfAtHigh, bottomCutHigh, z1],
                [-bottomHoleHalfAtHigh, bottomCutHigh, z1],
                [-bottomHoleHalfAtBottom, bottomCutLow, z1],
                [bottomHoleHalfAtBottom, bottomCutLow, z1],
                [0, 0, -1]
            );
        }
    }

    return builder.toGeometry();
}

function createSideMaterial(cacheKey, width, height, depth, slitEnabled) {
    if (USE_GLB_MATERIALS && __materialCache.has(cacheKey)) {
        return __materialCache.get(cacheKey).clone();
    }

    const material = createSciFiMaterial({
        side: THREE.FrontSide,
        transparent: slitEnabled ? false : true,
        opacity: slitEnabled ? 1.0 : 0.9,
        accentColor: 0x6be7ff,
        secondaryColor: 0x061733,
        edgeColor: 0xcdf8ff,
        emissiveColor: 0x54d7ff,
        emissiveIntensity: slitEnabled ? 0.42 : 0.56,
        metalness: slitEnabled ? 0.64 : 0.82,
        roughness: slitEnabled ? 0.24 : 0.1,
        clearcoat: slitEnabled ? 0.35 : 0.95,
        clearcoatRoughness: slitEnabled ? 0.28 : 0.16,
        transmission: slitEnabled ? 0 : 0.2,
        thickness: slitEnabled ? 0 : 1.8,
        iridescence: slitEnabled ? 0.3 : 0.62,
        sheen: slitEnabled ? 0.38 : 0.62,
        sheenColor: 0xd2ffff,
        envMapIntensity: slitEnabled ? 1.25 : 1.95,
        accentMix: 0.93,
        glowFalloff: 2.2,
        depthAccentStrength: slitEnabled ? 0.08 : 0.34,
        scanlineFrequency: (Math.PI * 2) / Math.max(height / 5, 1),
        scanlineStrength: slitEnabled ? 0.1 : 0.26,
        stripeFrequency: (Math.PI * 2) / Math.max(depth / 10, 1),
        stripeStrength: slitEnabled ? 0.12 : 0.6,
        rimIntensity: 0.68,
        gradientSharpness: 1.4,
        gradientBias: 0.02,
        fresnelBoost: 0.36,
        glintStrength: slitEnabled ? 0.08 : 0.24,
        noiseStrength: slitEnabled ? 0.015 : 0.05
    });

    material.depthWrite = true;
    material.depthTest = true;
    if (slitEnabled) {
        updateSciFiMaterialUniforms(material, SLIT_SHADER_STABILITY_UNIFORMS);
    }

    return material;
}

function createCapMaterial(cacheKey, width, height, depth, slitEnabled) {
    if (USE_GLB_MATERIALS && __materialCache.has(cacheKey)) {
        return __materialCache.get(cacheKey).clone();
    }

    const material = createSciFiMaterial({
        side: THREE.DoubleSide,
        transparent: slitEnabled ? false : true,
        opacity: slitEnabled ? 1.0 : 0.94,
        accentColor: 0x74f0ff,
        secondaryColor: 0x071c3d,
        edgeColor: 0xe1fcff,
        emissiveColor: 0x59dfff,
        emissiveIntensity: slitEnabled ? 0.45 : 0.62,
        metalness: slitEnabled ? 0.66 : 0.84,
        roughness: slitEnabled ? 0.26 : 0.09,
        clearcoat: slitEnabled ? 0.32 : 0.96,
        clearcoatRoughness: slitEnabled ? 0.3 : 0.14,
        transmission: slitEnabled ? 0 : 0.22,
        thickness: slitEnabled ? 0 : 1.85,
        iridescence: slitEnabled ? 0.32 : 0.65,
        sheen: slitEnabled ? 0.4 : 0.65,
        sheenColor: 0xe1ffff,
        envMapIntensity: slitEnabled ? 1.3 : 2.05,
        accentMix: 0.94,
        glowFalloff: 2.4,
        depthAccentStrength: slitEnabled ? 0.08 : 0.38,
        scanlineFrequency: (Math.PI * 2) / Math.max(height / 4, 1),
        scanlineStrength: slitEnabled ? 0.1 : 0.3,
        dimensions: { width, height, depth: Math.max(depth * 0.25, 1) },
        stripeFrequency: (Math.PI * 2) / Math.max(width / 6, 1),
        stripeStrength: slitEnabled ? 0.1 : 0.42,
        rimIntensity: 0.82,
        gradientSharpness: 1.5,
        gradientBias: 0.035,
        fresnelBoost: 0.45,
        glintStrength: slitEnabled ? 0.08 : 0.28,
        noiseStrength: slitEnabled ? 0.015 : 0.06
    });

    material.depthWrite = true;
    material.depthTest = true;
    if (slitEnabled) {
        updateSciFiMaterialUniforms(material, SLIT_SHADER_STABILITY_UNIFORMS);
    }

    return material;
}

function buildWeightMatrixGeometry({
    width,
    height,
    depth,
    topWidthFactor,
    cornerRadius,
    numberOfSlits,
    slitWidth,
    slitDepthFactor,
    slitBottomWidthFactor,
    slitTopWidthFactor
}) {
    const hasSlits = numberOfSlits > 0 && slitWidth > 0 && slitDepthFactor > 0;
    if (!hasSlits) {
        return buildSolidTrapezoidGeometry(width, height, depth, topWidthFactor, cornerRadius);
    }

    return buildFirstPrinciplesSlitWeightMatrixGeometry({
        width,
        height,
        depth,
        topWidthFactor,
        numberOfSlits,
        slitWidth,
        slitDepthFactor,
        slitBottomWidthFactor,
        slitTopWidthFactor
    });
}

export class WeightMatrixVisualization {
    constructor(
        data = null,
        position = new THREE.Vector3(0, 0, 0),
        width = 8,
        height = 4,
        depth = 30,
        topWidthFactor = 0.7,
        cornerRadius = 0.8,
        numberOfSlits = 0,
        slitWidth = 0.2,
        slitDepthFactor = 1.0,
        slitBottomWidthFactor = 0.9,
        slitTopWidthFactor = null,
        useInstancedSlices = true
    ) {
        this.group = new THREE.Group();
        this.group.position.copy(position);
        this.group.userData.label = 'Weight Matrix';

        this.width = width;
        this.height = height;
        this.depth = depth;
        this.topWidthFactor = topWidthFactor;
        this.cornerRadius = cornerRadius;
        this.numberOfSlits = numberOfSlits;
        this.slitWidth = slitWidth;
        this.slitDepthFactor = clamp01(slitDepthFactor);
        this.slitBottomWidthFactor = slitBottomWidthFactor;
        this.slitTopWidthFactor = slitTopWidthFactor !== null ? slitTopWidthFactor : slitBottomWidthFactor;
        this.useInstancedSlices = useInstancedSlices !== false;

        this.mesh = null;
        this.frontCapMesh = null;
        this.backCapMesh = null;

        this._createMesh();

        if (data) {
            this.updateData(data);
        }
    }

    _isSlitEnabled() {
        return this.numberOfSlits > 0 && this.slitWidth > 0 && this.slitDepthFactor > 0;
    }

    _getEffectiveCornerRadius() {
        return this._isSlitEnabled() ? 0 : this.cornerRadius;
    }

    _resolveGeometryDepth() {
        const slitEnabled = this._isSlitEnabled();
        if (!slitEnabled) return this.depth;

        const laneCount = Math.max(1, Math.floor(this.numberOfSlits || 1));
        if (laneCount < 2) return this.depth;

        // When callers pass the canonical lane-dependent depth ((N+1)*spacing),
        // match the historical instanced span (N*spacing) to preserve visual size.
        const expectedDepth = (laneCount + 1) * VECTOR_DEPTH_SPACING;
        const tolerance = Math.max(1, VECTOR_DEPTH_SPACING * 0.15);
        if (Math.abs(this.depth - expectedDepth) <= tolerance) {
            return laneCount * VECTOR_DEPTH_SPACING;
        }

        return this.depth;
    }

    _clearMesh() {
        const disposeMaterial = (mat) => {
            if (!mat) return;
            if (Array.isArray(mat)) {
                for (const m of mat) {
                    if (m && m.dispose) m.dispose();
                }
            } else if (mat.dispose) {
                mat.dispose();
            }
        };

        if (this.mesh) {
            this.group.remove(this.mesh);
            if (this.mesh.geometry) this.mesh.geometry.dispose();
            disposeMaterial(this.mesh.material);
            this.mesh = null;
        }

        if (this.frontCapMesh) {
            this.group.remove(this.frontCapMesh);
            if (this.frontCapMesh.geometry) this.frontCapMesh.geometry.dispose();
            disposeMaterial(this.frontCapMesh.material);
            this.frontCapMesh = null;
        }

        if (this.backCapMesh) {
            this.group.remove(this.backCapMesh);
            if (this.backCapMesh.geometry) this.backCapMesh.geometry.dispose();
            disposeMaterial(this.backCapMesh.material);
            this.backCapMesh = null;
        }
    }

    _createMesh() {
        this._clearMesh();

        const slitEnabled = this._isSlitEnabled();
        const geometryDepth = this._resolveGeometryDepth();

        const cacheKey = getCacheKey(
            this.width,
            this.height,
            geometryDepth,
            this.topWidthFactor,
            this.cornerRadius,
            this.numberOfSlits,
            this.slitWidth,
            this.slitDepthFactor,
            this.slitBottomWidthFactor,
            this.slitTopWidthFactor
        );

        const wantsInstancedSlices = USE_INSTANCED_MATRIX_SLICES
            && this.useInstancedSlices
            && !slitEnabled
            && this.depth > VECTOR_DEPTH_SPACING * 1.5;

        if (wantsInstancedSlices && !__geometryCache.has(cacheKey)) {
            this._createInstancedSlices();
            return;
        }

        const t0 = performance.now();

        let geometry;
        let cacheHit = false;
        if (__geometryCache.has(cacheKey)) {
            cacheHit = true;
            geometry = __geometryCache.get(cacheKey).clone();
        } else {
            geometry = buildWeightMatrixGeometry({
                width: this.width,
                height: this.height,
                depth: geometryDepth,
                topWidthFactor: this.topWidthFactor,
                cornerRadius: this._getEffectiveCornerRadius(),
                numberOfSlits: this.numberOfSlits,
                slitWidth: this.slitWidth,
                slitDepthFactor: this.slitDepthFactor,
                slitBottomWidthFactor: this.slitBottomWidthFactor,
                slitTopWidthFactor: this.slitTopWidthFactor
            });

            __geometryCache.set(cacheKey, geometry.clone());
        }

        const sideMaterial = createSideMaterial(cacheKey, this.width, this.height, geometryDepth, slitEnabled);

        this.mesh = new THREE.Mesh(geometry, sideMaterial);
        this.mesh.renderOrder = 0;
        this.group.add(this.mesh);

        if (!slitEnabled) {
            const capGeometry = createCapGeometry(
                this.width,
                this.height,
                this.topWidthFactor,
                this._getEffectiveCornerRadius()
            );

            const capFrontMaterial = createCapMaterial(cacheKey, this.width, this.height, this.depth, false);
            const capBackMaterial = createCapMaterial(cacheKey, this.width, this.height, this.depth, false);
            capFrontMaterial.polygonOffset = true;
            capFrontMaterial.polygonOffsetFactor = -1;
            capFrontMaterial.polygonOffsetUnits = -4;
            capBackMaterial.polygonOffset = true;
            capBackMaterial.polygonOffsetFactor = -1;
            capBackMaterial.polygonOffsetUnits = -4;

            const epsilon = 0.05;
            this.frontCapMesh = new THREE.Mesh(capGeometry.clone(), capFrontMaterial);
            this.frontCapMesh.position.z = geometryDepth / 2 + epsilon;
            this.frontCapMesh.renderOrder = 1;
            this.group.add(this.frontCapMesh);

            this.backCapMesh = new THREE.Mesh(capGeometry.clone(), capBackMaterial);
            this.backCapMesh.position.z = -geometryDepth / 2 - epsilon;
            this.backCapMesh.rotation.y = Math.PI;
            this.backCapMesh.renderOrder = 2;
            this.group.add(this.backCapMesh);

            capGeometry.dispose();
        }

        updateSciFiDimensions(this.mesh.material, {
            width: this.width,
            height: this.height,
            depth: geometryDepth
        });

        if (this.frontCapMesh) {
            updateSciFiDimensions(this.frontCapMesh.material, {
                width: this.width,
                height: this.height,
                depth: geometryDepth
            });
        }

        if (this.backCapMesh) {
            updateSciFiDimensions(this.backCapMesh.material, {
                width: this.width,
                height: this.height,
                depth: geometryDepth
            });
        }

        const dt = (performance.now() - t0).toFixed(1);
        console.log(`[Perf] WeightMatrixVisualization (${cacheHit ? 'cache' : 'built'}) – ${dt} ms.`);
    }

    _createInstancedSlices() {
        const sliceDepth = VECTOR_DEPTH_SPACING;
        const laneCount = Math.max(1, Math.floor(this.numberOfSlits || 1));
        const depthSpan = (laneCount - 1) * VECTOR_DEPTH_SPACING + sliceDepth;
        const sliceSlits = this._isSlitEnabled() ? 1 : 0;

        const sliceKey = getCacheKey(
            this.width,
            this.height,
            sliceDepth,
            this.topWidthFactor,
            this.cornerRadius,
            sliceSlits,
            this.slitWidth,
            this.slitDepthFactor,
            this.slitBottomWidthFactor,
            this.slitTopWidthFactor
        );

        let sliceGeometry;
        if (__geometryCache.has(sliceKey)) {
            sliceGeometry = __geometryCache.get(sliceKey);
        } else {
            sliceGeometry = buildWeightMatrixGeometry({
                width: this.width,
                height: this.height,
                depth: sliceDepth,
                topWidthFactor: this.topWidthFactor,
                cornerRadius: this._isSlitEnabled() ? 0 : this.cornerRadius,
                numberOfSlits: sliceSlits,
                slitWidth: this.slitWidth,
                slitDepthFactor: this.slitDepthFactor,
                slitBottomWidthFactor: this.slitBottomWidthFactor,
                slitTopWidthFactor: this.slitTopWidthFactor
            });

            __geometryCache.set(sliceKey, sliceGeometry.clone());
        }

        let stripGeometry;
        if (__sliceWallCache.has(sliceKey)) {
            stripGeometry = __sliceWallCache.get(sliceKey);
        } else {
            stripGeometry = stripSliceEndCaps(sliceGeometry, sliceDepth);
            __sliceWallCache.set(sliceKey, stripGeometry.clone());
        }

        const slitEnabled = this._isSlitEnabled();
        const sideMaterial = createSideMaterial(sliceKey, this.width, this.height, depthSpan, slitEnabled);

        const instancedGeometry = stripGeometry.clone();
        const inst = new THREE.InstancedMesh(instancedGeometry, sideMaterial, laneCount);

        const matrix = new THREE.Matrix4();
        for (let i = 0; i < laneCount; i++) {
            const z = (i - (laneCount - 1) / 2) * VECTOR_DEPTH_SPACING;
            matrix.makeTranslation(0, 0, z);
            inst.setMatrixAt(i, matrix);
        }
        inst.instanceMatrix.needsUpdate = true;

        const capKey = `${sliceKey}|caps|${depthSpan}`;
        let capGeoFront;
        let capGeoBack;
        if (__capFrontCache.has(capKey) && __capBackCache.has(capKey)) {
            capGeoFront = __capFrontCache.get(capKey);
            capGeoBack = __capBackCache.get(capKey);
        } else {
            const capGeometry = createCapGeometry(
                this.width,
                this.height,
                this.topWidthFactor,
                this._isSlitEnabled() ? 0 : this.cornerRadius
            );
            capGeoFront = capGeometry.clone();
            capGeoBack = capGeometry.clone();
            __capFrontCache.set(capKey, capGeoFront.clone());
            __capBackCache.set(capKey, capGeoBack.clone());
            capGeometry.dispose();
        }

        const capMatFront = createCapMaterial(sliceKey, this.width, this.height, depthSpan, slitEnabled);
        const capMatBack = createCapMaterial(sliceKey, this.width, this.height, depthSpan, slitEnabled);

        const capEps = 0.05;
        const frontCaps = new THREE.Mesh(capGeoFront.clone(), capMatFront);
        const backCaps = new THREE.Mesh(capGeoBack.clone(), capMatBack);

        frontCaps.position.z = depthSpan / 2 + capEps;
        backCaps.position.z = -depthSpan / 2 - capEps;
        backCaps.rotation.y = Math.PI;

        this.mesh = inst;
        this.frontCapMesh = frontCaps;
        this.backCapMesh = backCaps;

        this.group.add(inst);
        this.group.add(frontCaps);
        this.group.add(backCaps);

        this.mesh.renderOrder = 0;
        this.frontCapMesh.renderOrder = 1;
        this.backCapMesh.renderOrder = 2;

        const dims = { width: this.width, height: this.height, depth: depthSpan };
        updateSciFiDimensions(this.mesh.material, dims);
        updateSciFiDimensions(this.frontCapMesh.material, dims);
        updateSciFiDimensions(this.backCapMesh.material, dims);
    }

    setCapVisibility(showFront = true, showBack = true) {
        if (this.frontCapMesh) this.frontCapMesh.visible = !!showFront;
        if (this.backCapMesh) this.backCapMesh.visible = !!showBack;
    }

    updateData(data) {
        console.log('Updating weight matrix with data:', data);
    }

    updateGeometry(params) {
        this.width = params.width ?? this.width;
        this.height = params.height ?? this.height;
        this.depth = params.depth ?? this.depth;
        this.topWidthFactor = params.topWidthFactor ?? this.topWidthFactor;
        this.cornerRadius = params.cornerRadius ?? this.cornerRadius;
        this.numberOfSlits = params.numberOfSlits ?? this.numberOfSlits;
        this.slitWidth = params.slitWidth ?? this.slitWidth;

        this.slitDepthFactor = params.slitDepthFactor !== undefined
            ? clamp01(params.slitDepthFactor)
            : this.slitDepthFactor;

        this.slitBottomWidthFactor = params.slitBottomWidthFactor ?? this.slitBottomWidthFactor;
        this.slitTopWidthFactor = params.slitTopWidthFactor ?? this.slitTopWidthFactor;

        this._createMesh();
    }

    animatePulse(time) {
        if (!this.mesh) return;
        const scale = 1 + Math.sin(time * 5) * 0.05;
        this.mesh.scale.set(scale, scale, scale);
    }

    setPosition(x, y, z) {
        this.group.position.set(x, y, z);
    }

    setColor(color) {
        const applyColor = (mat) => {
            if (!mat) return;
            if (Array.isArray(mat)) {
                mat.forEach(m => updateSciFiMaterialColor(m, color));
            } else {
                updateSciFiMaterialColor(mat, color);
            }
        };

        applyColor(this.mesh?.material);
        applyColor(this.frontCapMesh?.material);
        applyColor(this.backCapMesh?.material);
    }

    setMaterialProperties(props) {
        const applyProps = (mat) => {
            if (!mat) return;

            if (props.metalness !== undefined) mat.metalness = props.metalness;
            if (props.roughness !== undefined) mat.roughness = props.roughness;
            if (props.clearcoat !== undefined) mat.clearcoat = props.clearcoat;
            if (props.clearcoatRoughness !== undefined) mat.clearcoatRoughness = props.clearcoatRoughness;
            if (props.iridescence !== undefined) mat.iridescence = props.iridescence;
            if (props.envMapIntensity !== undefined) mat.envMapIntensity = props.envMapIntensity;
            if (props.emissive !== undefined && mat.emissive) mat.emissive.set(props.emissive);
            if (props.emissiveIntensity !== undefined) mat.emissiveIntensity = props.emissiveIntensity;
            if (props.opacity !== undefined) mat.opacity = props.opacity;
            if (props.transparent !== undefined) mat.transparent = props.transparent;
            mat.needsUpdate = true;
        };

        const apply = (target) => {
            if (!target) return;
            if (Array.isArray(target)) {
                target.forEach(applyProps);
            } else {
                applyProps(target);
            }
        };

        apply(this.mesh?.material);
        apply(this.frontCapMesh?.material);
        apply(this.backCapMesh?.material);
    }

    setEmissive(color, intensity) {
        this.setMaterialProperties({ emissive: color, emissiveIntensity: intensity });
    }

    setOpacity(opacity) {
        this.setMaterialProperties({ opacity, transparent: true });
    }

    dispose() {
        this._clearMesh();
    }

    static registerPrecomputedGeometry(cacheKey, geometry, material = null) {
        if (!cacheKey || !geometry) return;
        const normalizedKey = mapLegacyCacheKeyToCurrent(cacheKey);
        const geometryClone = geometry.clone();

        if (!__geometryCache.has(normalizedKey)) {
            __geometryCache.set(normalizedKey, geometryClone.clone());
        }

        if (!__geometryCache.has(cacheKey)) {
            __geometryCache.set(cacheKey, geometryClone.clone());
        }

        if (material) {
            const materialClone = material.clone();
            if (!__materialCache.has(normalizedKey)) {
                __materialCache.set(normalizedKey, materialClone.clone());
            }
            if (!__materialCache.has(cacheKey)) {
                __materialCache.set(cacheKey, materialClone.clone());
            }
        }
    }
}
