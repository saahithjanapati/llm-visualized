import * as THREE from 'three';
import { CSG } from 'three-csg-ts'; // Import CSG
import { QUALITY_PRESET, NUM_VECTOR_LANES, VECTOR_DEPTH_SPACING, USE_GLB_MATERIALS } from '../utils/constants.js';
import { createSciFiMaterial, updateSciFiDimensions, updateSciFiMaterialColor } from '../utils/sciFiMaterial.js';

// ------------------------------------------------------------------
// Geometry cache (module-level) – keyed by a stringified set of the main
// parameters that influence the vertex buffer.  Sharing geometries across
// identical matrices avoids repeating expensive CSG operations and GPU
// uploads.
// ------------------------------------------------------------------
const __geometryCache = new Map();
// Separate caches for front & back cap geometries when using slice instancing
const __capFrontCache = new Map();
const __capBackCache  = new Map();
const __materialCache = new Map();
const SLIT_CLEANUP_FLAG = 'slitCleanupAppliedV2';
const TOP_BOTTOM_SMOOTH_FLAG = 'topBottomNormalsSmoothedV3';
const SLICE_CAPS_STRIPPED_FLAG = 'sliceEndCapsStripped';

function parseCacheKeyForSlits(cacheKey) {
    if (!cacheKey || typeof cacheKey !== 'string') return null;
    const parts = cacheKey.split('|');
    if (parts.length < 10) return null;
    const nums = parts.slice(0, 10).map(Number);
    if (nums.some(val => !Number.isFinite(val))) return null;
    return {
        width: nums[0],
        height: nums[1],
        depth: nums[2],
        topWidthFactor: nums[3],
        cornerRadius: nums[4],
        numberOfSlits: Math.max(0, Math.floor(nums[5])),
        slitWidth: nums[6],
        slitDepthFactor: nums[7],
        slitBottomWidthFactor: nums[8],
        slitTopWidthFactor: nums[9]
    };
}

function cleanSlitLedgeFaces(geometry, params) {
    if (!geometry) return geometry;
    if (geometry.userData && geometry.userData[SLIT_CLEANUP_FLAG]) return geometry;
    if (!params || params.numberOfSlits <= 0 || params.slitWidth <= 0 || params.slitDepthFactor <= 0) {
        geometry.userData = { ...(geometry.userData || {}), [SLIT_CLEANUP_FLAG]: true };
        return geometry;
    }

    const numSlits = Math.max(0, Math.floor(params.numberOfSlits));
    if (!numSlits) {
        geometry.userData = { ...(geometry.userData || {}), [SLIT_CLEANUP_FLAG]: true };
        return geometry;
    }

    const cutDepth = params.height * params.slitDepthFactor;
    const epsCsg = Math.max(0.02, params.height * 0.001);
    const slitBoxHeight = cutDepth + epsCsg * 2;
    if (!(slitBoxHeight > 0)) {
        geometry.userData = { ...(geometry.userData || {}), [SLIT_CLEANUP_FLAG]: true };
        return geometry;
    }
    const cutCenterY = (params.height / 2 + epsCsg) - (slitBoxHeight / 2);
    const yTopOfSlit = cutCenterY + slitBoxHeight / 2;
    const yBottomOfSlit = cutCenterY - slitBoxHeight / 2;

    const halfHeight = params.height / 2;
    const shapeBottomWidth = params.width;
    const shapeTopWidth = params.width * params.topWidthFactor;
    const widthAtY = (y) => {
        const yc = Math.max(-halfHeight, Math.min(halfHeight, y));
        const t = (yc + halfHeight) / (2 * halfHeight);
        return THREE.MathUtils.lerp(shapeBottomWidth, shapeTopWidth, t);
    };

    const insetX = Math.max(0.1, params.width * 0.001);
    const bottomWidth = Math.max(0, widthAtY(yBottomOfSlit) * params.slitBottomWidthFactor - insetX * 2);
    const topWidth = Math.max(0, widthAtY(yTopOfSlit) * params.slitTopWidthFactor - insetX * 2);

    const slitSpacing = params.depth / (numSlits + 1);
    const slitCenters = new Array(numSlits);
    for (let i = 0; i < numSlits; i++) {
        slitCenters[i] = -params.depth / 2 + slitSpacing * (i + 1);
    }

    const halfSlitDepth = params.slitWidth / 2;
    const extraX = Math.max(0.02, params.width * 0.001);
    const extraZ = Math.max(0.02, params.depth * 0.001);
    const topCullBand = Math.max(0.05, params.height * 0.01);
    const allowBottomCull = params.slitDepthFactor >= 0.95;

    const isPointInsideSlit = (px, py, pz) => {
        if (py < yBottomOfSlit - epsCsg || py > yTopOfSlit + epsCsg) return false;
        const t = THREE.MathUtils.clamp((py - yBottomOfSlit) / slitBoxHeight, 0, 1);
        const widthAt = THREE.MathUtils.lerp(bottomWidth, topWidth, t);
        if (!(widthAt > 0)) return false;
        if (Math.abs(px) > widthAt * 0.5 + extraX) return false;
        for (let i = 0; i < numSlits; i++) {
            if (Math.abs(pz - slitCenters[i]) <= halfSlitDepth + extraZ) return true;
        }
        return false;
    };

    const source = geometry.index ? geometry.toNonIndexed() : geometry;
    const posAttr = source.getAttribute('position');
    if (!posAttr) {
        geometry.userData = { ...(geometry.userData || {}), [SLIT_CLEANUP_FLAG]: true };
        if (source !== geometry) source.dispose();
        return geometry;
    }

    const pos = posAttr.array;
    const kept = [];
    for (let i = 0; i < pos.length; i += 9) {
        const ax = pos[i];
        const ay = pos[i + 1];
        const az = pos[i + 2];
        const bx = pos[i + 3];
        const by = pos[i + 4];
        const bz = pos[i + 5];
        const cx = pos[i + 6];
        const cy = pos[i + 7];
        const cz = pos[i + 8];

        const abx = bx - ax;
        const aby = by - ay;
        const abz = bz - az;
        const acx = cx - ax;
        const acy = cy - ay;
        const acz = cz - az;
        const nx = aby * acz - abz * acy;
        const ny = abz * acx - abx * acz;
        const nz = abx * acy - aby * acx;
        const nLen = Math.hypot(nx, ny, nz);

        let drop = false;
        if (nLen > 1e-9) {
            const nyNorm = ny / nLen;
            const isHorizontal = Math.abs(nyNorm) > 0.9;
            if (isHorizontal) {
                const insideA = isPointInsideSlit(ax, ay, az);
                const insideB = isPointInsideSlit(bx, by, bz);
                const insideC = isPointInsideSlit(cx, cy, cz);
                const insideCount = (insideA ? 1 : 0) + (insideB ? 1 : 0) + (insideC ? 1 : 0);

                if (insideCount > 0) {
                    const cxAvg = (ax + bx + cx) / 3;
                    const cyAvg = (ay + by + cy) / 3;
                    const czAvg = (az + bz + cz) / 3;
                    const centroidInside = isPointInsideSlit(cxAvg, cyAvg, czAvg);

                    if (centroidInside || insideCount === 3) {
                        if (allowBottomCull) {
                            drop = true;
                        } else {
                            const triTopY = Math.max(ay, by, cy);
                            if (triTopY >= halfHeight - topCullBand) drop = true;
                        }
                    }
                }
            }
        }

        if (!drop) {
            kept.push(
                ax, ay, az,
                bx, by, bz,
                cx, cy, cz
            );
        }
    }

    if (source !== geometry) source.dispose();
    if (kept.length === pos.length) {
        geometry.userData = { ...(geometry.userData || {}), [SLIT_CLEANUP_FLAG]: true };
        return geometry;
    }

    const cleaned = new THREE.BufferGeometry();
    cleaned.setAttribute('position', new THREE.Float32BufferAttribute(kept, 3));
    cleaned.computeVertexNormals();
    cleaned.computeBoundingSphere();
    cleaned.computeBoundingBox();
    cleaned.userData = { ...(geometry.userData || {}), [SLIT_CLEANUP_FLAG]: true };
    return cleaned;
}

function smoothTopBottomNormals(geometry, params) {
    if (!geometry || !params || !Number.isFinite(params.height)) return geometry;
    if (geometry.userData && geometry.userData[TOP_BOTTOM_SMOOTH_FLAG]) return geometry;

    let target = geometry;
    if (target.index) {
        const nonIndexed = target.toNonIndexed();
        nonIndexed.userData = { ...(target.userData || {}) };
        target = nonIndexed;
    }

    const posAttr = target.getAttribute('position');
    if (!posAttr) {
        target.userData = { ...(target.userData || {}), [TOP_BOTTOM_SMOOTH_FLAG]: true };
        return target;
    }

    target.computeVertexNormals();
    const normAttr = target.getAttribute('normal');
    if (!normAttr) {
        target.userData = { ...(target.userData || {}), [TOP_BOTTOM_SMOOTH_FLAG]: true };
        return target;
    }

    const hasSlits = params.numberOfSlits > 0 && params.slitDepthFactor > 0;
    if (!hasSlits) {
        target.userData = { ...(target.userData || {}), [TOP_BOTTOM_SMOOTH_FLAG]: true };
        return target;
    }

    const halfHeight = Math.abs(params.height) / 2;
    const yTol = Math.max(0.1, Math.abs(params.height) * 0.005);
    let touchedPositions = false;

    for (let i = 0; i < posAttr.count; i += 3) {
        const ax = posAttr.getX(i);
        const ay = posAttr.getY(i);
        const az = posAttr.getZ(i);
        const bx = posAttr.getX(i + 1);
        const by = posAttr.getY(i + 1);
        const bz = posAttr.getZ(i + 1);
        const cx = posAttr.getX(i + 2);
        const cy = posAttr.getY(i + 2);
        const cz = posAttr.getZ(i + 2);

        const topMatch = Math.abs(ay - halfHeight) <= yTol
            && Math.abs(by - halfHeight) <= yTol
            && Math.abs(cy - halfHeight) <= yTol;
        const bottomMatch = Math.abs(ay + halfHeight) <= yTol
            && Math.abs(by + halfHeight) <= yTol
            && Math.abs(cy + halfHeight) <= yTol;

        if (topMatch) {
            normAttr.setXYZ(i, 0, 1, 0);
            normAttr.setXYZ(i + 1, 0, 1, 0);
            normAttr.setXYZ(i + 2, 0, 1, 0);
            if (Math.abs(ay - halfHeight) > 1e-4 || Math.abs(by - halfHeight) > 1e-4 || Math.abs(cy - halfHeight) > 1e-4) {
                posAttr.setY(i, halfHeight);
                posAttr.setY(i + 1, halfHeight);
                posAttr.setY(i + 2, halfHeight);
                touchedPositions = true;
            }
        } else if (bottomMatch) {
            normAttr.setXYZ(i, 0, -1, 0);
            normAttr.setXYZ(i + 1, 0, -1, 0);
            normAttr.setXYZ(i + 2, 0, -1, 0);
            if (Math.abs(ay + halfHeight) > 1e-4 || Math.abs(by + halfHeight) > 1e-4 || Math.abs(cy + halfHeight) > 1e-4) {
                posAttr.setY(i, -halfHeight);
                posAttr.setY(i + 1, -halfHeight);
                posAttr.setY(i + 2, -halfHeight);
                touchedPositions = true;
            }
        }
    }

    if (touchedPositions) posAttr.needsUpdate = true;
    normAttr.needsUpdate = true;
    target.userData = { ...(target.userData || {}), [TOP_BOTTOM_SMOOTH_FLAG]: true };
    return target;
}

function stripSliceEndCaps(geometry, depth) {
    if (!geometry || !Number.isFinite(depth)) return geometry;
    if (geometry.userData && geometry.userData[SLICE_CAPS_STRIPPED_FLAG]) return geometry;

    let target = geometry;
    if (target.index) {
        const nonIndexed = target.toNonIndexed();
        nonIndexed.userData = { ...(target.userData || {}) };
        target = nonIndexed;
    }

    const posAttr = target.getAttribute('position');
    if (!posAttr) {
        target.userData = { ...(target.userData || {}), [SLICE_CAPS_STRIPPED_FLAG]: true };
        return target;
    }

    const pos = posAttr.array;
    const kept = [];
    const halfDepth = Math.abs(depth) / 2;
    const zTol = Math.max(0.05, Math.abs(depth) * 0.002);

    for (let i = 0; i < pos.length; i += 9) {
        const ax = pos[i];
        const ay = pos[i + 1];
        const az = pos[i + 2];
        const bx = pos[i + 3];
        const by = pos[i + 4];
        const bz = pos[i + 5];
        const cx = pos[i + 6];
        const cy = pos[i + 7];
        const cz = pos[i + 8];

        const abx = bx - ax;
        const aby = by - ay;
        const abz = bz - az;
        const acx = cx - ax;
        const acy = cy - ay;
        const acz = cz - az;

        const nx = aby * acz - abz * acy;
        const ny = abz * acx - abx * acz;
        const nz = abx * acy - aby * acx;
        const nLen = Math.hypot(nx, ny, nz);

        let drop = false;
        if (nLen > 1e-9) {
            const nzNorm = nz / nLen;
            if (Math.abs(nzNorm) > 0.9) {
                const czAvg = (az + bz + cz) / 3;
                if (Math.abs(Math.abs(czAvg) - halfDepth) <= zTol) {
                    drop = true;
                }
            }
        }

        if (!drop) {
            kept.push(
                ax, ay, az,
                bx, by, bz,
                cx, cy, cz
            );
        }
    }

    if (kept.length === pos.length) {
        target.userData = { ...(target.userData || {}), [SLICE_CAPS_STRIPPED_FLAG]: true };
        return target;
    }

    const cleaned = new THREE.BufferGeometry();
    cleaned.setAttribute('position', new THREE.Float32BufferAttribute(kept, 3));
    cleaned.computeVertexNormals();
    cleaned.computeBoundingSphere();
    cleaned.computeBoundingBox();
    cleaned.userData = { ...(target.userData || {}), [SLICE_CAPS_STRIPPED_FLAG]: true };
    return cleaned;
}

function getCacheKey(width, height, depth, topWidthFactor, cornerRadius, numberOfSlits, slitWidth, slitDepthFactor, slitBottomWidthFactor, slitTopWidthFactor) {
    return [width, height, depth, topWidthFactor, cornerRadius, numberOfSlits, slitWidth, slitDepthFactor, slitBottomWidthFactor, slitTopWidthFactor, QUALITY_PRESET].join('|');
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
        slitBottomWidthFactor = 0.9, // previously `slitWidthFactor`
        slitTopWidthFactor = null    // allow different top factor; defaults to bottom
    ) {
        this.group = new THREE.Group();
        this.group.position.copy(position);
        // Label for raycasting hover info
        this.group.userData.label = 'Weight Matrix';

        this.width = width;
        this.height = height;
        this.depth = depth;
        this.topWidthFactor = topWidthFactor; // Width of the top relative to the bottom
        this.cornerRadius = cornerRadius;
        this.numberOfSlits = numberOfSlits;
        this.slitWidth = slitWidth;
        this.slitDepthFactor = Math.max(0, Math.min(1, slitDepthFactor)); // Clamp between 0 and 1
        this.slitBottomWidthFactor = slitBottomWidthFactor; // bottom width factor
        this.slitTopWidthFactor = slitTopWidthFactor !== null ? slitTopWidthFactor : slitBottomWidthFactor; // default to bottom factor

        this.mesh = null; // Main trapezoid mesh (sides)
        this.frontCapMesh = null; // Front face mesh
        this.backCapMesh = null; // Back face mesh

        this._createMesh();

        if (data) {
            this.updateData(data);
        }
    }

    _clearMesh() {
        if (this.mesh) {
            this.group.remove(this.mesh);
            // Safely dispose geometry and material(s)
            if (this.mesh.geometry) {
                this.mesh.geometry.dispose();
            }
            if (this.mesh.material) {
                if (Array.isArray(this.mesh.material)) {
                    this.mesh.material.forEach(m => m.dispose());
                } else {
                    this.mesh.material.dispose();
                }
            }
            this.mesh = null;
        }
        
        // Clear caps
        if (this.frontCapMesh) {
            this.group.remove(this.frontCapMesh);
            if (this.frontCapMesh.geometry) this.frontCapMesh.geometry.dispose();
            // Material is shared, dispose only once with main mesh
            this.frontCapMesh = null;
        }
        if (this.backCapMesh) {
            this.group.remove(this.backCapMesh);
            if (this.backCapMesh.geometry) this.backCapMesh.geometry.dispose();
            // Material is shared
            this.backCapMesh = null;
        }
    }

    _createMesh() {
        this._clearMesh();

        // --------------------------------------------------------------
        // Fast path – if the requested depth covers multiple lanes we
        // create ONE thin slice (depth = VECTOR_DEPTH_SPACING) and
        // replicate it across lanes via an InstancedMesh.  This avoids
        // the extremely heavy CSG work required for deep geometries and
        // completely sidesteps the need for a matching pre-baked asset.
        // --------------------------------------------------------------
        const wantsInstancedSlices = this.depth > VECTOR_DEPTH_SPACING * 1.5;
        if (wantsInstancedSlices) {
            this._createInstancedSlices();
            return;
        }

        // --- Create Main Trapezoid Shape (used for extrusion AND caps) ---
        const shape = new THREE.Shape();

        const hw     = this.width  / 2;                         // half bottom width
        const hTopW  = (this.width * this.topWidthFactor) / 2;  // half top width
        const hh     = this.height / 2;                         // half height

        // Desired corner radius, clamped so that arcs never overlap.
        let cr = this.cornerRadius;
        // --- Determine safe maximum radius for all four corners ---
        // 1. Along the bottom edge we previously limited the radius by the
        //    horizontal difference (hw − hTopW).  This works when the *top* is
        //    *narrower* than the bottom, but collapses to 0 when the top is
        //    wider (the up-projection case).  In that scenario we can safely
        //    allow the radius to grow up to the half-width of the bottom
        //    edge itself.
        const horizontalDiff = hw - hTopW;            // positive ⇒ top narrower
        // If the top width equals the bottom width (horizontalDiff == 0) the
        // shape is a true rectangle and we should still allow rounded
        // corners.  In that case fall back to the half-width of the bottom
        // edge instead of clamping the radius to zero.
        const maxBottomRadius = horizontalDiff > 0 ? horizontalDiff : hw;

        // 2. The top edge imposes its own limit: the radius cannot exceed
        //    half the top width, otherwise the two arcs would overlap.
        const maxTopRadius = hTopW;

        // 3. Finally, the slanted sides limit the radius – it must not exceed
        //    half the length of the side edges, otherwise arcs would intersect
        //    before reaching the mid-point of the edge.
        const sideLen       = Math.hypot(hw - hTopW, this.height);
        const maxSideRadius = sideLen / 2;

        // Choose the smallest of the candidate limits so the arcs never
        // overlap or cross any edge.
        cr = Math.min(cr, maxBottomRadius, maxTopRadius, maxSideRadius);

        // If radius is effectively zero, keep original sharp trapezoid
        if (cr < 1e-4) {
            shape.moveTo(-hw, -hh);
            shape.lineTo(hw, -hh);
            shape.lineTo(hTopW, hh);
            shape.lineTo(-hTopW, hh);
            shape.closePath();
        } else {
            // Pre‑compute some helper points for each corner where the arc starts/ends
            // Bottom edge offsets
            const blStart = new THREE.Vector2(-hw + cr, -hh); // bottom‑left start
            const brStart = new THREE.Vector2(hw - cr,  -hh); // bottom‑right start
            // Top edge offsets
            const trEnd   = new THREE.Vector2(hTopW - cr,  hh); // top‑right end
            const tlEnd   = new THREE.Vector2(-hTopW + cr, hh); // top‑left end

            // Unit vectors along the two slanted sides
            const rightSideDir = new THREE.Vector2(hTopW - hw, 2*hh).normalize();
            // Vector that goes from the bottom‑left to the top‑left along the slanted edge
            // (positive X, positive Y).  We do NOT negate; we'll use ± later as needed.
            const leftSideDir  = new THREE.Vector2(-hTopW + hw, 2*hh).normalize();

            const brSidePt = new THREE.Vector2(hw, -hh).addScaledVector(rightSideDir, cr);  // point after rounding bottom‑right
            const trSidePt = new THREE.Vector2(hTopW, hh).addScaledVector(rightSideDir.clone().negate(), cr); // point before rounding top‑right

            // For the top‑left we move *backwards* along the edge, so we use the negative direction.
            const tlSidePt = new THREE.Vector2(-hTopW, hh).addScaledVector(leftSideDir.clone().negate(), cr); // point after rounding top‑left
            // For the bottom‑left we move inward/upward along the edge (positive direction).
            const blSidePt = new THREE.Vector2(-hw, -hh).addScaledVector(leftSideDir, cr);  // point before rounding bottom‑left

            // Start drawing
            shape.moveTo(blStart.x, blStart.y);
            // Bottom edge
            shape.lineTo(brStart.x, brStart.y);
            // Bottom‑right corner arc (quadratic approximation)
            shape.quadraticCurveTo(hw, -hh, brSidePt.x, brSidePt.y);
            // Right side
            shape.lineTo(trSidePt.x, trSidePt.y);
            // Top‑right corner arc
            shape.quadraticCurveTo(hTopW, hh, trEnd.x, trEnd.y);
            // Top edge
            shape.lineTo(tlEnd.x, tlEnd.y);
            // Top‑left corner arc
            shape.quadraticCurveTo(-hTopW, hh, tlSidePt.x, tlSidePt.y);
            // Left side
            shape.lineTo(blSidePt.x, blSidePt.y);
            // Bottom‑left corner arc
            shape.quadraticCurveTo(-hw, -hh, blStart.x, blStart.y);
            shape.closePath();
        }

        // Extrusion settings with rounded/filleted corners controlled by `cornerRadius`
        // Enabling bevel creates additional geometry around every edge of the
        // extruded shape giving the appearance of smooth, rounded corners on
        // the final 3‑D trapezoid.  The user‑provided `cornerRadius` controls
        // both how far the bevel extends out from each edge (`bevelSize`) and
        // how thick the bevel is (`bevelThickness`).  A few segments are added
        // to approximate a smooth curve.  Increase `bevelSegments` for an even
        // smoother fillet at the cost of additional geometry.
        const extrudeSettings = {
            steps: 1,
            depth: this.depth,
            bevelEnabled: false,
            // Beveling is turned off because the 2‑D shape already contains
            // true circular arcs for its corners. Disabling bevel avoids CSG
            // artefacts that prevented the slit boxes from cutting all the
            // way through the top and bottom faces.
            bevelThickness: this.cornerRadius * 1.2,
            bevelSize: this.cornerRadius * 1.2,
            bevelOffset: 0,
            // Make the smoothness scale with the radius: at least 6 segments
            // with 3 extra per unit radius.
            bevelSegments: Math.max(6, Math.round(this.cornerRadius * 3))
        };

        // --------------------------------------------------------------
        //  1. Check the cache – if we have already built geometry for an
        //     identical parameter set we can skip the heavy CSG work and
        //     clone the cached BufferGeometry.
        // --------------------------------------------------------------
        const cacheKey = getCacheKey(this.width,this.height,this.depth,this.topWidthFactor,this.cornerRadius,this.numberOfSlits,this.slitWidth,this.slitDepthFactor,this.slitBottomWidthFactor,this.slitTopWidthFactor);

        const t0 = performance.now();
        let baseMesh;
        let cacheHit = false;
        if (__geometryCache.has(cacheKey)) {
            cacheHit = true;
            const cachedGeo = __geometryCache.get(cacheKey);
            baseMesh = new THREE.Mesh(cachedGeo);
        } else {
            // Create initial geometry by extruding the shape
            const baseGeometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);
            baseGeometry.center();
            baseMesh = new THREE.Mesh(baseGeometry);
            // heavy CSG operations proceed below
        }

        // --- Create and Subtract Slits using CSG (for the side walls) ---
        let finalMesh = baseMesh; // Start with the base mesh

        // Only run the heavy CSG slit generation when we *didn't* hit the cache.
        if (!cacheHit && QUALITY_PRESET === 'high' && this.numberOfSlits > 0 && this.slitWidth > 0) {
            const slitSpacing = this.depth / (this.numberOfSlits + 1);

            // Calculate the actual depth of the cut based on the factor
            const cutDepth = this.height * this.slitDepthFactor;
            // Robust CSG: push the cutting box a hair ABOVE the top face and
            // extend it slightly so we avoid coplanar faces (which can lead to
            // Boolean artefacts like split/duplicated holes).
            const EPS_CSG = Math.max(0.02, this.height * 0.001);
            const slitBoxHeight = cutDepth + EPS_CSG * 2; // extend both ends a touch

            // Position so the top of the slit box sits just above the top face
            // (at +height/2 + EPS_CSG), eliminating coplanar ambiguity.
            const cutCenterY = (this.height / 2 + EPS_CSG) - (slitBoxHeight / 2);

            // Helper: compute the trapezoid width at an arbitrary Y (clamped to the body)
            const hh = this.height / 2;
            const shapeBottomWidth = this.width;                       // width at y = -hh
            const shapeTopWidth    = this.width * this.topWidthFactor; // width at y = +hh
            const widthAtY = (y) => {
                const yc = Math.max(-hh, Math.min(hh, y));
                const t = (yc + hh) / (2 * hh); // 0 at bottom, 1 at top
                return THREE.MathUtils.lerp(shapeBottomWidth, shapeTopWidth, t);
            };

            // Compute the top/bottom Y of the slit volume in the matrix’s local space
            const yTopOfSlit = cutCenterY + slitBoxHeight / 2;
            const yBottomOfSlit = cutCenterY - slitBoxHeight / 2;

            // Slight inward inset so we never breach the side walls due to numerical issues
            const INSET_X = Math.max(0.1, this.width * 0.001);

            // Compute cutter widths to match the body profile exactly at the
            // slit bounds, scaled by user factors and inset slightly.
            const dynamicBottomWidth = Math.max(0,
                widthAtY(yBottomOfSlit) * this.slitBottomWidthFactor - INSET_X * 2
            );
            const dynamicTopWidth = Math.max(0,
                widthAtY(yTopOfSlit) * this.slitTopWidthFactor - INSET_X * 2
            );

            for (let i = 0; i < this.numberOfSlits; i++) {
                const zPos = -this.depth / 2 + slitSpacing * (i + 1);

                let slitGeometry;

                if (Math.abs(dynamicBottomWidth - dynamicTopWidth) < 1e-4) {
                    // Simple rectangular slit (top == bottom)
                    slitGeometry = new THREE.BoxGeometry(dynamicBottomWidth, slitBoxHeight, this.slitWidth);
                } else {
                    // Create tapered geometry (trapezoidal prism) by modifying a box geometry
                    slitGeometry = new THREE.BoxGeometry(dynamicBottomWidth, slitBoxHeight, this.slitWidth, 1, 1, 1);
                    const posAttr = slitGeometry.attributes.position;
                    const halfH = slitBoxHeight / 2;
                    for (let v = 0; v < posAttr.count; v++) {
                        const y = posAttr.getY(v);
                        const t = (y + halfH) / slitBoxHeight; // 0 at bottom, 1 at top
                        const targetWidth = THREE.MathUtils.lerp(dynamicBottomWidth, dynamicTopWidth, t);
                        const scale = (dynamicBottomWidth > 0) ? (targetWidth / dynamicBottomWidth) : 1.0;
                        posAttr.setX(v, posAttr.getX(v) * scale);
                    }
                    posAttr.needsUpdate = true;
                    slitGeometry.computeVertexNormals();
                }
                
                // Material not needed for CSG
                const slitMesh = new THREE.Mesh(slitGeometry); 

                // Position the slit box so its center aligns with the desired cut midpoint
                slitMesh.position.set(0, cutCenterY, zPos);
                slitMesh.updateMatrix(); // IMPORTANT: Update matrix before CSG operation

                // Subtract the slit mesh from the current result
                finalMesh = CSG.subtract(finalMesh, slitMesh);
            }
        }

        // --- Finalize the Side Walls Mesh ---
        // Use two distinct materials so we can apply a polygon offset to the
        // *cap* faces only.  Offsetting the depth values of the caps nudges
        // them ever-so-slightly closer to the camera, eliminating the
        // z-fighting flicker that still appeared when zoomed far out.

        let sideMaterial;
        if (USE_GLB_MATERIALS && __materialCache.has(cacheKey)) {
            sideMaterial = __materialCache.get(cacheKey).clone();
        } else {
            sideMaterial = createSciFiMaterial({
                side: THREE.FrontSide,
                transparent: true,
                opacity: 0.9,
                accentColor: 0x6be7ff,
                secondaryColor: 0x061733,
                edgeColor: 0xcdf8ff,
                emissiveColor: 0x54d7ff,
                emissiveIntensity: 0.56,
                metalness: 0.82,
                roughness: 0.1,
                clearcoat: 0.95,
                clearcoatRoughness: 0.16,
                transmission: 0.2,
                thickness: 1.8,
                iridescence: 0.62,
                sheen: 0.62,
                sheenColor: 0xd2ffff,
                envMapIntensity: 1.95,
                accentMix: 0.93,
                glowFalloff: 2.2,
                depthAccentStrength: 0.34,
                scanlineFrequency: (Math.PI * 2) / Math.max(this.height / 5, 1),
                scanlineStrength: 0.0,
                stripeFrequency: (Math.PI * 2) / Math.max(this.depth / 10, 1),
                stripeStrength: 0.0,
                rimIntensity: 0.68,
                gradientSharpness: 1.4,
                gradientBias: 0.02,
                fresnelBoost: 0.36,
                glintStrength: 0.24,
                noiseStrength: 0.05
            });
        }

        const capMaterial = (USE_GLB_MATERIALS && __materialCache.has(cacheKey))
            ? sideMaterial.clone()
            : createSciFiMaterial({
                side: THREE.DoubleSide,
                transparent: true,
                opacity: 0.94,
                accentColor: 0x74f0ff,
                secondaryColor: 0x071c3d,
                edgeColor: 0xe1fcff,
                emissiveColor: 0x59dfff,
                emissiveIntensity: 0.62,
                metalness: 0.84,
                roughness: 0.09,
                clearcoat: 0.96,
                clearcoatRoughness: 0.14,
                transmission: 0.22,
                thickness: 1.85,
                iridescence: 0.65,
                sheen: 0.65,
                sheenColor: 0xe1ffff,
                envMapIntensity: 2.05,
                accentMix: 0.94,
                glowFalloff: 2.4,
                depthAccentStrength: 0.38,
                scanlineFrequency: (Math.PI * 2) / Math.max(this.height / 4, 1),
                scanlineStrength: 0.0,
                dimensions: { width: this.width, height: this.height, depth: Math.max(this.depth * 0.25, 1) },
                stripeFrequency: (Math.PI * 2) / Math.max(this.width / 6, 1),
                stripeStrength: 0.0,
                rimIntensity: 0.82,
                gradientSharpness: 1.5,
                gradientBias: 0.035,
                fresnelBoost: 0.45,
                glintStrength: 0.28,
                noiseStrength: 0.06
            });

        capMaterial.polygonOffset = true;
        capMaterial.polygonOffsetFactor = -1; // pull slightly forward
        capMaterial.polygonOffsetUnits  = -4;

        // Assign the final CSG result geometry and the material to the main mesh (sides)
        if (!cacheHit) {
            const cleanedGeometry = cleanSlitLedgeFaces(finalMesh.geometry, {
                width: this.width,
                height: this.height,
                depth: this.depth,
                topWidthFactor: this.topWidthFactor,
                cornerRadius: this.cornerRadius,
                numberOfSlits: this.numberOfSlits,
                slitWidth: this.slitWidth,
                slitDepthFactor: this.slitDepthFactor,
                slitBottomWidthFactor: this.slitBottomWidthFactor,
                slitTopWidthFactor: this.slitTopWidthFactor
            });
            if (cleanedGeometry !== finalMesh.geometry) {
                finalMesh.geometry.dispose();
                finalMesh.geometry = cleanedGeometry;
            }
        }

        const smoothedGeometry = smoothTopBottomNormals(finalMesh.geometry, {
            height: this.height,
            numberOfSlits: this.numberOfSlits,
            slitDepthFactor: this.slitDepthFactor
        });
        if (smoothedGeometry !== finalMesh.geometry) {
            finalMesh.geometry.dispose();
            finalMesh.geometry = smoothedGeometry;
        }

        this.mesh = finalMesh;
        this.mesh.material = sideMaterial;

        const dt = (performance.now() - t0).toFixed(1);
        console.log(`[Perf] WeightMatrixVisualization (${cacheHit ? 'cache' : 'built'}) – ${dt} ms.`);

        // Ensure transparent objects are rendered in a predictable order to
        // avoid flickering caused by Three.js' painter-style sorting.  By
        // explicitly rendering the side walls first, followed by the front
        // cap and finally the back cap, we remove the ambiguity that appears
        // when the camera is far away or at shallow angles.
        this.mesh.renderOrder = 0;          // side walls first

        // --------------------------------------------------------------
        // Store completed geometry in cache for future instances
        // (avoid re-computing CSG next time the same matrix appears).
        // --------------------------------------------------------------
        if (!__geometryCache.has(cacheKey)) {
            __geometryCache.set(cacheKey, this.mesh.geometry.clone());
        }

        // Add the side walls mesh (with holes) to the group
        this.group.add(this.mesh);

        // --- Add Front / Back Caps to ensure end faces are solid ---
        // After the CSG operations (slits) the original end faces may have
        // been lost, so we recreate them using the original 2‑D profile and
        // place them just outside the body (epsilon offset) to avoid z‑fighting.
        const capGeometry = new THREE.ShapeGeometry(shape);
        capGeometry.center();

        // Increase the cap offset to further separate it from the main body and
        // avoid depth-buffer precision issues that show up when the camera is
        // far away.  A larger gap (≈ 5 cm in world-space) is visually
        // imperceptible but removes the flicker caused by z-fighting.
        const epsilon = 0.05;
        this.frontCapMesh = new THREE.Mesh(capGeometry, capMaterial);
        this.frontCapMesh.position.z = this.depth / 2 + epsilon;
        this.frontCapMesh.renderOrder = 1;  // front cap after walls
        this.group.add(this.frontCapMesh);

        this.backCapMesh = new THREE.Mesh(capGeometry, capMaterial);
        this.backCapMesh.position.z = -this.depth / 2 - epsilon;
        this.backCapMesh.rotation.y = Math.PI; // flip so normals face outwards
        this.backCapMesh.renderOrder = 2;   // back cap last
        this.group.add(this.backCapMesh);

        updateSciFiDimensions(this.mesh.material, { width: this.width, height: this.height, depth: this.depth });
        updateSciFiDimensions(this.frontCapMesh.material, { width: this.width, height: this.height, depth: this.depth });
        updateSciFiDimensions(this.backCapMesh.material, { width: this.width, height: this.height, depth: this.depth });
    }

    // Hide or show caps to avoid visible seams when abutting multiple
    // matrices along Z. For a chain of matrices: show back cap on the first,
    // hide both on the middle ones, and show front cap on the last.
    setCapVisibility(showFront = true, showBack = true) {
        if (this.frontCapMesh) this.frontCapMesh.visible = !!showFront;
        if (this.backCapMesh)  this.backCapMesh.visible  = !!showBack;
    }

    /**
     * Build a single-lane slice (depth = VECTOR_DEPTH_SPACING) and then
     * replicate it across NUM_VECTOR_LANES via InstancedMesh.  This greatly
     * reduces CPU load compared to carving a deep geometry with many CSG
     * operations.
     */
    _createInstancedSlices() {
        const sliceDepth = VECTOR_DEPTH_SPACING;
        const sliceSlits = 1; // one slit per slice – channels separated by instancing

        // ----------------------------------------------------------
        // Re-use / build geometry for the single slice
        // ----------------------------------------------------------
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
        let tmp = null; // temp builder (may remain null if cache hit)

        // Attempt to pull slice + cap geometry from cache first
        if (__geometryCache.has(sliceKey)) {
            sliceGeometry = __geometryCache.get(sliceKey);
        } else {
            // Build a *temporary* WeightMatrixVisualization to generate the
            // geometry for a single slice.  The recursive call will *not*
            // enter the instanced-slice path because the depth is now small.
            tmp = new WeightMatrixVisualization(
                null,
                new THREE.Vector3(),
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
            sliceGeometry = tmp.mesh.geometry.clone();
            // Cache for future re-use
            __geometryCache.set(sliceKey, sliceGeometry);
            // We'll cache cap geometries below once they're cloned
        }

        const cleanedSliceGeometry = stripSliceEndCaps(sliceGeometry, sliceDepth);
        sliceGeometry = cleanedSliceGeometry !== sliceGeometry ? cleanedSliceGeometry : sliceGeometry;
        const smoothedSliceGeometry = smoothTopBottomNormals(sliceGeometry, {
            height: this.height,
            numberOfSlits: sliceSlits,
            slitDepthFactor: this.slitDepthFactor
        });
        if (smoothedSliceGeometry !== sliceGeometry) {
            sliceGeometry = smoothedSliceGeometry;
        }
        __geometryCache.set(sliceKey, sliceGeometry);

        // ----------------------------------------------------------
        // Material – clone defaults from standard path
        // ----------------------------------------------------------
        const sciFiSliceDims = { width: this.width, height: this.height, depth: sliceDepth };
        const sciFiStackDims = { width: this.width, height: this.height, depth: sliceDepth * NUM_VECTOR_LANES };
        let mat;
        if (USE_GLB_MATERIALS && __materialCache.has(sliceKey)) {
            mat = __materialCache.get(sliceKey).clone();
        } else {
            mat = createSciFiMaterial({
                side: THREE.DoubleSide,
                transparent: true,
                opacity: 0.9,
                accentColor: 0x6deaff,
                secondaryColor: 0x071836,
                edgeColor: 0xd4f9ff,
                emissiveColor: 0x53d9ff,
                emissiveIntensity: 0.54,
                metalness: 0.8,
                roughness: 0.11,
                clearcoat: 0.94,
                clearcoatRoughness: 0.16,
                transmission: 0.18,
                thickness: 1.7,
                iridescence: 0.6,
                sheen: 0.6,
                sheenColor: 0xd4ffff,
                envMapIntensity: 1.9,
                accentMix: 0.92,
                glowFalloff: 2.1,
                depthAccentStrength: 0.32,
                scanlineFrequency: (Math.PI * 2) / Math.max(this.height / 5, 1),
                scanlineStrength: 0.0,
                dimensions: sciFiStackDims,
                stripeFrequency: (Math.PI * 2) / Math.max(sliceDepth / 6, 1),
                stripeStrength: 0.0,
                rimIntensity: 0.66,
                gradientSharpness: 1.38,
                gradientBias: 0.025,
                fresnelBoost: 0.38,
                glintStrength: 0.22,
                noiseStrength: 0.05
            });
        }

        // ----------------------------------------------------------
        // Build InstancedMesh for side walls across all lanes
        // ----------------------------------------------------------
        const inst = new THREE.InstancedMesh(sliceGeometry, mat, NUM_VECTOR_LANES);
        const mtx = new THREE.Matrix4();
        for (let i = 0; i < NUM_VECTOR_LANES; i++) {
            const z = (i - (NUM_VECTOR_LANES - 1) / 2) * VECTOR_DEPTH_SPACING;
            mtx.makeTranslation(0, 0, z);
            inst.setMatrixAt(i, mtx);
        }
        inst.instanceMatrix.needsUpdate = true;

        // ----------------------------------------------------------
        // Front & back caps – clone geometries from the temporary build
        // ----------------------------------------------------------
        let capGeoFront, capGeoBack;
        if (__capFrontCache.has(sliceKey) && __capBackCache.has(sliceKey)) {
            capGeoFront = __capFrontCache.get(sliceKey);
            capGeoBack  = __capBackCache.get(sliceKey);
        } else {
            if (!tmp) {
                // Build a throw-away WeightMatrixVisualization to obtain cap geometry only
                tmp = new WeightMatrixVisualization(
                    null,
                    new THREE.Vector3(),
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
            }
            capGeoFront = tmp.frontCapMesh.geometry.clone();
            capGeoBack  = tmp.backCapMesh.geometry.clone();

            // Cache for next time
            __capFrontCache.set(sliceKey, capGeoFront);
            __capBackCache.set(sliceKey, capGeoBack);
        }

        const capMatFront = createSciFiMaterial({
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.94,
            accentColor: 0x74f0ff,
            secondaryColor: 0x071c3d,
            edgeColor: 0xe1fcff,
            emissiveColor: 0x59dfff,
            emissiveIntensity: 0.6,
            metalness: 0.84,
            roughness: 0.09,
            clearcoat: 0.96,
            clearcoatRoughness: 0.14,
            transmission: 0.22,
            thickness: 1.85,
            iridescence: 0.65,
            sheen: 0.65,
            sheenColor: 0xe1ffff,
            envMapIntensity: 2.05,
            accentMix: 0.94,
            glowFalloff: 2.4,
            depthAccentStrength: 0.38,
            scanlineFrequency: (Math.PI * 2) / Math.max(this.height / 4, 1),
            scanlineStrength: 0.0,
            dimensions: sciFiStackDims,
            stripeFrequency: (Math.PI * 2) / Math.max(this.width / 6, 1),
            stripeStrength: 0.0,
            rimIntensity: 0.82,
            gradientSharpness: 1.5,
            gradientBias: 0.035,
            fresnelBoost: 0.45,
            glintStrength: 0.28,
            noiseStrength: 0.06
        });
        const capMatBack  = createSciFiMaterial({
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.94,
            accentColor: 0x74f0ff,
            secondaryColor: 0x071c3d,
            edgeColor: 0xe1fcff,
            emissiveColor: 0x59dfff,
            emissiveIntensity: 0.6,
            metalness: 0.84,
            roughness: 0.09,
            clearcoat: 0.96,
            clearcoatRoughness: 0.14,
            transmission: 0.22,
            thickness: 1.85,
            iridescence: 0.65,
            sheen: 0.65,
            sheenColor: 0xe1ffff,
            envMapIntensity: 2.05,
            accentMix: 0.94,
            glowFalloff: 2.4,
            depthAccentStrength: 0.38,
            scanlineFrequency: (Math.PI * 2) / Math.max(this.height / 4, 1),
            scanlineStrength: 0.0,
            dimensions: sciFiStackDims,
            stripeFrequency: (Math.PI * 2) / Math.max(this.width / 6, 1),
            stripeStrength: 0.0,
            rimIntensity: 0.82,
            gradientSharpness: 1.5,
            gradientBias: 0.035,
            fresnelBoost: 0.45,
            glintStrength: 0.28,
            noiseStrength: 0.06
        });

        const stackDepth = sciFiStackDims.depth;
        const capOffset = 0.05;

        const frontCap = new THREE.Mesh(capGeoFront.clone(), capMatFront);
        frontCap.position.z = stackDepth / 2 + capOffset;
        const backCap = new THREE.Mesh(capGeoBack.clone(), capMatBack);
        backCap.position.z = -stackDepth / 2 - capOffset;
        backCap.rotation.y = Math.PI;

        // ----------------------------------------------------------
        // Store references and add to group
        // ----------------------------------------------------------
        this.mesh = inst;
        this.frontCapMesh = frontCap;
        this.backCapMesh  = backCap;

        this.group.add(inst);
        this.group.add(frontCap);
        this.group.add(backCap);

        updateSciFiDimensions(mat, sciFiStackDims);
        updateSciFiDimensions(capMatFront, sciFiStackDims);
        updateSciFiDimensions(capMatBack, sciFiStackDims);

        // Render ordering – caps slightly after side walls to minimise z-fighting
        this.mesh.renderOrder = 0;
        this.frontCapMesh.renderOrder = 1;
        this.backCapMesh.renderOrder  = 2;

        // Dispose of the temporary object to free GPU resources
        if (tmp) {
            tmp.dispose && tmp.dispose();
        }
    }

    updateData(data) {
        // Placeholder for updating visualization based on data
        console.log("Updating weight matrix with data:", data);
        // Example: Change color based on data
        // const averageValue = data.reduce((sum, val) => sum + val, 0) / data.length;
        // const colorIntensity = Math.min(1, Math.max(0, averageValue)); // Normalize
        // this.setColor(new THREE.Color().setHSL(0.6, 1.0, colorIntensity * 0.5 + 0.25));
    }

    // Method to update geometry based on new parameters
    updateGeometry(params) {
        // Update properties using nullish coalescing for defaults
        this.width = params.width ?? this.width;
        this.height = params.height ?? this.height;
        this.depth = params.depth ?? this.depth;
        this.topWidthFactor = params.topWidthFactor ?? this.topWidthFactor;
        this.cornerRadius = params.cornerRadius ?? this.cornerRadius;
        this.numberOfSlits = params.numberOfSlits ?? this.numberOfSlits;
        this.slitWidth = params.slitWidth ?? this.slitWidth;
        
        // Update new parameters (excluding slit color/opacity)
        // this.slitHeight = params.slitHeight ?? this.slitHeight; // Removed
        this.slitDepthFactor = params.slitDepthFactor !== undefined 
            ? Math.max(0, Math.min(1, params.slitDepthFactor)) // Clamp between 0 and 1
            : this.slitDepthFactor;
        this.slitBottomWidthFactor = params.slitBottomWidthFactor ?? this.slitBottomWidthFactor;
        this.slitTopWidthFactor = params.slitTopWidthFactor ?? this.slitTopWidthFactor;

        this._createMesh(); // Recreate the mesh with new parameters
    }

    // Optional animation method
    animatePulse(time) {
        if (!this.mesh) return;
        const scale = 1 + Math.sin(time * 5) * 0.05; // Gentle pulse
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
        applyColor(this.frontCapMesh?.material); // Apply to caps too
        applyColor(this.backCapMesh?.material);  // Apply to caps too
    }

    setMaterialProperties(props) {
        const applyProps = (mat) => {
            if (mat) {
                 if (Array.isArray(mat)) {
                    mat.forEach(m => {
                        if (props.metalness !== undefined) m.metalness = props.metalness;
                        if (props.roughness !== undefined) m.roughness = props.roughness;
                        if (props.emissive !== undefined) m.emissive.set(props.emissive);
                        if (props.emissiveIntensity !== undefined) m.emissiveIntensity = props.emissiveIntensity;
                        if (props.opacity !== undefined) m.opacity = props.opacity;
                        if (props.transparent !== undefined) m.transparent = props.transparent;
                    });
                } else {
                    if (props.metalness !== undefined) mat.metalness = props.metalness;
                    if (props.roughness !== undefined) mat.roughness = props.roughness;
                    if (props.emissive !== undefined) mat.emissive.set(props.emissive);
                    if (props.emissiveIntensity !== undefined) mat.emissiveIntensity = props.emissiveIntensity;
                    if (props.opacity !== undefined) mat.opacity = props.opacity;
                    if (props.transparent !== undefined) mat.transparent = props.transparent;
                }
            }
        };
        applyProps(this.mesh?.material);
        applyProps(this.frontCapMesh?.material); 
        applyProps(this.backCapMesh?.material);  
    }

    // Convenience method to set emissive color and intensity
    setEmissive(color, intensity) {
        this.setMaterialProperties({ emissive: color, emissiveIntensity: intensity });
    }

    // Convenience method to set opacity (and ensure transparency is enabled)
    setOpacity(opacity) {
        const wantsTransparency = opacity < 1.0;
        this.setMaterialProperties({ opacity: opacity, transparent: wantsTransparency });
    }

    /**
     * Register a pre-generated BufferGeometry so that future WeightMatrixVisualization
     * instances can reuse it immediately without running the heavy CSG build.
     * @param {string} cacheKey – The key returned by the internal getCacheKey(..) helper.
     * @param {THREE.BufferGeometry} geometry – A **non-indexed** BufferGeometry to reuse.
     */
    static registerPrecomputedGeometry(cacheKey, geometry, material = null) {
        if (!cacheKey || !geometry) return;
        let cleanedGeometry = geometry;
        const params = parseCacheKeyForSlits(cacheKey);
        if (params) {
            const updated = cleanSlitLedgeFaces(geometry, params);
            cleanedGeometry = updated || geometry;
            cleanedGeometry = smoothTopBottomNormals(cleanedGeometry, params);
        }
        if (!__geometryCache.has(cacheKey)) {
            __geometryCache.set(cacheKey, cleanedGeometry);
        }
        if (material && !__materialCache.has(cacheKey)) {
            __materialCache.set(cacheKey, material);
        }
    }
}
