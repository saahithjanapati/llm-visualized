import * as THREE from 'three';
import { CSG } from 'three-csg-ts';

// A visualization for the Layer Normalization operation.
// It renders an extruded ellipse (like a squashed cylinder) with a hollow interior
// and regularly-spaced pass-through holes on the *top* and *bottom* portions of
// its side walls so that vectors can travel vertically through the layer.
//
// The implementation purposefully mirrors the public interface of
// `WeightMatrixVisualization` so it can be dropped into existing scenes with a
// similar parameter set (width / height / depth …) and animated / coloured in
// the same way.

// ------------------------------------------------------------------
// Geometry cache – stores BufferGeometry objects keyed by the parameters
// that uniquely define a LayerNorm extrusion.  Re-using the same
// BufferGeometry across layers slashes CPU time because the expensive CSG
// subtraction and clean-up stages are executed only once per unique
// parameter set.  Multiple THREE.Mesh instances can safely share the same
// BufferGeometry as long as we do **not** modify the vertices afterwards –
// which we never do.
// ------------------------------------------------------------------

const __geometryCache = new Map();

function getCacheKey(width, height, depth, wallThickness, numberOfHoles, holeWidth, holeWidthFactor, segments) {
    return [width, height, depth, wallThickness, numberOfHoles, holeWidth, holeWidthFactor, segments].join('|');
}

export class LayerNormalizationVisualization {
    constructor(
        position = new THREE.Vector3(0, 0, 0),
        /* Major (X-axis) diameter of the outer ellipse */
        width = 8,
        /* Minor (Y-axis) diameter of the outer ellipse */
        height = 4,
        /* Extrusion distance along Z */
        depth = 10,
        /* Thickness of the ring wall.  Must be < min(width, height) / 2 */
        wallThickness = 0.4,
        /* How many vertical pass-through channels should be cut */
        numberOfHoles = 6,
        /* Z-thickness of each hole (affects how wide the slit appears) */
        holeWidth = 0.4,
        /* Controls the length of slits along the ellipse surface (0-1) */
        holeWidthFactor = 0.8,
        /* How many segments to approximate the ellipse – higher = smoother */
        segments = 64
    ) {
        // Public group that callers can add to their scene
        this.group = new THREE.Group();
        this.group.position.copy(position);

        // Store parameters so `updateGeometry` can recreate the mesh later
        this.width = width;
        this.height = height;
        this.depth = depth;
        this.wallThickness = wallThickness;
        this.numberOfHoles = numberOfHoles;
        this.holeWidth = holeWidth;
        this.holeWidthFactor = holeWidthFactor;
        this.segments = segments;

        // Hold reference to the mesh for later disposal / replacement
        this.mesh = null;

        this._createMesh();
    }

    // Clean up previous geometry & material to avoid GPU leaks
    _clearMesh() {
        if (this.mesh) {
            this.group.remove(this.mesh);
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
    }

    _createMesh() {
        this._clearMesh();

        // Radii of the outer ellipse
        const outerRadiusX = this.width / 2;
        const outerRadiusY = this.height / 2;

        // Ensure the wall thickness never creates negative radii
        const innerRadiusX = Math.max(outerRadiusX - this.wallThickness, 0.01);
        const innerRadiusY = Math.max(outerRadiusY - this.wallThickness, 0.01);

        // ==== STEP 1 — build a 2-D shape that represents an ellipse with a
        //             concentric elliptical *hole* (so the interior is hollow) ====
        const outerShape = new THREE.Shape();
        outerShape.absellipse(0, 0, outerRadiusX, outerRadiusY, 0, Math.PI * 2, false, 0);

        const innerPath = new THREE.Path();
        // Winding order must be opposite (clockwise) for a hole
        innerPath.absellipse(0, 0, innerRadiusX, innerRadiusY, 0, Math.PI * 2, true, 0);
        outerShape.holes.push(innerPath);

        // Check cache first – if a geometry with identical parameters was
        // already built we can skip **all** further processing and simply
        // reuse the cached BufferGeometry.
        const cacheKey = getCacheKey(this.width,this.height,this.depth,this.wallThickness,this.numberOfHoles,this.holeWidth,this.holeWidthFactor,this.segments);

        let finalMesh;

        if (__geometryCache.has(cacheKey)) {
            // Directly create a mesh that *shares* the BufferGeometry.  This
            // avoids cloning so all instances reference the same GPU buffers.
            const sharedGeo = __geometryCache.get(cacheKey);
            finalMesh = new THREE.Mesh(sharedGeo);
        } else {
            // ==== STEP 2 — extrude the 2-D ring into a 3-D "tube" ====
            const extrudeSettings = {
                steps: 1,
                depth: this.depth,
                bevelEnabled: false,
                curveSegments: this.segments
            };

            const baseGeometry = new THREE.ExtrudeGeometry(outerShape, extrudeSettings);
            baseGeometry.center(); // Align the geometry so the group's origin is in the middle

            finalMesh = new THREE.Mesh(baseGeometry);
        }

        // ==== STEP 3 — cut pass-through holes on the TOP and BOTTOM edges ====
        // Perform the expensive CSG subtraction **only** when we had to
        // build a fresh geometry.  If we are sharing a cached geometry the
        // slits were already carved out so we can skip this entire block.
        if (!__geometryCache.has(cacheKey) && this.numberOfHoles > 0 && this.holeWidth > 0) {
            const spacing = this.depth / (this.numberOfHoles + 1);
            // Width of the subtraction box – calculating a width that ensures
            // complete penetration through the elliptical surface at any height
            // We need to ensure the box is wide enough to cut through the ellipse 
            // at the point where the slit occurs
            const widthScaleFactor = 1 + (this.height / this.width);
            // Ensure we can create really wide slits when requested
            const boxWidth = this.width * this.holeWidthFactor * widthScaleFactor * 1.2;
            // Height of the subtraction box (vertical size of each slit)
            const maxSlitHeight = outerRadiusY; // centre-line limit so top & bottom never overlap
            const boxHeight = Math.min(this.wallThickness * 1.5 * this.holeWidthFactor, maxSlitHeight);

            for (let i = 0; i < this.numberOfHoles; i++) {
                const zPos = -this.depth / 2 + spacing * (i + 1);

                // Create two boxes per Z-slice: one for the top wall, one for bottom
                for (const sign of [1, -1]) { // 1 => top, -1 => bottom
                    const boxGeo = new THREE.BoxGeometry(boxWidth, boxHeight, this.holeWidth);
                    const boxMesh = new THREE.Mesh(boxGeo);
                    boxMesh.position.set(
                        0, // centred along X
                        sign * (outerRadiusY - boxHeight / 2), // near the top/bottom edge
                        zPos
                    );
                    // IMPORTANT: update matrix so CSG uses the transformed mesh
                    boxMesh.updateMatrix();

                    finalMesh = CSG.subtract(finalMesh, boxMesh);
                }
            }
        }

        // ==== STEP 4 (re-implemented) — mesh clean-up for freshly built geometry ====
        if (!__geometryCache.has(cacheKey)) {
            const g = finalMesh.geometry.toNonIndexed();
            const pos = g.getAttribute('position').array;
            const normals = g.getAttribute('normal').array;

            const keptPos = [];
            const keptNorm = [];

            for (let i = 0; i < pos.length; i += 9) {
                // Calculate centroid & outward radial vector (2-D)
                const cx = (pos[i] + pos[i+3] + pos[i+6]) / 3;
                const cy = (pos[i+1] + pos[i+4] + pos[i+7]) / 3;

                const rLen = Math.hypot(cx, cy);
                if (rLen < 1e-6) continue; // degenerate, ignore

                const rx = cx / rLen;
                const ry = cy / rLen;

                // Average normal (XY components)
                const nx = (normals[i]   + normals[i+3] + normals[i+6]) / 3;
                const ny = (normals[i+1] + normals[i+4] + normals[i+7]) / 3;

                // Keep triangles whose normals point mostly OUTWARD (dot > 0.3)
                const dotOut = nx * rx + ny * ry;
                if (dotOut < 0.3) continue; // inward or side-facing → drop

                // NEW FILTER — discard faces that sit *inside* the true outer
                // ellipse surface (these are the little "ledges" left at the
                // edges of each slit after the CSG subtraction).
                const theta = Math.atan2(cy, cx);
                const cosT  = Math.cos(theta);
                const sinT  = Math.sin(theta);
                const expectedR = (outerRadiusX * outerRadiusY) / Math.sqrt(
                    (outerRadiusY * cosT) * (outerRadiusY * cosT) +
                    (outerRadiusX * sinT) * (outerRadiusX * sinT)
                );
                // Adjust the culling margin so it scales with the overall
                // size of the geometry.  A fixed absolute value worked fine
                // for very small meshes but once the ellipse grows, the
                // geometric error between the ideal curve and the faceted
                // approximation increases proportionally to the radius.
                // If we keep using the tiny constant margin we end up
                // mistakenly deleting legitimate outer-wall faces.  We now
                // base the margin on the *sagitta* (chord error) of the
                // segmentation which is `R * (1 - cos(π / segments))` for a
                // circle of radius R.  The ellipse has two radii, so we use
                // the larger one as a safe upper-bound, and add a small
                // wall-thickness factor to keep behaviour reasonable when the
                // geometry is very small.

                const maxOuterRadius = Math.max(outerRadiusX, outerRadiusY);
                const chordError = maxOuterRadius * (1 - Math.cos(Math.PI / this.segments));

                // Safety factor so we never accidentally cull *true* outer
                // wall faces – tweakable but 1.1 works well in practise.
                const margin = Math.max(this.wallThickness * 0.05, chordError * 1.1);

                if (rLen < expectedR - margin) continue;

                // Keep the triangle
                keptPos.push(
                    pos[i], pos[i+1], pos[i+2],
                    pos[i+3], pos[i+4], pos[i+5],
                    pos[i+6], pos[i+7], pos[i+8]
                );
                keptNorm.push(
                    normals[i], normals[i+1], normals[i+2],
                    normals[i+3], normals[i+4], normals[i+5],
                    normals[i+6], normals[i+7], normals[i+8]
                );
            }

            const newGeom = new THREE.BufferGeometry();
            newGeom.setAttribute('position', new THREE.Float32BufferAttribute(keptPos, 3));
            newGeom.setAttribute('normal',   new THREE.Float32BufferAttribute(keptNorm, 3));
            newGeom.computeVertexNormals();
            newGeom.computeBoundingSphere();
            finalMesh.geometry.dispose();
            finalMesh.geometry = newGeom;
        }

        // ==== STEP 5 — assign material & add to the outer group ====
        const material = new THREE.MeshStandardMaterial({
            color: 0x00aa88,
            metalness: 0.15,
            roughness: 0.6,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.85
        });

        // After all geometry operations *and* potential caching we assign
        // material and add the mesh to the parent group.  When the geometry
        // was built from scratch we also remember it in the cache so future
        // LayerNorms can share it.

        this.mesh = finalMesh;
        this.mesh.material = material;

        if (!__geometryCache.has(cacheKey)) {
            // Clone before caching?  No – we store the *exact* BufferGeometry
            // instance so that all consumers share it.  This saves memory and
            // makes renderer uploads even faster.
            __geometryCache.set(cacheKey, this.mesh.geometry);
        }

        // Add to parent group so it appears in the scene
        this.group.add(this.mesh);
    }

    // Public helper to update geometry when parameters change (e.g. from a GUI)
    updateGeometry(params) {
        // Copy over provided keys to the instance so further updates stay in sync
        Object.assign(this, params);
        this._createMesh();
    }

    // Convenience wrappers to change appearance after creation
    setColor(color) {
        if (!this.mesh) return;
        const apply = (mat) => {
            mat.color.copy(color);
            mat.emissive.copy(color);
        };
        if (Array.isArray(this.mesh.material)) {
            this.mesh.material.forEach(apply);
        } else {
            apply(this.mesh.material);
        }
    }

    setMaterialProperties(props) {
        if (!this.mesh) return;
        const apply = (mat) => Object.assign(mat, props);
        if (Array.isArray(this.mesh.material)) {
            this.mesh.material.forEach(apply);
        } else {
            apply(this.mesh.material);
        }
    }

    // Clean up GPU resources when this object is removed from the scene
    dispose() {
        this._clearMesh();
    }
} 