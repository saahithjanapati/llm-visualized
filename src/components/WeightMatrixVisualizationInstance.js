import * as THREE from 'three';
import { CSG } from 'three-csg-ts';

// A lightweight copy of `WeightMatrixVisualization` that re-uses a *single* geometry
// instance across *all* objects.  This avoids repeating the very heavy CSG
// computation for every matrix whilst maintaining the same overall appearance.
//
// IMPORTANT:
//   • We assume **all** matrices are created with *identical* parameters
//     (width/height/depth etc.) as is the case in MultiHeadAttentionAnimation.
//   • If any parameter changes via `updateGeometry`, the shared geometry will be
//     recomputed once and propagated to every instance created *after* that.
//     Already-existing meshes are **not** retro-updated — that would require a
//     more complex observer pattern.  For the current usage pattern this is
//     sufficient.
export class WeightMatrixVisualizationInstance {
    // Static cache for the heavy, CSG-generated geometries.
    static _cachedGeometries = null; // { baseGeom, capGeom }
    static _cachedParams = null;     // keep track of params used for cache

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
        slitTopWidthFactor = null
    ) {
        this.group = new THREE.Group();
        this.group.position.copy(position);

        // Store properties in case someone calls `updateGeometry`
        this.width = width;
        this.height = height;
        this.depth = depth;
        this.topWidthFactor = topWidthFactor;
        this.cornerRadius = cornerRadius;
        this.numberOfSlits = numberOfSlits;
        this.slitWidth = slitWidth;
        this.slitDepthFactor = Math.max(0, Math.min(1, slitDepthFactor));
        this.slitBottomWidthFactor = slitBottomWidthFactor;
        this.slitTopWidthFactor = slitTopWidthFactor !== null ? slitTopWidthFactor : slitBottomWidthFactor;

        this.mesh = null;          // side walls mesh
        this.frontCapMesh = null;  // front face
        this.backCapMesh = null;   // back face

        this._createMesh();

        if (data) {
            this.updateData(data);
        }
    }

    // Helper util to compare param sets
    static _paramsKey(params) {
        // JSON stringify is fine here given the small param set
        return JSON.stringify(params);
    }

    _clearMesh() {
        if (this.mesh) {
            this.group.remove(this.mesh);
            // We **do not** dispose geometries: they are shared!
            if (this.mesh.material) {
                if (Array.isArray(this.mesh.material)) {
                    this.mesh.material.forEach(m => m.dispose());
                } else {
                    this.mesh.material.dispose();
                }
            }
            this.mesh = null;
        }

        if (this.frontCapMesh) {
            this.group.remove(this.frontCapMesh);
            // share geometry, dispose only material
            if (this.frontCapMesh.material) this.frontCapMesh.material.dispose();
            this.frontCapMesh = null;
        }

        if (this.backCapMesh) {
            this.group.remove(this.backCapMesh);
            if (this.backCapMesh.material) this.backCapMesh.material.dispose();
            this.backCapMesh = null;
        }
    }

    _createMesh() {
        this._clearMesh();

        // Check if we already have geometries cached for these parameters.
        const currentParams = {
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
        };
        const matchesCache = WeightMatrixVisualizationInstance._cachedGeometries &&
                              WeightMatrixVisualizationInstance._paramsKey(currentParams) === WeightMatrixVisualizationInstance._cachedParams;

        let baseGeometry, capShapeGeometry;

        if (matchesCache) {
            // Re-use cached geometries
            ({ baseGeometry, capShapeGeometry } = WeightMatrixVisualizationInstance._cachedGeometries);
        } else {
            // --- Compute geometries *once* and cache them ---
            // The following code is essentially the heavy part copied (with only
            // tiny modifications) from the original WeightMatrixVisualization.

            // Build 2-D trapezoid shape
            const shape = new THREE.Shape();

            const hw     = this.width  / 2;
            const hTopW  = (this.width * this.topWidthFactor) / 2;
            const hh     = this.height / 2;

            let cr = this.cornerRadius;
            const maxBottomRadius = Math.max(0, hw - hTopW);
            const sideLen = Math.hypot(hw - hTopW, this.height);
            const maxSideRadius  = sideLen / 2;
            cr = Math.min(cr, maxBottomRadius, maxSideRadius);

            if (cr < 1e-4) {
                shape.moveTo(-hw, -hh);
                shape.lineTo(hw, -hh);
                shape.lineTo(hTopW, hh);
                shape.lineTo(-hTopW, hh);
                shape.closePath();
            } else {
                const blStart = new THREE.Vector2(-hw + cr, -hh);
                const brStart = new THREE.Vector2(hw - cr,  -hh);
                const trEnd   = new THREE.Vector2(hTopW - cr,  hh);
                const tlEnd   = new THREE.Vector2(-hTopW + cr, hh);

                const rightSideDir = new THREE.Vector2(hTopW - hw, 2*hh).normalize();
                const leftSideDir  = new THREE.Vector2(-hTopW + hw, 2*hh).normalize();

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
            }

            const extrudeSettings = {
                steps: 1,
                depth: this.depth,
                bevelEnabled: false,
                bevelThickness: this.cornerRadius * 1.2,
                bevelSize: this.cornerRadius * 1.2,
                bevelOffset: 0,
                bevelSegments: Math.max(6, Math.round(this.cornerRadius * 3))
            };

            // Generate base geometry via extrusion
            const tempExtrude = new THREE.ExtrudeGeometry(shape, extrudeSettings);
            tempExtrude.center();

            let baseMesh = new THREE.Mesh(tempExtrude);

            if (this.numberOfSlits > 0 && this.slitWidth > 0) {
                const slitSpacing = this.depth / (this.numberOfSlits + 1);
                const cutDepth = this.height * this.slitDepthFactor;
                const slitBoxHeight = cutDepth + this.cornerRadius * 2 + 0.001;
                const cutCenterYTop = (this.height / 2) - (slitBoxHeight / 2);
                const cutCenterYBottom = (-this.height / 2) + (slitBoxHeight / 2);

                const widest = Math.max(this.width, this.width * this.topWidthFactor);
                const constantBottomWidth = (widest + this.cornerRadius * 2) * this.slitBottomWidthFactor;
                const constantTopWidth    = (widest + this.cornerRadius * 2) * this.slitTopWidthFactor;

                const useDualCuts = this.slitDepthFactor < 0.95;
                for (let i = 0; i < this.numberOfSlits; i++) {
                    const zPos = -this.depth / 2 + slitSpacing * (i + 1);
                    let slitGeometry;
                    if (Math.abs(constantBottomWidth - constantTopWidth) < 1e-4) {
                        slitGeometry = new THREE.BoxGeometry(constantBottomWidth, slitBoxHeight, this.slitWidth);
                    } else {
                        slitGeometry = new THREE.BoxGeometry(constantBottomWidth, slitBoxHeight, this.slitWidth, 1, 1, 1);
                        const posAttr = slitGeometry.attributes.position;
                        const halfH = slitBoxHeight / 2;
                        for (let v = 0; v < posAttr.count; v++) {
                            const y = posAttr.getY(v);
                            const t = (y + halfH) / slitBoxHeight;
                            const targetWidth = THREE.MathUtils.lerp(constantBottomWidth, constantTopWidth, t);
                            const scale = targetWidth / constantBottomWidth;
                            posAttr.setX(v, posAttr.getX(v) * scale);
                        }
                        posAttr.needsUpdate = true;
                        slitGeometry.computeVertexNormals();
                    }
                    const slitMeshTop = new THREE.Mesh(slitGeometry);
                    slitMeshTop.position.set(0, cutCenterYTop, zPos);
                    slitMeshTop.updateMatrix();
                    baseMesh = CSG.subtract(baseMesh, slitMeshTop);

                    if (useDualCuts) {
                        const slitMeshBottom = new THREE.Mesh(slitGeometry);
                        slitMeshBottom.position.set(0, cutCenterYBottom, zPos);
                        slitMeshBottom.updateMatrix();
                        baseMesh = CSG.subtract(baseMesh, slitMeshBottom);
                    }
                }
            }

            baseGeometry = baseMesh.geometry;
            baseGeometry.computeVertexNormals();

            // cap shape geometry (flat)
            capShapeGeometry = new THREE.ShapeGeometry(shape);
            capShapeGeometry.center();

            // Store in cache
            WeightMatrixVisualizationInstance._cachedGeometries = { baseGeometry, capShapeGeometry };
            WeightMatrixVisualizationInstance._cachedParams = WeightMatrixVisualizationInstance._paramsKey(currentParams);
        }

        // --- Create Meshes for *this* instance ---
        const sideMaterial = new THREE.MeshStandardMaterial({
            color: 0x0077ff,
            metalness: 0.1,
            roughness: 0.7,
            flatShading: false,
            side: THREE.FrontSide,
            transparent: true,
            opacity: 0.8
        });

        const capMaterial = sideMaterial.clone();
        capMaterial.polygonOffset = true;
        capMaterial.polygonOffsetFactor = -1;
        capMaterial.polygonOffsetUnits  = -4;

        // Side walls
        this.mesh = new THREE.Mesh(baseGeometry, sideMaterial);
        this.mesh.renderOrder = 0;
        this.group.add(this.mesh);

        // Caps use *the same* shape geometry.  We place them with a tiny epsilon
        // offset to avoid z-fighting exactly the same way the original visualisation did.
        const epsilon = 0.05;
        this.frontCapMesh = new THREE.Mesh(capShapeGeometry, capMaterial);
        this.frontCapMesh.position.z = this.depth / 2 + epsilon;
        this.frontCapMesh.renderOrder = 1;
        this.group.add(this.frontCapMesh);

        this.backCapMesh = new THREE.Mesh(capShapeGeometry, capMaterial);
        this.backCapMesh.position.z = -this.depth / 2 - epsilon;
        this.backCapMesh.rotation.y = Math.PI;
        this.backCapMesh.renderOrder = 2;
        this.group.add(this.backCapMesh);
    }

    // API compatibility layer --------------------------------------------------
    updateData(data) {
        console.log('Updating weight matrix (shared) with data:', data);
        // Same placeholder behaviour as parent class.
    }

    updateGeometry(params) {
        // Only recompute cache if *any* parameter differs from the cached set.
        const newParams = { ...params };
        const key = WeightMatrixVisualizationInstance._paramsKey(newParams);
        if (key !== WeightMatrixVisualizationInstance._cachedParams) {
            // Update instance properties then force rebuild & recache
            Object.assign(this, params);
            // Clear static cache so _createMesh will regenerate
            WeightMatrixVisualizationInstance._cachedGeometries = null;
            WeightMatrixVisualizationInstance._cachedParams = null;
        }
        // Rebuild mesh for this instance (will reuse or recompute cache)
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
                mat.forEach(m => m.color.set(color));
            } else {
                mat.color.set(color);
            }
        };
        applyColor(this.mesh.material);
        applyColor(this.frontCapMesh.material);
        applyColor(this.backCapMesh.material);
    }

    setMaterialProperties(props) {
        const applyProps = (mat) => {
            if (!mat) return;
            if (Array.isArray(mat)) {
                mat.forEach(m => {
                    if (props.metalness !== undefined) m.metalness = props.metalness;
                    if (props.roughness !== undefined) m.roughness = props.roughness;
                });
            } else {
                if (props.metalness !== undefined) mat.metalness = props.metalness;
                if (props.roughness !== undefined) mat.roughness = props.roughness;
            }
        };
        applyProps(this.mesh.material);
        applyProps(this.frontCapMesh.material);
        applyProps(this.backCapMesh.material);
    }
} 
