import * as THREE from 'three';
import { CSG } from 'three-csg-ts'; // Import CSG

export class WeightMatrixVisualization {
    constructor(data = null, position = new THREE.Vector3(0, 0, 0), width = 8, height = 4, depth = 30, topWidthFactor = 0.7, cornerRadius = 0.8, numberOfSlits = 0, slitWidth = 0.2, slitDepthFactor = 1.0, slitWidthFactor = 0.9) {
        this.group = new THREE.Group();
        this.group.position.copy(position);

        this.width = width;
        this.height = height;
        this.depth = depth;
        this.topWidthFactor = topWidthFactor; // Width of the top relative to the bottom
        this.cornerRadius = cornerRadius;
        this.numberOfSlits = numberOfSlits;
        this.slitWidth = slitWidth;
        this.slitDepthFactor = Math.max(0, Math.min(1, slitDepthFactor)); // Clamp between 0 and 1
        this.slitWidthFactor = slitWidthFactor; // Percentage of trapezoid width that slits occupy

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
        this._clearMesh(); // Clear previous mesh and resources

        // --- Create Main Trapezoid Shape (used for extrusion AND caps) ---
        const shape = new THREE.Shape();
        const hw = this.width / 2;
        const hTopW = (this.width * this.topWidthFactor) / 2;
        const hh = this.height / 2;
        const cr = this.cornerRadius;

        // Define the outer trapezoid profile
        shape.moveTo(-hw, -hh);          // Bottom Left
        shape.lineTo(hw, -hh);           // Bottom Right
        shape.lineTo(hTopW, hh);         // Top Right
        shape.lineTo(-hTopW, hh);        // Top Left
        shape.closePath();               // Close the shape

        // Extrusion settings
        const extrudeSettings = {
            steps: 1,
            depth: this.depth,
            bevelEnabled: false, // Disable bevel
            // bevelThickness: cr,
            // bevelSize: cr,
            // bevelOffset: -cr,
            // bevelSegments: 24 
        };

        // Create initial geometry by extruding the shape
        const baseGeometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);
        baseGeometry.center(); // Center the resulting geometry

        // Create a Mesh for the base geometry to use with CSG
        const baseMesh = new THREE.Mesh(baseGeometry); // Material is not needed for CSG operation itself

        // --- Create and Subtract Slits using CSG (for the side walls) ---
        let finalMesh = baseMesh; // Start with the base mesh

        if (this.numberOfSlits > 0 && this.slitWidth > 0) {
            const slitSpacing = this.depth / (this.numberOfSlits + 1);

            // Calculate the actual depth of the cut based on the factor
            const cutDepth = this.height * this.slitDepthFactor;
            // Make the subtraction box height match the cut depth (+ epsilon)
            const slitBoxHeight = cutDepth + 0.001; // Use exact depth + epsilon
            // Calculate the Y position for the center of the cut region
            const cutCenterY = (this.height / 2) - (cutDepth / 2);
            
            // Use constant average width for slits (based on diagnostic findings)
            const avgWidth = (this.width + (this.width * this.topWidthFactor)) / 2;
            const constantSlitBoxWidth = avgWidth * this.slitWidthFactor;

            for (let i = 0; i < this.numberOfSlits; i++) {
                const zPos = -this.depth / 2 + slitSpacing * (i + 1);

                // Create a box geometry for the slit with the CONSTANT width and EXACT cut depth (+epsilon)
                const slitGeometry = new THREE.BoxGeometry(constantSlitBoxWidth, slitBoxHeight, this.slitWidth);
                
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
        const material = new THREE.MeshStandardMaterial({
            color: 0x0077ff, // Initial color
            metalness: 0.1,
            roughness: 0.7,
            flatShading: false,
            side: THREE.DoubleSide // Keep DoubleSide for the extruded part in case caps aren't perfectly flush
        });

        // Assign the final CSG result geometry and the material to the main mesh (sides)
        this.mesh = finalMesh; 
        this.mesh.material = material;
        this.mesh.geometry.computeVertexNormals(); 

        // Add the side walls mesh (with holes) to the group
        this.group.add(this.mesh);

        // --- Create and Add Front/Back Caps ---
        const capGeometry = new THREE.ShapeGeometry(shape); // Use the original 2D shape
        capGeometry.center(); // Center it like the extruded geometry

        this.frontCapMesh = new THREE.Mesh(capGeometry, material);
        this.frontCapMesh.position.z = this.depth / 2; // Position at the front
        this.group.add(this.frontCapMesh);

        this.backCapMesh = new THREE.Mesh(capGeometry, material);
        this.backCapMesh.position.z = -this.depth / 2; // Position at the back
        this.backCapMesh.rotation.y = Math.PI; // Rotate to face outwards
        this.group.add(this.backCapMesh);
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
        this.slitWidthFactor = params.slitWidthFactor ?? this.slitWidthFactor;

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
            if (mat) {
                if (Array.isArray(mat)) {
                    mat.forEach(m => m.color.set(color));
                } else {
                    mat.color.set(color);
                }
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
                    });
                } else {
                    if (props.metalness !== undefined) mat.metalness = props.metalness;
                    if (props.roughness !== undefined) mat.roughness = props.roughness;
                }
            }
           
        };
        applyProps(this.mesh?.material);
        applyProps(this.frontCapMesh?.material); // Apply to caps too
        applyProps(this.backCapMesh?.material);  // Apply to caps too
    }
}