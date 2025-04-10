import * as THREE from 'three';

export class WeightMatrixVisualization {
    constructor(data = null, position = new THREE.Vector3(0, 0, 0), width = 8, height = 4, depth = 30, topWidthFactor = 0.7, cornerRadius = 0.8, numberOfSlits = 0, slitWidth = 0.2, slitHeight = 0, slitColor = 0x000000, slitOpacity = 0.9, slitWidthFactor = 0.9) {
        this.group = new THREE.Group();
        this.group.position.copy(position);

        this.width = width;
        this.height = height;
        this.depth = depth;
        this.topWidthFactor = topWidthFactor; // Width of the top relative to the bottom
        this.cornerRadius = cornerRadius;
        this.numberOfSlits = numberOfSlits;
        this.slitWidth = slitWidth;
        this.slitHeight = slitHeight; // 0 means auto height calculation
        this.slitColor = slitColor;
        this.slitOpacity = slitOpacity;
        this.slitWidthFactor = slitWidthFactor; // Percentage of trapezoid width that slits occupy

        this.mesh = null; // Main trapezoid mesh
        this.slitMeshes = []; // Array to store slit meshes

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

        // Remove and dispose of all slit meshes
        this.slitMeshes.forEach(slitMesh => {
            this.group.remove(slitMesh);
            if (slitMesh.geometry) slitMesh.geometry.dispose();
            if (slitMesh.material) slitMesh.material.dispose();
        });
        this.slitMeshes = [];
    }

    _createMesh() {
        this._clearMesh(); // Clear previous mesh and resources

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
            bevelEnabled: true,
            bevelThickness: cr,
            bevelSize: cr,
            bevelOffset: -cr,
            bevelSegments: 24 // Increase for smoother bevels
        };

        // Create geometry by extruding the shape (with potential holes)
        const geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);
        geometry.center(); // Center the resulting geometry

        // Create material (render both sides as holes make inside visible)
        const material = new THREE.MeshStandardMaterial({
            color: 0x0077ff, // Initial color
            metalness: 0.1,
            roughness: 0.7,
            flatShading: false,
            side: THREE.DoubleSide
        });

        // Create the mesh and add it to the group
        this.mesh = new THREE.Mesh(geometry, material);
        this.group.add(this.mesh);

        // --- Create horizontal slits if requested ---
        if (this.numberOfSlits > 0 && this.slitWidth > 0) {
            // Calculate spacing between slits
            const slitSpacing = this.depth / (this.numberOfSlits + 1);
            
            // Create a slightly transparent material for visualizing the slits
            const slitMaterial = new THREE.MeshBasicMaterial({
                color: this.slitColor,
                transparent: true,
                opacity: this.slitOpacity,
                side: THREE.DoubleSide
            });
            
            // Calculate standardized slit dimensions
            // If slitHeight is 0, make it slightly larger than the trapezoid height
            const actualSlitHeight = this.slitHeight <= 0 ? this.height + 0.2 : this.slitHeight;
            
            // Calculate standard width for all slits - use average of bottom and top width
            const avgWidth = (this.width + (this.width * this.topWidthFactor)) / 2;
            const standardSlitWidth = avgWidth * this.slitWidthFactor;
            
            for (let i = 0; i < this.numberOfSlits; i++) {
                // Calculate position along the depth
                const zPos = -this.depth/2 + slitSpacing * (i + 1);
                
                // Create a thin box for the slit with standardized width
                const slitGeometry = new THREE.BoxGeometry(standardSlitWidth, actualSlitHeight, this.slitWidth);
                
                const slitMesh = new THREE.Mesh(slitGeometry, slitMaterial);
                
                // Position the slit at the correct location along the depth
                slitMesh.position.set(0, 0, zPos);
                
                // Add the slit mesh to the group and store it
                this.group.add(slitMesh);
                this.slitMeshes.push(slitMesh);
            }
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
        
        // New parameters
        this.slitHeight = params.slitHeight ?? this.slitHeight;
        this.slitColor = params.slitColor ?? this.slitColor;
        this.slitOpacity = params.slitOpacity ?? this.slitOpacity;
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
        if (this.mesh && this.mesh.material) {
            const mat = this.mesh.material;
            // Handle potential array of materials (though ExtrudeGeometry usually creates one)
            if (Array.isArray(mat)) {
                mat.forEach(m => m.color.set(color));
            } else {
                mat.color.set(color);
            }
        }
    }

    setMaterialProperties(props) {
        if (this.mesh && this.mesh.material) {
            const applyProps = (mat) => {
                if (props.metalness !== undefined) mat.metalness = props.metalness;
                if (props.roughness !== undefined) mat.roughness = props.roughness;
            };
            const mat = this.mesh.material;
            // Handle potential array of materials
            if (Array.isArray(mat)) {
                mat.forEach(applyProps);
            } else {
                applyProps(mat);
            }
        }
    }

    // Update slit properties without recreating the whole mesh
    updateSlitProperties(props) {
        if (props.slitColor !== undefined || props.slitOpacity !== undefined) {
            this.slitMeshes.forEach(slitMesh => {
                if (props.slitColor !== undefined) {
                    this.slitColor = props.slitColor;
                    slitMesh.material.color.set(props.slitColor);
                }
                if (props.slitOpacity !== undefined) {
                    this.slitOpacity = props.slitOpacity;
                    slitMesh.material.opacity = props.slitOpacity;
                }
            });
        }
    }
}