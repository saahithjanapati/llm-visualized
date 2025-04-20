import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GUI } from 'three/examples/jsm/libs/lil-gui.module.min.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'; // Import EffectComposer
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';     // Import RenderPass
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'; // Import UnrealBloomPass
import { VectorVisualization } from '../components/VectorVisualization';
import { WeightMatrixVisualization } from '../components/WeightMatrixVisualization';
import { VECTOR_LENGTH } from '../utils/constants.js'; // Import VECTOR_LENGTH

// Maximum points per trail line (adjust for performance/length)
const MAX_TRAIL_POINTS = 1000; // Further increased buffer size

export function initMainScene(canvas) { // Renamed function here
    // --- Basic Three.js setup ---
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111); // Dark background

    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 10, 25); // Position camera to view both objects

    const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);

    // --- Post Processing Setup ---
    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));

    const bloomParams = {
        strength: 1.5, // Initial strength
        radius: 0.4,   // Initial radius
        threshold: 0.85 // Initial threshold
    };
    const bloomPass = new UnrealBloomPass(
        new THREE.Vector2(window.innerWidth, window.innerHeight),
        bloomParams.strength,
        bloomParams.radius,
        bloomParams.threshold
    );
    composer.addPass(bloomPass);


    // --- Lighting ---
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 10, 7.5);
    scene.add(directionalLight);

    // --- Controls ---
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // --- Keyboard Controls State ---
    const keysPressed = {};
    const moveSpeed = 0.5; // Original speed, can be reused or adjusted
    const panSpeed = 0.2; // Speed for panning
    const zoomSpeed = 0.5; // Speed for zooming
    const rotateSpeed = 0.02; // Adjust rotation speed as needed

    // --- Visualizations ---
    const allVectorVisualizations = []; // Array to hold all vector instances
    const allTrailLines = []; // Array to hold { line, geometry, material } for trails
    const allTrailPoints = []; // Array of arrays to hold points for each trail line [[x,y,z], ...]
    const vectorHeightOffset = 40; // Significantly increased offset

    // Material Definitions (Shared by Matrix & Vectors)
    const availableMaterials = {
        MeshBasicMaterial: THREE.MeshBasicMaterial,
        MeshLambertMaterial: THREE.MeshLambertMaterial,
        MeshPhongMaterial: THREE.MeshPhongMaterial,
        MeshStandardMaterial: THREE.MeshStandardMaterial,
        MeshNormalMaterial: THREE.MeshNormalMaterial,
        MeshDepthMaterial: THREE.MeshDepthMaterial,
        MeshToonMaterial: THREE.MeshToonMaterial,
    };
    const defaultMatrixMaterialName = 'MeshStandardMaterial';
    const defaultVectorMaterialName = 'MeshStandardMaterial'; // Default for vectors

    // Configure matrix with user specified defaults
    const matrixParams = {
        width: 57.5,
        height: 8, // Increased thickness
        depth: 72,
        topWidthFactor: 0.76,
        cornerRadius: 0.1, // Keeping previous default, adjust if needed
        numberOfSlits: 5,
        slitWidth: 2.0, // Increased slit width
        slitDepthFactor: 1.0,
        slitWidthFactor: 0.6,
        opacity: 1.0, // Added opacity parameter
        material: defaultMatrixMaterialName,
        color: 0xaaaaaa, // Default color
    };
    const matrixVis = new WeightMatrixVisualization(null, new THREE.Vector3(0, 0, 0), matrixParams.width, matrixParams.height, matrixParams.depth, matrixParams.topWidthFactor, matrixParams.cornerRadius, matrixParams.numberOfSlits, matrixParams.slitWidth, matrixParams.slitDepthFactor, matrixParams.slitWidthFactor);

    // Function to update matrix material
    function updateMatrixMaterial() {
        if (!matrixVis.group) return;

        // Find the first mesh to get its old material (for disposal)
        let oldMaterial = null;
        matrixVis.group.children.forEach(child => {
            if (child instanceof THREE.Mesh && child.material && !oldMaterial) {
                oldMaterial = child.material;
            }
        });

        const MaterialConstructor = availableMaterials[matrixParams.material];
        if (!MaterialConstructor) {
            console.error('Invalid material selected:', matrixParams.material);
            return;
        }

        const newMaterial = new MaterialConstructor({
            color: matrixParams.color,
            transparent: true, // Keep transparent for opacity control
            opacity: matrixParams.opacity,
            // Add specific properties for certain materials if needed
            // e.g., side: THREE.DoubleSide for MeshBasicMaterial if required
        });

        // Apply the new material to all mesh children
        matrixVis.group.children.forEach(child => {
            if (child instanceof THREE.Mesh) {
                child.material = newMaterial;
                child.material.needsUpdate = true; // Ensure material update is flagged
            }
        });

        // Dispose the old material (only once)
        if (oldMaterial && oldMaterial !== newMaterial && typeof oldMaterial.dispose === 'function') {
            oldMaterial.dispose();
        }
    }

    // Initial material setup
    updateMatrixMaterial();
    scene.add(matrixVis.group);


    // Define colors for animation (kept for potential future use)
    const brightYellow = new THREE.Color(0xFFFF00);
    const darkYellow = new THREE.Color(0xCCA000);
    const matrixCurrentColor = new THREE.Color(); // Color object to reuse for lerping
    const baseMatrixColor = new THREE.Color(); // To store the GUI base color

    // --- Vector Control Parameters ---
    const vectorControlParams = {
        opacity: 1.0,
        metalness: 0.3, // Default from VectorVisualization
        roughness: 0.5, // Default from VectorVisualization
        emissiveIntensity: 0.3, // Default from VectorVisualization
        material: defaultVectorMaterialName // Add material selection for vectors
    };

    // Function to update all vector materials
    function updateAllVectorMaterials(property, value) {
        allVectorVisualizations.forEach(vectorVis => {
            vectorVis.ellipses.forEach(ellipse => {
                if (ellipse.material) {
                    if (property === 'material') {
                        // Change material type
                        const MaterialConstructor = availableMaterials[value];
                        if (!MaterialConstructor) {
                            console.error('Invalid vector material selected:', value);
                            return; // Skip this ellipse if material type is invalid
                        }

                        const oldMaterial = ellipse.material;
                        const newMaterial = new MaterialConstructor({
                            // Preserve existing data-driven color & emissive
                            color: oldMaterial.color,
                            emissive: oldMaterial.emissive,
                            // Apply global controls
                            transparent: vectorControlParams.opacity < 1.0,
                            opacity: vectorControlParams.opacity,
                            metalness: vectorControlParams.metalness,
                            roughness: vectorControlParams.roughness,
                            emissiveIntensity: vectorControlParams.emissiveIntensity,
                        });
                        ellipse.material = newMaterial;
                        if (oldMaterial && typeof oldMaterial.dispose === 'function') {
                            oldMaterial.dispose(); // Dispose the old specific instance
                        }
                    } else {
                         // Update specific property on existing material
                        if (property === 'opacity') {
                            ellipse.material.opacity = value;
                            ellipse.material.transparent = value < 1.0; // Enable transparency only if needed
                        } else if (property === 'metalness') {
                            ellipse.material.metalness = value;
                        } else if (property === 'roughness') {
                            ellipse.material.roughness = value;
                        } else if (property === 'emissiveIntensity') {
                            ellipse.material.emissiveIntensity = value;
                        }
                    }
                    ellipse.material.needsUpdate = true;
                }
            });
        });
    }


    // --- Create and Position Vectors over Slits ---
    const slitSpacing = matrixParams.depth / (matrixParams.numberOfSlits + 1);
    // Define animation bounds
    const startY = -matrixParams.height / 2 - vectorHeightOffset; // Start below the matrix
    const endY = matrixParams.height / 2 + vectorHeightOffset;   // End above the matrix
    const animationDistance = endY - startY;

    for (let i = 0; i < matrixParams.numberOfSlits; i++) {
        // Generate unique data for each vector, matching VECTOR_LENGTH
        const vectorData = Array.from({ length: VECTOR_LENGTH }, () => (Math.random() - 0.5) * 10);

        // Use default constructor from VectorVisualization
        const vectorVis = new VectorVisualization(vectorData);

        // Apply initial vector control parameters (including material type now)
        const InitialVectorMaterial = availableMaterials[vectorControlParams.material];
        vectorVis.ellipses.forEach(ellipse => {
            if (ellipse.material) {
                 const oldMaterial = ellipse.material;
                 const newMaterial = new InitialVectorMaterial({
                     // Preserve initial data-driven color & emissive
                     color: oldMaterial.color,
                     emissive: oldMaterial.emissive,
                     // Apply initial global controls
                     transparent: vectorControlParams.opacity < 1.0,
                     opacity: vectorControlParams.opacity,
                     metalness: vectorControlParams.metalness,
                     roughness: vectorControlParams.roughness,
                     emissiveIntensity: vectorControlParams.emissiveIntensity,
                 });
                 ellipse.material = newMaterial;
                 if (oldMaterial && typeof oldMaterial.dispose === 'function') {
                     oldMaterial.dispose();
                 }
                 ellipse.material.needsUpdate = true;
             }
        });


        // Calculate Z position based on slit index
        const vectorZPos = -matrixParams.depth / 2 + slitSpacing * (i + 1);
        // Set initial position (start of animation)
        vectorVis.group.position.set(0, startY, vectorZPos);

        scene.add(vectorVis.group);
        allVectorVisualizations.push(vectorVis); // Add to array for cleanup

        // --- Create Trail Line for this Vector ---
        allTrailPoints.push([]); // Initialize points array for this trail
        const trailGeometry = new THREE.BufferGeometry();
        // Pre-allocate buffer large enough for max points
        const positions = new Float32Array(MAX_TRAIL_POINTS * 3);
        trailGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        const trailMaterial = new THREE.LineBasicMaterial({ color: 0x888888 }); // Gray color
        const trailLine = new THREE.Line(trailGeometry, trailMaterial);
        scene.add(trailLine);
        allTrailLines.push({ line: trailLine, geometry: trailGeometry, material: trailMaterial });
    }

    // --- GUI ---
    const gui = new GUI();

    // Lighting Folder
    const lightFolder = gui.addFolder('Lighting');
    lightFolder.add(ambientLight, 'intensity', 0, 2, 0.1).name('Ambient Intensity');
    lightFolder.add(directionalLight, 'intensity', 0, 2, 0.1).name('Directional Intensity');
    lightFolder.add(directionalLight.position, 'x', -20, 20, 0.5).name('Dir Light X');
    lightFolder.add(directionalLight.position, 'y', -20, 20, 0.5).name('Dir Light Y');
    lightFolder.add(directionalLight.position, 'z', -20, 20, 0.5).name('Dir Light Z');

    // Bloom Effect Folder
    const bloomFolder = gui.addFolder('Bloom Effect');
    bloomFolder.add(bloomParams, 'strength', 0, 3, 0.05).name('Strength').onChange((value) => { bloomPass.strength = value; });
    bloomFolder.add(bloomParams, 'radius', 0, 1, 0.01).name('Radius').onChange((value) => { bloomPass.radius = value; });
    bloomFolder.add(bloomParams, 'threshold', 0, 1, 0.01).name('Threshold').onChange((value) => { bloomPass.threshold = value; });
    // bloomFolder.open();

    // Matrix Folder
    const matrixFolder = gui.addFolder('Matrix');
    matrixFolder.add(matrixParams, 'material', Object.keys(availableMaterials)).name('Material').onChange(updateMatrixMaterial);
    matrixFolder.addColor(matrixParams, 'color').name('Color').onChange(updateMatrixMaterial);
    matrixFolder.add(matrixParams, 'numberOfSlits', 0, 20, 1).name('Number of Slits').onChange(() => { /* Complex update needed */ });
    matrixFolder.add(matrixParams, 'slitWidth', 0.1, 5.0, 0.05).name('Slit Width').onChange(() => matrixVis.updateGeometry(matrixParams));
    matrixFolder.add(matrixParams, 'slitDepthFactor', 0, 1, 0.01).name('Slit Depth Factor').onChange(() => matrixVis.updateGeometry(matrixParams));
    matrixFolder.add(matrixParams, 'slitWidthFactor', 0.1, 1, 0.01).name('Slit Width Factor').onChange(() => matrixVis.updateGeometry(matrixParams));
    matrixFolder.add(matrixParams, 'width', 1, 100, 0.5).name('Base Width').onChange(() => matrixVis.updateGeometry(matrixParams));
    matrixFolder.add(matrixParams, 'topWidthFactor', 0.1, 2, 0.01).name('Top Width Factor').onChange(() => matrixVis.updateGeometry(matrixParams));
    matrixFolder.add(matrixParams, 'cornerRadius', 0, 5, 0.05).name('Corner Radius').onChange(() => matrixVis.updateGeometry(matrixParams));
    matrixFolder.add(matrixParams, 'depth', 5, 100, 1).name('Depth').onChange(() => { /* Complex update needed */ });
    matrixFolder.add(matrixParams, 'opacity', 0, 1, 0.01).name('Opacity').onChange((value) => {
        matrixParams.opacity = value; // Store value
        updateMatrixMaterial(); // Recreate material with new opacity
    });
    matrixFolder.open();

    // Vectors Folder
    const vectorFolder = gui.addFolder('Vectors');
    vectorFolder.add(vectorControlParams, 'material', Object.keys(availableMaterials)).name('Material').onChange(value => updateAllVectorMaterials('material', value));
    vectorFolder.add(vectorControlParams, 'opacity', 0, 1, 0.01).name('Opacity').onChange(value => updateAllVectorMaterials('opacity', value));
    vectorFolder.add(vectorControlParams, 'metalness', 0, 1, 0.01).name('Metalness').onChange(value => updateAllVectorMaterials('metalness', value));
    vectorFolder.add(vectorControlParams, 'roughness', 0, 1, 0.01).name('Roughness').onChange(value => updateAllVectorMaterials('roughness', value));
    vectorFolder.add(vectorControlParams, 'emissiveIntensity', 0, 1, 0.01).name('Emissive Intensity').onChange(value => updateAllVectorMaterials('emissiveIntensity', value));
    // vectorFolder.open(); // Keep closed by default

    // --- Resize Handling ---
    window.addEventListener('resize', onWindowResize, false);
    function onWindowResize() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
        composer.setSize(window.innerWidth, window.innerHeight); // Update composer size
    }

    // --- Keyboard Event Listeners ---
    function onKeyDown(event) {
        keysPressed[event.key.toLowerCase()] = true;
        keysPressed[event.code] = true; // Also store by code for arrow keys
    }

    function onKeyUp(event) {
        keysPressed[event.key.toLowerCase()] = false;
        keysPressed[event.code] = false;
    }

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);

    const clock = new THREE.Clock(); // Clock for animation timing
    let previousY = startY; // Initialize previous Y position tracking

    // --- Animation Loop ---
    function animate() {
        requestAnimationFrame(animate);

        // --- Handle Keyboard Movement ---
        const cameraRight = new THREE.Vector3();
        camera.getWorldDirection(cameraRight); // Get forward direction first
        cameraRight.cross(camera.up).normalize(); // Calculate right vector
        const cameraUp = new THREE.Vector3().copy(camera.up); // Use world up for panning usually

        // WASD for Panning (Move camera and target together)
        if (keysPressed['w']) {
            const panOffset = cameraUp.clone().multiplyScalar(panSpeed);
            camera.position.add(panOffset);
            controls.target.add(panOffset);
        }
        if (keysPressed['s']) {
            const panOffset = cameraUp.clone().multiplyScalar(-panSpeed);
            camera.position.add(panOffset);
            controls.target.add(panOffset);
        }
        if (keysPressed['a']) {
            const panOffset = cameraRight.clone().multiplyScalar(-panSpeed);
            camera.position.add(panOffset);
            controls.target.add(panOffset);
        }
        if (keysPressed['d']) {
            const panOffset = cameraRight.clone().multiplyScalar(panSpeed);
            camera.position.add(panOffset);
            controls.target.add(panOffset);
        }

        // Arrow Up/Down for Tilting (Pitch - Adjusting OrbitControls target vertically)
        if (keysPressed['ArrowUp']) {
            const targetOffset = cameraUp.clone().multiplyScalar(rotateSpeed * 10); // Adjust target up
            controls.target.add(targetOffset);
        }
        if (keysPressed['ArrowDown']) {
            const targetOffset = cameraUp.clone().multiplyScalar(-rotateSpeed * 10); // Adjust target down
            controls.target.add(targetOffset);
        }

        // Arrow Left/Right for Rotating View Angle (Adjusting OrbitControls target horizontally)
        if (keysPressed['ArrowLeft']) {
            const targetOffset = cameraRight.clone().multiplyScalar(-rotateSpeed * 10); // Adjust target left
            controls.target.add(targetOffset);
        }
        if (keysPressed['ArrowRight']) {
            const targetOffset = cameraRight.clone().multiplyScalar(rotateSpeed * 10); // Adjust target right
            controls.target.add(targetOffset);
        }


        controls.update(); // Required if enableDamping is true

        const elapsedTime = clock.getElapsedTime();
        const animationDuration = 5; // seconds for one full loop (up and down implied by modulo)
        const loopTime = (elapsedTime % animationDuration) / animationDuration; // Normalized loop time (0 to 1)

        // Calculate current Y position using linear interpolation within the loop
        const currentY = startY + loopTime * animationDistance;

        // --- Check for Animation Reset ---
        const animationReset = currentY < previousY;
        if (animationReset) {
            // Clear points for all trails
            allTrailPoints.forEach(points => points.length = 0);
        }

        // --- Update Trail Lines ---
        allTrailLines.forEach((trail, index) => {
            const currentPoints = allTrailPoints[index];
            const vectorZPos = allVectorVisualizations[index].group.position.z; // Get Z for this vector
            const newPoint = [0, currentY, vectorZPos];

            // Add new point
            currentPoints.push(newPoint);

            // Update geometry attribute
            const positionAttribute = trail.geometry.getAttribute('position');
            for (let j = 0; j < currentPoints.length; j++) {
                positionAttribute.setXYZ(j, currentPoints[j][0], currentPoints[j][1], currentPoints[j][2]);
            }

            trail.geometry.setDrawRange(0, currentPoints.length);
            positionAttribute.needsUpdate = true;
        });

        // --- Matrix Color Animation ---
        // Check if the current material uses standard color property
        const canAnimateColor = matrixParams.material !== 'MeshNormalMaterial' && matrixParams.material !== 'MeshDepthMaterial';

        if (canAnimateColor && matrixVis.group) {
            const matrixBottomY = -matrixParams.height / 2;
            const matrixMidY = 0; // Center of the matrix
            const matrixTopY = matrixParams.height / 2;

            // Get base color from GUI and derive dark version
            baseMatrixColor.set(matrixParams.color);
            const outsideColor = baseMatrixColor.clone().multiplyScalar(0.2);

            let t = 0;
            if (currentY < matrixBottomY) {
                matrixCurrentColor.copy(outsideColor);
            } else if (currentY >= matrixBottomY && currentY < matrixMidY) {
                // Entering: Dark Base to Bright Yellow
                t = (currentY - matrixBottomY) / (matrixMidY - matrixBottomY);
                matrixCurrentColor.lerpColors(outsideColor, brightYellow, t);
            } else if (currentY >= matrixMidY && currentY < matrixTopY) {
                // Leaving: Bright Yellow to Dark Yellow
                t = (currentY - matrixMidY) / (matrixTopY - matrixMidY);
                matrixCurrentColor.lerpColors(brightYellow, darkYellow, t);
            } else { // currentY >= matrixTopY
                matrixCurrentColor.copy(darkYellow);
            }

            // Apply the calculated color to all matrix mesh materials
            matrixVis.group.children.forEach(child => {
                if (child instanceof THREE.Mesh && child.material) {
                     // Check if material has a color property before setting
                    if (child.material.color) {
                        child.material.color.copy(matrixCurrentColor);
                    }
                }
            });
        }

        // Update vector positions
        allVectorVisualizations.forEach(vectorVis => {
            vectorVis.group.position.y = currentY;
        });

        // Update previous Y for next frame
        previousY = currentY;

        // Use composer to render with post-processing effects
        composer.render();
        // renderer.render(scene, camera); // Replaced by composer.render()
    }

    animate(); // Start the animation loop

    // Return cleanup function
    return () => {
        window.removeEventListener('resize', onWindowResize);
        window.removeEventListener('keydown', onKeyDown); // Remove key listeners
        window.removeEventListener('keyup', onKeyUp);     // Remove key listeners
        controls.dispose();
        gui.destroy();

        // Dispose Bloom Pass Render Targets if they exist
        if (bloomPass && bloomPass.renderTargetBright) bloomPass.renderTargetBright.dispose();
        if (bloomPass && bloomPass.renderTargetBlur) bloomPass.renderTargetBlur.dispose();
        // We don't explicitly dispose the composer itself, but ensure passes are handled.

        // Dispose geometries and materials
        allVectorVisualizations.forEach(vec => vec.dispose()); // Dispose all vectors (handles ellipse materials/geometries)
        allTrailLines.forEach(trail => { // Dispose trails
            trail.geometry.dispose();
            trail.material.dispose();
        });

        // Dispose the final matrix material (find it on one of the meshes)
        let finalMatrixMaterial = null;
        if (matrixVis.group) {
            matrixVis.group.children.forEach(child => {
                if (child instanceof THREE.Mesh && child.material && !finalMatrixMaterial) {
                    finalMatrixMaterial = child.material;
                }
            });
        }
        if (finalMatrixMaterial && typeof finalMatrixMaterial.dispose === 'function') {
            finalMatrixMaterial.dispose();
        }

        matrixVis._clearMesh(); // Use existing clear method which handles matrix geometries
        // Add disposal for other objects if needed
        renderer.dispose();
    };
} 