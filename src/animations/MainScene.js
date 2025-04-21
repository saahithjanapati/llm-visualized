import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GUI } from 'three/examples/jsm/libs/lil-gui.module.min.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'; // Import EffectComposer
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';     // Import RenderPass
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'; // Import UnrealBloomPass
import { VectorVisualization } from '../components/VectorVisualization';
import { WeightMatrixVisualization } from '../components/WeightMatrixVisualization';
import { VECTOR_LENGTH } from '../utils/constants.js'; // Import VECTOR_LENGTH
import { mapValueToColor } from '../utils/colors.js'; // For vector addition color updates
import TWEEN from '@tweenjs/tween.js'; // Tweening library for addition animation

// Maximum points per trail line (adjust for performance/length)
const MAX_TRAIL_POINTS = 2000; // Further increased buffer size

export function initMainScene(canvas) { // Renamed function here
    // --- Basic Three.js setup ---
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111); // Dark background

    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 3000);
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
    controls.maxDistance = 2500; // prevent zooming beyond far plane

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
    const vectorHeightOffset = 60; // Start even lower so vectors appear further beneath the matrix

    // --- Branching / Duplicate Vector Setup ---
    // Arrays to hold duplicate vectors and their trails
    const branchedVectorVisualizations = [];
    const branchedTrailLines = [];
    const branchedTrailPoints = [];
    const additionPlayedFlags = [];

    // Configuration for the duplicate vector path (all durations are expressed as fraction of the full loop 0..1)
    const branchConfig = {
        branchStartT: 0.15,   // When (0..1) the branch occurs (still below the matrix)
        horzDurationT: 0.20,  // Fraction of the loop spent moving right
        vertDurationT: 0.30,  // Fraction spent moving up
        leftDurationT: 0.25,  // Fraction spent moving left to meet originals
        branchX: 45,          // How far to move to the right (in world units) – move further to avoid the matrix
        meetYOffset: 5        // Final Y offset above the original vector path when they meet
    };

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
        cornerRadius: 2, // Default corner radius
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

    // --- Vector Addition Animation (adapted from VectorAdditionAnimation.js) ---
    function startAdditionAnimation(vec1, vec2) {
        const duration = 400; // faster animation
        const flashDuration = 80;
        const delayBetweenCubes = 15;
        const vectorLength = vec1.ellipses.length;

        if (vectorLength !== vec2.ellipses.length) return;

        // Mark addition animation playing so main loop doesn't reset prematurely
        additionPlaying.active = true;
        additionPlaying.endTime = performance.now() + duration + flashDuration + vectorLength * delayBetweenCubes;

        console.log('[startAdditionAnimation] active tweens before creation:', TWEEN.getAll().length);

        for (let i = 0; i < vectorLength; i++) {
            const ellipse1 = vec1.ellipses[i];
            const ellipse2 = vec2.ellipses[i];
            if (!ellipse1 || !ellipse2) continue;

            // Force matrix updates before calculations
            vec1.group.updateMatrixWorld(true);
            vec2.group.updateMatrixWorld(true);

            const targetPosition = new THREE.Vector3();
            ellipse2.getWorldPosition(targetPosition);
            const localTarget = ellipse1.parent.worldToLocal(targetPosition.clone());

            // Debug for first cube of first addition run
            if (i === 0 && !additionPlaying.logged) {
                const sourceWorldPosition = new THREE.Vector3();
                ellipse1.getWorldPosition(sourceWorldPosition);
                console.log(`Debug (i=0):`);
                console.log(`  ellipse2 (target) world Y: ${targetPosition.y}`);
                console.log(`  ellipse1 (source) world Y: ${sourceWorldPosition.y}`);
                console.log(`  ellipse1.parent (branched group) world Y: ${ellipse1.parent.position.y}`);
                console.log(`  Calculated localTarget.y: ${localTarget.y}`);
                console.log(`  Initial ellipse1.position.y: ${ellipse1.position.y}`);
                additionPlaying.logged = true;
            }

            const moveTween = new TWEEN.Tween(ellipse1.position, true)
                .to({ y: localTarget.y }, duration)
                .easing(TWEEN.Easing.Quadratic.InOut)
                .delay(i * delayBetweenCubes)
                .onUpdate(() => {
                    if (i === 0) {
                        console.log('[MoveTween onUpdate] ellipse1.position.y =', ellipse1.position.y);
                    }
                })
                .onStart(() => {
                    if (i === 0 && !additionPlaying.tweenLogged) {
                        console.log('[MoveTween onStart] First tween started. From y=', ellipse1.position.y, 'to', localTarget.y);
                        additionPlaying.tweenLogged = true;
                    }
                })
                .onComplete(() => {
                    if (i === 0) console.log('[MoveTween onComplete] First tween completed');

                    // Flash effect on target cube
                    const originalColor = ellipse2.material.color.clone();
                    const originalEmissive = ellipse2.material.emissive ? ellipse2.material.emissive.clone() : new THREE.Color();
                    const originalIntensity = ellipse2.material.emissiveIntensity !== undefined ? ellipse2.material.emissiveIntensity : 0.0;

                    // Set to white for a bright flash (works well with bloom)
                    ellipse2.material.color.set(0xffffff);
                    if (ellipse2.material.emissive) {
                        ellipse2.material.emissive.set(0xffffff);
                    }
                    ellipse2.material.emissiveIntensity = 1.0;

                    // Tween on the material itself (dummy target) just for timing
                    new TWEEN.Tween(ellipse2.material, true)
                        .to({}, flashDuration)
                        .onComplete(() => {
                            // Calculate new summed value & color
                            const sum = vec1.data[i] + vec2.data[i];
                            vec2.data[i] = sum;
                            const newColor = mapValueToColor(sum);

                            // Restore cube material to new color
                            ellipse2.material.color.copy(newColor);
                            if (ellipse2.material.emissive) {
                                ellipse2.material.emissive.copy(newColor);
                            }
                            ellipse2.material.emissiveIntensity = originalIntensity;

                            // Hide the source cube after merging
                            ellipse1.visible = false;
                        })
                        .start();
                });

            moveTween.start();
            if (i === 0) {
                console.log("moveTween.start() called for i=0");
            }
        }

        console.log('[startAdditionAnimation] active tweens after creation:', TWEEN.getAll().length);
    }

    function resetVectorAddition(vec1, vec2) {
        const len = vec1.ellipses.length;
        for (let i = 0; i < len; i++) {
            const e1 = vec1.ellipses[i];
            const e2 = vec2.ellipses[i];
            if (e1) {
                e1.visible = true;
                e1.position.y = 0;
            }
            if (e2) {
                const color = mapValueToColor(vec2.data[i]);
                e2.material.color.copy(color);
                e2.material.emissive.copy(color);
                e2.material.emissiveIntensity = 0.3;
            }
        }
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
        // Store normalized data (to keep values near [-1,1]) for addition calculations
        vectorVis.data = vectorVis.layerNormalize(vectorData);

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

        // --- Create Duplicate Branched Vector ---
        const branchedVectorVis = new VectorVisualization(vectorData.slice()); // duplicate data
        branchedVectorVis.data = branchedVectorVis.layerNormalize(vectorData.slice());
        // Apply same initial material overrides as originals
        branchedVectorVis.ellipses.forEach((ellipse, idx) => {
            const origEllipse = vectorVis.ellipses[idx];
            if (origEllipse && origEllipse.material) {
                const clonedMat = origEllipse.material.clone();
                ellipse.material.dispose();
                ellipse.material = clonedMat;
            }
        });
        // Initial position identical to original
        branchedVectorVis.group.position.set(0, startY, vectorZPos);
        scene.add(branchedVectorVis.group);
        branchedVectorVisualizations.push(branchedVectorVis);
        additionPlayedFlags.push(false);

        // Trail for duplicate
        branchedTrailPoints.push([]);
        const branchedTrailGeometry = new THREE.BufferGeometry();
        const branchedPositions = new Float32Array(MAX_TRAIL_POINTS * 3);
        branchedTrailGeometry.setAttribute('position', new THREE.BufferAttribute(branchedPositions, 3));
        const branchedTrailMaterial = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.05 }); // Softer white
        const branchedTrailLine = new THREE.Line(branchedTrailGeometry, branchedTrailMaterial);
        scene.add(branchedTrailLine);
        // Seed branched trail with first point so it is always visible
        const initialBPos = [0, startY, vectorZPos];
        branchedTrailPoints[i].push(initialBPos);
        branchedTrailGeometry.getAttribute('position').setXYZ(0, ...initialBPos);
        branchedTrailGeometry.setDrawRange(0, 1);
        branchedTrailGeometry.computeBoundingSphere();
        branchedTrailLines.push({ line: branchedTrailLine, geometry: branchedTrailGeometry, material: branchedTrailMaterial });

        // --- Create Trail Line for this Vector ---
        allTrailPoints.push([]); // Initialize points array for this trail
        const trailGeometry = new THREE.BufferGeometry();
        // Pre-allocate buffer large enough for max points
        const positions = new Float32Array(MAX_TRAIL_POINTS * 3);
        trailGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        const trailMaterial = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.05 }); // Softer white
        const trailLine = new THREE.Line(trailGeometry, trailMaterial);
        scene.add(trailLine);
        // Seed original trail with first point as well
        const initialPos = [0, startY, vectorZPos];
        allTrailPoints[i].push(initialPos);
        trailGeometry.getAttribute('position').setXYZ(0, ...initialPos);
        trailGeometry.setDrawRange(0, 1);
        trailGeometry.computeBoundingSphere();
        allTrailLines.push({ line: trailLine, geometry: trailGeometry, material: trailMaterial });
    }

    // --- GUI ---
    const gui = new GUI({ closeFolders: true });
    gui.close();

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
    // matrixFolder.open(); // Keep closed by default

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

    const additionPlaying = { active: false, endTime: 0, logged: false, tweenLogged: false }; // Global flag to pause reset while addition runs

    // --- Post-Addition Rise Parameters ---
    const extraRiseDistance = 15; // How far to continue rising after combination
    const extraRiseDuration = 3;  // Seconds it takes to reach the extra height
    let extraRiseStartTime = null;

    // --- Animation Loop ---
    function animate() {
        requestAnimationFrame(animate);

        // --- Handle Keyboard Movement ---
        const cameraRight = new THREE.Vector3();
        camera.getWorldDirection(cameraRight); // Get forward direction first
        cameraRight.cross(camera.up).normalize(); // Calculate right vector
        const cameraUp = new THREE.Vector3().copy(camera.up); // Use world up for panning usually
        const cameraForward = new THREE.Vector3(); // Vector to store forward direction
        camera.getWorldDirection(cameraForward); // Get the camera's forward direction

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

        // Arrow Up/Down for Rotating View Angle (Adjusting OrbitControls target vertically)
        if (keysPressed['ArrowUp']) {
            const targetOffset = cameraUp.clone().multiplyScalar(rotateSpeed * 10); // Tilt view upward
            controls.target.add(targetOffset);
        }
        if (keysPressed['ArrowDown']) {
            const targetOffset = cameraUp.clone().multiplyScalar(-rotateSpeed * 10); // Tilt view downward
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

        // Update tweens (addition animations) with current time
        const updated = TWEEN.update();
        if (additionPlaying.active) {
            console.log('[animate] TWEEN.update returned', updated, 'active tweens:', TWEEN.getAll().length);
        }

        // Log position *after* tween update
        if (additionPlaying.active) {
            // Log position of the first ellipse of the first vector during the animation
            if (branchedVectorVisualizations[0] && branchedVectorVisualizations[0].ellipses[0]) {
                console.log("Animate loop: ellipse[0].position.y = ", branchedVectorVisualizations[0].ellipses[0].position.y);
            }
        }

        const elapsedTime = clock.getElapsedTime();
        const animationDuration = 5; // seconds to reach the meeting point once
        const loopTime = Math.min(elapsedTime / animationDuration, 1); // clamp 0..1, no cycling

        // Calculate current Y position using linear interpolation within the loop
        const currentY = startY + loopTime * animationDistance;

        // Mark addition finished when its expected time passes
        if (additionPlaying.active && performance.now() > additionPlaying.endTime) {
            additionPlaying.active = false;
        }

        // If all addition animations have played and finished, start the extra rise (once)
        if (!additionPlaying.active && extraRiseStartTime === null && additionPlayedFlags.every(f => f)) {
            extraRiseStartTime = clock.getElapsedTime();
        }

        // Calculate any extra rise offset based on time since extraRiseStartTime
        let extraRiseOffset = 0;
        if (extraRiseStartTime !== null) {
            const riseElapsed = clock.getElapsedTime() - extraRiseStartTime;
            const tRise = Math.min(riseElapsed / extraRiseDuration, 1); // 0..1
            extraRiseOffset = THREE.MathUtils.lerp(0, extraRiseDistance, tRise);
        }

        // --- Update Trail Lines for Original Vectors ---
        allTrailLines.forEach((trail, index) => {
            const currentPoints = allTrailPoints[index];
            const vectorZPos = allVectorVisualizations[index].group.position.z; // Z is constant per vector
            let trailY = allVectorVisualizations[index].group.position.y;

            // After the addition has played for this vector, extend its trail so it meets
            // the branched vector's trail at the rendez‑vous height (computed locally)
            if (additionPlayedFlags[index]) {
                const yMeet = startY + (branchConfig.branchStartT + branchConfig.horzDurationT + branchConfig.vertDurationT + branchConfig.leftDurationT) * animationDistance + branchConfig.meetYOffset;
                trailY = yMeet + extraRiseOffset;
            }

            const newPoint = [0, trailY, vectorZPos];

            if (currentPoints.length < MAX_TRAIL_POINTS) currentPoints.push(newPoint);

            // Update geometry attribute
            const positionAttribute = trail.geometry.getAttribute('position');
            for (let j = 0; j < currentPoints.length; j++) {
                positionAttribute.setXYZ(j, currentPoints[j][0], currentPoints[j][1], currentPoints[j][2]);
            }

            trail.geometry.setDrawRange(0, currentPoints.length);
            positionAttribute.needsUpdate = true;
            // Recompute bounding sphere so correct frustum‑culling as the trail grows
            trail.geometry.computeBoundingSphere();
        });

        // --- Update Branched Vector Positions & Trails ---
        const t1 = branchConfig.branchStartT;
        const t2 = t1 + branchConfig.horzDurationT;
        const t3 = t2 + branchConfig.vertDurationT;
        const t4 = t3 + branchConfig.leftDurationT;

        // Pre-compute key Y values along original path
        const yBranch = startY + t1 * animationDistance;
        const yAboveOriginal = startY + t4 * animationDistance + branchConfig.meetYOffset;

        branchedVectorVisualizations.forEach((bVecVis, index) => {
            const zPos = bVecVis.group.position.z; // constant Z same as original
            let xPos = 0;
            let yPos = currentY; // default follow original

            // First, check if this vector should trigger its addition animation
            if (!additionPlayedFlags[index] && loopTime >= t4) {
                // Reverse the order so units move **from** the bottom/original vectors **to** the branched/top vectors
                startAdditionAnimation(allVectorVisualizations[index], bVecVis);
                additionPlayedFlags[index] = true;
            }

            if (additionPlaying.active) {
                // Keep the branched vector locked in place while addition wave runs
                xPos = 0;
                yPos = yAboveOriginal;
            } else if (loopTime < t1) {
                // Before branching, follow original
                xPos = 0;
                yPos = startY + loopTime * animationDistance;
            } else if (loopTime >= t1 && loopTime < t2) {
                // Horizontal move right
                const localT = (loopTime - t1) / branchConfig.horzDurationT;
                xPos = THREE.MathUtils.lerp(0, branchConfig.branchX, localT);
                yPos = yBranch;
            } else if (loopTime >= t2 && loopTime < t3) {
                // Vertical move up
                const localT = (loopTime - t2) / branchConfig.vertDurationT;
                xPos = branchConfig.branchX;
                yPos = THREE.MathUtils.lerp(yBranch, yAboveOriginal, localT);
            } else if (loopTime >= t3 && loopTime < t4) {
                // Horizontal move left at constant Y
                const localT = (loopTime - t3) / branchConfig.leftDurationT;
                xPos = THREE.MathUtils.lerp(branchConfig.branchX, 0, localT);
                yPos = yAboveOriginal; // Keep Y constant during horizontal return for axis-aligned motion
            } else {
                // After meeting originals (x==0), stay put at the join position
                xPos = 0;
                yPos = yAboveOriginal; // Remain stationary until loop reset
            }

            // Update branched vector position
            bVecVis.group.position.set(xPos, yPos + extraRiseOffset, zPos);

            // --- Trail for branched ---
            const bTrail = branchedTrailLines[index];
            const bPoints = branchedTrailPoints[index];
            if (bPoints.length < MAX_TRAIL_POINTS) bPoints.push([xPos, yPos, zPos]);

            const bPosAttr = bTrail.geometry.getAttribute('position');
            for (let j = 0; j < bPoints.length; j++) {
                bPosAttr.setXYZ(j, bPoints[j][0], bPoints[j][1], bPoints[j][2]);
            }
            bTrail.geometry.setDrawRange(0, bPoints.length);
            bPosAttr.needsUpdate = true;
            // Update bounding sphere for branched trail as well
            bTrail.geometry.computeBoundingSphere();
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

        // Update original vector positions – stop once they reach below branched height
        const yOriginalStop = yAboveOriginal - branchConfig.meetYOffset;
        allVectorVisualizations.forEach(vectorVis => {
            // Allow vertical movement until reaching stop height
            let baseY;
            if (currentY < yOriginalStop) {
                baseY = currentY;
            } else {
                baseY = yOriginalStop;
            }

            vectorVis.group.position.y = baseY + extraRiseOffset;
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
        allTrailLines.forEach(trail => { // Dispose original trails
            trail.geometry.dispose();
            trail.material.dispose();
        });

        branchedVectorVisualizations.forEach(vec => vec.dispose());
        branchedTrailLines.forEach(trail => {
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