import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GUI } from 'three/examples/jsm/libs/lil-gui.module.min.js';
import { VectorVisualization } from '../components/VectorVisualization';
import { WeightMatrixVisualization } from '../components/WeightMatrixVisualization';
import { VECTOR_LENGTH } from '../utils/constants.js'; // Import VECTOR_LENGTH
import { scaleOpacityForDisplay } from '../utils/trailConstants.js';

// Maximum points per trail line (adjust for performance/length)
const MAX_TRAIL_POINTS = 1000; // Further increased buffer size

export function initVectorMatrixScene(canvas) {
    // --- Basic Three.js setup ---
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111); // Dark background

    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 10, 25); // Position camera to view both objects

    const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

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
    };
    const matrixVis = new WeightMatrixVisualization(null, new THREE.Vector3(0, 0, 0), matrixParams.width, matrixParams.height, matrixParams.depth, matrixParams.topWidthFactor, matrixParams.cornerRadius, matrixParams.numberOfSlits, matrixParams.slitWidth, matrixParams.slitDepthFactor, matrixParams.slitWidthFactor);
    scene.add(matrixVis.group);

    // Define colors for animation
    const darkGray = new THREE.Color(0x333333);
    const brightYellow = new THREE.Color(0xFFFF00);
    const darkYellow = new THREE.Color(0xCCA000);
    const matrixCurrentColor = new THREE.Color(); // Color object to reuse for lerping

    // --- Create and Position Vectors over Slits ---
    const slitSpacing = matrixParams.depth / (matrixParams.numberOfSlits + 1);
    // Define animation bounds
    const startY = -matrixParams.height / 2 - vectorHeightOffset; // Start below the matrix
    const endY = matrixParams.height / 2 + vectorHeightOffset;   // End above the matrix
    const animationDistance = endY - startY;

    for (let i = 0; i < matrixParams.numberOfSlits; i++) {
        // Generate unique data for each vector, matching VECTOR_LENGTH
        const vectorData = Array.from({ length: VECTOR_LENGTH }, () => (Math.random() - 0.5) * 10);
        const vectorParams = {
            numberOfElements: VECTOR_LENGTH, // Use VECTOR_LENGTH
            elementWidth: 1, // Match vector element width if needed
            spacing: 1.5,    // Match vector spacing if needed
            // Add other params if needed, using defaults otherwise
        };
        const vectorVis = new VectorVisualization(vectorData, vectorParams);

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

        const trailMaterial = new THREE.LineBasicMaterial({ color: 0x888888, transparent: true, opacity: scaleOpacityForDisplay(0.08) });
        const trailLine = new THREE.Line(trailGeometry, trailMaterial);
        scene.add(trailLine);
        allTrailLines.push({ line: trailLine, geometry: trailGeometry, material: trailMaterial });
    }

    // --- GUI (Optional - for parameter tuning) ---
    const gui = new GUI();
    // const vectorFolder = gui.addFolder('Vector'); // Remove vector folder

    const matrixFolder = gui.addFolder('Matrix');
    matrixFolder.add(matrixParams, 'numberOfSlits', 0, 20, 1).name('Number of Slits').onChange(() => { /* Need complex update logic now */ }); // Note: Changing slits requires recreating vectors
    matrixFolder.add(matrixParams, 'slitWidth', 0.1, 5.0, 0.05).name('Slit Width').onChange(() => matrixVis.updateGeometry(matrixParams)); // Increased max slit width
    matrixFolder.add(matrixParams, 'slitDepthFactor', 0, 1, 0.01).name('Slit Depth Factor').onChange(() => matrixVis.updateGeometry(matrixParams));
    matrixFolder.add(matrixParams, 'slitWidthFactor', 0.1, 1, 0.01).name('Slit Width Factor').onChange(() => matrixVis.updateGeometry(matrixParams));
    matrixFolder.add(matrixParams, 'width', 1, 100, 0.5).name('Base Width').onChange(() => matrixVis.updateGeometry(matrixParams));
    matrixFolder.add(matrixParams, 'topWidthFactor', 0.1, 2, 0.01).name('Top Width Factor').onChange(() => matrixVis.updateGeometry(matrixParams));
    matrixFolder.add(matrixParams, 'cornerRadius', 0, 5, 0.05).name('Corner Radius').onChange(() => matrixVis.updateGeometry(matrixParams));
    matrixFolder.add(matrixParams, 'depth', 5, 100, 1).name('Depth').onChange(() => { /* Need complex update logic now */ }); // Note: Changing depth requires repositioning vectors
    matrixFolder.open(); // Open matrix folder by default

    // --- Resize Handling ---
    window.addEventListener('resize', onWindowResize, false);
    function onWindowResize() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
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
        const matrixBottomY = -matrixParams.height / 2;
        const matrixMidY = 0; // Center of the matrix
        const matrixTopY = matrixParams.height / 2;

        let t = 0;
        if (currentY < matrixBottomY) {
            matrixCurrentColor.copy(darkGray);
        } else if (currentY >= matrixBottomY && currentY < matrixMidY) {
            // Entering: Gray to Bright Yellow
            t = (currentY - matrixBottomY) / (matrixMidY - matrixBottomY);
            matrixCurrentColor.lerpColors(darkGray, brightYellow, t);
        } else if (currentY >= matrixMidY && currentY < matrixTopY) {
            // Leaving: Bright Yellow to Dark Yellow
            t = (currentY - matrixMidY) / (matrixTopY - matrixMidY);
            matrixCurrentColor.lerpColors(brightYellow, darkYellow, t);
        } else { // currentY >= matrixTopY
            matrixCurrentColor.copy(darkYellow);
        }
        matrixVis.setColor(matrixCurrentColor);

        // Update vector positions
        allVectorVisualizations.forEach(vectorVis => {
            vectorVis.group.position.y = currentY;
        });

        // Update previous Y for next frame
        previousY = currentY;

        renderer.render(scene, camera);
    }

    animate(); // Start the animation loop

    // Return cleanup function
    return () => {
        window.removeEventListener('resize', onWindowResize);
        window.removeEventListener('keydown', onKeyDown); // Remove key listeners
        window.removeEventListener('keyup', onKeyUp);     // Remove key listeners
        controls.dispose();
        gui.destroy();
        // Dispose geometries and materials
        allVectorVisualizations.forEach(vec => vec.dispose()); // Dispose all vectors
        allTrailLines.forEach(trail => { // Dispose trails
            trail.geometry.dispose();
            trail.material.dispose();
        });
        matrixVis._clearMesh(); // Use existing clear method
        // Add disposal for other objects if needed
        renderer.dispose();
    };
} 