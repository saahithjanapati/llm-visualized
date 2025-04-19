import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GUI } from 'three/examples/jsm/libs/lil-gui.module.min.js';
import { VectorVisualization } from '../components/VectorVisualization';
import { WeightMatrixVisualization } from '../components/WeightMatrixVisualization';
import { VECTOR_LENGTH } from '../utils/constants.js'; // Import VECTOR_LENGTH

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
    renderer.setPixelRatio(window.devicePixelRatio);

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
    const moveSpeed = 0.5; // Adjust speed as needed
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

        const trailMaterial = new THREE.LineBasicMaterial({ color: 0x888888 }); // Gray color
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
        // WASD for Translation
        if (keysPressed['w']) {
            camera.translateZ(-moveSpeed);
        }
        if (keysPressed['s']) {
            camera.translateZ(moveSpeed);
        }
        if (keysPressed['a']) {
            camera.translateX(-moveSpeed);
        }
        if (keysPressed['d']) {
            camera.translateX(moveSpeed);
        }

        // Arrow Keys for Rotation (adjusting OrbitControls target)
        const lookDirection = new THREE.Vector3();
        camera.getWorldDirection(lookDirection);
        const rightDirection = new THREE.Vector3().crossVectors(lookDirection, camera.up).normalize();

        if (keysPressed['ArrowUp']) {
            // Adjust target upwards relative to camera view
            const upAdjustment = new THREE.Vector3().copy(camera.up).multiplyScalar(rotateSpeed * 10); // Scale factor for noticeable change
            controls.target.add(upAdjustment);
            // camera.position.add(upAdjustment); // Remove camera position change for rotation
        }
        if (keysPressed['ArrowDown']) {
            // Adjust target downwards relative to camera view
            const downAdjustment = new THREE.Vector3().copy(camera.up).multiplyScalar(-rotateSpeed * 10);
            controls.target.add(downAdjustment);
            // camera.position.add(downAdjustment); // Remove camera position change for rotation
        }
        if (keysPressed['ArrowLeft']) {
            // Adjust target leftwards relative to camera view
            const leftAdjustment = new THREE.Vector3().copy(rightDirection).multiplyScalar(-rotateSpeed * 10);
            controls.target.add(leftAdjustment);
             // camera.position.add(leftAdjustment); // Remove camera position change for rotation
        }
        if (keysPressed['ArrowRight']) {
            // Adjust target rightwards relative to camera view
            const rightAdjustment = new THREE.Vector3().copy(rightDirection).multiplyScalar(rotateSpeed * 10);
            controls.target.add(rightAdjustment);
            // camera.position.add(rightAdjustment); // Remove camera position change for rotation
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