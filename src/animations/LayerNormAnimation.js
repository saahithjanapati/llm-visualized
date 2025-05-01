import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { LayerNormalizationVisualization } from '../components/LayerNormalizationVisualization.js';
import { VectorVisualization } from '../components/VectorVisualization.js';
import { VECTOR_LENGTH } from '../utils/constants.js';

// A self-contained scene that demonstrates vectors rising through a
// Layer-Normalization solid.  The ring changes colour / transparency as
// vectors enter, pass through, and finally exit the layer.
export function initLayerNormAnimation(container) {
    // --- Scene / camera / renderer --------------------------------------------------
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 6, 25);

    let renderer;
    if (container instanceof HTMLCanvasElement) {
        renderer = new THREE.WebGLRenderer({ canvas: container, antialias: true });
    } else {
        renderer = new THREE.WebGLRenderer({ antialias: true });
        container.appendChild(renderer.domElement);
    }
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);

    // --- Controls -------------------------------------------------------------------
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // --- Lighting -------------------------------------------------------------------
    scene.add(new THREE.AmbientLight(0xffffff, 0.5));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(5, 10, 7);
    scene.add(dirLight);

    // --- LayerNorm solid -------------------------------------------------------------
    const lnParams = {
        width: 70,
        height: 27,
        depth: 72,
        wallThickness: 1.0,
        numberOfHoles: 5,
        holeWidth: 2.5,
        holeWidthFactor: 10
    };

    const layerNormVis = new LayerNormalizationVisualization(
        new THREE.Vector3(0, 0, 0),
        lnParams.width,
        lnParams.height,
        lnParams.depth,
        lnParams.wallThickness,
        lnParams.numberOfHoles,
        lnParams.holeWidth,
        lnParams.holeWidthFactor
    );
    scene.add(layerNormVis.group);

    // Initial material state – very transparent dark gray
    const darkGray = new THREE.Color(0x222222);
    layerNormVis.setMaterialProperties({ color: darkGray, transparent: true, opacity: 0.25 });

    // --- Vectors --------------------------------------------------------------------
    const vectors = [];
    const slitSpacing = lnParams.depth / (lnParams.numberOfHoles + 1);

    // Vertical travel bounds (some margin above / below the solid)
    const offset = 8; // distance outside the solid before/after
    const startY = -lnParams.height / 2 - offset;
    const endY   =  lnParams.height / 2 + offset;
    const travel = endY - startY;

    for (let i = 0; i < lnParams.numberOfHoles; i++) {
        const vectorData = Array.from({ length: VECTOR_LENGTH }, () => (Math.random() - 0.5) * 2);
        const vectorVis  = new VectorVisualization(vectorData);

        const zPos = -lnParams.depth / 2 + slitSpacing * (i + 1);
        vectorVis.group.position.set(0, startY, zPos);

        scene.add(vectorVis.group);
        vectors.push(vectorVis);
    }

    // --- Resize handling ------------------------------------------------------------
    window.addEventListener('resize', onWindowResize, false);
    function onWindowResize() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }

    // --- Animation loop -------------------------------------------------------------
    const clock = new THREE.Clock();
    const cycleSeconds = 5; // duration of one full rise (restart afterwards)
    const lightBlue = new THREE.Color(0x66ccff);
    const currentColor = new THREE.Color();

    function animate() {
        requestAnimationFrame(animate);
        controls.update();

        // --- Vector movement (simple looping) ---
        const elapsed = clock.getElapsedTime();
        const phase   = (elapsed % cycleSeconds) / cycleSeconds; // 0 → 1
        const currentY = startY + phase * travel;

        vectors.forEach(v => {
            v.group.position.y = currentY;
        });

        // --- Solid colour / opacity state machine ---
        const bottomY = -lnParams.height / 2;
        const midY    = 0;
        const topY    =  lnParams.height / 2;
        let opacity   = 0.25;

        if (currentY < bottomY) {
            currentColor.copy(darkGray);
            opacity = 0.25;
        } else if (currentY >= bottomY && currentY < midY) {
            const t = (currentY - bottomY) / (midY - bottomY);
            currentColor.lerpColors(darkGray, lightBlue, t);
            opacity = 0.25 + t * 0.15; // small fade-in
        } else if (currentY >= midY && currentY < topY) {
            const t = (currentY - midY) / (topY - midY);
            currentColor.copy(lightBlue);
            opacity = 0.4 + t * 0.3; // increase opacity while exiting
        } else { // above the solid
            currentColor.copy(lightBlue);
            // smoothly interpolate from exit opacity (0.4+0.3 = 0.7) to full opacity as vector rises from topY to endY using smoothstep
            const exitOpacity = 0.4 + 0.3; // 0.7
            // use Three.js smoothstep for easing
            const t2 = THREE.MathUtils.smoothstep(currentY, topY, endY);
            opacity = exitOpacity + t2 * (1 - exitOpacity);
        }

        layerNormVis.setMaterialProperties({ color: currentColor, opacity, transparent: true });

        renderer.render(scene, camera);
    }
    animate();

    // --- Cleanup --------------------------------------------------------------------
    return () => {
        window.removeEventListener('resize', onWindowResize);
        controls.dispose();
        vectors.forEach(v => v.dispose());
        layerNormVis.dispose();
        scene.traverse(obj => {
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) {
                if (Array.isArray(obj.material)) obj.material.forEach(mat => mat.dispose());
                else obj.material.dispose();
            }
        });
        renderer.dispose();
    };
} 