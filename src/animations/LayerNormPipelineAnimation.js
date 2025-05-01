import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { LayerNormalizationVisualization } from '../components/LayerNormalizationVisualization.js';
import { VectorNormalizationVisualization } from '../components/VectorNormalizationVisualization.js';
import { VectorVisualization } from '../components/VectorVisualization.js';
import { VECTOR_LENGTH } from '../utils/constants.js';
import { mapValueToColor } from '../utils/colors.js';

// NOTE: Requires global TWEEN.js (as the individual animation helpers rely on it)

export function initLayerNormPipelineAnimation(container) {
    // --- Scene basics --------------------------------------------------------------
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 15, 60);

    let renderer;
    if (container instanceof HTMLCanvasElement) {
        renderer = new THREE.WebGLRenderer({ canvas: container, antialias: true });
    } else {
        renderer = new THREE.WebGLRenderer({ antialias: true });
        container.appendChild(renderer.domElement);
    }
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);

    // --- Post Processing for bloom effect ----------------------------------------------
    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    const bloomPass = new UnrealBloomPass(
        new THREE.Vector2(window.innerWidth, window.innerHeight),
        1.2, 0.5, 0.8 // strength, radius, threshold - EXACTLY like in VectorAdditionAnimation.js
    );
    composer.addPass(bloomPass);

    // --- Controls ------------------------------------------------------------------
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    // --- Lights --------------------------------------------------------------------
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dir = new THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(10, 20, 10);
    scene.add(dir);

    // --- LayerNorm solid -----------------------------------------------------------
    const lnParams = {
        width: 60,
        height: 22,
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

    // Dark → blue opacity handling values
    const darkGray  = new THREE.Color(0x222222);
    const lightBlue = new THREE.Color(0x66ccff);
    const currentColor = new THREE.Color();
    let solidOpacity = 0.25;
    layerNormVis.setMaterialProperties({ color: darkGray, transparent: true, opacity: solidOpacity });

    // --- Vector sets ---------------------------------------------------------------
    const multiplyData  = Array.from({ length: VECTOR_LENGTH }, () => Math.random() * 2 - 1);
    const additionData  = Array.from({ length: VECTOR_LENGTH }, () => Math.random() * 2 - 1);

    const offsetY = 10; // vertical padding for moving vector outside solid
    const startY  = -lnParams.height / 2 - offsetY;
    const endY    =  lnParams.height / 2 + offsetY; // used for opacity ramp only
    // Start position for the addition vector – spawn lower inside the solid (quarter height above center)
    const addStartInsideY = lnParams.height / 4; // 1/4 of full height above center

    // Z-positions per slit
    const slitSpacing = lnParams.depth / (lnParams.numberOfHoles + 1);

    const lanes = []; // state objects for each slit

    for (let i = 0; i < lnParams.numberOfHoles; i++) {
        const zPos = -lnParams.depth / 2 + slitSpacing * (i + 1);

        // Moving vector – will be normalized then multiplied
        const movingVec = new VectorNormalizationVisualization(new THREE.Vector3(0, startY, zPos));
        movingVec.data = movingVec.normalizedData; // for multiplication calculation later
        scene.add(movingVec.group);

        // Static multiplication target in centre of solid
        const multTarget = new VectorVisualization(multiplyData, new THREE.Vector3(0, 0, zPos));
        multTarget.data = [...multiplyData];
        scene.add(multTarget.group);

        // Addition vector starting just inside the solid (will move downward)
        const addVec = new VectorVisualization(additionData, new THREE.Vector3(0, addStartInsideY, zPos));
        addVec.data = [...additionData];
        scene.add(addVec.group);

        lanes.push({
            zPos,
            movingVec,
            multTarget,
            addVec,
            // State flags
            normStarted: false,
            multStarted: false,
            multDone: false,
            addStarted: false,
            addDone: false
        });
    }

    // Helper: start multiplication, with callback when finished
    function startMultiplicationAnimation(vec1, vec2, onComplete) {
        const duration = 750;
        const vectorLength = vec1.ellipses.length;
        let moveTweensCompleted = 0;

        for (let i = 0; i < vectorLength; i++) {
            const e1 = vec1.ellipses[i];
            const e2 = vec2.ellipses[i];
            if (!e1 || !e2) continue;
            const targetPosWorld = new THREE.Vector3();
            e2.getWorldPosition(targetPosWorld);
            const localTarget = e1.parent.worldToLocal(targetPosWorld.clone());

            new TWEEN.Tween(e1.position)
                .to({ y: localTarget.y }, duration)
                .easing(TWEEN.Easing.Quadratic.InOut)
                .onComplete(() => {
                    moveTweensCompleted++;
                    if (moveTweensCompleted === vectorLength) {
                        triggerFlash();
                    }
                })
                .start();
        }

        function triggerFlash() {
            const flashDuration = 150;
            const original = [];
            for (let i = 0; i < vectorLength; i++) {
                const e2 = vec2.ellipses[i];
                original[i] = {
                    color: e2.material.color.clone(),
                    emissive: e2.material.emissive.clone(),
                    emissiveIntensity: e2.material.emissiveIntensity
                };
                e2.material.color.set(0xffffff);
                e2.material.emissive.set(0xffffff);
                e2.material.emissiveIntensity = 1.0;
            }
            new TWEEN.Tween({})
                .to({}, flashDuration)
                .onComplete(() => {
                    for (let i = 0; i < vectorLength; i++) {
                        const e1 = vec1.ellipses[i];
                        const e2 = vec2.ellipses[i];
                        const product = vec1.data[i] * vec2.data[i];
                        vec2.data[i] = product;
                        const col = mapValueToColor(product);
                        e2.material.color.copy(col);
                        e2.material.emissive.copy(col);
                        e2.material.emissiveIntensity = original[i].emissiveIntensity;
                        e1.visible = false;
                    }
                    if (onComplete) onComplete();
                })
                .start();
        }
    }

    // Helper: addition animation with callback - EXACT match to VectorAdditionAnimation.js
    function startAdditionAnimation(vec1, vec2, onComplete) {
        const duration = 750; 
        const flashDuration = 150;
        const delayBetweenCubes = 75; 
        const vectorLength = vec1.ellipses.length;
        let completed = 0;

        console.log("Starting vector addition animation sequence...");

        for (let i = 0; i < vectorLength; i++) {
            const ellipse1 = vec1.ellipses[i];
            const ellipse2 = vec2.ellipses[i];

            if (!ellipse1 || !ellipse2) continue;

            const targetPosition = new THREE.Vector3();
            ellipse2.getWorldPosition(targetPosition);
            const localTargetPosition = ellipse1.parent.worldToLocal(targetPosition.clone());

            // First movement phase
            const moveTween = new TWEEN.Tween(ellipse1.position)
                .to({ y: localTargetPosition.y }, duration)
                .easing(TWEEN.Easing.Quadratic.InOut)
                .delay(i * delayBetweenCubes)
                .onComplete(() => {
                    // Store original properties for restoration later
                    const originalColor = ellipse2.material.color.clone();
                    const originalEmissive = ellipse2.material.emissive.clone();
                    const originalIntensity = ellipse2.material.emissiveIntensity;

                    // Set to white for flash effect
                    ellipse2.material.color.set(0xffffff);
                    ellipse2.material.emissive.set(0xffffff);
                    ellipse2.material.emissiveIntensity = 1.0; // EXACT value from VectorAdditionAnimation

                    // Flash tween with dummy target - uses onComplete only
                    const flashTween = new TWEEN.Tween(ellipse2.material)
                        .to({}, flashDuration)
                        .onComplete(() => {
                            // Compute addition and update target
                            const sum = vec1.data[i] + vec2.data[i];
                            vec2.data[i] = sum;
                            
                            // Apply new color based on sum
                            const newColor = mapValueToColor(sum);
                            ellipse2.material.color.copy(newColor);
                            ellipse2.material.emissive.copy(newColor);
                            ellipse2.material.emissiveIntensity = originalIntensity;
                            
                            // Hide source ellipse
                            ellipse1.visible = false;
                            
                            completed++;
                            if (completed === vectorLength) {
                                startRisePhase();
                            }
                        });
                    flashTween.start();
                });
            moveTween.start();
        }

        // After addition complete, create separate rise animation
        function startRisePhase() {
            console.log("Addition complete - rising phase starting");
            
            // Store data and appearance for rising vector
            const resultData = [...vec2.data];
            const resultMaterials = vec2.ellipses.map(e => ({
                color: e.material.color.clone(),
                emissive: e.material.emissive.clone(),
                intensity: e.material.emissiveIntensity
            }));
            
            // Create rising vector copy
            const resultVec = new VectorVisualization(resultData, vec2.group.position.clone());
            resultVec.data = resultData;
            scene.add(resultVec.group);
            
            // Apply stored appearance
            for (let i = 0; i < vectorLength; i++) {
                if (resultMaterials[i] && resultVec.ellipses[i]) {
                    resultVec.ellipses[i].material.color.copy(resultMaterials[i].color);
                    resultVec.ellipses[i].material.emissive.copy(resultMaterials[i].emissive);
                    resultVec.ellipses[i].material.emissiveIntensity = resultMaterials[i].intensity;
                }
            }
            
            // Hide original vector
            vec2.group.visible = false;
            
            // Calculate rise with constant velocity
            const finalY = lnParams.height / 2 + offsetY / 2;
            const distance = finalY - resultVec.group.position.y;
            const riseSpeed = 5; // units per second
            const riseDuration = (distance / riseSpeed) * 1000;
            
            // Rise animation
            new TWEEN.Tween(resultVec.group.position)
                .to({ y: finalY }, riseDuration)
                .easing(TWEEN.Easing.Linear.None) // Constant velocity
                .onComplete(() => {
                    // Fade out
                    new TWEEN.Tween({ opacity: 1 })
                        .to({ opacity: 0 }, 1000)
                        .onUpdate(obj => {
                            resultVec.ellipses.forEach(ellipse => {
                                if (ellipse && ellipse.material) {
                                    ellipse.material.opacity = obj.opacity;
                                    ellipse.material.transparent = true;
                                }
                            });
                        })
                        .onComplete(() => {
                            scene.remove(resultVec.group);
                            resultVec.dispose();
                            if (onComplete) onComplete();
                        })
                        .start();
                })
                .start();
        }
    }

    // --- Animation loop ------------------------------------------------------------
    function onWindowResize() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
        composer.setSize(window.innerWidth, window.innerHeight);
    }
    window.addEventListener('resize', onWindowResize);

    const clock = new THREE.Clock();
    function animate() {
        requestAnimationFrame(animate);

        const time = performance.now();
        const deltaTime = clock.getDelta();

        // Opacity / colour update based on first lane's moving vector Y
        const sampleY = lanes[0].movingVec.group.position.y;
        const bottomY = -lnParams.height / 2;
        const midY    = 0;
        const topY    = lnParams.height / 2;

        if (sampleY < bottomY) {
            currentColor.copy(darkGray);
            solidOpacity = 0.25;
        } else if (sampleY >= bottomY && sampleY < midY) {
            const t = (sampleY - bottomY) / (midY - bottomY);
            currentColor.lerpColors(darkGray, lightBlue, t);
            solidOpacity = 0.25 + t * 0.15;
        } else if (sampleY >= midY && sampleY < topY) {
            const t = (sampleY - midY) / (topY - midY);
            currentColor.copy(lightBlue);
            solidOpacity = 0.4 + t * 0.3;
        } else {
            const exitOpacity = 0.7;
            const t2 = THREE.MathUtils.smoothstep(sampleY, topY, endY);
            currentColor.copy(lightBlue);
            solidOpacity = exitOpacity + t2 * (1 - exitOpacity);
        }
        layerNormVis.setMaterialProperties({ color: currentColor, opacity: solidOpacity, transparent: true });

        // --- Per-lane logic
        const riseSpeed = 6; // units per second until multiplication starts
        lanes.forEach(l => {
            // Start normalization at higher position (1/4 of the way from bottom to center)
            const normStartY = bottomY + (midY - bottomY) * 0.25; // 25% of the way from bottom to center
            if (!l.normStarted && l.movingVec.group.position.y >= normStartY) {
                l.movingVec.startAnimation();
                l.normStarted = true;
            }
            // Update normalization animation each frame
            l.movingVec.update(time);

            // If multiplication hasn't started, move up *only* when not actively normalizing
            const normAnimating = l.normStarted && l.movingVec.animationState.isAnimating;
            if (!l.multStarted && !normAnimating) {
                l.movingVec.group.position.y += riseSpeed * deltaTime;
            }

            // Trigger multiplication at center
            if (!l.multStarted && l.movingVec.group.position.y >= midY) {
                l.multStarted = true;
                // Stop vertical movement
                startMultiplicationAnimation(l.movingVec, l.multTarget, () => {
                    l.multDone = true;
                    l.movingVec.group.visible = false;
                });
            }

            // After multiplication done, trigger addition
            if (l.multDone && !l.addStarted) {
                l.addStarted = true;
                startAdditionAnimation(l.addVec, l.multTarget, () => {
                    l.addDone = true;
                    l.addVec.group.visible = false;
                });
            }
        });

        // Update tweens
        if (typeof TWEEN !== 'undefined' && TWEEN.update) TWEEN.update();

        controls.update();
        // Use composer instead of renderer directly for bloom effect
        composer.render();
    }
    animate();

    // Cleanup ----------------------------------------------------------------------
    return () => {
        window.removeEventListener('resize', onWindowResize);
        controls.dispose();
        lanes.forEach(l => {
            l.movingVec.dispose();
            l.multTarget.dispose();
            l.addVec.dispose();
        });
        layerNormVis.dispose();
        scene.traverse(obj => {
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) {
                if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
                else obj.material.dispose();
            }
        });
        // Dispose composer passes
        composer.passes.forEach(pass => { if (pass.dispose) pass.dispose(); });
        renderer.dispose();
    };
} 