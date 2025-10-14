import * as THREE from 'three';
import { CoreEngine } from './CoreEngine.js';
import Gpt2Layer from './layers/Gpt2Layer.js';
import { createRandomSource } from '../data/RandomActivationSource.js';
import {
    MLP_MATRIX_PARAMS_DOWN,
    EMBEDDING_MATRIX_PARAMS_VOCAB,
    TOP_EMBED_Y_GAP_ABOVE_TOWER,
    TOP_EMBED_Y_ADJUST,
    GLOBAL_ANIM_SPEED_MULT,
    ANIM_RISE_SPEED_ORIGINAL,
    LN_PARAMS,
    LN_NORM_START_FRACTION_FROM_BOTTOM,
    PRISM_ADD_ANIM_BASE_DURATION,
    PRISM_ADD_ANIM_BASE_FLASH_DURATION,
    PRISM_ADD_ANIM_BASE_DELAY_BETWEEN_PRISMS,
    PRISM_ADD_ANIM_SPEED_MULT,
    VECTOR_LENGTH_PRISM
} from '../utils/constants.js';
import { VectorVisualizationInstancedPrism } from '../components/VectorVisualizationInstancedPrism.js';
import { startPrismAdditionAnimation } from '../utils/additionUtils.js';
import { PrismLayerNormAnimation } from '../animations/PrismLayerNormAnimation.js';

function simplePrismMultiply(srcVec, tgtVec, onComplete) {
    for (let i = 0; i < VECTOR_LENGTH_PRISM; i++) {
        tgtVec.rawData[i] = (srcVec.rawData[i] || 0) * (tgtVec.rawData[i] || 0);
    }
    tgtVec.updateKeyColorsFromData(tgtVec.rawData, 30);
    if (onComplete) onComplete();
}

const COLOR_DARK_GRAY = new THREE.Color(0x333333);
const COLOR_LIGHT_YELLOW = new THREE.Color(0xffffff);
const COLOR_BRIGHT_YELLOW = new THREE.Color(0xffffff);

/**
 * LayerPipeline orchestrates a single bundle of vectors ("lanes") through an
 * arbitrary stack of GPT-2 transformer layers.  Unlike the old approach that
 * created a fresh bundle for every layer at T=0, this controller initialises
 * ONE set of vectors, then – once a layer finishes – hands those same Three
 * objects off to the next Gpt2Layer positioned above the previous.
 */
export class LayerPipeline extends EventTarget {
    /**
     * @param {HTMLCanvasElement} canvas – Render target for the CoreEngine.
     * @param {number}           numLayers – Total layers in the stack.
     * @param {Object}           [opts] – Additional CoreEngine options (camera, speed, etc.).
     * @param {() => any}        [opts.randomFactory] – Factory that produces a fresh random-source for each layer.
     */
    constructor(canvas, numLayers = 12, opts = {}) {
        super();
        if (!canvas) throw new Error('LayerPipeline requires a renderCanvas element');
        this._numLayers = Math.max(1, numLayers);
        this._canvas    = canvas;
        this._opts      = opts;
        this._randFactory = typeof opts.randomFactory === 'function' ? opts.randomFactory : createRandomSource;

        this._layers = [];
        this._currentLayerIdx = 0;

        this._autoCameraFollow = opts.autoCameraFollow !== false;
        this._autoCameraCenter = new THREE.Vector3();
        this._autoCameraOffsetScratch = new THREE.Vector3();
        this._autoCameraDesiredCameraOffset = new THREE.Vector3();
        this._autoCameraDesiredTargetOffset = new THREE.Vector3();
        this._hasAutoCameraOffsets = false;
        this._autoCameraManualDelayMs = 4000;
        this._autoCameraManualResumeAt = 0;
        this._autoCameraReturnActive = false;
        this._autoCameraReturnLerpAlpha = 0.12;
        this._autoCameraReturnSnapThresholdSq = 0.5 * 0.5;
        this._autoCameraReturnCameraScratch = new THREE.Vector3();
        this._autoCameraReturnTargetScratch = new THREE.Vector3();
        this._suppressControlsChange = false;
        this._cameraOffsetDiv = (typeof document !== 'undefined')
            ? document.getElementById('cameraOffsetOverlay')
            : null;
        this._controlsChangeHandler = null;
        this._onAutoCameraProgress = () => { this._maybeAutoCameraFocus(); };
        this._cameraOverlayRaf = null;
        this.addEventListener('progress', this._onAutoCameraProgress);

        // ------------------------------------------------------------------
        // Pre-create *all* layers so their static visuals are visible upfront.
        // Only the first layer is active immediately; higher layers remain
        // dormant until their turn, at which point we inject residual lanes.
        // ------------------------------------------------------------------

        const engineOpts = { ...opts };
        if (typeof engineOpts.cameraFarMargin !== 'number') {
            const DEFAULT_CAMERA_FAR_MARGIN = 40000;
            // Provide additional depth so tall transformer stacks remain visible
            // when the user zooms far away from the tower. The allowance scales
            // with layer count but never drops below the default margin.
            const approxTowerAllowance = Math.max(DEFAULT_CAMERA_FAR_MARGIN, this._numLayers * 1800);
            engineOpts.cameraFarMargin = approxTowerAllowance;
        }
        this._engine = new CoreEngine(canvas, [], engineOpts);

        if (this._engine?.controls) {
            this._controlsChangeHandler = () => {
                if (!this._autoCameraFollow || this._suppressControlsChange) {
                    return;
                }
                this._onUserCameraInteraction();
                this._updateCameraOffsetOverlay();
            };
            this._engine.controls.addEventListener('change', this._controlsChangeHandler);
        }

        for (let i = 0; i < this._numLayers; i++) {
            const rand = this._randFactory();
            const isActive = i === 0; // only first layer active initially
            const layer = new Gpt2Layer(i, rand, 0, /*externalLanes*/ null, /*onFinished*/ null, isActive);

            // Assign onFinished callback for chaining once layer becomes active
            layer.setOnFinished(() => this._advanceToNextLayer());
            layer.setProgressEmitter(this);

            layer.init(this._engine.scene);
            if (typeof this._engine.registerRaycastRoot === 'function') {
                this._engine.registerRaycastRoot(layer.root);
            }
            this._layers.push(layer);
            this._engine._layers.push(layer); // add to engine update list
        }

        // Ensure first layer has active callback wired before start
        this._layers[0].setOnFinished(() => this._advanceToNextLayer());
        this._layers[0].setProgressEmitter(this);

        this._maybeAutoCameraFocus({ immediate: true });
        if (this._autoCameraFollow) {
            this._startCameraOverlayLoop();
        }
    }

    /** Dispose and tear down Three resources */
    dispose() {
        if (this._onAutoCameraProgress) {
            this.removeEventListener('progress', this._onAutoCameraProgress);
            this._onAutoCameraProgress = null;
        }
        if (this._engine?.controls && this._controlsChangeHandler) {
            this._engine.controls.removeEventListener('change', this._controlsChangeHandler);
        }
        this._controlsChangeHandler = null;
        if (this._cameraOffsetDiv) {
            this._cameraOffsetDiv.style.display = 'none';
        }
        this._stopCameraOverlayLoop();
        if (this._engine) {
            this._engine.dispose();
        }
    }

    /** Return reference to internal CoreEngine (for advanced use-cases). */
    get engine() { return this._engine; }

    /** Enable or disable automatic camera tracking of the active layer. */
    setAutoCameraFollow(enabled, { immediate = false } = {}) {
        const nextValue = !!enabled;
        if (nextValue === this._autoCameraFollow) {
            if (nextValue) {
                if (immediate) {
                    this._updateCameraOffsetOverlay();
                }
            } else {
                this._updateCameraOffsetOverlay();
            }
            return;
        }
        this._autoCameraFollow = nextValue;
        this._cancelAutoCameraReturn();
        this._updateCameraOffsetOverlay();
        if (this._autoCameraFollow) {
            this._startCameraOverlayLoop();
        } else {
            this._clearAutoCameraOffsets();
            this._stopCameraOverlayLoop();
        }
    }

    /** Check whether automatic camera tracking is enabled. */
    isAutoCameraFollowEnabled() {
        return !!this._autoCameraFollow;
    }

    // ----------------------------------------------------------------------
    // Private helpers
    // ----------------------------------------------------------------------

    /**
     * Called when the currently active layer reports completion via its
     * `onFinished` callback.  This creates the next Gpt2Layer, injects the
     * existing lane bundle and registers its own completion hook.
     */
    _advanceToNextLayer() {
        this._currentLayerIdx += 1;
        this.dispatchEvent(new Event('progress'));
        if (this._currentLayerIdx >= this._numLayers) {
            // All layers processed – trigger final rise into top embedding
            try { this._animateRiseIntoTopEmbedding(); } catch (_) { /* optional */ }
            return;
        }

        // Grab lanes from the previous (just-completed) layer
        const prevLayer = this._layers[this._currentLayerIdx - 1];
        // Deactivate the previous layer to stop it updating the shared vectors
        if (prevLayer) {
            prevLayer.isActive = false;
        }
        const externalLanes = prevLayer.lanes;
        // Ensure residual trails remain continuous by reparenting any
        // world-space trails to the new engine scene (safety no-op if same).
        if (externalLanes && externalLanes.length) {
            externalLanes.forEach(lane => {
                // Prefer the dedicated world-space residual trail carried across lanes
                const trailRef = (lane && lane.originalTrail)
                    || (lane && lane.originalVec && lane.originalVec.userData && lane.originalVec.userData.trail);
                if (trailRef && typeof trailRef.reparent === 'function') {
                    trailRef.reparent(this._engine.scene);
                }
            });
        }

        const nextLayer = this._layers[this._currentLayerIdx];
        if (!nextLayer) return;

        nextLayer.activateWithLanes(externalLanes);

        this._maybeAutoCameraFocus({ immediate: true });

        // Now that the original residual vectors have been transferred, we can safely
        // hide the remaining heavy geometry in the previous layer to save GPU work.
        if (prevLayer && typeof prevLayer.hideDynamicGeometry === 'function') {
            prevLayer.hideDynamicGeometry();
        }
    }

    /**
     * After the last layer completes, raise residual vectors up into the top
     * vocabulary embedding position using the same placement logic as the test page.
     */
    _animateRiseIntoTopEmbedding() {
        const lastLayer = this._layers[this._numLayers - 1];
        if (!lastLayer || !Array.isArray(lastLayer.lanes) || !lastLayer.lanes.length) return;

        if (!lastLayer.mlpDown || !lastLayer.mlpDown.group) return;

        const targetYLocal = this._calculateTopEmbeddingTargetY(lastLayer);

        const lnInfo = this._findTopLayerNorm(lastLayer);

        this._animateResidualVectors(lastLayer, targetYLocal, lnInfo);
    }

    /**
     * Determine the target local Y position for residual vectors entering the
     * top vocabulary embedding and update MHSA animation boundaries.
     * @param {Gpt2Layer} lastLayer
     * @returns {number} Local-space Y coordinate where vectors should stop.
     */
    _calculateTopEmbeddingTargetY(lastLayer) {
        let targetYLocal = null;
        try {
            const scene = this._engine && this._engine.scene;
            let topEmbedObj = null;
            if (scene && typeof scene.traverse === 'function') {
                scene.traverse((obj) => {
                    if (topEmbedObj) return;
                    if (obj && obj.userData && obj.userData.label === 'Vocab Embedding (Top)') {
                        topEmbedObj = obj;
                    }
                });
            }
            if (topEmbedObj) {
                const centerWorld = new THREE.Vector3();
                topEmbedObj.getWorldPosition(centerWorld);
                const stopWorldY = centerWorld.y - EMBEDDING_MATRIX_PARAMS_VOCAB.height / 2 + 5;
                const localVec = new THREE.Vector3(0, stopWorldY, 0);
                lastLayer.root.worldToLocal(localVec);
                targetYLocal = localVec.y;
            }
        } catch (_) { /* fallback to formula below */ }

        if (targetYLocal == null) {
            const towerTopYLocal = lastLayer.mlpDown.group.position.y + MLP_MATRIX_PARAMS_DOWN.height / 2;
            const topVocabCenterYLocal = towerTopYLocal + TOP_EMBED_Y_GAP_ABOVE_TOWER + EMBEDDING_MATRIX_PARAMS_VOCAB.height / 2 + TOP_EMBED_Y_ADJUST;
            targetYLocal = topVocabCenterYLocal - EMBEDDING_MATRIX_PARAMS_VOCAB.height / 2 + 5;
        }

        try {
            if (lastLayer.mhsaAnimation) {
                lastLayer.mhsaAnimation.finalOriginalY = targetYLocal;
                lastLayer.mhsaAnimation.topEmbeddingStopY = targetYLocal;
                lastLayer.mhsaAnimation.postSplitRiseSpeed = ANIM_RISE_SPEED_ORIGINAL;
            }
            lastLayer.__topEmbedStopYLocal = targetYLocal;
        } catch (_) { /* no-op */ }

        return targetYLocal;
    }

    /**
     * Locate the optional top LayerNorm in the scene and compute useful
     * positional data for animation.
     * @param {Gpt2Layer} lastLayer
     * @returns {{lnTopGroup: THREE.Object3D, lnCenterY: number, lnBottomY: number}|null}
     *          LayerNorm group and position info if found.
     */
    _findTopLayerNorm(lastLayer) {
        let lnTopGroup = null;
        try {
            const scene = this._engine && this._engine.scene;
            if (scene && typeof scene.traverse === 'function') {
                scene.traverse(obj => {
                    if (!lnTopGroup && obj && obj.userData && obj.userData.label === 'LayerNorm (Top)') {
                        lnTopGroup = obj;
                    }
                });
            }
        } catch (_) { /* optional */ }

        if (!lnTopGroup) return null;

        const lnCenterWorld = new THREE.Vector3();
        lnTopGroup.getWorldPosition(lnCenterWorld);
        const lnCenterLocal = lnCenterWorld.clone();
        lastLayer.root.worldToLocal(lnCenterLocal);
        const lnCenterY = lnCenterLocal.y;
        const lnBottomY = lnCenterY - LN_PARAMS.height / 2;

        return { lnTopGroup, lnCenterY, lnBottomY };
    }

    /**
     * Apply the bright activated appearance to a LayerNorm group.
     * @param {THREE.Object3D} lnTopGroup
     */
    _activateLayerNormColor(lnTopGroup) {
        const white = new THREE.Color(0xffffff);
        lnTopGroup.traverse(obj => {
            if (obj.isMesh && obj.material) {
                const apply = mat => { mat.color.copy(white); mat.emissive.copy(white); mat.emissiveIntensity = 0.5; mat.transparent = false; mat.opacity = 1.0; };
                if (Array.isArray(obj.material)) obj.material.forEach(apply); else apply(obj.material);
            }
        });
    }

    /**
     * Animate residual vectors toward the vocab embedding, optionally passing
     * through the top LayerNorm pipeline when present.
     * @param {Gpt2Layer} lastLayer
     * @param {number} targetYLocal
     * @param {{lnTopGroup: THREE.Object3D, lnCenterY: number, lnBottomY: number}|null} lnInfo
     */
    _animateResidualVectors(lastLayer, targetYLocal, lnInfo) {
        if (lnInfo && lnInfo.lnTopGroup) {
            const { lnTopGroup, lnCenterY, lnBottomY } = lnInfo;
            const lnMeshes = [];
            lnTopGroup.traverse(obj => {
                if (obj && obj.isMesh && obj.material) {
                    lnMeshes.push(obj);
                }
            });

            const lnColorState = {
                highestY: -Infinity,
                locked: false,
                lockedColor: new THREE.Color(COLOR_BRIGHT_YELLOW),
                currentColor: new THREE.Color(COLOR_DARK_GRAY),
                currentOpacity: 1.0
            };
            const tempColor = new THREE.Color();
            const applyTopLnColor = () => {
                lnMeshes.forEach(mesh => {
                    const applyMaterial = mat => {
                        if (!mat) return;
                        if (mat.color) mat.color.copy(lnColorState.currentColor);
                        if (mat.emissive) mat.emissive.copy(lnColorState.currentColor);
                        mat.transparent = lnColorState.currentOpacity < 1.0;
                        mat.opacity = lnColorState.currentOpacity;
                        mat.needsUpdate = true;
                    };
                    if (Array.isArray(mesh.material)) {
                        mesh.material.forEach(applyMaterial);
                    } else {
                        applyMaterial(mesh.material);
                    }
                });
            };
            applyTopLnColor();

            const lnHeight = LN_PARAMS.height || 0;
            const lnTopY = lnCenterY + lnHeight / 2;
            const normStartY = lnBottomY + lnHeight * LN_NORM_START_FRACTION_FROM_BOTTOM;
            const exitTransitionRange = 5;
            const tmpWorldPos = new THREE.Vector3();
            const tmpLocalPos = new THREE.Vector3();

            const updateTopLnColor = (y) => {
                if (!Number.isFinite(y)) return;
                if (y > lnColorState.highestY) {
                    lnColorState.highestY = y;
                }

                const highest = lnColorState.highestY;
                if (lnColorState.locked) {
                    lnColorState.currentColor.copy(lnColorState.lockedColor);
                    lnColorState.currentOpacity = 1.0;
                } else {
                    if (highest >= lnBottomY && highest < lnCenterY) {
                        const denom = Math.max(lnCenterY - lnBottomY, 1e-6);
                        const t = (highest - lnBottomY) / denom;
                        tempColor.copy(COLOR_DARK_GRAY).lerp(COLOR_LIGHT_YELLOW, t);
                        lnColorState.currentColor.copy(tempColor);
                        lnColorState.currentOpacity = THREE.MathUtils.lerp(1.0, 0.6, t);
                    } else if (highest >= lnCenterY && highest < lnTopY) {
                        lnColorState.currentColor.copy(COLOR_LIGHT_YELLOW);
                        lnColorState.currentOpacity = 0.6;
                    } else if (highest >= lnTopY) {
                        const tRaw = (highest - lnTopY) / exitTransitionRange;
                        const t = Math.min(1, Math.max(0, tRaw));
                        tempColor.copy(COLOR_LIGHT_YELLOW).lerp(COLOR_BRIGHT_YELLOW, t);
                        lnColorState.currentColor.copy(tempColor);
                        lnColorState.currentOpacity = THREE.MathUtils.lerp(0.6, 1.0, t);
                    } else {
                        lnColorState.currentColor.copy(COLOR_DARK_GRAY);
                        lnColorState.currentOpacity = 1.0;
                    }

                    if (highest >= lnTopY + exitTransitionRange) {
                        lnColorState.locked = true;
                        lnColorState.lockedColor.copy(COLOR_BRIGHT_YELLOW);
                        lnColorState.currentColor.copy(lnColorState.lockedColor);
                        lnColorState.currentOpacity = 1.0;
                    }
                }

                applyTopLnColor();
            };

            const additionDuration = (PRISM_ADD_ANIM_BASE_DURATION + PRISM_ADD_ANIM_BASE_FLASH_DURATION + VECTOR_LENGTH_PRISM * PRISM_ADD_ANIM_BASE_DELAY_BETWEEN_PRISMS) / PRISM_ADD_ANIM_SPEED_MULT;

            const updateTrailPosition = (vector) => {
                if (!vector || !vector.userData || !vector.userData.trail) return;
                const trail = vector.userData.trail;
                if (typeof trail.update !== 'function') return;
                vector.group.getWorldPosition(tmpWorldPos);
                if (vector.userData.trailWorld) {
                    tmpLocalPos.copy(tmpWorldPos);
                    trail.update(tmpLocalPos);
                } else {
                    tmpLocalPos.copy(tmpWorldPos);
                    try {
                        const parentObject = (trail._line && trail._line.parent) || trail._scene || null;
                        if (parentObject && typeof parentObject.worldToLocal === 'function') {
                            parentObject.worldToLocal(tmpLocalPos);
                        }
                    } catch (_) {
                        // fall back to world position already copied into tmpLocalPos
                    }
                    trail.update(tmpLocalPos);
                }
            };

            lastLayer.lanes.forEach(lane => {
                const vec = lane && lane.originalVec;
                if (!vec || !vec.group) return;
                const startY = vec.group.position.y;
                if (!Number.isFinite(startY)) return;

                const zPos = lane.zPos || 0;
                const multVec = new VectorVisualizationInstancedPrism(vec.rawData.slice(), new THREE.Vector3(0, lnCenterY, zPos));
                lastLayer.root.add(multVec.group);
                multVec.group.visible = false;

                const addVec = new VectorVisualizationInstancedPrism(vec.rawData.slice(), new THREE.Vector3(0, lnCenterY + LN_PARAMS.height / 4, zPos));
                lastLayer.root.add(addVec.group);
                addVec.group.visible = false;

                const normAnim = new PrismLayerNormAnimation(vec);
                let normLoopActive = false;

                if (startY >= lnBottomY) {
                    multVec.group.visible = true;
                    addVec.group.visible = true;
                    lane.__topLnEntered = true;
                }

                const startFinalRise = (resVec) => {
                    const riseDist = Math.max(0, targetYLocal - resVec.group.position.y);
                    const durMs = (riseDist / (ANIM_RISE_SPEED_ORIGINAL * GLOBAL_ANIM_SPEED_MULT)) * 1000;
                    new TWEEN.Tween(resVec.group.position)
                        .to({ y: targetYLocal }, Math.max(100, durMs))
                        .easing(TWEEN.Easing.Quadratic.InOut)
                        .onUpdate(() => {
                            updateTopLnColor(resVec.group.position.y);
                            updateTrailPosition(resVec);
                            this.dispatchEvent(new Event('progress'));
                        })
                        .onComplete(() => {
                            updateTopLnColor(targetYLocal + exitTransitionRange);
                            updateTrailPosition(resVec);
                            this.dispatchEvent(new Event('progress'));
                        })
                        .start();
                };

                const beginMultiply = () => {
                    if (lane.__topLnMultStarted) return;
                    lane.__topLnMultStarted = true;
                    multVec.group.visible = true;
                    addVec.group.visible = true;

                    simplePrismMultiply(vec, multVec, () => {
                        updateTopLnColor(multVec.group.position.y);
                        vec.group.visible = false;
                        multVec.group.visible = false;

                        const resVec = new VectorVisualizationInstancedPrism(multVec.rawData.slice(), multVec.group.position.clone());
                        lastLayer.root.add(resVec.group);

                        if (multVec.group && multVec.group.parent) {
                            multVec.group.parent.remove(multVec.group);
                        }

                        resVec.userData = resVec.userData || {};
                        if (vec.userData && vec.userData.trail) {
                            resVec.userData.trail = vec.userData.trail;
                            resVec.userData.trailWorld = vec.userData.trailWorld;
                        }
                        lane.originalVec = resVec;

                        updateTopLnColor(resVec.group.position.y);
                        this.dispatchEvent(new Event('progress'));

                        startPrismAdditionAnimation(addVec, resVec, null, () => {
                            if (addVec.group) addVec.group.visible = false;
                            if (addVec.group && addVec.group.parent) {
                                addVec.group.parent.remove(addVec.group);
                            }
                        });

                        new TWEEN.Tween({ t: 0 })
                            .to({ t: 1 }, additionDuration)
                            .onUpdate(() => {
                                updateTopLnColor(resVec.group.position.y);
                                updateTrailPosition(resVec);
                                this.dispatchEvent(new Event('progress'));
                            })
                            .start();

                        setTimeout(() => {
                            updateTopLnColor(resVec.group.position.y);
                            startFinalRise(resVec);
                        }, additionDuration + 100);
                    });
                };

                const riseToCenter = () => {
                    const targetY = Math.max(vec.group.position.y, lnCenterY);
                    if (vec.group.position.y >= targetY - 0.01) {
                        vec.group.position.y = targetY;
                        beginMultiply();
                        return;
                    }

                    const distance = Math.max(0, targetY - vec.group.position.y);
                    const duration = (distance / (ANIM_RISE_SPEED_ORIGINAL * GLOBAL_ANIM_SPEED_MULT)) * 1000;
                    new TWEEN.Tween(vec.group.position)
                        .to({ y: targetY }, Math.max(100, duration))
                        .easing(TWEEN.Easing.Quadratic.InOut)
                        .onUpdate(() => {
                            updateTopLnColor(vec.group.position.y);
                            this.dispatchEvent(new Event('progress'));
                        })
                        .onComplete(() => {
                            updateTopLnColor(vec.group.position.y);
                            this.dispatchEvent(new Event('progress'));
                            beginMultiply();
                        })
                        .start();
                };

                const startNormalization = () => {
                    if (normLoopActive) return;
                    normLoopActive = true;
                    try {
                        normAnim.start(vec.rawData.slice());
                    } catch (_) {
                        normLoopActive = false;
                        riseToCenter();
                        return;
                    }

                    const runLoop = () => {
                        normAnim.update(0);
                        updateTopLnColor(vec.group.position.y);
                        this.dispatchEvent(new Event('progress'));
                        if (normAnim.isAnimating) {
                            requestAnimationFrame(runLoop);
                        } else {
                            normLoopActive = false;
                            riseToCenter();
                        }
                    };
                    runLoop();
                };

                const moveToNormStart = () => {
                    updateTopLnColor(vec.group.position.y);
                    const stageTarget = Math.max(vec.group.position.y, normStartY);

                    if (vec.group.position.y >= stageTarget - 0.01) {
                        vec.group.position.y = stageTarget;
                        startNormalization();
                        return;
                    }

                    const distance = Math.max(0, stageTarget - vec.group.position.y);
                    const duration = (distance / (ANIM_RISE_SPEED_ORIGINAL * GLOBAL_ANIM_SPEED_MULT)) * 1000;
                    new TWEEN.Tween(vec.group.position)
                        .to({ y: stageTarget }, Math.max(100, duration))
                        .easing(TWEEN.Easing.Quadratic.InOut)
                        .onUpdate(() => {
                            updateTopLnColor(vec.group.position.y);
                            if (!lane.__topLnEntered && vec.group.position.y >= lnBottomY) {
                                lane.__topLnEntered = true;
                                multVec.group.visible = true;
                                addVec.group.visible = true;
                            }
                            this.dispatchEvent(new Event('progress'));
                        })
                        .onComplete(() => {
                            updateTopLnColor(vec.group.position.y);
                            this.dispatchEvent(new Event('progress'));
                            startNormalization();
                        })
                        .start();
                };

                moveToNormStart();
            });

            return;
        }

        lastLayer.lanes.forEach(lane => {
            const vec = lane && lane.originalVec;
            if (!vec || !vec.group) return;
            const startY = vec.group.position.y;
            if (typeof startY !== 'number' || !isFinite(startY)) return;
            if (startY >= targetYLocal - 0.01) return;

            const riseDist = Math.max(0, targetYLocal - startY);
            const durMs = (riseDist / (ANIM_RISE_SPEED_ORIGINAL * GLOBAL_ANIM_SPEED_MULT)) * 1000;

            new TWEEN.Tween(vec.group.position)
                .to({ y: targetYLocal }, Math.max(100, durMs))
                .easing(TWEEN.Easing.Quadratic.InOut)
                .onUpdate(() => this.dispatchEvent(new Event('progress')))
                .onComplete(() => this.dispatchEvent(new Event('progress')))
                .start();
        });
    }

    _resolveActiveLanePosition(targetVec = null) {
        const layers = this._layers;
        if (!Array.isArray(layers) || layers.length === 0) {
            return { laneIndex: -1, laneCount: 0 };
        }

        const layerIndex = Math.min(this._currentLayerIdx, layers.length - 1);
        const layer = layers[layerIndex];
        const lanes = Array.isArray(layer?.lanes) ? layer.lanes : [];
        const laneCount = lanes.length;
        if (!laneCount) {
            return { laneIndex: -1, laneCount };
        }

        const laneIndex = Math.min(laneCount - 1, Math.floor(laneCount / 2));
        const lane = lanes[laneIndex];
        const vecGroup = lane?.originalVec?.group;
        if (!vecGroup || typeof vecGroup.getWorldPosition !== 'function') {
            return { laneIndex: -1, laneCount };
        }

        if (targetVec) {
            vecGroup.getWorldPosition(targetVec);
        }

        return { laneIndex, laneCount };
    }

    _clearAutoCameraOffsets() {
        this._hasAutoCameraOffsets = false;
        this._autoCameraDesiredCameraOffset.set(0, 0, 0);
        this._autoCameraDesiredTargetOffset.set(0, 0, 0);
    }

    _cancelAutoCameraReturn() {
        this._autoCameraReturnActive = false;
        this._autoCameraManualResumeAt = 0;
    }

    _getNow() {
        if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
            return performance.now();
        }
        return Date.now();
    }

    _onUserCameraInteraction() {
        if (!this._hasAutoCameraOffsets) {
            this._captureAutoCameraOffsets();
        }
        this._autoCameraManualResumeAt = this._getNow() + this._autoCameraManualDelayMs;
        this._autoCameraReturnActive = false;
    }

    _captureAutoCameraOffsets(existingReference = null) {
        const engine = this._engine;
        const camera = engine?.camera;
        if (!camera) return false;

        const reference = existingReference || this._autoCameraCenter;
        if (!existingReference) {
            const laneInfo = this._resolveActiveLanePosition(reference);
            if (laneInfo.laneIndex < 0) {
                this._clearAutoCameraOffsets();
                return false;
            }
        }

        this._autoCameraDesiredCameraOffset.copy(camera.position).sub(reference);
        if (!Number.isFinite(this._autoCameraDesiredCameraOffset.x)
            || !Number.isFinite(this._autoCameraDesiredCameraOffset.y)
            || !Number.isFinite(this._autoCameraDesiredCameraOffset.z)) {
            this._clearAutoCameraOffsets();
            return false;
        }

        const controls = engine?.controls;
        if (controls && controls.target) {
            this._autoCameraDesiredTargetOffset.copy(controls.target).sub(reference);
        } else {
            this._autoCameraDesiredTargetOffset.set(0, 0, 0);
        }

        this._hasAutoCameraOffsets = true;
        return true;
    }

    _applyAutoCamera(reference, { smooth = false } = {}) {
        if (!this._autoCameraFollow || !this._hasAutoCameraOffsets) {
            return;
        }

        const engine = this._engine;
        const camera = engine?.camera;
        if (!engine || !camera) return;

        if (!Number.isFinite(reference?.x) || !Number.isFinite(reference?.y) || !Number.isFinite(reference?.z)) {
            return;
        }

        if (!Number.isFinite(this._autoCameraDesiredCameraOffset.x)
            || !Number.isFinite(this._autoCameraDesiredCameraOffset.y)
            || !Number.isFinite(this._autoCameraDesiredCameraOffset.z)) {
            return;
        }

        const desiredCamera = this._autoCameraReturnCameraScratch;
        desiredCamera.copy(reference).add(this._autoCameraDesiredCameraOffset);

        const controls = engine.controls;
        const hasControlsTarget = controls && controls.target;
        const desiredTarget = hasControlsTarget ? this._autoCameraReturnTargetScratch.copy(reference).add(this._autoCameraDesiredTargetOffset) : null;

        if (smooth) {
            this._suppressControlsChange = true;
            try {
                camera.position.lerp(desiredCamera, this._autoCameraReturnLerpAlpha);

                if (hasControlsTarget && desiredTarget) {
                    controls.target.lerp(desiredTarget, this._autoCameraReturnLerpAlpha);
                    if (typeof controls.update === 'function') {
                        controls.update();
                    }
                }

                if (typeof engine.notifyCameraUpdated === 'function') {
                    engine.notifyCameraUpdated();
                }
            } finally {
                this._suppressControlsChange = false;
            }

            const cameraClose = camera.position.distanceToSquared(desiredCamera) <= this._autoCameraReturnSnapThresholdSq;
            let targetClose = true;
            if (hasControlsTarget && desiredTarget) {
                targetClose = controls.target.distanceToSquared(desiredTarget) <= this._autoCameraReturnSnapThresholdSq;
            }

            if (cameraClose && targetClose) {
                this._suppressControlsChange = true;
                try {
                    camera.position.copy(desiredCamera);
                    if (hasControlsTarget && desiredTarget) {
                        controls.target.copy(desiredTarget);
                        if (typeof controls.update === 'function') {
                            controls.update();
                        }
                    }
                    if (typeof engine.notifyCameraUpdated === 'function') {
                        engine.notifyCameraUpdated();
                    }
                } finally {
                    this._suppressControlsChange = false;
                }
                this._autoCameraReturnActive = false;
            }
            return;
        }

        this._suppressControlsChange = true;
        try {
            camera.position.copy(desiredCamera);

            if (hasControlsTarget && desiredTarget) {
                controls.target.copy(desiredTarget);
                if (typeof controls.update === 'function') {
                    controls.update();
                }
            }

            if (typeof engine.notifyCameraUpdated === 'function') {
                engine.notifyCameraUpdated();
            }
        } finally {
            this._suppressControlsChange = false;
        }
        this._autoCameraReturnActive = false;
    }

    _updateCameraOffsetOverlay() {
        const overlay = this._cameraOffsetDiv;
        if (!overlay) return;

        if (!this._autoCameraFollow) {
            overlay.style.display = 'none';
            return;
        }

        const engine = this._engine;
        const camera = engine?.camera;
        if (!camera) {
            overlay.style.display = 'block';
            overlay.textContent = 'Offset vs Residual Lane —\nΔx: —\nΔy: —\nΔz: —';
            this._clearAutoCameraOffsets();
            return;
        }

        const reference = this._autoCameraCenter;
        const { laneIndex } = this._resolveActiveLanePosition(reference);
        if (laneIndex < 0 || !Number.isFinite(reference.x) || !Number.isFinite(reference.y) || !Number.isFinite(reference.z)) {
            overlay.style.display = 'block';
            overlay.textContent = 'Offset vs Residual Lane —\nΔx: —\nΔy: —\nΔz: —';
            this._clearAutoCameraOffsets();
            return;
        }

        if (!this._hasAutoCameraOffsets) {
            this._captureAutoCameraOffsets(reference);
        }

        if (this._autoCameraManualResumeAt > 0) {
            const now = this._getNow();
            if (now >= this._autoCameraManualResumeAt) {
                this._autoCameraManualResumeAt = 0;
                this._autoCameraReturnActive = true;
            }
        }

        const manualActive = this._autoCameraManualResumeAt > 0;
        if (!manualActive || this._autoCameraReturnActive) {
            this._applyAutoCamera(reference, { smooth: this._autoCameraReturnActive });
        }

        const offset = this._autoCameraOffsetScratch;
        offset.copy(camera.position).sub(reference);
        const format = (value) => (Number.isFinite(value) ? value.toFixed(2) : '—');
        const laneLabel = Number.isInteger(laneIndex) && laneIndex >= 0 ? (laneIndex + 1) : '—';

        overlay.style.display = 'block';
        overlay.textContent = `Offset vs Residual Lane ${laneLabel}\nΔx: ${format(offset.x)}\nΔy: ${format(offset.y)}\nΔz: ${format(offset.z)}`;
    }

    _maybeAutoCameraFocus({ immediate = false } = {}) {
        // Retained for compatibility with existing progress hooks; now only updates the debug overlay.
        if (!this._autoCameraFollow && !immediate) {
            this._updateCameraOffsetOverlay();
            return;
        }
        this._updateCameraOffsetOverlay();
    }

    _startCameraOverlayLoop() {
        if (this._cameraOverlayRaf !== null) return;
        if (typeof requestAnimationFrame !== 'function') return;

        const tick = () => {
            if (!this._autoCameraFollow) {
                this._stopCameraOverlayLoop();
                return;
            }
            this._updateCameraOffsetOverlay();
            this._cameraOverlayRaf = requestAnimationFrame(tick);
        };

        this._cameraOverlayRaf = requestAnimationFrame(tick);
    }

    _stopCameraOverlayLoop() {
        if (this._cameraOverlayRaf !== null && typeof cancelAnimationFrame === 'function') {
            cancelAnimationFrame(this._cameraOverlayRaf);
        }
        this._cameraOverlayRaf = null;
    }
}

/** Convenience helper mirroring CoreEngine.startEngine signature */
export function startPipeline(canvas, numLayers = 12, opts = {}) {
    const pipeline = new LayerPipeline(canvas, numLayers, opts);
    return () => pipeline.dispose();
} 
