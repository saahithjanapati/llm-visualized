import * as THREE from 'three';
import { FontLoader } from 'three/examples/jsm/loaders/FontLoader.js';
import { WeightMatrixVisualization } from '../../components/WeightMatrixVisualization.js';
import { LayerNormalizationVisualization } from '../../components/LayerNormalizationVisualization.js';
import { VectorVisualizationInstancedPrism } from '../../components/VectorVisualizationInstancedPrism.js';
import { getLayerNormParamData } from '../../data/layerNormParams.js';
import {
    EMBEDDING_MATRIX_PARAMS_VOCAB,
    EMBEDDING_MATRIX_PARAMS_POSITION,
    LN_PARAMS,
    LAYER_NORM_1_Y_POS,
    MLP_MATRIX_PARAMS_DOWN,
    INACTIVE_COMPONENT_COLOR,
    EMBEDDING_BOTTOM_TOP_ALIGN_OFFSET_FROM_LN1_BOTTOM,
    EMBEDDING_BOTTOM_Y_ADJUST,
    EMBEDDING_BOTTOM_VOCAB_X_OFFSET,
    EMBEDDING_BOTTOM_PAIR_GAP_X,
    EMBEDDING_BOTTOM_POS_X_OFFSET,
    TOP_EMBED_VOCAB_X_OFFSET,
    TOP_EMBED_Y_GAP_ABOVE_TOWER,
    TOP_EMBED_Y_ADJUST,
    TOP_LN_TO_TOP_EMBED_GAP,
    USE_PHYSICAL_MATERIALS,
    VECTOR_LENGTH_PRISM
} from '../../utils/constants.js';
import {
    MHA_FINAL_Q_COLOR,
    POSITION_EMBED_COLOR,
    MHSA_MATRIX_INITIAL_RESTING_COLOR,
    TOP_EMBED_BASE_EMISSIVE,
    TOP_EMBED_MAX_EMISSIVE
} from '../../animations/LayerAnimationConstants.js';
import { appState } from '../../state/appState.js';
import { applyPhysicalMaterialsToScene } from '../../utils/materialUtils.js';
import { TOKEN_CHIP_STYLE, POSITION_CHIP_STYLE } from './config.js';
import { formatTokenLabel } from './tokenLabels.js';
import { addTopLogitBars, revealTopLogitBars } from './topLogitBars.js';

const LN_PARAM_MONOCHROME = {
    type: 'monochromatic',
    baseHue: 0,
    saturation: 0,
    minLightness: 0.03,
    maxLightness: 0.88,
    useData: true,
    valueMin: -1.8,
    valueMax: 1.8
};

// Build a rounded rectangle shape used for the token chip body.
function buildRoundedRectShape(width, height, radius) {
    const clampedRadius = Math.max(0, Math.min(radius, Math.min(width, height) / 2 - 1));
    const halfW = width / 2;
    const halfH = height / 2;
    const shape = new THREE.Shape();
    shape.moveTo(-halfW + clampedRadius, -halfH);
    shape.lineTo(halfW - clampedRadius, -halfH);
    shape.quadraticCurveTo(halfW, -halfH, halfW, -halfH + clampedRadius);
    shape.lineTo(halfW, halfH - clampedRadius);
    shape.quadraticCurveTo(halfW, halfH, halfW - clampedRadius, halfH);
    shape.lineTo(-halfW + clampedRadius, halfH);
    shape.quadraticCurveTo(-halfW, halfH, -halfW, halfH - clampedRadius);
    shape.lineTo(-halfW, -halfH + clampedRadius);
    shape.quadraticCurveTo(-halfW, -halfH, -halfW + clampedRadius, -halfH);
    shape.closePath();
    return shape;
}

// Create a 3D chip group with optional text mesh, plus a cached size in userData.
function createTokenChip(label, font, style) {
    let textGeo = null;
    let textShapes = null;
    let textDepth = 0;
    let bounds = null;
    const capOffset = 0.05;
    if (font && label.trim().length) {
        const desiredDepth = Number.isFinite(style.textDepth) ? style.textDepth : 0;
        const chipDepth = Number.isFinite(style.depth) ? style.depth : desiredDepth;
        textDepth = Number.isFinite(chipDepth) ? chipDepth + capOffset * 2 : desiredDepth;
        textShapes = font.generateShapes(label, style.textSize, 2);
        textGeo = new THREE.ExtrudeGeometry(textShapes, {
            depth: textDepth,
            curveSegments: 4,
            bevelEnabled: false
        });
        textGeo.computeBoundingBox();
        textGeo.computeVertexNormals();
        textGeo.translate(0, 0, -textDepth / 2);
        textGeo.computeBoundingBox();
        bounds = textGeo.boundingBox;
    }
    let textWidth = 0;
    let textHeight = 0;
    if (bounds && Number.isFinite(bounds.max.x) && Number.isFinite(bounds.min.x)) {
        textWidth = Math.max(0, bounds.max.x - bounds.min.x);
        textHeight = Math.max(0, bounds.max.y - bounds.min.y);
    }

    const chipWidth = Math.max(style.minWidth, textWidth + style.padding);
    const chipHeight = typeof style.height === 'number' && Number.isFinite(style.height)
        ? style.height
        : Math.max(style.minHeight, textHeight + style.padding);
    const chipRadius = Math.min(style.cornerRadius, Math.min(chipWidth, chipHeight) / 2 - 1);
    const chipShape = buildRoundedRectShape(chipWidth, chipHeight, chipRadius);
    const chipGeo = new THREE.ExtrudeGeometry(chipShape, {
        depth: style.depth,
        bevelEnabled: false
    });
    chipGeo.translate(0, 0, -style.depth / 2);
    chipGeo.computeVertexNormals();

    const chipMat = new THREE.MeshStandardMaterial({
        color: 0xf2e8d5,
        roughness: 0.35,
        metalness: 0.15,
        side: THREE.DoubleSide
    });
    const chipMesh = new THREE.Mesh(chipGeo, chipMat);

    const group = new THREE.Group();
    group.add(chipMesh);

    // Add thin caps to avoid transparent edges when viewed from the sides.
    const capMat = chipMat.clone();
    capMat.polygonOffset = false;
    capMat.polygonOffsetFactor = 0;
    capMat.polygonOffsetUnits = 0;
    const capGeo = new THREE.ShapeGeometry(chipShape);
    capGeo.computeVertexNormals();
    const frontCap = new THREE.Mesh(capGeo, capMat);
    frontCap.position.z = style.depth / 2 + capOffset;
    const backCap = new THREE.Mesh(capGeo, capMat);
    backCap.position.z = -style.depth / 2 - capOffset;
    backCap.rotation.y = Math.PI;
    group.add(frontCap, backCap);

    // Optional 3D text mesh, centered to span the chip depth.
    if (textGeo && textWidth > 0 && textHeight > 0) {
        const textBaseMat = new THREE.MeshBasicMaterial({
            color: 0xffffff,
            side: THREE.DoubleSide,
            depthWrite: true,
            depthTest: true,
            polygonOffset: true,
            polygonOffsetFactor: -0.5,
            polygonOffsetUnits: -0.5
        });
        const textCullMat = textBaseMat.clone();
        textCullMat.colorWrite = false;
        textCullMat.depthWrite = false;
        textCullMat.transparent = true;
        textCullMat.opacity = 0;
        const textGroup = new THREE.Group();
        const textMesh = new THREE.Mesh(textGeo, [textCullMat, textBaseMat]);
        textGroup.add(textMesh);
        if (textShapes) {
            const faceGeo = new THREE.ShapeGeometry(textShapes);
            faceGeo.computeVertexNormals();
            const faceOffset = 0.02;
            const frontFace = new THREE.Mesh(faceGeo, textBaseMat);
            frontFace.position.z = textDepth / 2 + faceOffset;
            const backFace = new THREE.Mesh(faceGeo, textBaseMat);
            backFace.position.z = -textDepth / 2 - faceOffset;
            textGroup.add(frontFace, backFace);
        }
        const centerX = (bounds.min.x + bounds.max.x) / 2;
        const centerY = (bounds.min.y + bounds.max.y) / 2;
        textGroup.position.set(-centerX, -centerY, 0);
        group.add(textGroup);
    } else if (textGeo) {
        textGeo.dispose();
    }

    if (typeof style.scale === 'number' && Number.isFinite(style.scale) && style.scale > 0) {
        group.scale.setScalar(style.scale);
    }
    const scaleFactor = (typeof style.scale === 'number' && Number.isFinite(style.scale) && style.scale > 0)
        ? style.scale
        : 1;
    group.userData.size = { width: chipWidth * scaleFactor, height: chipHeight * scaleFactor };
    return group;
}

// Temporarily stage the camera near the chips, then optionally return to tower view.
function stageChipCamera(pipeline, startPos, startTarget, endPos, endTarget, holdMs, returnMs) {
    const engine = pipeline?.engine;
    if (!engine || !engine.camera || !engine.controls) return;
    const hadAutoFollow = typeof pipeline.isAutoCameraFollowEnabled === 'function'
        ? pipeline.isAutoCameraFollowEnabled()
        : false;
    if (hadAutoFollow && typeof pipeline.setAutoCameraFollow === 'function') {
        pipeline.setAutoCameraFollow(false, { immediate: true });
    }

    engine.camera.position.copy(startPos);
    engine.controls.target.copy(startTarget);
    engine.notifyCameraUpdated();
    engine.controls.update();

    const restoreAutoFollow = () => {
        if (hadAutoFollow && typeof pipeline.setAutoCameraFollow === 'function') {
            pipeline.setAutoCameraFollow(true, { immediate: true });
        }
    };

    const shouldReturn = Number.isFinite(returnMs) && returnMs > 0 && endPos && endTarget;
    if (!shouldReturn) {
        if (hadAutoFollow) {
            const delayMs = Math.max(0, holdMs);
            if (delayMs > 0) {
                setTimeout(restoreAutoFollow, delayMs);
            } else {
                restoreAutoFollow();
            }
        }
        return;
    }

    if (typeof TWEEN !== 'undefined') {
        new TWEEN.Tween(engine.camera.position)
            .to({ x: endPos.x, y: endPos.y, z: endPos.z }, returnMs)
            .delay(holdMs)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(() => engine.notifyCameraUpdated())
            .start();

        new TWEEN.Tween(engine.controls.target)
            .to({ x: endTarget.x, y: endTarget.y, z: endTarget.z }, returnMs)
            .delay(holdMs)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(() => {
                engine.notifyCameraUpdated();
                engine.controls.update();
            })
            .onComplete(restoreAutoFollow)
            .start();
    } else {
        setTimeout(() => {
            engine.camera.position.copy(endPos);
            engine.controls.target.copy(endTarget);
            engine.notifyCameraUpdated();
            engine.controls.update();
            restoreAutoFollow();
        }, holdMs);
    }
}

export function addEmbeddingAndTokenChips({
    pipeline,
    laneCount,
    activationSource,
    laneTokenIndices,
    tokenLabels,
    positionLabels,
    cameraReturnPosition,
    cameraReturnTarget,
    numLayers
}) {
    if (!pipeline || !pipeline.engine || !pipeline.engine.scene) return null;

    const engine = pipeline.engine;
    const rootGroup = new THREE.Group();
    rootGroup.name = 'EmbeddingAndTokenChips';
    let disposed = false;
    let topLogitProgressHandler = null;
    let chipRemovalProgressHandler = null;
    let chipEntryTimeoutId = null;

    const disposeObject = (obj) => {
        if (!obj) return;
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
            const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
            materials.forEach((mat) => mat && mat.dispose && mat.dispose());
        }
        if (Array.isArray(obj.children)) {
            obj.children.forEach((child) => disposeObject(child));
        }
    };

    const addToRoot = (obj) => {
        if (!obj) return;
        if (disposed) {
            disposeObject(obj);
            return;
        }
        rootGroup.add(obj);
    };

    engine.scene.add(rootGroup);
    if (typeof engine.registerRaycastRoot === 'function') {
        engine.registerRaycastRoot(rootGroup);
    }

    try {
        const headBlue = new THREE.Color(MHA_FINAL_Q_COLOR);
        const positionGreen = new THREE.Color(POSITION_EMBED_COLOR);
        const residualYBase = LAYER_NORM_1_Y_POS - LN_PARAMS.height / 2 + EMBEDDING_BOTTOM_TOP_ALIGN_OFFSET_FROM_LN1_BOTTOM;
        const bottomVocabCenterY = residualYBase - EMBEDDING_MATRIX_PARAMS_VOCAB.height / 2 + EMBEDDING_BOTTOM_Y_ADJUST;
        const vocabX = 0 + EMBEDDING_BOTTOM_VOCAB_X_OFFSET;
        const vocabBottom = new WeightMatrixVisualization(
            null,
            new THREE.Vector3(vocabX, bottomVocabCenterY, 0),
            EMBEDDING_MATRIX_PARAMS_VOCAB.width,
            EMBEDDING_MATRIX_PARAMS_VOCAB.height,
            EMBEDDING_MATRIX_PARAMS_VOCAB.depth,
            EMBEDDING_MATRIX_PARAMS_VOCAB.topWidthFactor,
            EMBEDDING_MATRIX_PARAMS_VOCAB.cornerRadius,
            EMBEDDING_MATRIX_PARAMS_VOCAB.numberOfSlits,
            EMBEDDING_MATRIX_PARAMS_VOCAB.slitWidth,
            EMBEDDING_MATRIX_PARAMS_VOCAB.slitDepthFactor,
            EMBEDDING_MATRIX_PARAMS_VOCAB.slitBottomWidthFactor,
            EMBEDDING_MATRIX_PARAMS_VOCAB.slitTopWidthFactor
        );
        vocabBottom.group.userData.label = 'Vocab Embedding';
        vocabBottom.setColor(headBlue);
        vocabBottom.setMaterialProperties({
            opacity: 1.0,
            transparent: false,
            emissiveIntensity: TOP_EMBED_BASE_EMISSIVE + TOP_EMBED_MAX_EMISSIVE
        });
        addToRoot(vocabBottom.group);

        const gapX = EMBEDDING_BOTTOM_PAIR_GAP_X;
        const posX = (EMBEDDING_MATRIX_PARAMS_VOCAB.width / 2)
            + (EMBEDDING_MATRIX_PARAMS_POSITION.width / 2)
            + gapX
            + EMBEDDING_BOTTOM_POS_X_OFFSET
            + EMBEDDING_BOTTOM_VOCAB_X_OFFSET;
        const vocabBottomY = bottomVocabCenterY - EMBEDDING_MATRIX_PARAMS_VOCAB.height / 2;
        const bottomPosCenterY = vocabBottomY + EMBEDDING_MATRIX_PARAMS_POSITION.height / 2;
        const posBottom = new WeightMatrixVisualization(
            null,
            new THREE.Vector3(posX, bottomPosCenterY, 0),
            EMBEDDING_MATRIX_PARAMS_POSITION.width,
            EMBEDDING_MATRIX_PARAMS_POSITION.height,
            EMBEDDING_MATRIX_PARAMS_POSITION.depth,
            EMBEDDING_MATRIX_PARAMS_POSITION.topWidthFactor,
            EMBEDDING_MATRIX_PARAMS_POSITION.cornerRadius,
            EMBEDDING_MATRIX_PARAMS_POSITION.numberOfSlits,
            EMBEDDING_MATRIX_PARAMS_POSITION.slitWidth,
            EMBEDDING_MATRIX_PARAMS_POSITION.slitDepthFactor,
            EMBEDDING_MATRIX_PARAMS_POSITION.slitBottomWidthFactor,
            EMBEDDING_MATRIX_PARAMS_POSITION.slitTopWidthFactor
        );
        posBottom.group.userData.label = 'Positional Embedding';
        posBottom.setColor(positionGreen);
        posBottom.setMaterialProperties({
            opacity: 1.0,
            transparent: false,
            emissiveIntensity: TOP_EMBED_BASE_EMISSIVE + TOP_EMBED_MAX_EMISSIVE
        });
        addToRoot(posBottom.group);

        // Precompute Z positions for each lane so chips align with vector lanes.
        const laneSpacing = LN_PARAMS.depth / (laneCount + 1);
        const laneZs = [];
        for (let i = 0; i < laneCount; i++) {
            laneZs.push(-LN_PARAMS.depth / 2 + laneSpacing * (i + 1));
        }

        const vocabMatrixBottomY = bottomVocabCenterY - EMBEDDING_MATRIX_PARAMS_VOCAB.height / 2;
        const posMatrixBottomY = bottomPosCenterY - EMBEDDING_MATRIX_PARAMS_POSITION.height / 2;
        const bottomEmbeddingVectorStartY = bottomVocabCenterY + EMBEDDING_MATRIX_PARAMS_VOCAB.height / 2;
        const laneSpanMs = TOKEN_CHIP_STYLE.riseDelay * Math.max(0, laneCount - 1);
        const vocabRiseDuration = TOKEN_CHIP_STYLE.riseDuration * (TOKEN_CHIP_STYLE.vocabSlowdown || 1);
        const posRiseDuration = TOKEN_CHIP_STYLE.riseDuration * (TOKEN_CHIP_STYLE.positionSlowdown || 1);
        const maxChipRiseDuration = Math.max(vocabRiseDuration, posRiseDuration) + laneSpanMs;
        const chipStartZOffset = TOKEN_CHIP_STYLE.zOffset;
        const chipFontLoader = new FontLoader();
        const animatedEmbeddingChips = [];
        let chipsEnteredEmbedding = false;
        let chipsRemovedFromEmbedding = false;
        const clearChipEntryTimeout = () => {
            if (chipEntryTimeoutId) {
                clearTimeout(chipEntryTimeoutId);
                chipEntryTimeoutId = null;
            }
        };
        const removeAnimatedEmbeddingChips = () => {
            if (chipsRemovedFromEmbedding) return;
            chipsRemovedFromEmbedding = true;
            animatedEmbeddingChips.forEach((chip) => {
                if (!chip) return;
                if (chip.parent) {
                    chip.parent.remove(chip);
                }
                disposeObject(chip);
            });
            animatedEmbeddingChips.length = 0;
        };
        const haveBottomVectorsRisenFromEmbedding = () => {
            const firstLayer = pipeline?._layers?.[0];
            const lanes = Array.isArray(firstLayer?.lanes) ? firstLayer.lanes : [];
            if (!lanes.length) return false;
            return lanes.some((lane) => {
                const y = lane?.originalVec?.group?.position?.y;
                return Number.isFinite(y) && y > bottomEmbeddingVectorStartY + 0.25;
            });
        };
        const maybeRemoveEmbeddedChips = () => {
            if (disposed || chipsRemovedFromEmbedding || !chipsEnteredEmbedding) return;
            if (!haveBottomVectorsRisenFromEmbedding()) return;
            removeAnimatedEmbeddingChips();
            if (chipRemovalProgressHandler && pipeline && typeof pipeline.removeEventListener === 'function') {
                pipeline.removeEventListener('progress', chipRemovalProgressHandler);
                chipRemovalProgressHandler = null;
            }
        };
        const armEmbeddedChipRemoval = () => {
            clearChipEntryTimeout();
            const entryDelayMs = (typeof TWEEN !== 'undefined') ? Math.max(0, maxChipRiseDuration + 60) : 0;
            if (entryDelayMs <= 0) {
                chipsEnteredEmbedding = true;
                maybeRemoveEmbeddedChips();
                return;
            }
            chipEntryTimeoutId = setTimeout(() => {
                chipEntryTimeoutId = null;
                if (disposed) return;
                chipsEnteredEmbedding = true;
                maybeRemoveEmbeddedChips();
            }, entryDelayMs);
        };
        if (pipeline && typeof pipeline.addEventListener === 'function') {
            chipRemovalProgressHandler = () => {
                maybeRemoveEmbeddedChips();
            };
            pipeline.addEventListener('progress', chipRemovalProgressHandler);
        }
        const registerChip = (chip) => addToRoot(chip);
        const spawnTokenChips = (font) => {
            if (disposed) return;
            // Animate the token chips upward from a resting "stash" position.
            const cameraStartPos = new THREE.Vector3(
                vocabX + EMBEDDING_MATRIX_PARAMS_VOCAB.width * 0.4,
                bottomVocabCenterY - EMBEDDING_MATRIX_PARAMS_VOCAB.height * 0.8,
                EMBEDDING_MATRIX_PARAMS_VOCAB.depth * 1.8
            );
            const cameraStartTarget = new THREE.Vector3(
                vocabX,
                bottomVocabCenterY + EMBEDDING_MATRIX_PARAMS_VOCAB.height * 0.2,
                0
            );
            const adjustedHoldMs = maxChipRiseDuration + TOKEN_CHIP_STYLE.cameraHoldMs;
            stageChipCamera(
                pipeline,
                cameraStartPos,
                cameraStartTarget,
                cameraReturnPosition,
                cameraReturnTarget,
                adjustedHoldMs,
                TOKEN_CHIP_STYLE.cameraReturnMs
            );

            const labels = tokenLabels.slice(0, laneCount).map(formatTokenLabel);
            labels.forEach((label, idx) => {
                const chip = createTokenChip(label, font, TOKEN_CHIP_STYLE);
                const chipLabel = `Token: ${label}`;
                chip.userData.label = chipLabel;
                chip.name = chipLabel;
                const chipHeight = chip.userData.size.height;
                const targetY = vocabMatrixBottomY + chipHeight / 2 + TOKEN_CHIP_STYLE.inset;
                const targetZ = laneZs[idx];
                const startZ = targetZ + chipStartZOffset;

                const staticY = vocabMatrixBottomY - chipHeight / 2 - TOKEN_CHIP_STYLE.staticGap;
                const staticZ = targetZ + TOKEN_CHIP_STYLE.staticZOffset;
                const startY = staticY;

                chip.position.set(vocabX, startY, startZ);
                registerChip(chip);
                animatedEmbeddingChips.push(chip);

                // Keep a static chip below the stack for context once the animated one rises.
                const staticChip = createTokenChip(label, font, TOKEN_CHIP_STYLE);
                staticChip.userData.label = chipLabel;
                staticChip.name = chipLabel;
                staticChip.position.set(vocabX, staticY, staticZ);
                registerChip(staticChip);

                if (typeof TWEEN !== 'undefined') {
                    new TWEEN.Tween(chip.position)
                        .to({ y: targetY, z: targetZ }, vocabRiseDuration)
                        .delay(idx * TOKEN_CHIP_STYLE.riseDelay)
                        .easing(TWEEN.Easing.Quadratic.Out)
                        .start();
                } else {
                    chip.position.y = targetY;
                    chip.position.z = targetZ;
                }
            });
        };
        const spawnPositionChips = (font) => {
            if (disposed) return;
            const posRiseDuration = TOKEN_CHIP_STYLE.riseDuration * (TOKEN_CHIP_STYLE.positionSlowdown || 1);
            const style = POSITION_CHIP_STYLE;
            const labels = positionLabels.slice(0, laneCount);
            labels.forEach((label, idx) => {
                const chip = createTokenChip(label, font, style);
                const chipLabel = `Position: ${label}`;
                chip.userData.label = chipLabel;
                chip.name = chipLabel;
                const chipHeight = chip.userData.size.height;
                const targetY = posMatrixBottomY + chipHeight / 2 + style.inset;
                const targetZ = laneZs[idx];
                const startZ = targetZ + chipStartZOffset;

                const staticY = posMatrixBottomY - chipHeight / 2 - style.staticGap;
                const staticZ = targetZ + style.staticZOffset;
                const startY = staticY;

                chip.position.set(posX, startY, startZ);
                registerChip(chip);
                animatedEmbeddingChips.push(chip);

                // Static chip for the position stream, matching the token chips below.
                const staticChip = createTokenChip(label, font, style);
                staticChip.userData.label = chipLabel;
                staticChip.name = chipLabel;
                staticChip.position.set(posX, staticY, staticZ);
                registerChip(staticChip);

                if (typeof TWEEN !== 'undefined') {
                    new TWEEN.Tween(chip.position)
                        .to({ y: targetY, z: targetZ }, posRiseDuration)
                        .delay(idx * TOKEN_CHIP_STYLE.riseDelay)
                        .easing(TWEEN.Easing.Quadratic.Out)
                        .start();
                } else {
                    chip.position.y = targetY;
                    chip.position.z = targetZ;
                }
            });
        };
        chipFontLoader.load(
            'https://threejs.org/examples/fonts/helvetiker_regular.typeface.json',
            (font) => {
                if (disposed) return;
                spawnTokenChips(font);
                spawnPositionChips(font);
                armEmbeddedChipRemoval();
                applyPhysicalMaterialsToScene(engine.scene, USE_PHYSICAL_MATERIALS);
            },
            undefined,
            (err) => {
                if (disposed) return;
                console.warn('Token chip font failed to load, rendering without labels.', err);
                spawnTokenChips(null);
                spawnPositionChips(null);
                armEmbeddedChipRemoval();
                applyPhysicalMaterialsToScene(engine.scene, USE_PHYSICAL_MATERIALS);
            }
        );

        // Top-of-tower LayerNorm + vocab projection marker.
        const lastLayer = pipeline._layers[numLayers - 1];
        if (lastLayer && lastLayer.mlpDown && lastLayer.mlpDown.group) {
            pipeline._topLnParamPlaceholders = null;
            const tmp = new THREE.Vector3();
            lastLayer.mlpDown.group.getWorldPosition(tmp);
            const towerTopY = tmp.y + MLP_MATRIX_PARAMS_DOWN.height / 2;
            const topGap = TOP_EMBED_Y_GAP_ABOVE_TOWER;
            const topLnCenterY = towerTopY + topGap + LN_PARAMS.height / 2;
            const lnTop = new LayerNormalizationVisualization(
                new THREE.Vector3(0 + TOP_EMBED_VOCAB_X_OFFSET, topLnCenterY, 0),
                LN_PARAMS.width,
                LN_PARAMS.height,
                LN_PARAMS.depth,
                LN_PARAMS.wallThickness,
                LN_PARAMS.numberOfHoles,
                LN_PARAMS.holeWidth,
                LN_PARAMS.holeWidthFactor
            );
            lnTop.group.userData.label = 'LayerNorm (Top)';
            lnTop.setColor(new THREE.Color(INACTIVE_COMPONENT_COLOR));
            lnTop.setMaterialProperties({ opacity: 1.0, transparent: false, emissiveIntensity: 0.05 });
            addToRoot(lnTop.group);

            // Final LayerNorm parameter placeholders (gamma/beta), visible in grayscale
            // before activation, matching per-layer LayerNorm behavior.
            try {
                const instanceCount = Number.isFinite(lastLayer?._getBaseVectorLength?.())
                    ? Math.max(1, Math.floor(lastLayer._getBaseVectorLength()))
                    : Math.max(1, Math.floor(lastLayer?.lanes?.[0]?.originalVec?.instanceCount || VECTOR_LENGTH_PRISM));
                const finalScaleDataRaw = getLayerNormParamData(lastLayer.index, 'final', 'scale', instanceCount);
                const finalShiftDataRaw = getLayerNormParamData(lastLayer.index, 'final', 'shift', instanceCount);
                const finalScaleData = (Array.isArray(finalScaleDataRaw) && finalScaleDataRaw.length)
                    ? finalScaleDataRaw
                    : new Array(instanceCount).fill(1);
                const finalShiftData = (Array.isArray(finalShiftDataRaw) && finalShiftDataRaw.length)
                    ? finalShiftDataRaw
                    : new Array(instanceCount).fill(0);
                const finalParamParent = lastLayer.raycastRoot || lastLayer.root;
                const addYOffset = LN_PARAMS.height * 0.25;
                const scalePlaceholders = new Array(laneCount);
                const shiftPlaceholders = new Array(laneCount);

                for (let laneIdx = 0; laneIdx < laneCount; laneIdx++) {
                    const zPos = laneZs[laneIdx];
                    const scaleWorldPos = new THREE.Vector3(
                        TOP_EMBED_VOCAB_X_OFFSET,
                        topLnCenterY + 3.3,
                        zPos
                    );
                    const shiftWorldPos = new THREE.Vector3(
                        TOP_EMBED_VOCAB_X_OFFSET,
                        topLnCenterY + addYOffset,
                        zPos
                    );
                    const scaleLocalPos = scaleWorldPos.clone();
                    const shiftLocalPos = shiftWorldPos.clone();
                    lastLayer.root.worldToLocal(scaleLocalPos);
                    lastLayer.root.worldToLocal(shiftLocalPos);

                    const scaleVec = new VectorVisualizationInstancedPrism(
                        finalScaleData.slice(),
                        scaleLocalPos,
                        30,
                        instanceCount
                    );
                    scaleVec.group.userData.label = 'Final LN Scale (gamma)';
                    scaleVec.group.userData.skipVisible = true;
                    if (scaleVec.mesh) {
                        scaleVec.mesh.userData = scaleVec.mesh.userData || {};
                        scaleVec.mesh.userData.skipVisible = true;
                    }
                    scaleVec.group.visible = true;
                    {
                        const n = Math.min(30, Math.max(1, scaleVec.rawData.length || 1));
                        scaleVec.updateKeyColorsFromData(scaleVec.rawData, n, LN_PARAM_MONOCHROME, scaleVec.rawData);
                    }
                    finalParamParent.add(scaleVec.group);
                    scalePlaceholders[laneIdx] = scaleVec;

                    const shiftVec = new VectorVisualizationInstancedPrism(
                        finalShiftData.slice(),
                        shiftLocalPos,
                        30,
                        instanceCount
                    );
                    shiftVec.group.userData.label = 'Final LN Shift (beta)';
                    shiftVec.group.userData.skipVisible = true;
                    if (shiftVec.mesh) {
                        shiftVec.mesh.userData = shiftVec.mesh.userData || {};
                        shiftVec.mesh.userData.skipVisible = true;
                    }
                    shiftVec.group.visible = true;
                    {
                        const n = Math.min(30, Math.max(1, shiftVec.rawData.length || 1));
                        shiftVec.updateKeyColorsFromData(shiftVec.rawData, n, LN_PARAM_MONOCHROME, shiftVec.rawData);
                    }
                    finalParamParent.add(shiftVec.group);
                    shiftPlaceholders[laneIdx] = shiftVec;
                }

                pipeline._topLnParamPlaceholders = {
                    layerIndex: lastLayer.index,
                    scale: scalePlaceholders,
                    shift: shiftPlaceholders
                };
            } catch (_) {
                pipeline._topLnParamPlaceholders = null;
            }

            const topVocabCenterY = topLnCenterY
                + (LN_PARAMS.height / 2)
                + TOP_LN_TO_TOP_EMBED_GAP
                + (EMBEDDING_MATRIX_PARAMS_VOCAB.height / 2)
                + TOP_EMBED_Y_ADJUST;
            const vocabTop = new WeightMatrixVisualization(
                null,
                new THREE.Vector3(0 + TOP_EMBED_VOCAB_X_OFFSET, topVocabCenterY, 0),
                EMBEDDING_MATRIX_PARAMS_VOCAB.width,
                EMBEDDING_MATRIX_PARAMS_VOCAB.height,
                EMBEDDING_MATRIX_PARAMS_VOCAB.depth,
                EMBEDDING_MATRIX_PARAMS_VOCAB.topWidthFactor,
                EMBEDDING_MATRIX_PARAMS_VOCAB.cornerRadius,
                EMBEDDING_MATRIX_PARAMS_VOCAB.numberOfSlits,
                EMBEDDING_MATRIX_PARAMS_VOCAB.slitWidth,
                EMBEDDING_MATRIX_PARAMS_VOCAB.slitDepthFactor,
                EMBEDDING_MATRIX_PARAMS_VOCAB.slitBottomWidthFactor,
                EMBEDDING_MATRIX_PARAMS_VOCAB.slitTopWidthFactor
            );
            vocabTop.group.rotation.z = Math.PI;
            vocabTop.group.userData.label = 'Vocab Embedding (Top)';
            vocabTop.setColor(new THREE.Color(MHSA_MATRIX_INITIAL_RESTING_COLOR));
            vocabTop.setMaterialProperties({
                opacity: 1.0,
                transparent: false,
                emissiveIntensity: TOP_EMBED_BASE_EMISSIVE,
                metalness: 0.4,
                roughness: 0.35,
                clearcoat: 0.45,
                clearcoatRoughness: 0.4,
                iridescence: 0.25,
                envMapIntensity: 0.9
            });
            appState.vocabTopRef = vocabTop;
            addToRoot(vocabTop.group);
            const vocabTopPos = new THREE.Vector3();
            vocabTop.group.getWorldPosition(vocabTopPos);
            const topLogitBars = addTopLogitBars({
                activationSource,
                laneTokenIndices,
                laneZs,
                vocabCenter: vocabTopPos,
                scene: rootGroup,
                engine: null
            });
            if (topLogitBars && pipeline && typeof pipeline.addEventListener === 'function') {
                const isTopEmbeddingTraversalComplete = () => {
                    const lastLayerRef = pipeline._layers?.[numLayers - 1];
                    if (!lastLayerRef) return false;
                    const exitY = Number.isFinite(lastLayerRef.__topEmbedExitYLocal)
                        ? lastLayerRef.__topEmbedExitYLocal
                        : lastLayerRef.__topEmbedStopYLocal;
                    if (!Number.isFinite(exitY)) return false;
                    const lanes = Array.isArray(lastLayerRef.lanes) ? lastLayerRef.lanes : [];
                    if (!lanes.length) return false;
                    return lanes.every((lane) => {
                        const y = lane?.originalVec?.group?.position?.y;
                        return Number.isFinite(y) && y >= exitY - 0.5;
                    });
                };
                const maybeReveal = () => {
                    if (!isTopEmbeddingTraversalComplete()) {
                        return false;
                    }
                    const immediate = typeof pipeline.isSkipToEndActive === 'function'
                        ? pipeline.isSkipToEndActive()
                        : false;
                    revealTopLogitBars(topLogitBars, { immediate });
                    return true;
                };
                const onProgress = () => {
                    if (maybeReveal()) {
                        pipeline.removeEventListener('progress', onProgress);
                    }
                };
                topLogitProgressHandler = onProgress;
                pipeline.addEventListener('progress', onProgress);
                onProgress();
            }
        }
    } catch (_) {
        // Optional – embedding visuals are non-critical.
    }

    const dispose = () => {
        if (disposed) return;
        disposed = true;
        if (chipEntryTimeoutId) {
            clearTimeout(chipEntryTimeoutId);
            chipEntryTimeoutId = null;
        }
        if (chipRemovalProgressHandler && pipeline && typeof pipeline.removeEventListener === 'function') {
            pipeline.removeEventListener('progress', chipRemovalProgressHandler);
            chipRemovalProgressHandler = null;
        }
        if (topLogitProgressHandler && pipeline && typeof pipeline.removeEventListener === 'function') {
            pipeline.removeEventListener('progress', topLogitProgressHandler);
            topLogitProgressHandler = null;
        }
        if (engine && typeof engine.removeRaycastRoot === 'function') {
            engine.removeRaycastRoot(rootGroup);
        }
        if (rootGroup.parent) {
            rootGroup.parent.remove(rootGroup);
        }
        disposeObject(rootGroup);
        const topLnPlaceholders = pipeline && pipeline._topLnParamPlaceholders;
        if (topLnPlaceholders) {
            const refs = [
                ...(Array.isArray(topLnPlaceholders.scale) ? topLnPlaceholders.scale : []),
                ...(Array.isArray(topLnPlaceholders.shift) ? topLnPlaceholders.shift : [])
            ];
            refs.forEach((vec) => {
                if (!vec) return;
                if (vec.group && vec.group.parent) {
                    vec.group.parent.remove(vec.group);
                }
                if (typeof vec.dispose === 'function') {
                    try { vec.dispose(); } catch (_) { /* best-effort cleanup */ }
                }
            });
            pipeline._topLnParamPlaceholders = null;
        }
        if (appState.vocabTopRef) {
            appState.vocabTopRef = null;
        }
    };

    return { root: rootGroup, dispose };
}
