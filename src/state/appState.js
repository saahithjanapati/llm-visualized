import {
    LAYER_NORM_1_Y_POS,
    LN_PARAMS,
    EMBEDDING_BOTTOM_TOP_ALIGN_OFFSET_FROM_LN1_BOTTOM,
    EMBEDDING_BOTTOM_Y_ADJUST,
    EMBEDDING_MATRIX_PARAMS_VOCAB
} from '../utils/constants.js';

function computeTowerBaseY() {
    const topOfBottomEmbedding = (LAYER_NORM_1_Y_POS - LN_PARAMS.height / 2 + EMBEDDING_BOTTOM_TOP_ALIGN_OFFSET_FROM_LN1_BOTTOM)
        + EMBEDDING_BOTTOM_Y_ADJUST;
    return topOfBottomEmbedding - EMBEDDING_MATRIX_PARAMS_VOCAB.height;
}

export class AppState {
    constructor() {
        this.skipIntro = true;
        this.introActive = true;
        this.introRaf = 0;
        this.introCleaned = false;
        this.modalPaused = false;
        this.userPaused = false;
        this.vocabTopRef = null;
        this.topEmbedActivated = false;
        this.showEquations = true;
        this.lastEqKey = '';
        this.showHdrBackground = false;
        this.showSkyStars = true;
        this.environmentTexture = null;
        this.initialPipelineBackground = null;
        this.initialPipelineBackgroundCaptured = false;
        this.initialIntroBackground = null;
        this.initialIntroBackgroundCaptured = false;
        this.introSceneRef = null;
        this.towerBaseY = computeTowerBaseY();
    }

    applyEnvironmentBackground(pipeline, introScene = null) {
        if (introScene) {
            this.introSceneRef = introScene;
        }

        const desiredTexture = (this.showHdrBackground && this.environmentTexture)
            ? this.environmentTexture
            : null;

        const introTarget = introScene || this.introSceneRef;
        if (introTarget) {
            if (!this.initialIntroBackgroundCaptured) {
                const current = introTarget.background ?? null;
                this.initialIntroBackground = (current && typeof current.clone === 'function')
                    ? current.clone()
                    : current;
                this.initialIntroBackgroundCaptured = true;
            }
            introTarget.background = desiredTexture ?? this.initialIntroBackground ?? null;
        }

        if (pipeline?.engine?.scene) {
            const pipelineScene = pipeline.engine.scene;
            if (!this.initialPipelineBackgroundCaptured) {
                const current = pipelineScene.background ?? null;
                this.initialPipelineBackground = (current && typeof current.clone === 'function')
                    ? current.clone()
                    : current;
                this.initialPipelineBackgroundCaptured = true;
            }
            pipelineScene.background = desiredTexture ?? this.initialPipelineBackground ?? null;
        }
    }

    applySkyStars(pipeline) {
        if (!pipeline?.engine?.setSkyStarsEnabled) return;
        const layerCount = Array.isArray(pipeline?._layers) ? pipeline._layers.length : 0;
        const approxTowerHeight = Math.max(0, layerCount * 1600);
        const minHeight = 800;
        const maxHeight = Math.max(14000, approxTowerHeight + 8000);
        const maxRadius = Math.max(9000, approxTowerHeight * 0.5 + 9000);
        const starCount = Math.max(800, Math.floor(1600 + layerCount * 40));

        pipeline.engine.setSkyStarsEnabled(this.showSkyStars, {
            baseY: this.towerBaseY,
            minHeight,
            maxHeight,
            minRadius: 700,
            maxRadius,
            starCount,
            rotationSpeed: 0.12
        });
    }
}
export const appState = new AppState();
export default appState;
