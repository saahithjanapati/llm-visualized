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
        this.sciFiModeEnabled = false;
        this.environmentTexture = null;
        this.initialPipelineBackground = null;
        this.initialPipelineBackgroundCaptured = false;
        this.initialIntroBackground = null;
        this.initialIntroBackgroundCaptured = false;
        this.introSceneRef = null;
    }

    applyEnvironmentBackground(pipeline, introScene = null) {
        if (introScene) {
            this.introSceneRef = introScene;
        }

        const desiredTexture = (this.showHdrBackground && !this.sciFiModeEnabled && this.environmentTexture)
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
            if (!this.sciFiModeEnabled || desiredTexture) {
                introTarget.background = desiredTexture ?? this.initialIntroBackground ?? null;
            }
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
            if (!this.sciFiModeEnabled || desiredTexture) {
                pipelineScene.background = desiredTexture ?? this.initialPipelineBackground ?? null;
            }
        }
    }

    applySciFiMode(pipeline, introScene = null) {
        if (pipeline?.engine?.setSciFiModeEnabled) {
            pipeline.engine.setSciFiModeEnabled(!!this.sciFiModeEnabled);
        }

        // When sci-fi mode toggles it can change how the background should be
        // resolved (e.g. HDRI backgrounds are suppressed while the starfield
        // dome is active), so re-apply the background preferences now.
        this.applyEnvironmentBackground(pipeline, introScene);
    }
}
export const appState = new AppState();
export default appState;
