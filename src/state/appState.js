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
        this.equationsSuppressed = false;
        this.lastEqKey = '';
        this.showHdrBackground = false;
        this.showPromptTokenStrip = true;
        this.autoCameraFollow = true;
        this.showCameraDebug = false;
        this.showFollowViewInspector = false;
        this.devMode = false;
        this.showPerfOverlay = false;
        this.kvCacheModeEnabled = false;
        this.kvCachePrefillActive = false;
        this.kvCachePassIndex = 0;
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
}
export const appState = new AppState();
export default appState;
