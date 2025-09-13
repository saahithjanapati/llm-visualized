export class AppState {
    constructor() {
        this.skipIntro = true;
        this.introActive = true;
        this.introRaf = 0;
        this.introCleaned = false;
        this.modalPaused = false;
        this.vocabTopRef = null;
        this.topEmbedActivated = false;
        this.showEquations = true;
        this.lastEqKey = '';
    }
}
export const appState = new AppState();
export default appState;
