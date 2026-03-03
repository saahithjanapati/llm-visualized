import { appState } from '../state/appState.js';
import { getLogitTokenColorCss, resolveLogitTokenSeed } from '../app/gpt-tower/logitColor.js';

const STRIP_ID = 'promptTokenStrip';

function normalizeTokenText(token) {
    if (token === null || token === undefined) return '';
    const raw = String(token);
    if (!raw.length) return '';
    return raw
        .replace(/\r/g, '')
        .replace(/\n/g, '\u21b5')
        .replace(/\t/g, '\u21e5')
        .replace(/ /g, '\u00A0');
}

function buildDom() {
    let root = document.getElementById(STRIP_ID);
    if (!root) {
        root = document.createElement('div');
        root.id = STRIP_ID;
        root.dataset.visible = 'false';
        root.innerHTML = `
            <div class="prompt-token-strip__header">
                <span class="prompt-token-strip__title">Tokenized Prompt</span>
                <span class="prompt-token-strip__count" data-role="count"></span>
            </div>
            <div class="prompt-token-strip__tokens" data-role="tokens"></div>
        `;
        document.body.appendChild(root);
    }
    return {
        root,
        countEl: root.querySelector('[data-role="count"]'),
        tokensEl: root.querySelector('[data-role="tokens"]')
    };
}

function setBodyVisibilityFlag(visible) {
    if (typeof document === 'undefined' || !document.body) return;
    if (visible) {
        document.body.dataset.promptTokenStripVisible = 'true';
    } else {
        delete document.body.dataset.promptTokenStripVisible;
    }
}

function isDetailPanelSuppressingStripOnMobile() {
    if (typeof window === 'undefined' || typeof document === 'undefined' || !document.body) return false;
    if (!document.body.classList.contains('detail-mobile-focus')) return false;
    if (typeof window.matchMedia === 'function') {
        return window.matchMedia('(max-aspect-ratio: 1/1), (max-width: 880px)').matches;
    }
    return window.innerWidth <= 880 || window.innerHeight <= window.innerWidth;
}

export function initPromptTokenStrip({ onTokenClick = null } = {}) {
    if (typeof document === 'undefined') {
        return {
            update: () => {},
            dispose: () => {}
        };
    }

    const dom = buildDom();
    let lastSignature = '';
    let tokenEntries = [];
    let stripEnabled = appState.showPromptTokenStrip !== false;
    let bodyClassObserver = null;

    const updateVisibility = () => {
        const shouldShow = stripEnabled
            && tokenEntries.length > 0
            && !isDetailPanelSuppressingStripOnMobile();
        dom.root.dataset.visible = shouldShow ? 'true' : 'false';
        setBodyVisibilityFlag(shouldShow);
    };

    const setEnabled = (enabled) => {
        stripEnabled = !!enabled;
        appState.showPromptTokenStrip = stripEnabled;
        updateVisibility();
    };

    const handleTokenClick = (event) => {
        const target = event?.target?.closest?.('[data-token-entry-index]');
        if (!target || !dom.tokensEl.contains(target)) return;
        const rawIndex = Number(target.dataset.tokenEntryIndex);
        if (!Number.isFinite(rawIndex)) return;
        const entry = tokenEntries[Math.floor(rawIndex)];
        if (!entry) return;
        if (typeof onTokenClick === 'function') {
            onTokenClick({ ...entry });
        }
    };

    const handleVisibilityChanged = (event) => {
        const detail = event?.detail || null;
        const enabled = (detail && typeof detail.enabled === 'boolean')
            ? detail.enabled
            : (appState.showPromptTokenStrip !== false);
        setEnabled(enabled);
    };

    dom.tokensEl.addEventListener('click', handleTokenClick);
    if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
        window.addEventListener('promptTokenStripVisibilityChanged', handleVisibilityChanged);
    }
    if (typeof MutationObserver !== 'undefined' && document.body) {
        bodyClassObserver = new MutationObserver((mutationList) => {
            const classChanged = mutationList.some((mutation) => (
                mutation?.type === 'attributes' && mutation.attributeName === 'class'
            ));
            if (classChanged) {
                updateVisibility();
            }
        });
        bodyClassObserver.observe(document.body, {
            attributes: true,
            attributeFilter: ['class']
        });
    }

    const render = ({ tokenLabels = [], tokenIndices = null, tokenIds = null } = {}) => {
        const labels = Array.isArray(tokenLabels) ? tokenLabels : [];
        const indices = Array.isArray(tokenIndices) ? tokenIndices : [];
        const ids = Array.isArray(tokenIds) ? tokenIds : [];
        const entries = labels
            .map((tokenLabel, laneIndex) => {
                const label = (tokenLabel === null || tokenLabel === undefined)
                    ? ''
                    : String(tokenLabel);
                if (!label.length) return null;
                const tokenIndex = Number.isFinite(indices[laneIndex]) ? Math.floor(indices[laneIndex]) : null;
                const tokenId = Number.isFinite(ids[laneIndex]) ? Math.floor(ids[laneIndex]) : null;
                return {
                    laneIndex,
                    tokenIndex,
                    tokenId,
                    tokenLabel: label
                };
            })
            .filter(Boolean);

        if (!entries.length) {
            dom.tokensEl.innerHTML = '';
            if (dom.countEl) dom.countEl.textContent = '';
            tokenEntries = [];
            lastSignature = '';
            updateVisibility();
            return;
        }

        const signature = entries
            .map((entry) => `${entry.laneIndex}|${entry.tokenIndex}|${entry.tokenId}|${entry.tokenLabel}`)
            .join('\u241f');
        tokenEntries = entries;
        if (signature === lastSignature) {
            updateVisibility();
            return;
        }
        lastSignature = signature;

        const fragment = document.createDocumentFragment();
        entries.forEach((entry, index) => {
            const seed = resolveLogitTokenSeed({
                token_id: entry.tokenId,
                token: entry.tokenLabel
            }, index);
            const tokenEl = document.createElement('button');
            tokenEl.type = 'button';
            tokenEl.className = 'prompt-token-strip__token';
            tokenEl.style.setProperty('--token-color-border', getLogitTokenColorCss(seed, 0.92));
            tokenEl.style.setProperty('--token-color-fill', getLogitTokenColorCss(seed, 0.2));
            tokenEl.style.setProperty('--token-color-fill-hover', getLogitTokenColorCss(seed, 0.28));
            tokenEl.textContent = normalizeTokenText(entry.tokenLabel);
            tokenEl.dataset.tokenEntryIndex = String(index);
            fragment.appendChild(tokenEl);
        });

        dom.tokensEl.replaceChildren(fragment);
        if (dom.countEl) {
            dom.countEl.textContent = `${entries.length} token${entries.length === 1 ? '' : 's'}`;
        }
        updateVisibility();
    };

    return {
        update: ({ tokenLabels = [], tokenIndices = null, tokenIds = null } = {}) => {
            render({ tokenLabels, tokenIndices, tokenIds });
        },
        setEnabled,
        dispose: () => {
            setBodyVisibilityFlag(false);
            tokenEntries = [];
            dom.tokensEl.removeEventListener('click', handleTokenClick);
            if (typeof window !== 'undefined' && typeof window.removeEventListener === 'function') {
                window.removeEventListener('promptTokenStripVisibilityChanged', handleVisibilityChanged);
            }
            if (bodyClassObserver) {
                bodyClassObserver.disconnect();
                bodyClassObserver = null;
            }
            if (dom.root && dom.root.parentElement) {
                dom.root.parentElement.removeChild(dom.root);
            }
        }
    };
}

export default initPromptTokenStrip;
