import {
    resolveHoverLabelSubtitle,
    resolveHoverTokenContext
} from '../engine/coreHoverTokenContext.js';
import {
    applyPromptTokenChipColors,
    formatTokenChipDisplayText
} from '../utils/tokenChipStyleUtils.js';

const DEFAULT_TOKEN_CHIP_FONT_SIZE = '11px';

function createHoverLabelRoot(documentRef, {
    zIndex = 12,
    tokenChipFontSize = DEFAULT_TOKEN_CHIP_FONT_SIZE
} = {}) {
    const root = documentRef.createElement('div');
    root.className = 'scene-hover-label';
    Object.assign(root.style, {
        position: 'fixed',
        top: '0px',
        left: '0px',
        padding: '6px 10px',
        fontFamily: 'monospace',
        fontSize: '14px',
        color: '#fff',
        background: 'rgba(20,20,20,0.72)',
        backdropFilter: 'blur(6px)',
        WebkitBackdropFilter: 'blur(6px)',
        borderRadius: '8px',
        pointerEvents: 'none',
        zIndex: String(Math.max(1, Math.floor(zIndex))),
        whiteSpace: 'nowrap',
        display: 'none'
    });
    root.style.setProperty('--detail-font-chip', tokenChipFontSize);
    return root;
}

function setHidden(node, hidden) {
    if (!node) return;
    node.hidden = !!hidden;
}

export function createHoverLabelOverlay({
    documentRef = typeof document !== 'undefined' ? document : null,
    parent = documentRef?.body || null,
    offsetX = 12,
    offsetY = 12,
    zIndex = 12,
    tokenChipFontSize = DEFAULT_TOKEN_CHIP_FONT_SIZE
} = {}) {
    if (!documentRef || !parent) {
        return {
            element: null,
            show: () => false,
            hide: () => {},
            destroy: () => {}
        };
    }

    const root = createHoverLabelRoot(documentRef, {
        zIndex,
        tokenChipFontSize
    });
    const content = documentRef.createElement('div');
    content.className = 'scene-hover-label__content';
    const topRow = documentRef.createElement('div');
    topRow.className = 'scene-hover-label__top-row';
    const labelText = documentRef.createElement('span');
    labelText.className = 'scene-hover-label__text';
    const separator = documentRef.createElement('span');
    separator.className = 'scene-hover-label__separator';
    separator.textContent = '-';
    const tokenChip = documentRef.createElement('span');
    tokenChip.className = 'detail-subtitle-token-chip scene-hover-label__token-chip';
    tokenChip.setAttribute('aria-hidden', 'true');
    const detailText = documentRef.createElement('span');
    detailText.className = 'scene-hover-label__detail-text';
    detailText.setAttribute('aria-hidden', 'true');
    const subtitle = documentRef.createElement('span');
    subtitle.className = 'scene-hover-label__subtitle';
    subtitle.setAttribute('aria-hidden', 'true');

    topRow.append(labelText, separator, tokenChip, detailText);
    content.append(topRow, subtitle);
    root.appendChild(content);
    parent.appendChild(root);

    function render({
        label = '',
        info = null,
        object = null,
        activationSource = null
    } = {}) {
        const safeLabel = String(label || '');
        if (!safeLabel.length) return false;

        const detailContext = resolveHoverTokenContext({
            label: safeLabel,
            info,
            object,
            activationSource
        });
        const subtitleText = resolveHoverLabelSubtitle({
            label: safeLabel,
            info,
            object
        });

        if (detailContext?.suppressHoverLabel === true) {
            labelText.textContent = '';
            tokenChip.textContent = '';
            tokenChip.removeAttribute('title');
            detailText.textContent = '';
            subtitle.textContent = '';
            setHidden(labelText, true);
            setHidden(separator, true);
            setHidden(tokenChip, true);
            setHidden(detailText, true);
            setHidden(subtitle, true);
            return false;
        }

        const showDetail = !!detailContext;
        const showTokenChip = detailContext?.detailKind === 'token-chip';
        const showDetailText = detailContext?.detailKind === 'position-text';
        const showPrimaryLabel = detailContext?.showPrimaryLabel !== false;
        const showSubtitle = typeof subtitleText === 'string' && subtitleText.length > 0;
        const primaryLabelText = (
            typeof detailContext?.primaryLabelText === 'string'
            && detailContext.primaryLabelText.length
        )
            ? detailContext.primaryLabelText
            : safeLabel;

        labelText.textContent = showPrimaryLabel ? primaryLabelText : '';
        subtitle.textContent = showSubtitle ? subtitleText : '';
        setHidden(labelText, !showPrimaryLabel);
        setHidden(separator, !showDetail || !showPrimaryLabel);
        setHidden(tokenChip, !showTokenChip);
        setHidden(detailText, !showDetailText);
        setHidden(subtitle, !showSubtitle);

        if (!showDetail) {
            tokenChip.textContent = '';
            tokenChip.removeAttribute('title');
            detailText.textContent = '';
            return true;
        }

        if (showDetailText) {
            tokenChip.textContent = '';
            tokenChip.removeAttribute('title');
            detailText.textContent = detailContext.detailText || '';
            return true;
        }

        detailText.textContent = '';
        tokenChip.textContent = formatTokenChipDisplayText(
            detailContext.tokenLabel,
            detailContext.tokenIndex
        );
        tokenChip.title = detailContext.tokenLabel;
        applyPromptTokenChipColors(tokenChip, {
            tokenText: detailContext.tokenLabel,
            tokenIndex: detailContext.tokenIndex,
            tokenId: detailContext.tokenId
        });
        return true;
    }

    return {
        element: root,
        show({
            clientX = 0,
            clientY = 0,
            label = '',
            info = null,
            object = null,
            activationSource = null
        } = {}) {
            const rendered = render({
                label,
                info,
                object,
                activationSource
            });
            if (!rendered) {
                root.style.display = 'none';
                return false;
            }
            root.style.left = `${Math.round((Number(clientX) || 0) + offsetX)}px`;
            root.style.top = `${Math.round((Number(clientY) || 0) + offsetY)}px`;
            root.style.display = 'block';
            return true;
        },
        hide() {
            root.style.display = 'none';
        },
        destroy() {
            if (root.parentElement) {
                root.parentElement.removeChild(root);
            }
        }
    };
}
