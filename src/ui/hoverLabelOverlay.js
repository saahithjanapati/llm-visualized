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

function createAttentionDetailRow(documentRef) {
    const row = documentRef.createElement('div');
    row.className = 'scene-hover-label__attention-row';

    const role = documentRef.createElement('span');
    role.className = 'scene-hover-label__attention-role';

    const chip = documentRef.createElement('span');
    chip.className = 'detail-subtitle-token-chip scene-hover-label__attention-chip';
    chip.setAttribute('aria-hidden', 'true');

    const position = documentRef.createElement('span');
    position.className = 'scene-hover-label__attention-position';
    position.setAttribute('aria-hidden', 'true');

    row.append(role, chip, position);
    return { row, role, chip, position };
}

function createAttentionMetricRow(documentRef) {
    const row = documentRef.createElement('div');
    row.className = 'scene-hover-label__attention-metric';

    const role = documentRef.createElement('span');
    role.className = 'scene-hover-label__attention-metric-role';

    const value = documentRef.createElement('span');
    value.className = 'scene-hover-label__attention-metric-value';

    row.append(role, value);
    return { row, role, value };
}

function renderAttentionDetailRow(rowParts, rowContext = null) {
    if (!rowParts) return false;

    const hasToken = !!rowContext && (
        typeof rowContext.tokenLabel === 'string' && rowContext.tokenLabel.length > 0
        || Number.isFinite(rowContext.tokenIndex)
    );
    setHidden(rowParts.row, !hasToken);
    if (!hasToken) {
        rowParts.role.textContent = '';
        rowParts.chip.textContent = '';
        rowParts.chip.removeAttribute('title');
        rowParts.position.textContent = '';
        return false;
    }

    rowParts.role.textContent = rowContext.roleLabel || 'Token';
    rowParts.chip.textContent = formatTokenChipDisplayText(
        rowContext.tokenLabel,
        rowContext.tokenIndex
    );
    rowParts.chip.title = rowContext.tokenLabel || '';
    applyPromptTokenChipColors(rowParts.chip, {
        tokenText: rowContext.tokenLabel,
        tokenIndex: rowContext.tokenIndex,
        tokenId: rowContext.tokenId
    });
    rowParts.position.textContent = rowContext.positionText
        ? `(${rowContext.positionText})`
        : '';
    return true;
}

function renderAttentionMetricRow(rowParts, metric = null) {
    if (!rowParts) return false;
    const roleLabel = typeof metric?.roleLabel === 'string' ? metric.roleLabel : '';
    const valueText = typeof metric?.valueText === 'string' ? metric.valueText : '';
    const visible = roleLabel.length > 0 && valueText.length > 0;
    setHidden(rowParts.row, !visible);
    if (!visible) {
        rowParts.role.textContent = '';
        rowParts.value.textContent = '';
        return false;
    }
    rowParts.role.textContent = roleLabel;
    rowParts.value.textContent = valueText;
    return true;
}

function setAttentionDetailOrder(content, subtitle, attentionDetails, subtitleFirst) {
    if (!content || !subtitle || !attentionDetails) return;
    // Attention score hovers read more naturally when head/layer context appears
    // before the source/target token pair block.
    if (subtitleFirst) {
        if (subtitle.nextSibling !== attentionDetails) {
            content.insertBefore(subtitle, attentionDetails);
        }
        return;
    }
    if (attentionDetails.nextSibling !== subtitle) {
        content.insertBefore(attentionDetails, subtitle);
    }
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
    separator.textContent = '•';
    const tokenChip = documentRef.createElement('span');
    tokenChip.className = 'detail-subtitle-token-chip scene-hover-label__token-chip';
    tokenChip.setAttribute('aria-hidden', 'true');
    const detailText = documentRef.createElement('span');
    detailText.className = 'scene-hover-label__detail-text';
    detailText.setAttribute('aria-hidden', 'true');
    const attentionDetails = documentRef.createElement('div');
    attentionDetails.className = 'scene-hover-label__attention-details';
    attentionDetails.setAttribute('aria-hidden', 'true');
    const sourceRow = createAttentionDetailRow(documentRef);
    const targetRow = createAttentionDetailRow(documentRef);
    const metricRows = Array.from({ length: 3 }, () => createAttentionMetricRow(documentRef));
    const attentionMetrics = documentRef.createElement('div');
    attentionMetrics.className = 'scene-hover-label__attention-metrics';
    attentionMetrics.setAttribute('aria-hidden', 'true');
    metricRows.forEach((rowParts) => attentionMetrics.appendChild(rowParts.row));
    attentionDetails.append(attentionMetrics, sourceRow.row, targetRow.row);
    const subtitle = documentRef.createElement('span');
    subtitle.className = 'scene-hover-label__subtitle';
    subtitle.setAttribute('aria-hidden', 'true');

    topRow.append(labelText, separator, tokenChip, detailText);
    content.append(topRow, attentionDetails, subtitle);
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
            setAttentionDetailOrder(content, subtitle, attentionDetails, false);
            labelText.textContent = '';
            tokenChip.textContent = '';
            tokenChip.removeAttribute('title');
            detailText.textContent = '';
            renderAttentionDetailRow(sourceRow, null);
            renderAttentionDetailRow(targetRow, null);
            metricRows.forEach((rowParts) => renderAttentionMetricRow(rowParts, null));
            setHidden(attentionMetrics, true);
            subtitle.textContent = '';
            setHidden(labelText, true);
            setHidden(separator, true);
            setHidden(tokenChip, true);
            setHidden(detailText, true);
            setHidden(attentionDetails, true);
            setHidden(subtitle, true);
            return false;
        }

        const showDetail = !!detailContext;
        const showTokenChip = detailContext?.detailKind === 'token-chip';
        const showDetailText = detailContext?.detailKind === 'position-text';
        const showAttentionDetails = detailContext?.detailKind === 'attention-token-pair';
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
        setAttentionDetailOrder(content, subtitle, attentionDetails, showAttentionDetails);
        setHidden(labelText, !showPrimaryLabel);
        setHidden(separator, !showDetail || !showPrimaryLabel || showAttentionDetails);
        setHidden(tokenChip, !showTokenChip);
        setHidden(detailText, !showDetailText);
        setHidden(attentionDetails, !showAttentionDetails);
        setHidden(subtitle, !showSubtitle);

        if (!showDetail) {
            tokenChip.textContent = '';
            tokenChip.removeAttribute('title');
            detailText.textContent = '';
            renderAttentionDetailRow(sourceRow, null);
            renderAttentionDetailRow(targetRow, null);
            metricRows.forEach((rowParts) => renderAttentionMetricRow(rowParts, null));
            setHidden(attentionMetrics, true);
            return true;
        }

        if (showAttentionDetails) {
            tokenChip.textContent = '';
            tokenChip.removeAttribute('title');
            detailText.textContent = '';
            const rows = Array.isArray(detailContext.attentionRows)
                ? detailContext.attentionRows
                : [];
            const metrics = Array.isArray(detailContext.attentionMetrics)
                ? detailContext.attentionMetrics
                : [];
            const sourceVisible = renderAttentionDetailRow(sourceRow, rows[0] || null);
            const targetVisible = renderAttentionDetailRow(targetRow, rows[1] || null);
            let metricsVisible = false;
            metricRows.forEach((rowParts, index) => {
                metricsVisible = renderAttentionMetricRow(rowParts, metrics[index] || null) || metricsVisible;
            });
            setHidden(attentionMetrics, !metricsVisible);
            setHidden(attentionDetails, !(sourceVisible || targetVisible || metricsVisible));
            return true;
        }

        if (showDetailText) {
            tokenChip.textContent = '';
            tokenChip.removeAttribute('title');
            detailText.textContent = detailContext.detailText || '';
            renderAttentionDetailRow(sourceRow, null);
            renderAttentionDetailRow(targetRow, null);
            metricRows.forEach((rowParts) => renderAttentionMetricRow(rowParts, null));
            setHidden(attentionMetrics, true);
            return true;
        }

        detailText.textContent = '';
        renderAttentionDetailRow(sourceRow, null);
        renderAttentionDetailRow(targetRow, null);
        metricRows.forEach((rowParts) => renderAttentionMetricRow(rowParts, null));
        setHidden(attentionMetrics, true);
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
        move({
            clientX = 0,
            clientY = 0
        } = {}) {
            if (root.style.display === 'none') return false;
            root.style.left = `${Math.round((Number(clientX) || 0) + offsetX)}px`;
            root.style.top = `${Math.round((Number(clientY) || 0) + offsetY)}px`;
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
