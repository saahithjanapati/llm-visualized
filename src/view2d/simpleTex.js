const SIMPLE_TEX_SYMBOLS = Object.freeze({
    times: '×',
    cdot: '·',
    bullet: '•',
    alpha: 'α',
    beta: 'β',
    gamma: 'γ',
    delta: 'δ',
    epsilon: 'ε',
    theta: 'θ',
    lambda: 'λ',
    mu: 'μ',
    pi: 'π',
    sigma: 'σ',
    phi: 'φ',
    psi: 'ψ',
    omega: 'ω'
});

const SIMPLE_TEX_TEXT_COMMANDS = new Set([
    'mathrm',
    'mathsf',
    'mathbf',
    'mathit',
    'mathtt',
    'text',
    'operatorname'
]);

function resolveSimpleTexRunsInternal(input = '', variant = 'base', state = { nextSqrtGroupId: 0 }) {
    const safeInput = typeof input === 'string' ? input : '';
    const runs = [];
    let index = 0;
    let buffer = '';

    const flush = () => {
        if (!buffer.length) return;
        const lastRun = runs[runs.length - 1];
        if (lastRun?.variant === variant) {
            lastRun.text += buffer;
        } else {
            runs.push({
                variant,
                text: buffer
            });
        }
        buffer = '';
    };

    const readCommand = (startIndex) => {
        let cursor = startIndex;
        let name = '';
        while (cursor < safeInput.length && /[a-zA-Z]/.test(safeInput[cursor])) {
            name += safeInput[cursor];
            cursor += 1;
        }
        return {
            name,
            index: cursor
        };
    };

    const readGroup = (startIndex, nextVariant = variant) => {
        if (safeInput[startIndex] !== '{') {
            if (startIndex >= safeInput.length) {
                return {
                    runs: [],
                    index: startIndex
                };
            }
            return {
                runs: [{
                    variant: nextVariant,
                    text: safeInput[startIndex]
                }],
                index: startIndex + 1
            };
        }

        let depth = 0;
        let cursor = startIndex + 1;
        let content = '';
        while (cursor < safeInput.length) {
            const char = safeInput[cursor];
            if (char === '{') {
                depth += 1;
                content += char;
                cursor += 1;
                continue;
            }
            if (char === '}') {
                if (depth === 0) {
                    break;
                }
                depth -= 1;
                content += char;
                cursor += 1;
                continue;
            }
            content += char;
            cursor += 1;
        }

        return {
            runs: resolveSimpleTexRunsInternal(content, nextVariant, state),
            index: cursor < safeInput.length ? cursor + 1 : cursor
        };
    };

    while (index < safeInput.length) {
        const char = safeInput[index];

        if (char === '\\') {
            const { name, index: commandEnd } = readCommand(index + 1);
            if (!name.length) {
                buffer += '\\';
                index += 1;
                continue;
            }
            index = commandEnd;
            if (SIMPLE_TEX_TEXT_COMMANDS.has(name) && safeInput[index] === '{') {
                flush();
                const group = readGroup(index, variant);
                runs.push(...group.runs);
                index = group.index;
                continue;
            }
            if (name === 'sqrt') {
                flush();
                const group = readGroup(index, variant);
                const sqrtGroupId = state.nextSqrtGroupId;
                state.nextSqrtGroupId += 1;
                runs.push({
                    variant,
                    text: '√',
                    sqrtGroupId,
                    sqrtRole: 'radical'
                });
                runs.push(...group.runs.map((run) => ({
                    ...run,
                    sqrtGroupId,
                    sqrtRole: 'radicand'
                })));
                index = group.index;
                continue;
            }
            buffer += SIMPLE_TEX_SYMBOLS[name] || name;
            continue;
        }

        if (char === '_') {
            flush();
            const nextVariant = 'subscript';
            const group = safeInput[index + 1] === '{'
                ? readGroup(index + 1, nextVariant)
                : readGroup(index + 1, nextVariant);
            runs.push(...group.runs);
            index = group.index;
            continue;
        }

        if (char === '^') {
            flush();
            const nextVariant = 'superscript';
            const group = safeInput[index + 1] === '{'
                ? readGroup(index + 1, nextVariant)
                : readGroup(index + 1, nextVariant);
            runs.push(...group.runs);
            index = group.index;
            continue;
        }

        if (char === '{' || char === '}') {
            index += 1;
            continue;
        }

        if (char === '~') {
            buffer += ' ';
            index += 1;
            continue;
        }

        buffer += char;
        index += 1;
    }

    flush();
    return runs.filter((run) => typeof run.text === 'string' && run.text.length);
}

function resolveSimpleTexRuns(input = '', variant = 'base') {
    return resolveSimpleTexRunsInternal(input, variant, { nextSqrtGroupId: 0 });
}

export function hasSimpleTexMarkup(input = '') {
    return typeof input === 'string' && /[\\_^{}]/.test(input);
}

export function resolveSimpleTexPlainText(input = '') {
    const safeInput = typeof input === 'string' ? input : '';
    if (!safeInput.length) return '';
    return resolveSimpleTexRuns(safeInput)
        .map((run) => run.text)
        .join('')
        .replace(/\s+/g, ' ')
        .trim();
}

export function drawSimpleTex(ctx, input = '', {
    x = 0,
    y = 0,
    fontSize = 12,
    fontWeight = 500,
    fontFamily = 'ui-monospace, SFMono-Regular, Menlo, monospace',
    color = '#fff',
    textAlign = 'center',
    subscriptScale = 0.72,
    superscriptScale = 0.72,
    subscriptOffset = 0.3,
    superscriptOffset = 0.34
} = {}) {
    if (!ctx) return;
    const safeInput = typeof input === 'string' ? input : '';
    if (!safeInput.length) return;

    const runs = resolveSimpleTexRuns(safeInput);
    if (!runs.length) return;

    ctx.save();
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = color;

    const measuredRuns = runs.map((run) => {
        const scale = run.variant === 'subscript'
            ? subscriptScale
            : (run.variant === 'superscript' ? superscriptScale : 1);
        const runFontSize = Math.max(1, fontSize * scale);
        ctx.font = `${fontWeight} ${runFontSize}px ${fontFamily}`;
        return {
            ...run,
            fontSize: runFontSize,
            width: ctx.measureText(run.text).width
        };
    });

    const totalWidth = measuredRuns.reduce((sum, run) => sum + run.width, 0);
    let cursorX = x;
    if (textAlign === 'center') {
        cursorX -= totalWidth / 2;
    } else if (textAlign === 'right' || textAlign === 'end') {
        cursorX -= totalWidth;
    }

    const positionedRuns = measuredRuns.map((run) => {
        let baselineY = y;
        if (run.variant === 'subscript') {
            baselineY += fontSize * subscriptOffset;
        } else if (run.variant === 'superscript') {
            baselineY -= fontSize * superscriptOffset;
        }
        const positionedRun = {
            ...run,
            x: cursorX,
            baselineY
        };
        cursorX += run.width;
        return positionedRun;
    });

    positionedRuns.forEach((run) => {
        if (run.sqrtRole === 'radical') return;
        ctx.font = `${fontWeight} ${run.fontSize}px ${fontFamily}`;
        ctx.fillText(run.text, run.x, run.baselineY);
    });

    const sqrtGroups = new Map();
    positionedRuns.forEach((run) => {
        if (!Number.isFinite(run.sqrtGroupId)) return;
        const group = sqrtGroups.get(run.sqrtGroupId) || {
            radical: null,
            radicandRuns: []
        };
        if (run.sqrtRole === 'radical') {
            group.radical = run;
        } else if (run.sqrtRole === 'radicand') {
            group.radicandRuns.push(run);
        }
        sqrtGroups.set(run.sqrtGroupId, group);
    });

    sqrtGroups.forEach(({ radical, radicandRuns }) => {
        if (!radical || !radicandRuns.length) return;
        const firstRadicand = radicandRuns[0];
        const lastRadicand = radicandRuns[radicandRuns.length - 1];
        const lineStartX = Math.max(
            radical.x + (radical.width * 0.78),
            firstRadicand.x - (fontSize * 0.08)
        );
        const lineEndX = lastRadicand.x + lastRadicand.width + (fontSize * 0.04);
        const lineY = y - (fontSize * 0.44);
        const leadX = radical.x + (radical.width * 0.08);
        const hookX = radical.x + (radical.width * 0.28);
        const valleyX = radical.x + (radical.width * 0.48);
        const riseY = y - (fontSize * 0.16);
        const hookY = y + (fontSize * 0.24);
        const valleyY = y + (fontSize * 0.36);

        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(1, fontSize * 0.08);
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();
        ctx.moveTo(leadX, y + (fontSize * 0.02));
        ctx.lineTo(hookX, hookY);
        ctx.lineTo(valleyX, valleyY);
        ctx.lineTo(lineStartX, riseY);
        ctx.lineTo(lineStartX, lineY);
        ctx.lineTo(lineEndX, lineY);
        ctx.stroke();
        ctx.restore();
    });

    ctx.restore();
}
