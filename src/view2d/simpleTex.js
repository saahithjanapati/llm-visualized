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
    'text',
    'operatorname'
]);

function resolveSimpleTexRuns(input = '', variant = 'base') {
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
            runs: resolveSimpleTexRuns(content, nextVariant),
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

    measuredRuns.forEach((run) => {
        ctx.font = `${fontWeight} ${run.fontSize}px ${fontFamily}`;
        let baselineY = y;
        if (run.variant === 'subscript') {
            baselineY += fontSize * subscriptOffset;
        } else if (run.variant === 'superscript') {
            baselineY -= fontSize * superscriptOffset;
        }
        ctx.fillText(run.text, cursorX, baselineY);
        cursorX += run.width;
    });

    ctx.restore();
}
