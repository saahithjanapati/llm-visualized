import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(scriptDir, '..');
const sourceDir = resolve(projectRoot, 'public');
const runtimeCaptureDir = resolve(projectRoot, 'src/assets/captures');
const captureFiles = ['capture.json', 'capture_2.json'];

async function syncCaptureAsset(fileName) {
    const sourcePath = resolve(sourceDir, fileName);
    const targetPath = resolve(runtimeCaptureDir, fileName);
    const rawJson = await readFile(sourcePath, 'utf8');
    const minifiedJson = JSON.stringify(JSON.parse(rawJson));
    await writeFile(targetPath, `${minifiedJson}\n`, 'utf8');
}

await mkdir(runtimeCaptureDir, { recursive: true });
await Promise.all(captureFiles.map(syncCaptureAsset));
