import { stripTypeScriptTypes } from 'node:module';
import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
await mkdir(path.join(root, 'dist'), { recursive: true });
for (const name of await readdir(path.join(root, 'src'))) {
  if (!name.endsWith('.ts')) continue;
  const source = await readFile(path.join(root, 'src', name), 'utf8');
  const output = stripTypeScriptTypes(source, { mode: 'strip', sourceUrl: name });
  await writeFile(path.join(root, 'dist', name.replace(/\.ts$/, '.js')), output, 'utf8');
}
console.log('Built portable Living Home ESM plugin.');
