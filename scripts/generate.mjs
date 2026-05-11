#!/usr/bin/env node
// Reads nvidia/<name>/README.md + overrides/<name>.md → writes dist/skills/dgx-spark-<name>/SKILL.md
// Overrides provide hand-curated frontmatter description and extra body sections (Related, etc.).
// Generator-owned content is bounded by GENERATED markers and rewritten on every run; override
// content is appended verbatim and preserved across regenerations.

import { readdir, readFile, writeFile, mkdir, rm } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { join, dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const NVIDIA = join(REPO, 'nvidia')
const OVERRIDES = join(REPO, 'overrides')
const SKILLS_OUT = join(REPO, 'skills')

async function main() {
  // skills/ is entirely generator-owned — hand edits belong in overrides/
  await rm(SKILLS_OUT, { recursive: true, force: true })
  await mkdir(SKILLS_OUT, { recursive: true })

  const leafNames = (await readdir(NVIDIA, { withFileTypes: true }))
    .filter(d => d.isDirectory())
    .map(d => d.name)
    .sort()

  await writeIndex()
  for (const name of leafNames) await writeLeaf(name)

  console.log(`✓ Generated ${leafNames.length + 1} skills in ${SKILLS_OUT}`)
  console.log(`  • dgx-spark (index)`)
  console.log(`  • ${leafNames.length} leaves: ${leafNames.slice(0, 3).join(', ')}, ...`)
}

async function writeIndex() {
  const override = await readOverride('_index')
  if (!override) {
    throw new Error('overrides/_index.md is required (contains the catalog and relationship graph)')
  }
  const content = `---
name: dgx-spark
description: ${inlineDescription(override.fm.description)}
---

${override.body.trim()}
`
  await writeSkill('dgx-spark', content)
}

async function writeLeaf(name) {
  const readme = await readFile(join(NVIDIA, name, 'README.md'), 'utf8')
  const override = await readOverride(name)
  const description = override?.fm.description ?? fallbackDescription(name, readme)
  const generated = extractGeneratedBody(name, readme)

  const content = `---
name: dgx-spark-${name}
description: ${inlineDescription(description)}
---

<!-- GENERATED:BEGIN from nvidia/${name}/README.md -->
${generated}
<!-- GENERATED:END -->
${override?.body ? '\n' + override.body.trim() + '\n' : ''}`

  await writeSkill(`dgx-spark-${name}`, content)
}

function extractGeneratedBody(name, readme) {
  const title = firstMatch(readme, /^#\s+(.+)$/m) ?? name
  const tagline = firstMatch(readme, /^>\s+(.+)$/m) ?? ''
  const basicIdea = extractSection(readme, 'Basic idea') || extractSection(readme, 'Overview')
  const accomplish = extractSection(readme, "What you'll accomplish")
  const duration = firstMatch(readme, /\*\*Duration\*\*:\s*(.+)$/m)
  const risk = firstMatch(readme, /\*\*Risk level\*\*:\s*(.+)$/m)

  const parts = [`# ${title}`]
  if (tagline) parts.push(`> ${tagline}`)
  if (basicIdea) parts.push(basicIdea)
  if (accomplish) parts.push(`**Outcome**: ${accomplish}`)
  if (duration || risk) {
    const meta = []
    if (duration) meta.push(`Duration: ${duration}`)
    if (risk) meta.push(`Risk: ${risk}`)
    parts.push(meta.join(' · '))
  }
  parts.push(`**Full playbook**: \`${join(NVIDIA, name, 'README.md')}\``)
  return parts.join('\n\n')
}

function fallbackDescription(name, readme) {
  const tagline = firstMatch(readme, /^>\s+(.+)$/m)
  if (tagline) return `${tagline} — on NVIDIA DGX Spark. Use when setting up ${name} on Spark hardware.`
  return `Set up ${name} on NVIDIA DGX Spark. Use when the user wants to install or configure ${name} on Spark hardware.`
}

function inlineDescription(desc) {
  // YAML-safe single-line description. Our parser doesn't handle multi-line, so collapse + escape quotes.
  const collapsed = desc.replace(/\s+/g, ' ').trim()
  if (collapsed.includes(':') || collapsed.includes('#')) {
    return JSON.stringify(collapsed) // YAML accepts double-quoted strings
  }
  return collapsed
}

function firstMatch(s, re) {
  const m = s.match(re)
  return m ? m[1].trim() : null
}

function extractSection(md, heading) {
  const re = new RegExp(`##\\s+${heading}\\s*\\n+([\\s\\S]*?)(?=\\n##\\s|\\n---|$)`, 'i')
  const m = md.match(re)
  if (!m) return ''
  return m[1].trim().split('\n').slice(0, 8).join('\n').trim()
}

async function readOverride(name) {
  const path = join(OVERRIDES, `${name}.md`)
  if (!existsSync(path)) return null
  const text = await readFile(path, 'utf8')
  return parseFrontmatter(text)
}

function parseFrontmatter(md) {
  const m = md.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/)
  if (!m) return { fm: {}, body: md }
  const fm = {}
  for (const line of m[1].split('\n')) {
    const idx = line.indexOf(':')
    if (idx === -1) continue
    const key = line.slice(0, idx).trim()
    let value = line.slice(idx + 1).trim()
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1)
    }
    fm[key] = value
  }
  return { fm, body: m[2] }
}

async function writeSkill(name, content) {
  const dir = join(SKILLS_OUT, name)
  await mkdir(dir, { recursive: true })
  await writeFile(join(dir, 'SKILL.md'), content)
}

main().catch(err => {
  console.error('generate failed:', err)
  process.exit(1)
})
