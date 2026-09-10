# Example Test Playbook

> Example/template playbook used to validate the client-hw-playbooks pipeline.

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)

---

## Overview

## Basic idea

This is an **example playbook** that demonstrates the folder layout and the
metadata files required to publish a playbook in the API Catalog. Copy this
folder and replace the values to bootstrap a new playbook.

## What you'll accomplish

- Understand the required files: `conf.yaml`, `ux-conf.yaml`, and `ux/`.
- See how `tabs`, `resources`, and `cta` render in the Catalog UI.

## What to know before starting

- Basic familiarity with YAML.
- How the CI `.models` list maps each folder to a pipeline job.

## Prerequisites

- A folder under `nvidia/<playbook-name>/` matching the entry in `.gitlab-ci.yml`.

## Instructions

## Step 1. Copy this example

Duplicate `nvidia/playbook-test/` and rename it to your playbook name
(keep the `playbook-` prefix).

```bash
cp -r nvidia/playbook-test nvidia/playbook-<your-name>
```

## Step 2. Update metadata

Edit `conf.yaml` and `ux-conf.yaml` so `catalog_name`, `name`, and
`artifactName` match the new folder name.

## Step 3. Register in CI

Add `nvidia/playbook-<your-name>` to the `.models` list in `.gitlab-ci.yml`
so the pipeline picks it up.
