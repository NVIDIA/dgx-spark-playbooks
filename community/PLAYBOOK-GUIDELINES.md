# Playbook Guidelines

This document provides detailed guidelines for creating high-quality playbooks for the DGX Spark Playbooks repository.

## Table of Contents

- [Structure Requirements](#structure-requirements)
- [README.md Template](#readmemd-template)
- [Content Standards](#content-standards)
- [Code and Script Guidelines](#code-and-script-guidelines)
- [Asset Guidelines](#asset-guidelines)
- [Testing Requirements](#testing-requirements)
- [Maintenance Guidelines](#maintenance-guidelines)

## Structure Requirements

Each playbook must follow this directory structure:

```
community/your-playbook-name/
├── README.md                 # Main playbook content
├── assets/                   # Supporting files
│   ├── images/              # Screenshots, diagrams
│   ├── configs/             # Configuration files
│   ├── scripts/             # Shell scripts, Python files
│   └── examples/            # Sample code, datasets
```

### Directory Naming Convention

- Use lowercase with hyphens for directory names
- Be descriptive but concise (e.g., `flux-finetuning`, `multi-agent-chatbot`)
- Avoid abbreviations unless they are widely recognized
- Match the primary technology or framework name

## README.md Template

Your playbook's `README.md` should include:

```markdown
# Your Playbook Title

Brief description of what this playbook accomplishes.

## Prerequisites

- Hardware requirements
- Software dependencies
- Required access/accounts

## Overview

Detailed explanation of the technology and use case.

## Instructions

### Step 1: Setup Environment
[Detailed instructions with commands]

### Step 2: Install Dependencies
[More instructions]

### Step 3: Configuration
[Configuration steps]

### Step 4: Running the Application
[Execution instructions]

## Verification

How to verify the setup is working correctly.

## Troubleshooting

Common issues and their solutions.

## Next Steps

- Additional resources
- Related playbooks
- Advanced configurations

## Resources

- Official documentation links
- Community forums
- Related repositories
```

## Content Standards

### ✅ Do's

- **Be comprehensive**: Include all necessary steps from start to finish
- **Test thoroughly**: Verify all commands work on a clean DGX Spark system
- **Use clear language**: Write for users with varying experience levels
- **Include verification**: Provide ways to confirm each step worked
- **Add troubleshooting**: Document common issues and solutions
- **Keep updated**: Ensure compatibility with latest software versions
- **Use consistent formatting**: Follow markdown best practices
- **Provide context**: Explain why each step is necessary
- **Include performance expectations**: Document expected completion times
- **Add security considerations**: Highlight any security implications

### ❌ Don'ts

- **Assume knowledge**: Don't skip "obvious" steps
- **Use relative paths**: Always use absolute paths where possible
- **Include secrets**: Never commit API keys, passwords, or tokens
- **Copy without testing**: All content must be validated on DGX Spark
- **Use deprecated methods**: Always use current best practices
- **Skip error handling**: Always include error scenarios
- **Use unclear pronouns**: Be specific about what "it" or "this" refers to
- **Mix different approaches**: Stick to one method per playbook

### Writing Style

- Use active voice ("Run the command" vs "The command should be run")
- Write in second person ("you will..." not "one should...")
- Use present tense for instructions
- Be consistent with terminology throughout
- Use numbered lists for sequential steps
- Use bullet points for non-sequential items

## Code and Script Guidelines

### General Requirements

- All scripts must be tested on DGX Spark
- Include comprehensive comments
- Use error handling and validation
- Follow language-specific style guides
- Include requirements/dependencies files
- Document any system modifications

## Asset Guidelines

### Images

- **Format**: Use PNG for screenshots, SVG for diagrams when possible
- **Resolution**: Maximum 1920px width, optimize for web viewing
- **Naming**: Use descriptive names (e.g., `login-screen.png`, `architecture-diagram.svg`)
- **Alt text**: Include meaningful descriptions for accessibility
- **Size**: Keep under 500KB when possible, compress if necessary

### Scripts

- Include shebang lines (`#!/bin/bash`, `#!/usr/bin/env python3`)
- Set appropriate execute permissions (`chmod +x script.sh`)
- Use `.sh` extension for shell scripts
- Include usage comments at the top
- Test on DGX Spark before committing

### Configuration Files

- Provide both template (`.template`) and example (`.example`) versions
- Remove sensitive information from examples
- Include validation scripts where applicable
- Document all required and optional parameters

### Example Code and Datasets

- Keep datasets small (< 10MB) or provide download instructions
- Include data licenses and attribution
- Provide sample inputs and expected outputs
- Test all example code before including

## Testing Requirements

### Validation Checklist

Before submitting a playbook, ensure:

- [ ] **Clean installation test**: Tested on a fresh DGX Spark system
- [ ] **All commands work**: Every command executes without errors
- [ ] **Links are valid**: All URLs and file references work
- [ ] **Prerequisites listed**: All dependencies documented
- [ ] **Verification steps**: Ways to confirm success at each stage
- [ ] **Troubleshooting tested**: Common issues documented and solutions verified
- [ ] **Performance noted**: Expected execution times documented
- [ ] **Resource usage**: Memory/disk requirements specified

### Test Environment

- Use a clean DGX Spark installation
- Document the specific DGX Spark software version used
- Test with minimal privileges (don't assume root access)
- Verify network connectivity requirements
- Test with different hardware configurations if applicable

### Documentation Testing

- [ ] Spelling and grammar check
- [ ] Markdown formatting validation
- [ ] Image loading verification
- [ ] Code syntax highlighting works
- [ ] Internal links function correctly
- [ ] External links are accessible

## Maintenance Guidelines

### Version Compatibility

- Test with latest DGX Spark software releases
- Update deprecated commands and methods
- Maintain compatibility matrices when applicable
- Document breaking changes clearly

### Content Updates

- Review playbooks quarterly for accuracy
- Update software versions and download links
- Refresh screenshots when UI changes
- Validate external resources remain available

### Community Feedback

- Respond to issues and questions promptly
- Incorporate user feedback and improvements
- Track common problems for FAQ updates
- Monitor performance and optimization opportunities

### Deprecation Process

When a playbook becomes outdated:

1. Add deprecation notice to the README
2. Provide migration path to newer alternatives
3. Set timeline for removal (minimum 6 months)
4. Update main repository README
5. Archive the playbook with clear historical context

## Quality Assurance

### Pre-Submission Review

Contributors should self-review using this checklist:

- [ ] Follows directory structure requirements
- [ ] Uses provided README template
- [ ] Meets all content standards
- [ ] Includes proper error handling
- [ ] Assets are optimized and properly named
- [ ] All testing requirements met
- [ ] Documentation is clear and complete

### Continuous Improvement

- Gather user feedback through GitHub issues
- Monitor playbook usage and success rates
- Update based on new DGX Spark features
- Benchmark performance improvements
- Share best practices across playbooks

---

For questions about these guidelines, please open a GitHub Discussion or contact the maintainers.
