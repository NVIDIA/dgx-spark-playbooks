# Contributing to DGX Spark Playbooks

Thank you for your interest in contributing to the DGX Spark Playbooks! This repository provides step-by-step guides for AI/ML workloads on NVIDIA DGX Spark devices, and we welcome contributions from the community.

## Table of Contents

- [Types of Contributions](#types-of-contributions)
- [Issue Tracking](#issue-tracking)
  - [Pull Requests](#pull-requests)
  - [Signing Your Work](#signing-your-work)
- [Playbook Guidelines](#playbook-guidelines)
- [License](#license)
- [Questions?](#questions)


## Types of Contributions

We welcome the following types of contributions:

### 🆕 New Playbooks
- Step-by-step guides for AI/ML frameworks not yet covered
- Optimization techniques for existing frameworks
- Integration guides between multiple tools
- Performance benchmarking tutorials
Refer to [community/PLAYBOOK-GUIDELINES.md](community/PLAYBOOK-GUIDELINES.md) for publishing guidelines.

### 📝 Documentation Improvements
- Corrections to existing playbooks
- Additional troubleshooting sections
- Enhanced examples and code snippets
- Better explanations of complex concepts

### 🐛 Bug Fixes
- Corrections to incorrect commands or configurations
- Updates for deprecated software versions
- Fixes to broken links or references

### 🔧 Other Ideas
- Create a discussion topic!

## Issue Tracking

We use GitHub Issues to track bugs, feature requests, and general discussions about the DGX Spark Playbooks.

### Before Opening an Issue

1. **Search existing issues** to avoid duplicates
2. **Test on the latest version** of DGX Spark software
3. **Gather system information** (DGX Spark version, hardware config, etc.)

### Bug Reports

When reporting bugs, please include:

```markdown
**Playbook**: [Name of the affected playbook]
**DGX Spark Version**: [e.g., 24.10]
**Hardware**: [GPU model, memory, etc.]
**Steps to Reproduce**:
1. Step one
2. Step two
3. ...

**Expected Behavior**: [What should happen]
**Actual Behavior**: [What actually happens]
**Error Messages**: [Full error messages and logs]
**Additional Context**: [Screenshots, related issues, etc.]
```

### Feature Requests

For new playbook requests or enhancements:

```markdown
**Feature Type**: [New Playbook / Enhancement / Integration]
**Framework/Tool**: [Name and version]
**Use Case**: [Why this would be valuable]
**Priority**: [High / Medium / Low]
**Additional Context**: [Links, examples, related work]
```

#### Pull Requests
Developer workflow for code contributions is as follows:

1. Developers must first [fork](https://help.github.com/en/articles/fork-a-repo) the [upstream](https://github.com/nvidia/dgx-spark-playbooks) DGX Spark Playbook repository.

2. Git clone the forked repository and push changes to the personal fork.

  ```bash
git clone https://github.com/YOUR_USERNAME/YOUR_FORK.git dgx-spark-playbooks
# Checkout the targeted branch and commit changes
# Push the commits to a branch on the fork (remote).
git push -u origin <local-branch>:<remote-branch>
  ```

3. Once the code changes are staged on the fork and ready for review, a [Pull Request](https://help.github.com/en/articles/about-pull-requests) (PR) can be [requested](https://help.github.com/en/articles/creating-a-pull-request) to merge the changes from a branch of the fork into a selected branch of upstream.
  * Exercise caution when selecting the source and target branches for the PR.
  * Creation of a PR creation kicks off the code review process.
  * Atleast one repository owner will be assigned for the review.
  * While under review, mark your PRs as work-in-progress by prefixing the PR title with [WIP].

4. Since there is no CI/CD process in place yet, the PR will be accepted and the corresponding issue closed only after adequate testing has been completed, manually, by the developer and/or repository owner reviewing the change.


#### Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1
    
    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129
    
    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```
    Developer's Certificate of Origin 1.1
    
    By making a contribution to this project, I certify that:
    
    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or
    
    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or
    
    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.
    
    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```

## Playbook Guidelines

For detailed information on creating high-quality playbooks, please refer to [community/PLAYBOOK-GUIDELINES.md](community/PLAYBOOK-GUIDELINES.md).

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project. See [LICENSE](LICENSE) for details.

## Questions?

If you have questions about contributing, please:
1. Check existing [GitHub Discussions](../../discussions)
2. Review this contributing guide thoroughly
3. Open a new discussion with the "Contributing" category
4. Tag maintainers if you need urgent assistance

Thank you for helping make DGX Spark more accessible to the AI/ML community! 🚀
