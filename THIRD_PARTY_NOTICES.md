# Third-party dependencies and attribution

Dependencies are installed separately; their source distributions, wheels and binaries are not bundled here.
Their own licenses and bundled-component notices apply independently of the status of original research code.

| Dependency | Main project license |
|---|---|
| NumPy | BSD |
| pandas | BSD 3-Clause |
| SciPy | BSD 3-Clause; binary distributions include additional notices |
| scikit-image | Modified BSD |
| Matplotlib | PSF-based license |
| PyTorch | BSD-style license; binary distributions include additional notices |
| PyYAML | MIT |

These labels were checked against installed dependency metadata, not a complete legal audit of every transitive
binary component. Refer to the license files in the versions actually installed. No third-party notice found in
the selected source was removed. No pretrained third-party weights are included.

The focal-frequency loss file describes a lightweight implementation based on the ICCV 2021 method, rather than
a bundled copy of an upstream package. Fourier operators, U-Net and FiLM are methodological references, not
evidence of code copyright provenance. Scientific attribution is not a substitute for confirming rights in any
adapted implementation. That review remains open before licensing or public release.
