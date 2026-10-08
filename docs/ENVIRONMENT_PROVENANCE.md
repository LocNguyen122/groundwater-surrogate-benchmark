# Environment provenance and limits

CPU CI: Ubuntu 24.04, Python 3.11, CPU-only PyTorch 2.6.0, and pinned packages in
`requirements-cpu.txt`. Actions are pinned to immutable revisions and runtime package versions are printed.
This compatibility environment is distinct from the historical training runtime.

The manuscript records A100 40 GB training, RTX 3090 Ti inference, FP16 dynamic scaling for U-Net variants,
FP32 for the operator, optimizer and seed settings. Exact per-run Python/PyTorch/CUDA driver/build versions
are absent from the reviewed config records; no complete GPU lockfile or validated container is supplied.
Generic project environment instructions are not evidence of exact historical run versions.

Recorded simulator names are MODFLOW-2005, MT3D-USGS 1.1.0 and SGSIM through the generator utility. Generator
binary/build flags, complete solver/compiler/runtime provenance and authorized execution assets are not
included. Do not infer a validated end-to-end environment from this document or a successful CPU check.

The environment gap remains open for independent field-based reproduction. No dependencies, simulators or
GPU environments were installed or reconstructed to prepare this snapshot.
