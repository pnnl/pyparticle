# PyParticle Documentation

← Back to Index

Welcome to the PyParticle documentation. PyParticle is a focused Python library for constructing aerosol particle populations, attaching species-level physical properties (including wavelength-dependent refractive indices), building per-particle optical objects (morphologies such as homogeneous spheres and core-shell) and aggregating per-particle optics to population-level optical coefficients (scattering, absorption, extinction, asymmetry).

This scaffold includes a Quickstart, API summary, user guides for populations/optics/species/visualization, and developer instructions for adding new factories. The docs are written for two audiences:

- Primary: users who will build populations and query optical coefficients.
- Secondary: developers who will extend the package by adding population factories or optics morphologies.

Tested with: see `environment.yml` in the project root (create a conda env using that file before running examples or tests).

Quick links

- Quickstart: `docs/quickstart.md`
- API summary: `docs/api_summary.md`
- User Guide (populations, optics, species, viz): `docs/user_guide/`
- Developer Guide (factories + tests): `docs/developer/`

If you need a narrower README per module later, we can add module-level READMEs; for now everything is centralized under `docs/` for discoverability.
