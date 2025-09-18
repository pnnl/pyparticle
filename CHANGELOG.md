# Changelog

All notable changes to this project will be documented in this file.

The format is intended to follow Keep a Changelog and Semantic Versioning.

## [Unreleased]

- Prepare packaging metadata and CI for PyPI publishing
 - Add validation for PartMC and MAM4 builders to ensure `timestep` is an integer index; improved error messages when NetCDF files or indices are missing.
 - Note: PartMC and MAM4 population builders interpret `timestep` as an integer model output index and are not yet fully synchronized in their internal assumptions (e.g., species lists and wet/dry diameter handling). They do work in the AMBRS workflows, but expect further refinement to fully align mode/species semantics across builders.

## [0.1.0] - 2025-09-15

- Initial packaged release (metadata, setup files, README install docs)
