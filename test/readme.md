Combining documentation and testing: develop in `marimo` to create docs but also integrate testing with `pytest`.


# Regression Testing
General workflow (to enable iteration):
1. Create a script/notebook (that produces results)
2. Pick some outputs that should not change (assumed to be correct).
Map them to a data structure that will be saved.
3. Run test program to initialize above data
4. Make changes to the code.
5. Run the test program again to assert data did not change. 
6. Commit code changes.

Implementation:
1. `cd` into [this dir](.).
2. Use `marimo` to edit notebooks. Edit 'library' code.
3. Run `pytest`. Regression testing data lives in [regression](./data/regression).

