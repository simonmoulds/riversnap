.PHONY: docs docs-clean examples examples-clean

EXAMPLES_DIR := examples
DOCS_DIR := docs
SPHINX_BUILDDIR := $(DOCS_DIR)/_build

examples:
	@echo "Executing notebooks in $(EXAMPLES_DIR)..."
	jupyter nbconvert --to notebook --execute $(EXAMPLES_DIR)/*.ipynb --inplace \
		--ExecutePreprocessor.timeout=600

docs:
	@echo "Building Sphinx docs..."
	$(MAKE) -C $(DOCS_DIR) html

docs-clean:
	rm -rf $(SPHINX_BUILDDIR)

examples-clean:
	@echo "Removing notebook outputs..."
	jupyter nbconvert --clear-output --inplace $(EXAMPLES_DIR)/*.ipynb

# One command to do everything (recommended)
docs-all: examples docs
