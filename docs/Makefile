SPHINXOPTS    = -W --keep-going
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

SPHINXAPIDOC  = sphinx-apidoc
MISCDOCFLAGS  = -f -T
DOCOUTPUT     = md
SRCPROJECT    = ../src/ark
OUTPUTDIR     = _markdown
IGNORE        = ../tests/*

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXAPIDOC) $(MISCDOCFLAGS) -s $(DOCOUTPUT) -o $(OUTPUTDIR) $(SRCPROJECT) $(IGNORE)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	git checkout main -- _rtd/landing.md
