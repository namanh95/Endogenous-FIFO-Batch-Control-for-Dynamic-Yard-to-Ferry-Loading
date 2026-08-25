# Endogenous-FIFO-Batch-Control-for-Dynamic-Yard-to-Ferry-Loading
## Reproduce

```bash
python3 -m unittest discover -s tests -v
python3 code/run_paper_experiments.py --output results --workers 4
pdflatex Manuscript_MDPI_Special_Issue.tex
pdflatex Manuscript_MDPI_Special_Issue.tex
```

The experiments use 20 fixed seeds per regime--intensity cell, paired arrival paths, a common 14-unit execution-time budget, and stratified paired bootstrap intervals. Expected run time depends on hardware; HBC-MPC is intentionally more expensive than the myopic controller.
