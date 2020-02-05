.PHONY: quantex_interim_report_1.pdf all clean

all: quantex_interim_report_1.pdf

quantex_interim_report_1.pdf: quantex_interim_report_1.tex library.bib evaluations.tex design.tex
	latexmk -pdf quantex_interim_report_1.tex

clean:
	latexmk -pdf -CA
