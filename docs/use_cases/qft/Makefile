.PHONY: quantex_qft_use_case.pdf all clean

all: quantex_qft_use_case.pdf

quantex_qft_use_case.pdf: quantex_qft_use_case.tex library.bib
	latexmk -pdf quantex_qft_use_case.tex

clean:
	latexmk -pdf -CA
