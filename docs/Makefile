


all: getbib paper.pdf

clean:
	rm paper.pdf

BIBHOME=${HOME}/Documents/
getbib:
	cp "${BIBHOME}/connectomics motif.bib" . 
	cp "${BIBHOME}/connectomics motif-relational models.bib" . 

techreport.pdf: techreport.tex 	
	pdflatex techreport.tex
	biber techreport
	pdflatex techreport.tex

paper.pdf: paper.tex 	
	pdflatex paper.tex
	biber paper
	pdflatex paper.tex

dropbox:
	cp -r * ~/Dropbox/jonaskording-connectomics 
