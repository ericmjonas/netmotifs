


all: getbib paper.pdf

clean:
	rm paper.pdf paper.bcf paper.blg paper.aux paper.bbl paper.log paper.out paper.run.xml 


BIBHOME=${HOME}/Documents/
getbib:
	cp "${BIBHOME}/connectomics motif.bib" . 
	cp "${BIBHOME}/connectomics motif-relational models.bib" . 

techreport.pdf: techreport.tex 	
	pdflatex techreport.tex
	bibtex8 techreport
	pdflatex techreport.tex

paper.pdf: paper.tex 	
	pdflatex paper.tex
	biber paper
	pdflatex paper.tex

dropbox:
	cp -r * ~/Dropbox/jonaskording-connectomics 

debug: paper.pdf
	biber paper | grep "didn't" 
	grep "LaTeX Warning" paper.log

