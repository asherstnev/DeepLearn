################################################################################################
# installation
# CPU-only!
if (1) {
  install.packages("drat", repos="https://cran.rstudio.com")
  drat:::addRepo("dmlc")
  install.packages("mxnet")
  
  install.packages("RnavGraphImageData")
  source("https://bioconductor.org/biocLite.R")
  biocLite("EBImage")
}