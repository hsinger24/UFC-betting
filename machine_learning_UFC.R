########## Setting working directory and imports ##########
setwd("~/Desktop/UFC-betting")
library(MASS)
library(dplyr)
library(caret)
library(reshape2)
library(fastDummies)
library(deepnet)

########## Importing data and splitting test and train ##########

# Reading in data
data = read.csv('mma_data.csv')
results = c(0,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,-1,1,1,1,1,-1,1,0,0)
data[, 'result'] = results
data = data[!(data$result==-1),]
