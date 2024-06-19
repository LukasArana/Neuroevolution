library("ggsci")
library("ggplot2")
library("gridExtra")
library(viridis)
library(ggsci)
library(cowplot)
#library(tidyverse)
setwd("/home/lufanta/Desktop/tfg/nofn")
library(scales)


lims = list("pendulum" = c(-106.9528, 100), # Best result on page
            "mountain_car_cont" = c(90, 100),
            "mountain_car" = c(-110, 100),
            "lunar" = c(200, 100),
            "cart" = c(500, 100),
            "acrobot" = c(-42.37 + 4.83,100),
            "DoubleInvertedPendulum" = c(9000, 100),
            "InvertedPendulum" = c(1000, 1000))# Best result ,n page

envname_list <- c("acrobot", "cart", "lunar", "mountain_car", "mountain_car_cont", "pendulum", "InvertedPendulum", "DoubleInvertedPendulum")
alg_list <- c("cma","cmaFC", "neat", "cma_neat", "FullyRandom", "newCMA")
alg_list <- c("cmaFC","neat", "FullyRandom")
data_dir <- "results/data/pruebaRandom/"
seed <- 3
n <- 20
data_perf <- list()
data_arch <- list()
for (alg in alg_list){
  data_arch_list <- list()
  data_alg_list <- list()
  for (envname in envname_list){
    # Loop through each file and read it as a csv.
    for (i in 0:(n -1)) {
      file_data <- paste0(data_dir, envname, "/", alg, "_", envname, "_", seed, "_", i, ".txt") # nolint: line_length_linter.
      csv <- read.csv(file_data, header = TRUE)  
      # Assuming the files are in the working directory
      data_alg_list[[i + 1]] <- csv

      if (alg == "neat"){
        file_arch <- paste0(data_dir, envname, "/", alg, "_", envname, "_", seed, "_", i, "_nn.csv")
        csv <- read.csv(file_arch, header = TRUE)  # Assuming the files are in the working directory
        data_arch_list[[i + 1]] <- csv
        data_arch[[envname]] <- data_arch_list
       # print(data_arch_list[[i+1]])
      }

    }
    data_perf[[envname]][[alg]] <- data_alg_list
  }
}

source("src/plot_funcs.R")
first = TRUE
par(mfrow = c(2, 1))  # Set up a 2x1 grid for plots
for (i in envname_list){
  legend = FALSE
  if (i == "acrobot"){
    legend = TRUE
  }
  if (first){
    perf <- plot_all(data_perf, i, alg_list, legend)  + coord_cartesian(xlim = c(0, 15000))
    neurons <- plot_chars(data_arch, i, "weight")+ coord_cartesian(xlim = c(0, 15000))
    p <- grid.arrange(neurons, perf, ncol = 1)
    folder = "results/first/"
    
  } else{
    p <- plot_all(data_perf, i, alg_list, legend)
    folder = "results/fourth/"
  }
  ggsave(paste0(folder, i, ".pdf"), p,width = 4, height = 5)
  
}


for (i in envname_list){
  legend = FALSE
  if (i == "acrobot"){
    legend = TRUE
  }
  
  p <- plot_all(data_perf, i, alg_list, legend)
  folder = "results/fourth/"
  
  ggsave(paste0(folder, i, ".pdf"), p,width = 4, height = 2.5)
}
'
med_quan <- function(data){
  n <- 20
  f_neat <- lapply(data, function(x) x$f)
  neat_m <- matrix(0, nrow = length(f_neat), ncol = n)
  for (j in seq(1, n)){
    for (i in seq(1, length(f_neat))){
      neat_m[i, j] <- f_neat[[i]][j] 
    }
  }
  quantiles_neat <- array(dim = c(2, ncol(neat_m)))
  medians_neat <- array(dim = c(1, ncol(neat_m)))
  
  # Calculate and store quantiles for all columns
  for (i in 1:ncol(neat_m)) {
    quantiles <- quantile(neat_m[, i], probs = c(0.2, 0.8))
    median_ <- median(neat_m[, i])
    quantiles_neat[, i] <- quantiles
    medians_neat[, i] <- median_
  }
  
  return( list("quan" = quantiles_neat, "med" = medians_neat))
}
'

