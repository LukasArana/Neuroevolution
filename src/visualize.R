library("ggsci")
library("ggplot2")
library("gridExtra")
library(viridis)
library(ggsci)
setwd("/home/walle/Desktop/TFG/nofn")

source("plot_funcs.R")


lims = list("pendulum" = c(-106.9528, 100), # Best result on page
            "mountain_car_cont" = c(90, 100),
            "mountain_car" = c(-110, 100),
            "lunar" = c(200, 100),
            "cart" = c(500, 100),
            "acrobot" = c(-42.37 + 4.83,100),
            "DoubleInvertedPendulum" = c(9000, 100),
            "InvertedPendulum" = c(1000, 1000))# Best result ,n page

envname_list <- c("acrobot", "cart", "lunar", "mountain_car", "mountain_car_cont", "pendulum", "InvertedPendulum", "DoubleInvertedPendulum")

alg_list <- c("cma", "neat")
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
      file_data <- paste0("results/data/prueba/", envname, "/", alg, "_", envname, "_", seed, "_", i, ".txt")
      csv <- read.csv(file_data, header = TRUE)  # Assuming the files are in the working directory
      data_alg_list[[i + 1]] <- csv

      if (alg == "neat"){
        file_arch <- paste0("results/data/prueba/", envname, "/", alg, "_", envname, "_", seed, "_", i, "_nn.csv")
        csv <- read.csv(file_arch, header = TRUE)  # Assuming the files are in the working directory
        data_arch_list[[i + 1]] <- csv
      }

    }
    data_perf[[envname]][[alg]] <- data_alg_list
    data_arch[[envname]] <- data_arch_list
  }
}

par(mfrow = c(2, 1))  # Set up a 2x1 grid for plots
env <- "DoubleInvertedPendulum"

perf <- plot_perf(data_perf, env)  + coord_cartesian(xlim = c(0, 18000))
#neurons <- plot_chars(data_arch, env, "neurons")
weights <- plot_chars(data_arch, env, "weight") + coord_cartesian(xlim = c(0, 18000)) 

grid.arrange(weights, perf, ncol = 1)


