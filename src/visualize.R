library(ggplot2)
library(viridis)
library(ggsci)
setwd("/home/walle/Desktop/TFG/nofn")

source("plot_funcs.R")


lims = list("pendulum" = c(-106.9528, 100), # Best result on page
            "mountain_car_cont" = c(90, 100),
            "mountain_car" = c(-110, 100),
            "lunar" = c(200, 100),
            "cart" = c(500, 100),
            "acrobot" = c(-42.37 + 4.83,100))# Best result ,n page
envname_list <- c("acrobot", "cart", "lunar", "mountain_car", "mountain_car_cont", "pendulum")
envname_list <- c("cart", "mountain_car", "mountain_car_cont", "pendulum")
alg_list <- c("cma", "neat")
seed <- 3
n <- 20
data_perf <- list()
data_arch <- list()
for (alg in alg_list){
  data_arch_list <- list()
  data_alg_list <- list()
  for (envname in envname_list){
    # Loop through each file and read it as a csv
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


f <- function(data_arch, env, char){
  data <- data_arch[[env]]
  evals <- data[[1]]$evaluations
  envs <- c("pendulum", "mountain_car_cont", "mountain_car", "lunar", "cart", "acrobot")
  cma_data <- data.frame("weights" = c(51, 41, 63, 143, 72,  103), 
                         "neurons" = c(3 + 1 + 10,
                                       2 + 1 + 10,
                                       2 + 3 + 10,
                                       8 + 4 + 10,
                                       4 + 2 + 10,
                                       10 + 3 + 6))
  
  if (char == "neurons"){
    char_cma <- cma_data$neurons[match(env, envs)]
    char_neat <- lapply(data, function(x) x$neurons)
  } else if (char == "weight"){
    char_cma <- cma_data$weights[match(env, envs)]
    char_neat <- lapply(data, function(x) x$weight)
  }else if (char == "fitness"){
    char_neat <- lapply(neat, function(x) x$fitness)
  }
  
  
  char_m <- matrix(0, nrow = length(data), ncol = length(char_neat[[1]]))
  for (j in seq(1, length(char_neat[[1]]))){
    for (i in seq(1, length(char_neat))){
      char_m[i, j] <- char_neat[[i]][j] 
    }
  }
  quantiles_n <- array(dim = c(2, ncol(char_m)))
  medians_n <- array(dim = c(1, ncol(char_m)))
  
  # Calculate and store quantiles for all columns
  for (i in 1:ncol(char_m)) {
    quantiles <- quantile(char_m[, i], probs = c(0, 1))
    median_ <- median(char_m[, i])
    quantiles_n[, i] <- quantiles
    medians_n[, i] <- median_
  }
  
  p <- ggplot() 
  evals <- seq(1, 20000, 1000)
  # Add ribbons and median lines for each column
  plot_data <- data.frame(x = evals,
                          median_neat = c(medians_n),
                          q25_neat = quantiles_n[1, ],
                          q75_neat = quantiles_n[2, ])
  
  viridis_pal <- viridis(6)
  
  p <- ggplot(plot_data, aes(x = x)) +
    geom_ribbon(aes(ymin = q25_neat, ymax = q75_neat, fill = "Min - Max NEAT"), alpha = 0.3, show.legend = TRUE) +
    geom_line(aes(y = median_neat, color = "Median NEAT"), size = 1, show.legend = TRUE) +
    geom_line(aes(y = rep(char_cma, length(median_neat)), color = "CMA"), size = 1, show.legend = TRUE) +
    labs(x = "Evaluations", y = char) +
    #scale_fill_manual(values = viridis_pal, name = NULL) +
    #scale_color_manual(values = viridis_pal, name = NULL) +
    #scale_fill_manual(values = pal_npg("nrc",  alpha = 0.6)(5), name = NULL) +  # Using ggsci palette
    #scale_color_manual(values = pal_npg("nrc",  alpha = 0.6)(5),name = NULL) +  # Using ggsci palette
    theme(text = element_text(size = 14),  # Increase font size
          axis.title = element_text(face = "bold"),  # Bold axis titles
          legend.title = element_text(face = "bold"),  # Bold legend titles
          legend.text = element_text(size = 12),  # Increase legend font size
          legend.position = "bottom",  # Move legend to bottom
          panel.grid.major = element_blank(),  # Remove major grid lines
          panel.grid.minor = element_blank(),  # Remove minor grid lines
          panel.border = element_blank(),  # Remove panel border
          plot.title = element_text(hjust = 0.5)) +  # Center plot title
    theme(text = element_text(size = 12)) +# Print the plot
    scale_x_continuous(breaks = evals) +
    scale_y_continuous(breaks = seq(min(plot_data$q25_neat), min(plot_data$q75_neat) + 100, by = 5)) + 
    scale_linetype_manual(name = "Legend", values = "dotted", guide = "none")
  print(p)
}

f(data_arch, "lunar", "weight")

#plot_perf(data_perf, "lunar")



