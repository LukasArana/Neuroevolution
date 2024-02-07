setwd("/home/walle/Desktop/TFG/nofn")
file <- 'results/data/final/acrobot/cma_acrobot_2_0.txt'
data_initial <- read.csv(file)
envname_list <- c("acrobot", "cart", "lunar", "mountain_car", "mountain_car_cont", "pendulum")
alg_list <- c("cma", "neat")
seed <- 2
n <- 20
data_list <- list()
for (alg in alg_list){
  data_alg_list <- list()
  for (envname in envname_list){
    # Loop through each file and read it as a csv
    for (i in 0:(n -1)) {
      file_name <- paste0("results/data/final/", envname, "/", alg, "_", envname, "_", seed, "_", i, ".txt")
      csv <- read.csv(file_name, header = TRUE)  # Assuming the files are in the working directory
      data_alg_list[[i + 1]] <- csv
    }
    data_list[[envname]][[alg]] <- data_alg_list
  }
}