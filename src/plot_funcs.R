
names = c("pendulum" = "Pendulum",
          "mountain_car_cont" = "Mountain Car Continuous",
          "mountain_car" = "Mountain Car",
          "lunar" = "Lunar",
          "cart" = "Cart",
          "acrobot" = "Acrobot",
          "InvertedPendulum" = "Inverted Pendulum",
          "DoubleInvertedPendulum" = "Double Inverted Pendulum")
alg_list_tot <- c("cmaHidden","cma", "neat", "cma-neat", "random", "cmaes")
alg_names <- c(
                "cmaFC" = "cma", 
                "neat" = "neat", 
                "cma"=  "cmaHidden", 
                "FullyRandom" = "random",
                "newCMA" = "cmaes", 
                "cma_neat" = "cma-neat")
n_algorithms <- length(alg_list_tot)
my_palette = setNames(object = scales::hue_pal()(n_algorithms), 
                      nm = alg_list_tot)


med_quan <- function(data){
  n <- 15
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

plot_perf<- function(data_list, env, legend){
  data <- data_list[[env]]
  print(data)
  neat <- data$neat
  cma <- data$cmaFC
  random <- data$random
  evals <- cma[[1]]$evaluations
  
  
  neat <- med_quan(data$neat)
  quantiles_neat <- neat$"quan"
  medians_neat <- neat$"med"
  print('neat')
  print(quantiles_neat)
  print(medians_neat)
  
  cma <- med_quan(data$cmaFC)
  quantiles_cma <- cma$"quan"
  medians_cma <- cma$"med"
  print('cmaFC')
  print(quantiles_cma)
  print(medians_cma)
  
  
  random <- med_quan(data$FullyRandom)
  quantiles_random <- random$"quan"
  medians_random <- random$"med"

  quantiles <- cbind(quantiles_cma, quantiles_neat, quantiles_random)
  medians <- c(medians_cma, medians_neat, medians_random)
  evals <- as.numeric(evals[1:16])

  df <- data.frame(x = evals,
                   medians= medians,
                   q25 = quantiles[1, ],
                   q75 = quantiles[2, ],
                   Algorithms =  c( rep("Cma", 16), rep("Neat", 16), rep("Random", 16)),
                   max = rep(lims[[env]][1], times = length(medians)))
  # Create a ggplot
  y_min = min(df$q25) - min(df$q25) %% 10
  y_max = min(df$max)
  p <- ggplot() 
  p <-  ggplot(df, aes(x = x, fill = Algorithms, color = Algorithms, linetype = Algorithms)) +
    geom_line(aes(y = medians), size = 0.75, show.legend = TRUE) +
    geom_line(aes(y = max),linetype = "longdash", alpha = 0.5, color = 'black', size = 0.75, show.legend = TRUE) +  

    geom_ribbon(aes(ymin = q25, ymax = q75), alpha = 0.3)+
    labs(x = "Evaluations", y = "Reward") +
    scale_x_continuous(breaks = seq(min(df$x) - 1 + 1000, max(df$x) , by = 2000)) +
    scale_y_continuous(breaks = seq(y_min, 
                                    y_max, 
                                    by = as.integer((y_max - y_min)/ 5)))+ 
    theme_classic()
  if (legend == TRUE){
    p <- p + theme(legend.background = NULL, legend.position = c(0.8,0.45), legend.key.size = unit(0.4, 'cm'), legend.title = element_text(size=9))
    legend_colors <- ggplot_build(p)$data[[1]]$fill
  }
  else{
      p <- p + theme(legend.position="none")
  }

  return(p)  
}

plot_chars <- function(data_arch, env, char){
  print(env)
  data <- data_arch[[env]]
  envs <- c("pendulum", "mountain_car_cont", "mountain_car", "lunar", "cart", "acrobot", "InvertedPendulum", "DoubleInvertedPendulum")
  cma_data <- data.frame("Neurons" = c(3, 2, 6, 32, 8,  30, 40,10), 
                         "Weights" = c(3 * 1 ,
                                       2 * 1,
                                       2 * 3,
                                       8 * 4,
                                       4 * 2,
                                       6 * 3,
                                       1 * 4,
                                       11 * 1))
  
  if (char == "neurons"){
    char_cma <- cma_data$Neurons[match(env, envs)]
    char_neat <- lapply(data, function(x) x$neurons)
  } else if (char == "weight"){
    char_cma <- cma_data$Weights[match(env, envs)]
    char_neat <- lapply(data, function(x) x$weight)
  }else if (char == "fitness"){
    char_neat <- lapply(neat, function(x) x$fitness)
  }
  print("length")
  print(length(char_neat[[1]]))
  char_m <- matrix(0, nrow = length(data), ncol = length(char_neat[[1]]))
  print(char_m)
  for (j in seq(1, length(char_neat[[1]]))){
    for (i in seq(1, length(char_neat))){
      char_m[i, j] <- char_neat[[i]][j] 
    }
  }
  print(char_m)
  quantiles_n <- array(dim = c(2, ncol(char_m)))
  medians_n <- array(dim = c(1, ncol(char_m)))
  
  # Calculate and store quantiles for all columns
  #print(char_m)
  for (i in 1:ncol(char_m)) {
    quantiles <- quantile(char_m[, i], probs = c(0.2, 0.8))
    median_ <- median(char_m[, i])
    quantiles_n[, i] <- quantiles
    medians_n[, i] <- median_
  }
  
  print(quantiles_n[1, ])
  print(quantiles_n[2, ])
  
  p <- ggplot() 
  evals <- seq(1, 20000, 1000)
  # Add ribbons and median lines for each column
  plot_data <- data.frame(x = evals,
                          median_neat = c(medians_n),
                          q25_neat = quantiles_n[1, ],
                          q75_neat = quantiles_n[2, ],
                          cma = rep(char_cma, length(medians_n)),
                          Algorithms =  c(rep("neat", 20)))
  viridis_pal <- viridis(6)
  maxy <- char_cma +15
  miny <- 0
  p <- ggplot(plot_data, aes(x = x, fill=Algorithms, color = Algorithms, linetype = Algorithms)) +
    geom_ribbon(aes(ymin = q25_neat, ymax = q75_neat), alpha = 0.3, show.legend = FALSE, linetype = "dashed") +
    geom_line(aes(y = median_neat), alpha = 0.5, size = 1, show.legend = FALSE) +
    geom_line(aes(y = cma), size = 0.75, color = my_palette[["cma"]], alpha = 0.7,show.legend = FALSE) +
    labs(x = "Evaluations", y = "Number of weights") + 
    scale_x_continuous(breaks = seq(min(plot_data$x) - 1 + 1000, max(plot_data$x) , by = 2000)) +
    scale_y_continuous(breaks = seq(miny, maxy , by = as.integer((maxy - miny)/4)), limits = c(miny, maxy)) + 
    theme_classic() + scale_fill_manual(values = my_palette) + scale_color_manual(values = my_palette)  
  # Print hexadecimal codes

  return(p)
}


plot_all<- function(data_list, env, algs, legend){
  data <- data_list[[env]]
  quantiles <- c()
  
  
  medians <- c()
  algsName <- c()
  for (i in algs){
    data_new <- med_quan(data[[i]])
    med <- data_new$"med"
    quan <- data_new$"quan"
    quantiles <- cbind(quantiles, quan)
    medians <- c(medians, med)
    algsName <- c(algsName, rep(alg_names[i], 15))
  }

  evals <- seq(1, 19001, 1000)
  evals <- as.numeric(evals[1:15])
  
  #Set color
  
  
  df <- data.frame(x = evals,
                 medians= medians,
                 q25 = quantiles[1, ],
                 q75 = quantiles[2, ],
                 Algorithms = algsName,
                 max = rep(lims[[env]][1], times = length(medians)))
  # Create a ggplot\\
  y_min = min(df$q25) - min(df$q25) %% 10
  y_max = min(df$max)
  p <- ggplot() 
  p <-  ggplot(df, aes(x = x, fill = Algorithms, color = Algorithms, linetype = Algorithms)) +
    geom_line(aes(y = medians), size = 0.75, show.legend = TRUE) +
    geom_line(aes(y = max),linetype = "longdash", alpha = 0.5, color = 'black', size = 0.75, show.legend = TRUE) +  
    
    geom_ribbon(aes(ymin = q25, ymax = q75), alpha = 0.3)+
    labs(x = "Evaluations", y = "Reward") +
    scale_x_continuous(breaks = seq(min(df$x) - 1 + 1000, max(df$x) , by = 2000)) +
    scale_y_continuous(breaks = seq(y_min, 
                                    y_max, 
                                    by = as.integer((y_max - y_min)/ 5)))+ 
    theme_classic()+ scale_color_manual(values = my_palette) + scale_fill_manual(values = my_palette)
    
  if (legend == TRUE){
    p <- p + theme(legend.background = NULL, legend.position = c(0.8,0.45), legend.key.size = unit(0.4, 'cm'), legend.title = element_text(size=9))
    legend_colors <- ggplot_build(p)$data[[1]]$fill
  }
  else{
    p <- p + theme(legend.position="none")
  }
  
    #theme(legend.position = "top", 
     #     legend.title = element_blank(),
      #    legend.background = element_rect(fill = 'transparent'), 
       #   legend.key.size = unit(0.5, 'cm'))  #print("a")
  
  #legend_colors <- ggplot_build(p)$data[[1]]$fill
  return(p)  
}


plot_box<- function(data_list, env, algs, legend){
  data <- data_list[[env]]
  quantiles <- c()
  evals <- c()
  f <- c()
  alg_list <- c()
  for (alg in algs){
    for (i in 1:20){
      evals <- c(evals, seq(1, 15000, 1000))
      f <- c(f, data[[alg[1]]][[i]]$f[1:15])
      alg <- c(alg,rep(alg, 15))
    }
  }
  df <- data.frame(x = evals,
                   y = data,
                   Algorithms = algs,
                   max = rep(lims[[env]][1], times = length(evals)))

  #Create a ggplot
  p <- ggplot() 
  p <-  ggplot(df, aes(x = x, fill = Algorithms, color = Algorithms, linetype = Algorithms)) +
    geom_boxplot(size = 0.75, show.legend = TRUE) +

    labs(x = "Evaluations", y = "Reward") +
    scale_y_continuous(breaks = seq(y_min, 
                                    y_max, 
                                    by = as.integer((y_max - y_min)/ 5)))+ 
    theme_classic()+ scale_color_manual(values = my_palette) + scale_fill_manual(values = my_palette)
  
  if (legend == TRUE){
    p <- p + theme(legend.background = NULL, legend.position = c(0.8,0.45), legend.key.size = unit(0.4, 'cm'), legend.title = element_text(size=9))
    legend_colors <- ggplot_build(p)$data[[1]]$fill
  }
  else{
    p <- p + theme(legend.position="none")
  }
  
  #theme(legend.position = "top", 
  #     legend.title = element_blank(),
  #    legend.background = element_rect(fill = 'transparent'), 
  #   legend.key.size = unit(0.5, 'cm'))  #print("a")
  
  #legend_colors <- ggplot_build(p)$data[[1]]$fill
  return(p)  
}
