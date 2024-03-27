
names = c("pendulum" = "Pendulum",
          "mountain_car_cont" = "Mountain Car Continuous",
          "mountain_car" = "Mountain Car",
          "lunar" = "Lunar",
          "cart" = "Cart",
          "acrobot" = "Acrobot",
          "InvertedPendulum" = "Inverted Pendulum",
          "DoubleInvertedPendulum" = "Double Inverted Pendulum")

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

plot_perf<- function(data_list, env){
  data <- data_list[[env]]
  neat <- data$neat
  cma <- data$cma
  random <- data$random
  evals <- cma[[1]]$evaluations
  
  
  neat <- med_quan(data$neat)
  quantiles_neat <- neat$"quan"
  medians_neat <- neat$"med"
  cma <- med_quan(data$cma)
  quantiles_cma <- cma$"quan"
  medians_cma <- cma$"med"

  
  random <- med_quan(data$random)
  quantiles_random <- random$"quan"
  medians_random <- random$"med"
  
  
  quantiles <- cbind(quantiles_cma, quantiles_neat, quantiles_random)
  medians <- c(medians_cma, medians_neat, medians_random)
  #quantiles <- cbind(quantiles_cma)
  #medians <- c(medians_cma)
  
  print("CMA")
  print(quantiles_cma)
  print(medians_cma)
  df <- data.frame(x = evals,
                   medians= medians,
                   q25 = quantiles[1, ],
                   q75 = quantiles[2, ],
                   Algorithms =  c( rep("Cma", 20), rep("Neat", 20), rep("Random", 20)),
                   #Algorithms =  c( rep("Cma", 20)),
                   max = rep(lims[[env]][1], times = length(medians_cma) * 3))
  # Create a ggplot
  y_min = min(df$q25) - min(df$q25) %% 10
  y_max = min(df$max)
  p <- ggplot() 
  p <-  ggplot(df, aes(x = x, fill = Algorithms, color = Algorithms, linetype = Algorithms)) +
    geom_line(aes(y = medians), size = 0.75, show.legend = TRUE) +
    geom_line(aes(y = max),linetype = "longdash", alpha = 0.5, color = 'black', size = 0.75, show.legend = TRUE) +  
    geom_line(aes(y = max),linetype = "longdash", alpha = 0.5, color = 'black', size = 0.75, show.legend = TRUE) +  
    
    geom_ribbon(aes(ymin = q25, ymax = q75), alpha = 0.3)+
    labs(x = "Evaluations", y = "Reward") +
    scale_x_continuous(breaks = seq(min(df$x) - 1, max(df$x) , by = 2000)) +
    scale_y_continuous(breaks = seq(y_min, 
                                    y_max, 
                                    by = as.integer((y_max - y_min)/ 5)))+ 
    theme_classic()+
    #theme(legend.position="none")
    theme(legend.position = c(0.77,0.35), legend.key.size = unit(0.5, 'cm'), legend.title = element_text(size=9))
  legend_colors <- ggplot_build(p)$data[[1]]$fill

  return(p)  
}

plot_chars <- function(data_arch, env, char){
  data <- data_arch[[env]]
  envs <- c("pendulum", "mountain_car_cont", "mountain_car", "lunar", "cart", "acrobot", "InvertedPendulum", "DoubleInvertedPendulum")
  cma_data <- data.frame("Weights" = c(51, 41, 63, 143, 72,  103, 61,131 ), 
                         "Neurons" = c(3 + 1 + 10,
                                       2 + 1 + 10,
                                       2 + 3 + 10,
                                       8 + 4 + 10,
                                       4 + 2 + 10,
                                       10 + 3 + 6,
                                       10 + 4 + 1,
                                       10 + 1 + 1 ))
  
  if (char == "neurons"){
    char_cma <- cma_data$Neurons[match(env, envs)]
    char_neat <- lapply(data, function(x) x$neurons)
  } else if (char == "weight"){
    char_cma <- cma_data$Weights[match(env, envs)]
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
    quantiles <- quantile(char_m[, i], probs = c(0.1, 0.9))
    median_ <- median(char_m[, i])
    quantiles_n[, i] <- quantiles
    medians_n[, i] <- median_
  }
  
  p <- ggplot() 
  #print(evals)
  evals <- seq(1, 19001, 1000)
  # Add ribbons and median lines for each column
  plot_data <- data.frame(x = evals,
                          median_neat = c(medians_n),
                          q25_neat = quantiles_n[1, ],
                          q75_neat = quantiles_n[2, ],
                          cma = rep(char_cma, length(medians_n)),
                          Algorithms =  c(rep("Neat", 20)))
  viridis_pal <- viridis(6)
  maxy <- max(plot_data$cma)
  miny <- min(plot_data$q25_neat) - min(plot_data$q25_neat) %% 20
  p <- ggplot(plot_data, aes(x = x, fill=Algorithms, color = Algorithms, linetype = Algorithms)) +
    geom_ribbon(aes(ymin = q25_neat, ymax = q75_neat), alpha = 0.3, show.legend = FALSE, linetype = "dashed") +
    geom_line(aes(y = median_neat), alpha = 0.5, size = 1, show.legend = FALSE) +
    geom_line(aes(y = cma), size = 0.75, color = "#F8766D", alpha = 0.7,show.legend = FALSE) +
    labs(x = "Evaluations", y = "Number of weights") + 
    scale_x_continuous(breaks = seq(min(plot_data$x) - 1, max(plot_data$x) , by = 2000)) +
    scale_y_continuous(breaks = seq(miny, maxy, by = as.integer((maxy - miny)/4)), limits = c(min(plot_data$q25_neat) - min(plot_data$q25_neat) %% 20, max(plot_data$cma)+5)) + 
    theme_classic() + scale_fill_manual(values = c("#00BFC4")) + scale_color_manual(values = c("#00BFC4"))  
  # Print hexadecimal codes

  return(p)
}


plot_all<- function(data_list, env, algs){
  data <- data_list[[env]]
  #print(data)
  quantiles <- c()
  
  medians <- c()
  algsName <- c()
  for (i in algs){
    data_new <- med_quan(data[[i]])
    med <- data_new$"med"
    quan <- data_new$"quan"
    quantiles <- cbind(quantiles, quan)
    medians <- c(medians, med)
    algsName <- c(algsName, rep(i, 15))
  }
  evals <- seq(1, 15000, 1000)
  #Set color
  alg_list <- c("cma", "neat", "random", "cmaFC", "FullyRandom", "newCMA", "cma_neat")
  n_algorithms <- length(alg_list)
  palette1_named = setNames(object = scales::hue_pal()(n_algorithms), 
                            nm = alg_list)
  
  
  
  df <- data.frame(x = evals,
                 medians= medians,
                 q25 = quantiles[1, ],
                 q75 = quantiles[2, ],
                 Algorithms = algsName,
                 max = rep(lims[[env]][1], times = length(medians)))
  # Create a ggplot\\
  y_min = min(df$q25) - min(df$q25) %% 10
  y_max = min(df$max)
  p = ggplot(df, aes(x = x, color = Algorithms, linetype = Algorithms, fill = Algorithms)) +
    geom_line(aes(y = medians), size = 0.75, show.legend = TRUE) +
    geom_ribbon(aes(ymin = q25, ymax = q75), alpha = 0.3, show.legend = FALSE, linetype = "dashed") +
    geom_line(aes(y = max),linetype = "longdash", alpha = 0.5, color = 'black', size = 0.75, show.legend = TRUE) +  
    labs(x = "Evaluations", y = "Reward") +
    scale_x_continuous(breaks = seq(min(df$x) - 1, max(df$x) , by = 2000)) +
    scale_y_continuous(breaks = seq(y_min, 
                                    y_max, 
                                    by = as.integer((y_max - y_min)/ 5)))+ 
    theme_classic() +
    scale_color_manual(values = palette1_named)+
    scale_fill_manual(values = palette1_named) + 
    #theme(legend.position = c(0.77,0.35), legend.key.size = unit(0.5, 'cm'), legend.title = element_text(size=9))
    #theme(legend.position="none")
    theme()  +     guides(color = guide_legend(override.aes = list(linetype = 1)), 
                          linetype = "none")

    #theme(legend.position = "top", 
     #     legend.title = element_blank(),
      #    legend.background = element_rect(fill = 'transparent'), 
       #   legend.key.size = unit(0.5, 'cm'))  #print("a")
  
  #legend_colors <- ggplot_build(p)$data[[1]]$fill
  return(p)  
}
