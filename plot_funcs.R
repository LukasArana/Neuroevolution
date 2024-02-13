plot_perf<- function(data_list, env){
  data <- data_list[[env]]
  print(data)
  print(data$neat)
  neat <- data$neat
  cma <- data$cma
  evals <- neat[[1]]$evaluations
  f_neat <- lapply(neat, function(x) x$f)
  f_cma <- lapply(cma, function(x) x$f)
  neat_m <- matrix(0, nrow = length(f_neat), ncol = length(f_neat[[1]]))
  for (j in seq(1, length(f_neat[[1]]))){
    for (i in seq(1, length(f_neat))){
      neat_m[i, j] <- f_neat[[i]][j] 
    }
  }
  
  cma_m <- matrix(0, nrow = length(f_cma), ncol = length(f_cma[[1]]))
  for (j in seq(1, length(f_cma[[1]]))){
    for (i in seq(1, length(f_cma))){
      cma_m[i, j] <- f_cma[[i]][j] 
    }
  }
  
  quantiles_neat <- array(dim = c(2, ncol(neat_m)))
  medians_neat <- array(dim = c(1, ncol(neat_m)))
  
  quantiles_cma <- array(dim = c(2, ncol(cma_m)))
  medians_cma <- array(dim = c(1, ncol(cma_m)))
  
  
  # Calculate and store quantiles for all columns
  for (i in 1:ncol(neat_m)) {
    quantiles <- quantile(neat_m[, i], probs = c(0.05, 0.95))
    median_ <- median(neat_m[, i])
    quantiles_neat[, i] <- quantiles
    medians_neat[, i] <- median_
  }
  
  
  for (i in 1:ncol(cma_m)) {
    quantiles <- quantile(cma_m[, i], probs = c(0.05, 0.95))
    median_ <- median(cma_m[, i])
    quantiles_cma[, i] <- quantiles
    medians_cma[, i] <- median_
  }
  
  # Create a ggplot
  p <- ggplot() 
  
  # Add ribbons and median lines for each column
  plot_data <- data.frame(x = evals,
                          median_neat = c(medians_neat),
                          q25_neat = quantiles_neat[1, ],
                          q75_neat = quantiles_neat[2, ],
                          median_cma = c(medians_cma),
                          q25_cma = quantiles_cma[1, ],
                          q75_cma = quantiles_cma[2, ],
                          max = rep(lims[[env]][1], times = length(medians_cma)))
  viridis_pal <- viridis(6)
  
  p <- ggplot(plot_data, aes(x = x)) +
    geom_line(aes(y = max, color = "Max"), linetype = "dotted", size = 1, show.legend = TRUE) +
    geom_ribbon(aes(ymin = q25_cma, ymax = q75_cma, fill = "%5 - %95 CMA"), alpha = 0.3, show.legend = TRUE) +
    geom_ribbon(aes(ymin = q25_neat, ymax = q75_neat, fill = "%5 - %95 NEAT"), alpha = 0.3, show.legend = TRUE) +
    geom_line(aes(y = median_neat, color = "Median NEAT"), size = 1, show.legend = TRUE) +
    geom_line(aes(y = median_cma, color = "Median CMA"), size = 1, show.legend = TRUE) +
    labs(x = "Evaluations", y = "Reward") +
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
    scale_x_continuous(breaks = seq(min(plot_data$x) -1, max(plot_data$x) + 1, by = 1000)) +
    scale_y_continuous(breaks = seq(as.integer(min(plot_data$q25_neat)), min(plot_data$max) + 100, by = 100)) + 
    scale_linetype_manual(name = "Legend", values = "dotted", guide = "none")
  print(p)
}


plot_chars<- function(data_list, env){
  data <- data_list[[env]]
  neat <- data$neat
  cma <- data$cma
  evals <- neat[[1]]$evaluations
  f_neat <- lapply(neat, function(x) x$f)
  f_cma <- lapply(cma, function(x) x$f)
  
  }