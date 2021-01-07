# Packages
library(tidyverse)
library(gganimate)
library(cowplot)
library(data.table)
library(gridExtra)

# Set Working Directory
setwd("~/Box Sync/Big Data Bowl/Data")

# Data
week1 <- read_csv("week1.csv")
target_probs <- read_csv("target_probs.csv")
completion_probs <- read_csv("completion_probs.csv")
exp_yards_gained <- read_csv("exp_yards_gained.csv")
pred_epa <- read_csv("pred_epa.csv")


# Plot - Expected Target Probability
sample_target <- week1 %>% 
  filter(gameId == 2018090600 & playId == 75 & nflId %in% c(2495454,2533040,2543583,2555415,2559033) & frameId > 10 & frameId <= 35) %>%
  left_join(filter(target_probs,gameId == 2018090600 & playId == 75 & nflId %in% c(2495454,2533040,2543583,2555415,2559033)),by = c("gameId","playId","frameId","nflId"))

g1 <- ggplot(data = sample_target %>% mutate(x = 120 -x),aes(x=x,y=y,color = target_prob,group = nflId)) +
  geom_point(size = 4) +
  scale_colour_gradient2(low = "blue",
                             mid = "white",
                             high = "red",
                             midpoint = 0.5,
                             space = "Lab",
                             na.value = "grey50",
                             guide = "colourbar",
                             aesthetics = "colour") +
  geom_segment(x = 30,xend = 30,y = 0,yend = 52,color = "black") +
  annotate("text",x = 30.5,y = 45.5,label = "LOS",size = 5) +
  annotate("text",x = 40,y = 44,label = "Julio Jones",size = 5) +
  annotate("text",x = 38,y = 35,label = "Mohamed Sanu",size = 5) +
  annotate("text",x = 37.5,y = 22,label = "Austin Hooper",size = 5) +
  annotate("text",x = 26,y = 20.75,label = "Ricky Ortiz",size = 5) +
  annotate("text",x = 27,y = 29.5,label = "Devonta Freeman",size = 5) +
  ggtitle("3.1 - Target Probability") +
  xlab("") +
  ylab("") +
  labs(color='Target Prob') +
  theme_bw() +
  theme(plot.title = element_text(size = 18)) +
  #ylim(43,46) +
  #xlim(80,95) +
  coord_flip()

# Plot - Expected Completion Probability
sample_comp_prob <- week1 %>% 
  filter(gameId == 2018090600 & playId == 75 & nflId %in% c(2495454,2533040,2543583,2555415,2559033) & frameId > 10 & frameId <= 36) %>%
  left_join(filter(completion_probs,gameId ==  2018090600 & playId == 75 & nflId %in% c(2495454,2533040,2543583,2555415,2559033)),by = c("gameId","playId","frameId","nflId"))

g2 <- ggplot(data = sample_comp_prob %>% mutate(x = 120 - x),aes(x=x,y=y,color = completion_probs,group = nflId)) +
  geom_point(size = 4) +
  geom_line() +
  scale_colour_gradient2(low = "blue",
                         mid = "white",
                         high = "red",
                         midpoint = 0.5,
                         space = "Lab",
                         na.value = "grey50",
                         guide = "colourbar",
                         aesthetics = "colour",
                         limits = c(0,1)) +
  geom_segment(x = 30,xend = 30,y = 0,yend = 52,color = "black") +
  annotate("text",x = 30.5,y = 45.5,label = "LOS",size = 5) +
  annotate("text",x = 40,y = 44,label = "Julio Jones",size = 5) +
  annotate("text",x = 38,y = 35,label = "Mohamed Sanu",size = 5) +
  annotate("text",x = 37.5,y = 22,label = "Austin Hooper",size = 5) +
  annotate("text",x = 26,y = 20.75,label = "Ricky Ortiz",size = 5) +
  annotate("text",x = 27,y = 29.5,label = "Devonta Freeman",size = 5) +
  ggtitle("3.2 - Completion Probability") +
  xlab("") +
  ylab("") +
  labs(color='Comp Prob') +
  theme_bw() +
  theme(plot.title = element_text(size = 18)) +
  #ylim(43,46) +
  #xlim(80,95) +
  coord_flip()

# Plot - Expected Yards Gained
sample_exp_yards_gained <- week1 %>% 
  filter(gameId == 2018090600 & playId == 75 & nflId %in% c(2495454,2533040,2543583,2555415,2559033) & frameId > 10 & frameId <= 36) %>%
  left_join(filter(exp_yards_gained,gameId ==  2018090600 & playId == 75 & nflId %in% c(2495454,2533040,2543583,2555415,2559033)),by = c("gameId","playId","frameId","nflId"))  #%>%
  #left_join(filter(completion_probs,gameId ==  2018090600 & playId == 75 & nflId %in% c(2495454,2533040,2543583,2555415,2559033)),by = c("gameId","playId","frameId","nflId")) %>%
  #left_join(filter(target_probs,gameId == 2018090600 & playId == 75 & nflId %in% c(2495454,2533040,2543583,2555415,2559033)),by = c("gameId","playId","frameId","nflId")) %>%
  #mutate(exp_yards_gained = target_prob * exp_yards_gained * completion_probs)
  
g3 <- ggplot(data = sample_exp_yards_gained %>% mutate(x = 120 - x),aes(x=x,y=y,color = exp_yards_gained,group = nflId)) +
  geom_point(size = 4) +
  geom_line() +
  scale_colour_gradient2(low = "blue",
                         mid = "white",
                         high = "red",
                         midpoint = 12,
                         space = "Lab",
                         na.value = "grey50",
                         guide = "colourbar",
                         aesthetics = "colour",
                         limits = c(0,31)) +
  geom_segment(x = 30,xend = 30,y = 0,yend = 52,color = "black") +
  annotate("text",x = 30.5,y = 45.5,label = "LOS",size = 5) +
  annotate("text",x = 40,y = 44,label = "Julio Jones",size = 5) +
  annotate("text",x = 38,y = 35,label = "Mohamed Sanu",size = 5) +
  annotate("text",x = 37.5,y = 22,label = "Austin Hooper",size = 5) +
  annotate("text",x = 26,y = 20.75,label = "Ricky Ortiz",size = 5) +
  annotate("text",x = 27,y = 29.5,label = "Devonta Freeman",size = 5) +
  ggtitle("3.3 - Expected Yards Gained") +
  xlab("") +
  ylab("") +
  labs(color='Exp Yards Gained') +
  theme_bw() +
  theme(plot.title = element_text(size = 18)) +
  #ylim(43,46) +
  #xlim(80,95) +
  coord_flip()

# Plot - Pred EPA
sample_pred_epa <- week1 %>% 
  filter(gameId == 2018090600 & playId == 75 & nflId %in% c(2495454,2533040,2543583,2555415,2559033) & frameId > 10 & frameId <= 36) %>%
  left_join(filter(testEPA_,gameId ==  2018090600 & playId == 75 & nflId %in% c(2495454,2533040,2543583,2555415,2559033)),by = c("gameId","playId","frameId","nflId"))

g4 <- ggplot(data = sample_pred_epa %>% mutate(x = 120 - x),aes(x=x,y=y,color = pred_epa,group = nflId)) +
  geom_point(size = 4) +
  geom_line() +
  scale_colour_gradient2(low = "blue",
                         mid = "white",
                         high = "red",
                         midpoint = 0,
                         space = "Lab",
                         na.value = "grey50",
                         guide = "colourbar",
                         aesthetics = "colour",
                         limits = c(-2.5,2.5)) +
  geom_segment(x = 30,xend = 30,y = 0,yend = 52,color = "black") +
  annotate("text",x = 30.5,y = 45.5,label = "LOS",size = 5) +
  annotate("text",x = 40,y = 44,label = "Julio Jones",size = 5) +
  annotate("text",x = 38,y = 35,label = "Mohamed Sanu",size = 5) +
  annotate("text",x = 37.5,y = 22,label = "Austin Hooper",size = 5) +
  annotate("text",x = 26,y = 20.75,label = "Ricky Ortiz",size = 5) +
  annotate("text",x = 27,y = 29.5,label = "Devonta Freeman",size = 5) +
  ggtitle("3.4 - Predicted EPA") +
  xlab("") +
  ylab("") +
  labs(color='Pred EPA') +
  theme_bw() +
  theme(plot.title = element_text(size = 18)) +
  #ylim(43,46) +
  #xlim(80,95) +
  coord_flip()



#g1
#g2
#g3
#g4 

# Multi-plot
grid.arrange(g1,g2,g3,g4,ncol = 2, nrow = 2)
