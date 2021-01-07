# Packages
library(tidyverse)
library(xgboost)
library(data.table)

# Set Working Directory
setwd("~/Box Sync/Big Data Bowl/Data")
# Data - this will take a minute. It loads all weeks
files <- list.files(pattern = "week*")
# rearrange file names -- NOTE this is not necessary
# files <- files[c(6,18,20,22,24,26,28,30,32,8,10:16)]
# read into a temp list
temp <- lapply(files, fread, sep=",")
# combine list into dataframe
week <- rbindlist( temp )
# read in players db
players <- read.csv("players.csv", stringsAsFactors=FALSE)
# read in games db
games <- read.csv("games.csv", stringsAsFactors=FALSE)
# read in plays db
plays <- read.csv("plays.csv", stringsAsFactors=FALSE)
# Read in targeted receivers data
targetedReceiver <- read.csv("targetedReceiver.csv", stringsAsFactors=FALSE)
# Target Probs -- from Target_Prob_Model.R
target_probs <- read_csv("target_probs.csv")
# Completion Probs -- from Completion_Prob_Model.R
completion_probs <- read_csv("completion_probs.csv")

# Read and Join datasets with variables
files <- list.files(pattern = "*exp_comp_vars_db")
# Temp files into a list
temp <- lapply(files, fread, sep=",")
# combine list into dataframe
all_week_pass_db  <- rbindlist( temp )
# filter out duplicates
all_week_pass_db <- all_week_pass_db %>% distinct()
  

# Create a database filtered only on the frame in which the ball was released from the QBs hand, the receiver was targeted and pass was completed
all_week_yards_gained_db <- all_week_pass_db %>%
  # filter on ball release
  inner_join(week %>% distinct() %>% filter(event %in% c("pass_forward","pass_shovel")) %>% select(gameId,playId,frameId,nflId,playDirection),by = c("gameId","playId","frameId","nflId")) %>%
  # join targeted receiver db to get target on route
  left_join(targetedReceiver %>% distinct() %>% mutate(targeted = 1),by = c("gameId","playId","nflId"="targetNflId")) %>%
  # filter out none targeted receivers
  filter(!is.na(targeted)) %>%
  # join plays to get passResult, yards gained, yards to go, down, LOS
  left_join(plays %>% distinct() %>% select(gameId,playId,passResult,offensePlayResult,absoluteYardlineNumber,yardsToGo,down,quarter),by = c("gameId","playId")) %>%
  rename(yards_gained = offensePlayResult) %>%
  # filter on just completed passes
  filter(passResult == "C") %>%
  # adjust x coordinate
  mutate(x = case_when(playDirection == "left"~120 - x,
                       TRUE~x),
         x_Def1 = case_when(playDirection == "left"~120 - x_Def1,
                            TRUE~x_Def1),
         x_Def2 = case_when(playDirection == "left"~120 - x_Def2,
                            TRUE~x_Def2),
         x_Def3 = case_when(playDirection == "left"~120 - x_Def3,
                            TRUE~x_Def3),
         pass_x = case_when(playDirection == "left"~120 - pass_x,
                            TRUE~pass_x)) %>%
  # make binary variables and receiver distance from los adjusted for by play direction
  mutate(receiver_dist_line_of_scrim = case_when(playDirection == "left"~x - (120 - absoluteYardlineNumber),
                                                 playDirection == "right"~x - absoluteYardlineNumber),
         first_down = ifelse(down == 1,1,0),
         second_down = ifelse(down == 2,1,0),
         third_down = ifelse(down == 3,1,0),
         fourth_down = ifelse(down == 4,1,0),
         first_quarter = ifelse(quarter == 1,1,0),
         second_quarter = ifelse(quarter == 2,1,0),
         third_quarter = ifelse(quarter == 3,1,0),
         fourth_quarter = ifelse(quarter == 4,1,0),
         ot = ifelse(quarter == 5,1,0)) %>%
  select(-playDirection,-targeted,-completed,-passResult,-absoluteYardlineNumber,-down,-quarter)

# Add QB position, speed, orientation, and direction
all_week_yards_gained_db <- all_week_yards_gained_db %>%
  left_join(week %>% 
              distinct() %>%
              filter(position == "QB") %>% 
              group_by(gameId,playId,frameId) %>%
              filter(row_number() <= 1) %>%
              ungroup() %>%
              mutate(x = case_when(playDirection == "left"~120 - x,
                                   TRUE~x)) %>%
              select(gameId,playId,frameId,x,y,o,dir,s) %>% 
              rename(QB_x = x,
                     QB_y = y,
                     QB_s = s,
                     QB_o = o,
                     QB_dir = dir), by = c("gameId","playId","frameId"))

# Add target and completion probabilities
all_week_yards_gained_db <- all_week_yards_gained_db %>%
  left_join(target_probs %>% select(gameId,playId,frameId,nflId,target_prob),by = c("gameId","playId","frameId","nflId")) %>%
  left_join(completion_probs %>% select(gameId,playId,frameId,nflId,completion_probs),by = c("gameId","playId","frameId","nflId"))

trainExpYards_ <- all_week_yards_gained_db %>%
  arrange(gameId,playId,frameId) %>%
  filter(complete.cases(.))

trainExpYards <- all_week_yards_gained_db %>%
  arrange(gameId,playId,frameId) %>%
  select(-gameId,-playId,-frameId,-nflId) %>%
  filter(complete.cases(.)) %>%
  as.data.frame()

# rearrange variables 
trainExpYards <- trainExpYards[,c(38,41:49,39,40,1:37,50:ncol(trainExpYards))]

# Scale Variables but not binary variables
trainExpYards[,c(11:ncol(trainExpYards))] = trainExpYards[,c(11:ncol(trainExpYards))] %>% mutate_all(~(scale(.) %>% as.vector))

# XGBOOST Hypertune
best_param = list()
best_seednumber = 1234
best_rmse = Inf
best_rmse_index = 0
start = Sys.time()
for (iter in 1:2) {
  param <- list(objective = "reg:squarederror",
                eval.metric = "rmse",
                max_depth = sample(6:10, 1),
                eta = runif(1, .01, .3),
                gamma = runif(1, 0.0, 0.2), 
                subsample = runif(1, .6, .9),
                colsample_bytree = runif(1, .5, .8), 
                min_child_weight = sample(1:40, 1),
                max_delta_step = sample(1:10, 1)
  )
  cv.nround = 1000
  cv.nfold = 5
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  mdcv <- xgb.cv(data = model.matrix(yards_gained ~ .,data = trainExpYards),
                 label = as.vector(trainExpYards$yards_gained), 
                 params = param, 
                 nthread=6, 
                 nfold=cv.nfold, 
                 nrounds=cv.nround,
                 verbose = F, 
                 #early.stop.round=10, 
                 maximize=FALSE)
  
  min_rmse = min(mdcv$evaluation_log[, "test_rmse_mean"])
  min_rmse_index = which.min(mdcv$evaluation_log[, test_rmse_mean])
  
  if (min_rmse < best_rmse) {
    best_rmse = min_rmse
    best_rmse_index = min_rmse_index
    best_seednumber = seed.number
    best_param = param
  }
  
  print(iter)
  
}

#best_rmse 7.095303
# Mean  rmse = 104.8805
mean((trainExpYards$yards_gained - rep(mean(trainExpYards$yards_gained ),length(trainExpYards$yards_gained)))^2) 

# Optimal parameters from tuning
# best_seednumber = 5781
#best_param <- list(objective = "reg:squarederror",
#              eval.metric = "rmse",
#              max_depth = 9,
#              eta = 0.1367493,
#              gamma = 0.1616799, 
#              subsample = 0.649484,
#              colsample_bytree = 0.525788, 
#              min_child_weight = 33,
#              max_delta_step = 6
#)

# Train on all data with optimial best_param
mod_boost_trained_yards_gained <- xgboost(data = model.matrix(yards_gained ~ .,data = trainExpYards),
                                 label = as.vector(trainExpYards$yards_gained),
                                 params = best_param, 
                                 nthread=6,
                                 nround = 1000,
                                 verbose = F, 
                                 #early.stop.round=10, 
                                 maximize=FALSE)

# Variable Importance Matrix
importance_matrix <- xgb.importance(feature_names = colnames(trainExpYards), model = mod_boost_trained_yards_gained)
write.csv(importance_matrix,"yards_gained_importance_matrix.csv",row.names = F)

# ggplot of variable importance
gg <- xgb.ggplot.importance(importance_matrix, measure = "Gain",rel_to_first = F)
gg + ylab("Gain") +  theme(legend.position = "none") +xlab("") + theme(axis.text.y = element_text(size = 9))
y_boost_yards_gained = predict(mod_boost_trained_yards_gained,newdata = model.matrix(yards_gained ~ .,data = trainExpYards)) 

# Create Data Frame to Predict on all frames for all plays across all games
all_week_yards_gained_db_extra <- all_week_pass_db %>%
  inner_join(week %>% distinct() %>% select(gameId,playId,frameId,nflId,playDirection),by = c("gameId","playId","frameId","nflId")) %>%
  left_join(plays %>% distinct() %>% select(gameId,playId,absoluteYardlineNumber,yardsToGo,down,quarter),by = c("gameId","playId")) %>%
  mutate(yards_gained = 0) %>%
  mutate(x = case_when(playDirection == "left"~120 - x,
                       TRUE~x),
         x_Def1 = case_when(playDirection == "left"~120 - x_Def1,
                            TRUE~x_Def1),
         x_Def2 = case_when(playDirection == "left"~120 - x_Def2,
                            TRUE~x_Def2),
         x_Def3 = case_when(playDirection == "left"~120 - x_Def3,
                            TRUE~x_Def3),
         pass_x = case_when(playDirection == "left"~120 - pass_x,
                            TRUE~pass_x)) %>%
  # make binary variables and receiver distance from los adjusted for by play direction
  mutate(receiver_dist_line_of_scrim = case_when(playDirection == "left"~x - (120 - absoluteYardlineNumber),
                                                 playDirection == "right"~x - absoluteYardlineNumber),
         first_down = ifelse(down == 1,1,0),
         second_down = ifelse(down == 2,1,0),
         third_down = ifelse(down == 3,1,0),
         fourth_down = ifelse(down == 4,1,0),
         first_quarter = ifelse(quarter == 1,1,0),
         second_quarter = ifelse(quarter == 2,1,0),
         third_quarter = ifelse(quarter == 3,1,0),
         fourth_quarter = ifelse(quarter == 4,1,0),
         ot = ifelse(quarter == 5,1,0)) %>%
  select(-playDirection,-completed,-absoluteYardlineNumber,-down,-quarter)

# Add QB data
all_week_yards_gained_db_extra <- all_week_yards_gained_db_extra %>%
  left_join(week %>% 
              distinct() %>%
              filter(position == "QB") %>% 
              group_by(gameId,playId,frameId) %>%
              filter(row_number() <= 1) %>%
              ungroup() %>%
              select(gameId,playId,frameId,x,y,o,dir,s) %>% 
              rename(QB_x = x,
                     QB_y = y,
                     QB_s = s,
                     QB_o = o,
                     QB_dir = dir), by = c("gameId","playId","frameId"))

# Add Target and Completion Probability
all_week_yards_gained_db_extra <- all_week_yards_gained_db_extra %>%
  left_join(target_probs %>% select(gameId,playId,frameId,nflId,target_prob),by = c("gameId","playId","frameId","nflId")) %>%
  left_join(completion_probs %>% select(gameId,playId,frameId,nflId,completion_probs),by = c("gameId","playId","frameId","nflId"))

testExpYards_ <- all_week_yards_gained_db_extra %>%
  arrange(gameId,playId,frameId) %>%
  filter(complete.cases(.))

testExpYards <- all_week_yards_gained_db_extra %>%
  arrange(gameId,playId,frameId) %>%
  select(-gameId,-playId,-frameId,-nflId) %>%
  filter(complete.cases(.)) %>%
  as.data.frame()

testExpYards <-  testExpYards[,c(39,41:49,38,40,1:37,50:ncol(testExpYards))]

testExpYards[,c(11:ncol(testExpYards))] = testExpYards[,c(11:ncol(testExpYards))] %>% mutate_all(~(scale(.) %>% as.vector))

y_boost_test_yards_gained_atpass = predict(mod_boost_trained_yards_gained,newdata = model.matrix(yards_gained ~ .,data = testExpYards)) #predict

# Add to temp dataframe
testExpYards_$exp_yards_gained <- y_boost_test_yards_gained_atpass

# select only gameId, playId, frameId, nflId and exp_yards_gained
testExpYards_ <- testExpYards_ %>% select(gameId,playId,frameId,nflId,exp_yards_gained)

# Save results
write.csv(testExpYards_,"exp_yards_gained.csv",row.names = F)

