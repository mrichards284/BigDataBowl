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
all_week_epa_db <- all_week_pass_db %>%
  # join with week to filter on event being the time the ball was released
  inner_join(week %>% distinct() %>% filter(event %in% c("pass_forward","pass_shovel")) %>% select(gameId,playId,frameId,nflId,playDirection),by = c("gameId","playId","frameId","nflId")) %>%
  # join targeted receiver db
  left_join(targetedReceiver %>% distinct() %>% mutate(targeted = 1),by = c("gameId","playId","nflId"="targetNflId")) %>%
  # filter only on targeted receivers
  filter(!is.na(targeted)) %>%
  # join plays db to get epa for training
  left_join(plays %>% distinct() %>% select(gameId,playId,epa,absoluteYardlineNumber,yardsToGo,down,quarter),by = c("gameId","playId")) %>%
  # adjust x coordinates
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
  select(-playDirection,-targeted,-completed,-absoluteYardlineNumber,-down,-quarter)

# Add QB position, speed, orientation, and direction
all_week_epa_db <- all_week_epa_db %>%
  left_join(week %>% 
              distinct() %>%
              # filter on QB
              filter(position == "QB") %>% 
              # some plays have more than 1 QB on the field, we will just take the first -- might be some error
              group_by(gameId,playId,frameId) %>%
              filter(row_number() <= 1) %>%
              ungroup() %>%
              select(gameId,playId,frameId,x,y,o,dir,s) %>% 
              rename(QB_x = x,
                     QB_y = y,
                     QB_s = s,
                     QB_o = o,
                     QB_dir = dir), by = c("gameId","playId","frameId"))

# Add Target and Completion Probabilities
all_week_epa_db <- all_week_epa_db %>%
  left_join(target_probs %>% select(gameId,playId,frameId,nflId,target_prob),by = c("gameId","playId","frameId","nflId")) %>%
  left_join(completion_probs %>% select(gameId,playId,frameId,nflId,completion_probs),by = c("gameId","playId","frameId","nflId"))

# temp training db with gameId, playId, frameId to join with week dataset later
trainEPA_ <- all_week_epa_db %>%
  arrange(gameId,playId,frameId) %>%
  filter(complete.cases(.))

# training db 
trainEPA <- all_week_epa_db %>%
  arrange(gameId,playId,frameId) %>%
  select(-gameId,-playId,-frameId,-nflId) %>%
  filter(complete.cases(.)) %>%
  as.data.frame()

# rearrange columns -- DOUBLE CHECK before running
trainEPA <- trainEPA[,c(38,41:49,39,40,1:37,50:ncol(trainEPA))]

# Scale Variables
trainEPA[,c(11:ncol(trainEPA))] = trainEPA[,c(11:ncol(trainEPA))] %>% mutate_all(~(scale(.) %>% as.vector))

# XGG=BOOST Hypertune 
best_param = list()
best_seednumber = 1234
best_rmse = Inf
best_rmse_index = 0
start = Sys.time()
for (iter in 1:2) {
  param <- list(objective = "reg:squarederror",
                eval.metric = "rmse",
                metrics = "rmse",
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
  mdcv <- xgb.cv(data = model.matrix(epa ~ .,data = trainEPA),
                 label = as.vector(trainEPA$epa), 
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

best_rmse # 1.430931
mean((trainEPA$epa - rep(mean(trainEPA$epa ),length(trainEPA$epa)))^2) #2.393671

#best_param <- list(objective = "reg:squarederror",
#              eval.metric = "rmse",
#              max_depth = 10,
#              eta = 0.2026603,
#              gamma = 0.08454456, 
#              subsample = 0.8202024,
#              colsample_bytree = 0.6267184, 
#              min_child_weight = 25,
#              max_delta_step = 2
#)

# Train with best parameters
mod_boost_trained_epa <- xgboost(data = model.matrix(epa ~ .,data = trainEPA),
                             label = as.vector(trainEPA$epa),
                             params = best_param, 
                             nthread=6,
                             nround = 1000,
                             verbose = F, 
                             #early.stop.round=10, 
                             maximize=FALSE)
# Variable importance matrix
importance_matrix <- xgb.importance(feature_names = colnames(trainEPA), model = mod_boost_trained_epa)
write.csv(importance_matrix,"epa_importance_matrix.csv",row.names = F)

# ggplot of variable importance
gg <- xgb.ggplot.importance(importance_matrix, measure = "Gain",rel_to_first = F)
gg + ylab("Gain") +  theme(legend.position = "none") +xlab("") + theme(axis.text.y = element_text(size = 9))
# predicted values
y_boost_epa = predict(mod_boost_trained_epa,newdata = model.matrix(epa ~ .,data = trainEPA)) 

# Predict on all frames for all plays across all games.
all_week_epa_db_extra <- all_week_pass_db %>%
  inner_join(week %>% distinct() %>% select(gameId,playId,frameId,nflId,playDirection),by = c("gameId","playId","frameId","nflId")) %>%
  left_join(plays %>% distinct() %>% select(gameId,playId,absoluteYardlineNumber,yardsToGo,down,quarter),by = c("gameId","playId")) %>%
  mutate(epa = 0) %>%
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

# Add QB position, speed, orientation and direction
all_week_epa_db_extra <- all_week_epa_db_extra %>%
  left_join(week %>% 
              distinct() %>%
              # filter on QB
              filter(position == "QB") %>% 
              # some plays have more than 1 QB on the field, we will just take the first -- might be some error
              group_by(gameId,playId,frameId) %>%
              filter(row_number() <= 1) %>%
              ungroup() %>%
              select(gameId,playId,frameId,x,y,o,dir,s) %>% 
              rename(QB_x = x,
                     QB_y = y,
                     QB_s = s,
                     QB_o = o,
                     QB_dir = dir), by = c("gameId","playId","frameId"))

# Add target and completion probs
all_week_epa_db_extra <- all_week_epa_db_extra %>%
  distinct() %>%
  left_join(target_probs %>% distinct() %>% select(gameId,playId,frameId,nflId,target_prob),by = c("gameId","playId","frameId","nflId")) %>%
  left_join(completion_probs %>% distinct() %>% select(gameId,playId,frameId,nflId,completion_probs),by = c("gameId","playId","frameId","nflId")) %>%
  distinct()

# temp test dataset
testEPA_ <- all_week_epa_db_extra %>%
  arrange(gameId,playId,frameId) %>%
  filter(complete.cases(.)) %>%
  select(gameId,playId,frameId,nflId)

# test dataset
testEPA <- all_week_epa_db_extra %>%
  arrange(gameId,playId,frameId) %>%
  select(-gameId,-playId,-frameId,-nflId) %>%
  filter(complete.cases(.)) %>%
  as.data.frame()

# rearrange columns
testEPA <- testEPA[,c(39,41:49,38,40,1:37,50:ncol(testEPA))]

# Scale variables
testEPA[,c(11:ncol(testEPA))] = testEPA[,c(11:ncol(testEPA))] %>% mutate_all(~(scale(.) %>% as.vector))

# Predict on all variables
y_boost_test_epa_atpass = predict(mod_boost_trained_epa,newdata = model.matrix(epa ~ .,data = testEPA)) #predict

# Add to test EPA temp
testEPA_$pred_epa <- y_boost_test_epa_atpass

# Save Results
write.csv(testEPA_,"pred_epa.csv",row.names = F)





