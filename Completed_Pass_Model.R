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

# Read and Join datasets with variables
files <- list.files(pattern = "*exp_comp_vars_db")
# Temp files into a list
temp <- lapply(files, fread, sep=",")
# combine list into dataframe
all_week_pass_db  <- rbindlist( temp )
# filter out duplicates
all_week_pass_db <- all_week_pass_db %>% distinct()


# Create a database filtered only on the frame in which the ball was released from the QBs hand and the receiver was targeted
all_week_completed_db <- all_week_pass_db %>%
  # filter on frames where the ball was released
  inner_join(week %>% distinct() %>% filter(event %in% c("pass_forward","pass_shovel")) %>% select(gameId,playId,frameId,nflId,playDirection),by = c("gameId","playId","frameId","nflId")) %>%
  # only targeted receivers
  left_join(targetedReceiver %>% distinct() %>% mutate(targeted = 1),by = c("gameId","playId","nflId"="targetNflId")) %>%
  filter(!is.na(targeted)) %>%
  # get passResult (for completions)
  left_join(plays %>% distinct() %>% select(gameId,playId,passResult),by = c("gameId","playId")) %>%
  # create completed binary variable
  mutate(completed = ifelse(passResult == "C",1,0)) %>%
  # adjust x coordinate for play Direction
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
  select(-playDirection,-passResult,-targeted)

# Add QB variables
all_week_completed_db <- all_week_completed_db %>%
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

# Add target probability
all_week_completed_db <- all_week_completed_db %>%
  left_join(target_probs %>% select(gameId,playId,frameId,nflId,target_prob),by = c("gameId","playId","frameId","nflId"))

# Train temp database with gameId, playId, frameId, nflId for joining back to week database
trainCompleted_ <- all_week_completed_db %>%
  arrange(gameId,playId,frameId) %>%
  filter(complete.cases(.)) %>%
  select(gameId,playId,frameId,nflId,completed)

# Train database only with variables for modeling
trainCompleted <- all_week_completed_db %>%
  arrange(gameId,playId,frameId) %>%
  select(-gameId,-playId,-frameId,-nflId) %>%
  filter(complete.cases(.)) %>%
  as.data.frame()

# Scale Variables
trainCompleted[,c(2:ncol(trainCompleted))] = trainCompleted[,c(2:ncol(trainCompleted))] %>% mutate_all(~(scale(.) %>% as.vector))

# XGBOOST Hypertuning
best_param = list()
best_seednumber = 1234
best_logloss = Inf
best_logloss_index = 0
start = Sys.time()
for (iter in 1:2) {
  param <- list(objective = "binary:logistic",
                eval.metric = "logloss",
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
  mdcv <- xgb.cv(data = model.matrix(completed ~ .,data = trainCompleted),
                 label = as.vector(trainCompleted$completed), 
                 params = param, 
                 nthread=6, 
                 nfold=cv.nfold, 
                 nrounds=cv.nround,
                 verbose = F, 
                 #early.stop.round=10, 
                 maximize=FALSE)
  
  min_logloss = min(mdcv$evaluation_log[, "test_logloss_mean"])
  min_logloss_index = which.min(mdcv$evaluation_log[, test_logloss_mean])
  
  if (min_logloss < best_logloss) {
    best_logloss = min_logloss
    best_logloss_index = min_logloss_index
    best_seednumber = seed.number
    best_param = param
  }
  
  print(iter)
  
}

# best_logloss 0.53825
# best_seednumber = 4343
#best_param <- list(objective = "binary:logistic",
#                   max_depth = 8,
#                   eta = 0.04281647,
#                   gamma = 0.1845386, 
#                   subsample = 0.719794,
#                   colsample_bytree = 0.5148373, 
#                   min_child_weight = 19,
#                   max_delta_step = 2
#)

LogLoss = function(actual, predicted, eps = 1e-15) {
  predicted = pmin(pmax(predicted, eps), 1-eps) 
  return(- (sum(actual * log(predicted) + (1 - actual) * log(1 - predicted))) / length(actual))
}
# Log Loss of mean prediction -- 0.6450836 fairly significant improvement! 
LogLoss(actual = trainCompleted$completed,predicted = rep(mean(trainCompleted$completed),nrow(trainCompleted))) 


# Trained Model on all data with tuned parameters
mod_boost_trained_completed <- xgboost(data = model.matrix(completed ~ .,data = trainCompleted),
                             label = as.vector(trainCompleted$completed),
                             params = best_param, 
                             nthread=6,
                             nround = 1000,
                             verbose = F, 
                             #early.stop.round=10, 
                             maximize=FALSE)
# Variable importance matrix
importance_matrix <- xgb.importance(feature_names = colnames(trainCompleted), model = mod_boost_trained_completed)
write.csv(importance_matrix,"completion_prob_importance_matrix.csv",row.names = F)
# ggplot variable importance matrix
gg <- xgb.ggplot.importance(importance_matrix, measure = "Gain",rel_to_first = F)
gg + ylab("Gain") +  theme(legend.position = "none") +xlab("") + theme(axis.text.y = element_text(size = 9))
# trained predictions at time of pass
y_boost_atpass = predict(mod_boost_trained_completed,newdata = model.matrix(completed ~ .,data = trainCompleted)) #predict
trainCompleted_$exp_comp_prob <- y_boost_atpass
# Save results at pass
write.csv(trainCompleted_,"completion_probs_atpass.csv",row.names = F)

# Predict on all frames for all receivers -- not just targeted and event == pass_forward / pass_shovel
all_week_completed_db_extra <- all_week_pass_db %>%
  inner_join(week %>% distinct() %>% select(gameId,playId,frameId,nflId,playDirection),by = c("gameId","playId","frameId","nflId")) %>%
  mutate(completed = 0) %>%
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
  select(-playDirection)

# Add QB position, speed, orientation and direction
all_week_completed_db_extra <- all_week_completed_db_extra %>% 
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

# Add target probability
all_week_completed_db_extra <- all_week_completed_db_extra %>%
  left_join(target_probs %>% select(gameId,playId,frameId,nflId,target_prob),by = c("gameId","playId","frameId","nflId"))

# Create temp test data frame w/ gameId, playId, frameId, and nflId to join back to week dataset
testCompleted_ <- all_week_completed_db_extra %>%
  arrange(gameId,playId,frameId) %>%
  filter(complete.cases(.)) %>%
  select(gameId,playId,frameId,nflId)

# Create test data frame
testCompleted <- all_week_completed_db_extra %>%
  arrange(gameId,playId,frameId) %>%
  select(-gameId,-playId,-frameId,-nflId) %>%
  filter(complete.cases(.)) %>%
  as.data.frame()

# Scale Variables
testCompleted[,c(2:ncol(testCompleted))] = testCompleted[,c(2:ncol(testCompleted))] %>% mutate_all(~(scale(.) %>% as.vector))

# Predict on all test dataset
y_boost_test_atpass = predict(mod_boost_trained_completed,newdata = model.matrix(completed ~ .,data = testCompleted)) 

# Add Completion probabilities to temp test dataset to save and join to week dataset
testCompleted_$completion_probs <- y_boost_test_atpass

# Save results
write.csv(testCompleted_,"completion_probs.csv",row.names = F)

