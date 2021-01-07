# Packages
library(tidyverse)
library(xgboost)
library(data.table)

# Set Working Directory
setwd("~/Box Sync/Big Data Bowl/Data")
# Data - this will take a minute. It loads all weeks
files <- list.files(pattern = "week*")
# rearrange file names
files <- files[c(6,18,20,22,24,26,28,30,32,8,10:16)]
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

# Read and Join datasets with variables
files <- list.files(pattern = "*exp_comp_vars_db")
# Temp files into a list
temp <- lapply(files, fread, sep=",")
# combine list into dataframe
all_week_pass_db  <- rbindlist( temp )
# filter out duplicates
all_week_pass_db <- all_week_pass_db %>% distinct()


# Filter all_week_pass_db to only by where the ball was released, adjust x coordinates, create response variable.
all_week_targeted_db <- all_week_pass_db %>%
  # Join with week to get ONLY the frames where the ball was released from the QBs hand
  inner_join(week %>% distinct() %>% filter(event %in% c("pass_forward","pass_shovel")) %>% select(gameId,playId,frameId,nflId,playDirection),by = c("gameId","playId","frameId","nflId")) %>%
  # Join with targeted Receiver data frame
  left_join(targetedReceiver %>% distinct() %>% mutate(targeted = 1),by = c("gameId","playId","nflId"="targetNflId")) %>%
  # create binary variable -- receiver targeterd == 1, not targeted == 0
  mutate(targeted = ifelse(is.na(targeted),0,targeted)) %>%
  # Adjust x coordinate by playDirection
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


# Add QB location, orientation, speed and direction
all_week_targeted_db <- all_week_targeted_db %>%
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

# transfer the data from long to wide so that for each frame instead of 5 rows there is 1 row with info on the 5 different receivers
all_week_targeted_db_wide <- all_week_targeted_db %>%
  # Select independent variables and response
  select(gameId,playId,frameId,t,
         x,y,dir,o,s,
         dist_to_targeted_Def1,x_Def1,y_Def1,o_Def1,dir_Def1,s_Def1,
         dist_to_targeted_Def2,x_Def2,y_Def2,o_Def2,dir_Def2,s_Def2,
         dist_to_targeted_Def3,x_Def3,y_Def3,o_Def3,dir_Def3,s_Def3,
         pass_dist,pass_x,pass_y,
         QB_x,QB_y,QB_o,QB_dir,
         targeted) %>%
  group_by(gameId,playId,frameId,t) %>%
  # sort by pass distance 
  arrange(gameId,playId,frameId,t,pass_dist) %>%
  # filter so only 5 receivers
  filter(row_number() <= 5) %>%
  # label each receiver
  mutate(target = paste0("rec_",row_number())) %>%
  # long to wide
  pivot_wider(names_from = target,values_from = c(x,y,dir,o,s,
                                                  dist_to_targeted_Def1,x_Def1,y_Def1,o_Def1,dir_Def1,s_Def1,
                                                  dist_to_targeted_Def2,x_Def2,y_Def2,o_Def2,dir_Def2,s_Def2,
                                                  dist_to_targeted_Def3,x_Def3,y_Def3,o_Def3,dir_Def3,s_Def3,
                                                  pass_dist,pass_x,pass_y,
                                                  QB_x,QB_y,QB_o,QB_dir,
                                                  targeted)) %>%
  ungroup() %>%
  # create response variable
  mutate(targeted = case_when(targeted_rec_1 == 1~1,
                              targeted_rec_2 == 1~2,
                              targeted_rec_3 == 1~3,
                              targeted_rec_4 == 1~4,
                              targeted_rec_5 == 1~5)) %>%
  # remove multiples of QB and football data
  select(-targeted_rec_1,-targeted_rec_2,-targeted_rec_3,-targeted_rec_4,-targeted_rec_5,
         -QB_x_rec_2,-QB_y_rec_2,-QB_o_rec_2,-QB_dir_rec_2,
         -QB_x_rec_3,-QB_y_rec_3,-QB_o_rec_3,-QB_dir_rec_3,
         -QB_x_rec_4,-QB_y_rec_4,-QB_o_rec_4,-QB_dir_rec_4,
         -QB_x_rec_5,-QB_y_rec_5,-QB_o_rec_5,-QB_dir_rec_5,
         -pass_x_rec_2,-pass_y_rec_2,
         -pass_x_rec_3,-pass_y_rec_3,
         -pass_x_rec_4,-pass_y_rec_4,
         -pass_x_rec_5,-pass_y_rec_5) 

# Create temp trainTargeted with gameId,playId,frameId so we can join later
trainTargeted_ <- all_week_targeted_db_wide %>%
  arrange(gameId,playId,frameId) %>%
  filter(complete.cases(.))

# Create training data frame -- filter only on complete cases
trainTargeted <- all_week_targeted_db_wide %>%
  arrange(gameId,playId,frameId) %>%
  # remove gameId, playId, frameId -- frameId is accounted for by the variable t
  select(-gameId,-playId,-frameId) %>%
  filter(complete.cases(.)) %>%
  as.data.frame()

# rearrange columns to targeted (response) is first variable
trainTargeted <- trainTargeted[,c(ncol(trainTargeted),1:(ncol(trainTargeted)-1))]

# Scale variables
trainTargeted[,c(2:ncol(trainTargeted))] = trainTargeted[,c(2:ncol(trainTargeted))] %>% mutate_all(~(scale(.) %>% as.vector))

# Subtract 1 from target for XGBOOST model
trainTargeted <- trainTargeted %>% mutate(targeted = targeted - 1)

# Hyper tune parameters using XGBOOST
num_cv = 1
best_param = list()
best_seednumber = 1234
best_logloss = Inf
best_logloss_index = 0
start = Sys.time()
for (iter in 1:num_cv) {
  param <- list("objective" = "multi:softprob",
                "eval_metric" = "mlogloss",
                "num_class" = 5,
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
  mdcv <- xgb.cv(data = model.matrix(targeted ~ .,data = trainTargeted),
                 label = as.vector(trainTargeted$targeted), 
                 params = param, 
                 nthread=6, 
                 nfold=cv.nfold, 
                 nrounds=cv.nround,
                 verbose = F, 
                 #early.stop.round=10, 
                 maximize=FALSE)
  
  min_logloss = min(mdcv$evaluation_log[, "test_mlogloss_mean"])
  min_logloss_index = which.min(mdcv$evaluation_log[, test_mlogloss_mean])
  
  if (min_logloss < best_logloss) {
    best_logloss = min_logloss
    best_logloss_index = min_logloss_index
    best_seednumber = seed.number
    best_param = param
  }
  
  print(iter)
  
}

stop = Sys.time()
stop - start # Time difference of 1.016085 hours

# best_seednumber

#best_param <- list("objective" = "multi:softprob",
#              "eval_metric" = "mlogloss",
#              "num_class" = 5,
#              max_depth = 7,
#              eta = 0.2250507,
#              gamma = 0.1420197, 
#              subsample = 0.6236883,
#              colsample_bytree = 0.5625715, 
#              min_child_weight = 23,
#              max_delta_step = 7
#)

best_param

# best log loss 0.908667
# Mean m log loss -- 1.609438 -- significant improvment!
#MultiLogLoss(y_true = xgb.pred$label, y_pred = matrix(rep(0.2,nrow(xgb.pred)*5),ncol = 5))


# Train model with best_params
mod_boost_trained_target <- xgboost(data = model.matrix(targeted ~ .,data = trainTargeted),
                             label = as.vector(trainTargeted$targeted),
                             params = best_param, 
                             nthread=6,
                             nround = 1000,
                             verbose = F, 
                             #early.stop.round=10, 
                             maximize=FALSE)

# Variable Importance Matrix
importance_matrix <- xgb.importance(feature_names = colnames(trainTargeted), model = mod_boost_trained_target)
write.csv(importance_matrix,"target_prob_importance_matrix.csv",row.names = F)

# Creat ggplot of importance matrix
gg <- xgb.ggplot.importance(importance_matrix, measure = "Gain",rel_to_first = F)
gg + 
  ylab("Gain") +  
  xlab("") + 
  theme_bw() + 
  theme(legend.position = "none") +
  theme(axis.text.y = element_text(size = 9))

# Make Predictions on Training Set
y_boost_target = predict(mod_boost_trained_target,newdata = model.matrix(targeted ~ .,data = trainTargeted),reshape=T) #predict
# Create data frame
xgb.pred = as.data.frame(y_boost_target)
# Assign column names to each receiver
colnames(xgb.pred) = c("rec_1","rec_2","rec_3","rec_4","rec_5")
# Make prediction based on max probability in each row
xgb.pred$prediction = apply(xgb.pred,1,function(x) colnames(xgb.pred)[which.max(x)])
# Assign approriate label
xgb.pred$label = c("rec_1","rec_2","rec_3","rec_4","rec_5")[trainTargeted$targeted+1]
# table of predicted vs actual
table(xgb.pred$prediction,xgb.pred$label)
# View results
result = sum(xgb.pred$prediction==xgb.pred$label)/nrow(xgb.pred)
print(paste("Final Accuracy =",sprintf("%1.2f%%", 100*result)))
# MLogLoss
# MultiLogLoss(y_true = xgb.pred$label, y_pred = xgb.pred[,1:5])

# Again the Mean has m log loss of 1.609438
# MultiLogLoss(y_true = xgb.pred$label, y_pred = matrix(rep(0.2,nrow(xgb.pred)*5),ncol = 5))

# Predict on all frames (not just training data) --  same script as above but just on all data
##### WARNING --  May take longer to run #######
all_week_targeted_db_extra <- all_week_pass_db %>%
  inner_join(week %>% distinct() %>% select(gameId,playId,frameId,nflId,playDirection),by = c("gameId","playId","frameId","nflId")) %>%
  left_join(targetedReceiver %>% distinct() %>%  mutate(targeted = 1),by = c("gameId","playId","nflId"="targetNflId")) %>%
  mutate(targeted = ifelse(is.na(targeted),0,targeted)) %>%
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

all_week_targeted_db_extra <- all_week_targeted_db_extra %>%
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

all_week_targeted_db_extra_wide <- all_week_targeted_db_extra %>%
  arrange(gameId,playId,frameId) %>%
  select(gameId,playId,frameId,t,
         x,y,dir,o,s,
         dist_to_targeted_Def1,x_Def1,y_Def1,o_Def1,dir_Def1,s_Def1,
         dist_to_targeted_Def2,x_Def2,y_Def2,o_Def2,dir_Def2,s_Def2,
         dist_to_targeted_Def3,x_Def3,y_Def3,o_Def3,dir_Def3,s_Def3,
         pass_dist,pass_x,pass_y,
         QB_x,QB_y,QB_o,QB_dir) %>%
  group_by(gameId,playId,frameId,t) %>%
  arrange(gameId,playId,frameId,t,pass_dist) %>%
  filter(row_number() <= 5) %>%
  mutate(target = paste0("rec_",row_number())) %>%
  pivot_wider(names_from = target,values_from = c(x,y,dir,o,s,
                                                  dist_to_targeted_Def1,x_Def1,y_Def1,o_Def1,dir_Def1,s_Def1,
                                                  dist_to_targeted_Def2,x_Def2,y_Def2,o_Def2,dir_Def2,s_Def2,
                                                  dist_to_targeted_Def3,x_Def3,y_Def3,o_Def3,dir_Def3,s_Def3,
                                                  pass_dist,pass_x,pass_y,
                                                  QB_x,QB_y,QB_o,QB_dir)) %>%
  ungroup() %>%
  mutate(targeted = 0) %>%
  select(-QB_x_rec_2,-QB_y_rec_2,-QB_o_rec_2,-QB_dir_rec_2,
         -QB_x_rec_3,-QB_y_rec_3,-QB_o_rec_3,-QB_dir_rec_3,
         -QB_x_rec_4,-QB_y_rec_4,-QB_o_rec_4,-QB_dir_rec_4,
         -QB_x_rec_5,-QB_y_rec_5,-QB_o_rec_5,-QB_dir_rec_5,
         -pass_x_rec_2,-pass_y_rec_2,
         -pass_x_rec_3,-pass_y_rec_3,
         -pass_x_rec_4,-pass_y_rec_4,
         -pass_x_rec_5,-pass_y_rec_5)

testTargeted_ <- all_week_targeted_db_extra_wide %>%
  filter(complete.cases(.))

testTargeted <- all_week_targeted_db_extra_wide %>%
  select(-gameId,-playId,-frameId) %>%
  filter(complete.cases(.)) %>%
  as.data.frame()

testTargeted <- testTargeted[,c(ncol(testTargeted),1:(ncol(testTargeted)-1))]

testTargeted[,c(2:ncol(testTargeted))] = testTargeted[,c(2:ncol(testTargeted))] %>% mutate_all(~(scale(.) %>% as.vector))

#predict on all data
y_boost_target = predict(mod_boost_trained_target,newdata = model.matrix(targeted ~ .,data = testTargeted),reshape=T) 

# Create a data frame that is long (not wide) to store predictions -- run similar code to above to make sure all rows are properply filtered
all_week_targeted_db_extra_wPreds <- all_week_targeted_db_extra %>%
  select(gameId,playId,frameId,t,nflId,
         x,y,dir,o,s,
         dist_to_targeted_Def1,x_Def1,y_Def1,o_Def1,dir_Def1,s_Def1,
         dist_to_targeted_Def2,x_Def2,y_Def2,o_Def2,dir_Def2,s_Def2,
         dist_to_targeted_Def3,x_Def3,y_Def3,o_Def3,dir_Def3,s_Def3,
         pass_dist,pass_x,pass_y,
         QB_x,QB_y,QB_o,QB_dir,
         targeted) %>%
  group_by(gameId,playId,frameId,t) %>%
  arrange(pass_dist) %>%
  filter(row_number() <= 5) %>%
  ungroup() %>%
  filter(complete.cases(.)) %>%
  select(gameId,playId,frameId,nflId,pass_dist) %>%
  left_join(all_week_targeted_db_extra_wide  %>% rename(targeted_wide = targeted),by = c("gameId","playId","frameId")) %>%
  filter(complete.cases(.)) %>%
  select(gameId,playId,frameId,nflId,pass_dist) %>%
  arrange(gameId,playId,frameId,pass_dist)

xgb.pred = as.data.frame(y_boost_target)
colnames(xgb.pred) = c("rec_1","rec_2","rec_3","rec_4","rec_5")

xgb.pred$prediction = apply(xgb.pred,1,function(x) colnames(xgb.pred)[which.max(x)])

# predict in long format
all_week_targeted_db_extra_wPreds$target_prob <- xgb.pred[,1:5] %>% t %>% as.matrix %>% as.vector

# write target probs out to csv
write.csv(all_week_targeted_db_extra_wPreds,"target_probs.csv",row.names = F)


