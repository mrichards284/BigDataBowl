# Packages
library(tidyverse)

# Set Working Directory
setwd("~/Box Sync/Big Data Bowl/Data")

# Data
man_all <- read_csv("man_all_20201222.csv")
target_probs <- read_csv("target_probs.csv")
completion_probs <- read_csv("completion_probs.csv")
completion_probs_atpass <- read_csv("completion_probs_atpass.csv")
exp_yards_gained <- read_csv("exp_yards_gained.csv")
pred_epa <- read_csv("pred_epa.csv")
players <- read.csv("players.csv", stringsAsFactors=FALSE)

# make sure no duplicates
man_all <- man_all %>% 
  distinct()

# Aggregate Expected Completed Pass
completion_probs_agg <- completion_probs %>%
  group_by(gameId,playId,nflId) %>%
  summarize(avg_comp_prob = mean(completion_probs),
            max_comp_prob = max(completion_probs)) %>%
  ungroup() %>%
  as.data.frame()

# Aggregate Expected Yards Gained
exp_yards_gained_agg <- exp_yards_gained %>%
  group_by(gameId,playId,nflId) %>%
  summarize(avg_exp_yards_gained = mean(exp_yards_gained),
            max_exp_yards_gained = max(exp_yards_gained)) %>%
  ungroup() %>%
  as.data.frame()

# Aggregate Predicted EPA
pred_epa_agg <- pred_epa %>%
  group_by(gameId,playId,nflId) %>%
  summarize(avg_pred_epa = mean(pred_epa),
            max_pred_epa = max(pred_epa)) %>%
  ungroup() %>%
  as.data.frame()

# Join datasets
man_all <- man_all %>%
  left_join(completion_probs_atpass %>% select(-completed,-frameId), by = c("gameId","playId","covering"="nflId")) %>%
  left_join(completion_probs_agg, by = c("gameId","playId","covering"="nflId")) %>%
  left_join(exp_yards_gained_agg, by = c("gameId","playId","covering"="nflId")) %>%
  left_join(pred_epa_agg, by = c("gameId","playId","covering"="nflId"))

# After Pass
man_all_after_pass_agg <- man_all %>%
  mutate(completed = ifelse(completed == "C",1,ifelse(completed %in% c("I","IN"),0,NA))) %>%
  filter(dist_snap <= 12) %>%
  filter(isDefensivePI == FALSE | is.na(isDefensivePI)) %>%
  group_by(nflId) %>%
  summarize(dpoe = sum(exp_comp_prob - completed,na.rm = T),
            comp_perc = sum(completed,na.rm = T)/sum(targeted,na.rm=T),
            avg_epa = mean(epa,na.rm=T)) %>%
  ungroup() %>%
  left_join(players %>% select(nflId,displayName,position),by = c("nflId")) %>%
  select(nflId,displayName,position,dpoe,comp_perc) %>%
  left_join(man_all %>%
              mutate(completed = ifelse(completed == "C",1,ifelse(completed %in% c("I","IN"),0,NA))) %>%
              filter(dist_snap <= 12) %>%
              group_by(nflId) %>%
              summarize(avg_epa = mean(epa,na.rm=T),
                        times_targeted = sum(targeted,na.rm=T),
                        plays = n(),
                        perc_targeted = sum(targeted,na.rm=T)/n()) %>%
              ungroup() %>%
              select(nflId,avg_epa,times_targeted,plays,perc_targeted),by = c("nflId")) %>%
  filter(position == "CB" & plays >= 100 & times_targeted >= 15) %>%
  #filter(times_targeted > 20) %>%
  arrange(-dpoe)

write.csv(man_all_after_pass_agg,"man_all_after_pass_agg.csv",row.names = T)

# Before Pass
man_all_before_pass_agg <- man_all %>%
  mutate(completed = ifelse(completed == "C",1,ifelse(completed %in% c("I","IN"),0,NA))) %>%
  filter(dist_snap <= 12) %>%
  #filter(isDefensivePI == FALSE | is.na(isDefensivePI)) %>%
  group_by(nflId) %>%
  summarize(avg_dist = mean(avg_dist,na.rm = T),
            avg_max_dist = mean(max_dist,na.rm = T),
            avg_exp_comp_prob = mean(avg_comp_prob,na.rm = T),
            avg_max_exp_comp_prob = mean(max_comp_prob,na.rm = T),
            avg_exp_yards = mean(avg_exp_yards_gained,na.rm = T),
            avg_max_exp_yards = mean(max_exp_yards_gained,na.rm = T),
            avg_pred_epa = mean(avg_pred_epa,na.rm = T),
            avg_max_pred_epa = mean(max_pred_epa,na.rm = T),
            times_targeted = sum(targeted,na.rm=T),
            plays = n(),
            perc_targeted = sum(targeted,na.rm=T)/n()) %>%
  ungroup() %>%
  left_join(players %>% select(nflId,displayName,position),by = c("nflId")) %>%
  filter( plays >= 100 & position == "CB" & times_targeted >= 15) %>%
  select(nflId,displayName,position,avg_dist,avg_max_dist,avg_exp_comp_prob,avg_max_exp_comp_prob,avg_max_exp_yards,avg_max_pred_epa,
         times_targeted,plays,perc_targeted) %>%
  arrange(-avg_max_pred_epa) 

write.csv(man_all_before_pass_agg,"man_all_before_pass_agg.csv",row.names = F)

# Join before and after pass
man_all_agg <- man_all_before_pass_agg %>%
  left_join(man_all_after_pass_agg %>% select(nflId,dpoe,comp_perc,avg_epa),by = c("nflId"))

write.csv(man_all_agg,"man_all_agg.csv",row.names = F)

# Correlation
man_cor_afterpass <- man_all %>%
  mutate(completed = ifelse(completed == "C",1,ifelse(completed %in% c("I","IN"),0,NA))) #%>%
  #filter(isDefensivePI == FALSE | is.na(isDefensivePI)) %>%
  #filter(dist_snap <= 4)


cor(man_cor_afterpass[,c("avg_dist","max_dist","exp_comp_prob","avg_comp_prob","max_comp_prob","avg_exp_yards_gained","max_exp_yards_gained","avg_pred_epa","max_pred_epa","completed","epa")],use="complete.obs")


# After Pass - Press Coverage
man_all_press_after_pass_agg <- man_all %>%
  mutate(completed = ifelse(completed == "C",1,ifelse(completed %in% c("I","IN"),0,NA))) %>%
  filter(dist_snap <= 4) %>%
  filter(isDefensivePI == FALSE | is.na(isDefensivePI)) %>%
  group_by(nflId) %>%
  summarize(dpoe = sum(exp_comp_prob - completed,na.rm = T),
            comp_perc = sum(completed,na.rm = T)/sum(targeted,na.rm=T),
            avg_epa = mean(epa,na.rm=T)) %>%
  ungroup() %>%
  left_join(players %>% select(nflId,displayName,position),by = c("nflId")) %>%
  select(nflId,displayName,position,dpoe,comp_perc) %>%
  left_join(man_all %>%
              mutate(completed = ifelse(completed == "C",1,ifelse(completed %in% c("I","IN"),0,NA))) %>%
              filter(dist_snap <= 4) %>%
              group_by(nflId) %>%
              summarize(avg_epa = mean(epa,na.rm=T),
                        times_targeted = sum(targeted,na.rm=T),
                        plays = n(),
                        perc_targeted = sum(targeted,na.rm=T)/n()) %>%
              ungroup() %>%
              select(nflId,avg_epa,times_targeted,plays,perc_targeted),by = c("nflId")) %>%
  filter(position == "CB" & plays >= 100 & times_targeted >= 15) %>%
  #filter(times_targeted > 20) %>%
  arrange(-dpoe)

write.csv(man_all_press_after_pass_agg,"man_all_press_after_pass_agg.csv",row.names = T)

# Before Pass - Press Coverage
man_all_press_before_pass_agg <- man_all %>%
  mutate(completed = ifelse(completed == "C",1,ifelse(completed %in% c("I","IN"),0,NA))) %>%
  filter(dist_snap <= 4) %>%
  #filter(isDefensivePI == FALSE | is.na(isDefensivePI)) %>%
  group_by(nflId) %>%
  summarize(avg_dist = mean(avg_dist,na.rm = T),
            avg_max_dist = mean(max_dist,na.rm = T),
            avg_exp_comp_prob = mean(avg_comp_prob,na.rm = T),
            avg_max_exp_comp_prob = mean(max_comp_prob,na.rm = T),
            avg_exp_yards = mean(avg_exp_yards_gained,na.rm = T),
            avg_max_exp_yards = mean(max_exp_yards_gained,na.rm = T),
            avg_pred_epa = mean(avg_pred_epa,na.rm = T),
            avg_max_pred_epa = mean(max_pred_epa,na.rm = T),
            times_targeted = sum(targeted,na.rm=T),
            plays = n(),
            perc_targeted = sum(targeted,na.rm=T)/n()) %>%
  ungroup() %>%
  left_join(players %>% select(nflId,displayName,position),by = c("nflId")) %>%
  filter( plays >= 100 & position == "CB" & times_targeted >= 15) %>%
  select(nflId,displayName,position,avg_dist,avg_max_dist,avg_exp_comp_prob,avg_max_exp_comp_prob,avg_max_exp_yards,avg_max_pred_epa,
         times_targeted,plays,perc_targeted) %>%
  arrange(-avg_max_pred_epa) 

write.csv(man_all_press_before_pass_agg,"man_all_press_before_pass_agg.csv",row.names = F)

# Join before and after pass
man_all_press_agg <- man_all_press_agg %>%
  left_join(man_all_press_after_pass_agg %>% select(nflId,dpoe,comp_perc,avg_epa),by = c("nflId"))

write.csv(man_all_press_agg,"man_all_press_agg.csv",row.names = F)


# Correlation
man_cor_afterpass <- man_all %>%
  mutate(completed = ifelse(completed == "C",1,ifelse(completed %in% c("I","IN"),0,NA))) %>%
  #filter(isDefensivePI == FALSE | is.na(isDefensivePI)) %>%
  filter(dist_snap <= 4)

cor(man_cor_afterpass[,c("avg_dist","max_dist","exp_comp_prob","avg_comp_prob","max_comp_prob","avg_exp_yards_gained","max_exp_yards_gained","avg_pred_epa","max_pred_epa","completed","epa")],use="complete.obs")
