# Packages
library(tidyverse)
library(data.table)

# Set Working Directory
setwd("~/Box Sync/Big Data Bowl/Data/DataLabeled")

# Data - this will take a minute. It loads all weeks
files <- list.files(pattern = "week*")
temp <- lapply(files, fread, sep=",")
taggedData <- rbindlist( temp )
#taggedData <- read_csv("DataLabeled/week1.csv") %>%
#  select(-X1) # man/zone

# Change working directory again
setwd("~/Box Sync/Big Data Bowl/Data")


target_probs <- read_csv("target_probs.csv")
completion_probs <- read_csv("completion_probs.csv")
completion_probs_atpass <- read_csv("completion_probs_atpass.csv")
exp_yards_gained <- read_csv("exp_yards_gained.csv")
pred_epa <- read_csv("pred_epa.csv")
players <- read_csv("players.csv")
plays <- read_csv("plays.csv")
targetedReceiver <- read_csv("targetedReceiver.csv")
# Man / Zone
week_all_zone_man <- read_csv("all_weeks_zone_man.csv")
zone_all <- week_all_zone_man %>% filter(coverage_type == "zone")


# Add nflId to covering, remove offensive players and add covering x and y to get distance calculation 
taggedData_covering <- taggedData %>%
  left_join(taggedData %>%
  filter(position %in% c("QB","RB","HB","FB","WR","TE")) %>%
  select(gameId,playId,nflId,jerseyNumber) %>%
  distinct() %>%
  rename(covering = nflId),by = c("gameId","playId","tag"="jerseyNumber")) %>%
  filter(!is.na(covering)) %>%
  left_join(taggedData %>% select(gameId,frameId,playId,nflId,x,y) %>% rename(covering_x = x,covering_y = y),by = c("gameId","playId","frameId","covering"="nflId")) %>%
  mutate(dist = sqrt((x - covering_x)^2 + (y - covering_y)^2))

# Add Completion Prob, Exp Yards Gained, and Predicted EPA
taggedData_covering <- taggedData_covering %>%
  select(-index,-a,-dir,-dis,-o,-playDirection,-jerseyNumber,-s,-tag,-x,-y,-time,-team,-route,-covering_x,-covering_y) %>%
  left_join(completion_probs,c("gameId","playId","frameId","covering"="nflId")) %>%
  left_join(completion_probs_atpass,c("gameId","playId","frameId","covering"="nflId")) %>%
  left_join(exp_yards_gained,c("gameId","playId","frameId","covering"="nflId")) %>%
  left_join(pred_epa,c("gameId","playId","frameId","covering"="nflId")) %>%
  left_join(targetedReceiver %>% mutate(targeted = 1),by = c("gameId","playId","covering"="targetNflId")) %>%
  left_join(plays %>% select(gameId,playId,epa,isDefensivePI),by = c("gameId","playId"))

# Roll up to play level
taggedData_covering_agg <- taggedData_covering %>%
  distinct() %>%
  group_by(gameId,playId,nflId) %>%
  summarize(completed = sum(completed,na.rm=T),
            targeted = max(targeted,na.rm = T),
            exp_comp_prob = max(exp_comp_prob,na.rm = T),
            avg_comp_prob = mean(completion_probs,na.rm=T),
            max_comp_prob = max(completion_probs,na.rm = T),
            max_exp_yards = max(exp_yards_gained,na.rm = T),
            avg_pred_epa = mean(pred_epa,na.rm = T),
            max_pred_epa = max(pred_epa,na.rm = T)) %>%
  mutate(exp_comp_prob = ifelse(targeted!=1,NA,exp_comp_prob),
         targeted = ifelse(targeted!=1,NA,targeted)) %>%
  mutate(exp_comp_prob = ifelse(exp_comp_prob >= 0,exp_comp_prob,NA),
         max_comp_prob = ifelse(is.na(avg_comp_prob),NA,max_comp_prob),
         max_pred_epa = ifelse(is.na(avg_pred_epa),NA,max_pred_epa),
         max_exp_yards = ifelse(is.na(avg_pred_epa),NA,max_exp_yards))  %>%
  left_join(plays %>% select(gameId,playId,epa,isDefensivePI),by = c("gameId","playId")) %>%
  mutate(epa = ifelse(targeted == 1,epa,NA),
         isDefensivePI = ifelse(targeted == 1,isDefensivePI,NA))

# Join to only players in zone
zone_all <- zone_all %>%
  left_join(taggedData_covering_agg,by = c("gameId","playId","nflId")) %>%
  left_join(players %>% select(nflId,displayName,position),by = c("nflId"))

# Zone Analysis
zone_all_after_pass_agg <- zone_all %>%
  filter(isDefensivePI == FALSE | is.na(isDefensivePI)) %>%
  group_by(nflId,displayName,position) %>%
  summarize(dpoe = sum(exp_comp_prob - completed,na.rm = T),
            comp_perc = sum(completed,na.rm = T)/sum(targeted,na.rm=T)) %>%
            #avg_epa = mean(epa,na.rm=T)) %>%
  ungroup() %>%
  #left_join(players %>% select(nflId,displayName,position),by = c("nflId")) %>%
  select(nflId,displayName,position,dpoe,comp_perc) %>%
  left_join(zone_all %>%
              group_by(nflId) %>%
              summarize(avg_epa = mean(epa,na.rm=T),
                        times_targeted = sum(targeted,na.rm=T),
                        plays = n(),
                        perc_targeted = sum(targeted,na.rm=T)/n()) %>%
              ungroup() %>%
              select(nflId,avg_epa,times_targeted,plays,perc_targeted),by = c("nflId")) %>%
  filter(position %in% c("CB","DB","S","FS","SS") & plays >= 100 & times_targeted >= 15) %>%
  #filter(times_targeted > 20) %>%
  arrange(-dpoe)

write.csv(zone_all_after_pass_agg,"zone_all_after_pass_agg.csv",row.names = F)

zone_all_before_pass_agg <- zone_all %>%
  group_by(nflId,displayName,position) %>%
  summarize(avg_exp_comp_prob = mean(avg_comp_prob,na.rm = T),
            avg_max_exp_comp_prob = mean(max_comp_prob,na.rm = T),
            avg_max_exp_yards = mean(max_exp_yards,na.rm = T),
            avg_pred_epa = mean(avg_pred_epa,na.rm = T),
            avg_max_pred_epa = mean(max_pred_epa,na.rm = T),
            times_targeted = sum(targeted,na.rm=T),
            plays = n(),
            perc_targeted = sum(targeted,na.rm=T)/n()) %>%
  ungroup() %>%
  #left_join(players %>% select(nflId,displayName,position),by = c("nflId")) %>%
  filter(position %in% c("CB","DB","S","FS","SS") & plays >= 100 & times_targeted >= 15) %>%
  select(nflId,displayName,position,
         avg_exp_comp_prob,avg_max_exp_comp_prob,avg_max_exp_yards,avg_max_pred_epa,
         times_targeted,plays,perc_targeted) %>%
  arrange(-avg_max_pred_epa) 

write.csv(zone_all_before_pass_agg,"zone_all_before_pass_agg.csv",row.names = F)

# Combine Before and After Pass
zone_all_agg <- zone_all_before_pass_agg %>%
  left_join(zone_all_after_pass_agg %>% select(nflId,dpoe,comp_perc,avg_epa),by = c("nflId"))
# Save Results
write.csv(zone_all_agg,"zone_all_agg.csv",row.names = F)

# Correlation among metrics
cor(zone_all[,c("exp_comp_prob","avg_comp_prob","max_comp_prob","max_exp_yards","avg_pred_epa","max_pred_epa","completed","epa")],use="complete.obs")



