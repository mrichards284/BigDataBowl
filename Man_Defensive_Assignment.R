# Packages
library(tidyverse)
library(gganimate)
library(cowplot)

# Set Working Directory
setwd("~/Box Sync/Big Data Bowl/Data")

# Data 
# weeks data - this will take a minute. It loads all weeks
files <- list.files(pattern = "week*")
#files <- files[6:length(files)]
#files <- files[c(1,10:17,2:9)]
temp <- lapply(files, fread, sep=",")
week <- rbindlist( temp )
# player data
players <- read.csv("players.csv", stringsAsFactors=FALSE)
# games data
games <- read.csv("games.csv", stringsAsFactors=FALSE)
# plays data
plays <- read.csv("plays.csv", stringsAsFactors=FALSE)
# coverages data
coverages <- read.csv("coverages_wk1.csv", stringsAsFactors=FALSE)
# targeted Receiver data
targetedReceiver <- read.csv("targetedReceiver.csv", stringsAsFactors=FALSE)
# Man / Zone Data
week_all_zone_man <- read.csv("all_weeks_zone_man.csv", stringsAsFactors=FALSE)

# Add event tag
week <- week %>% 
  inner_join(games %>% select(-gameDate,-gameTimeEastern,-week)) %>% 
  inner_join(plays)
man_all <- week_all_zone_man %>% filter(coverage_type == "man")
man_all$covering <- NA



# Identify target based on the player they were closest to for most of the route
pb <- txtProgressBar(min = 1, max = nrow(man_all), style = 3)
for (i in 1:nrow(man_all)){
  
  
  frameId_min <- filter(week, gameId == man_all[i,"gameId"] & playId == man_all[i,"playId"] & nflId == man_all[i,"nflId"] & event == "ball_snap") %>%
    select(frameId) %>%
    slice_min(order_by = frameId,n = 1) %>%
    filter(row_number() <= 1) %>%
    as.matrix %>%
    as.vector
  
  frameId_max <- filter(week, gameId == man_all[i,"gameId"] & playId == man_all[i,"playId"] & nflId == man_all[i,"nflId"] & event %in% c("pass_forward","qb_sack","qb_strip_sack","pass_shovel","out_of_bounds")) %>%
    slice_min(order_by = frameId,n = 1) %>%
    filter(row_number() <= 1) %>%
    select(frameId) %>%
    as.matrix %>%
    as.vector
  
  man_data <- week %>%
    filter(gameId == man_all[i,"gameId"] & playId == man_all[i,"playId"] & nflId == man_all[i,"nflId"] & between(frameId,frameId_min,frameId_max)) %>%
    rename(man_x = x, man_y = y) %>%
    select(gameId,playId,frameId,man_x,man_y)
  
  man_all$covering[i] <- week %>%
    filter(between(frameId,frameId_min,frameId_max)  & gameId == man_all[i,"gameId"] & playId == man_all[i,"playId"] & displayName != 'Football' &
             position %in% c("QB","WR","RB","TE","FB","HB")) %>%
    left_join(man_data,by = c("gameId","playId","frameId")) %>%
    group_by(gameId,playId,nflId,displayName) %>%
    summarise(avg_dist = mean(sqrt((x - man_x)^2 + (y - man_y)^2)),.groups = "keep") %>%
    ungroup() %>%
    select(gameId,playId,nflId,displayName,avg_dist) %>%
    slice_min(order_by = avg_dist,n=1) %>%
    select(nflId) %>%
    as.matrix %>%
    as.vector
  
  setTxtProgressBar(pb, i)
  
}

man_all$avg_dist <- NA
man_all$max_dist <- NA
man_all$dist_snap <- NA
man_all$dist_pass <- NA
man_all$route <- NA
man_all$targeted <- NA
man_all$completed <- NA
man_all$epa <- NA
man_all$isDefensivePI <- NA
man_all$dist_pass_arrival <- NA
man_all$break_diff <- NA
pb <- txtProgressBar(min = 1, max = nrow(man_all), style = 3)
for (i in 1:nrow(man_all)){
  
  frameId_pass <- week %>%
    filter(gameId == man_all[i,"gameId"] & playId == man_all[i,"playId"]) %>% 
    filter(event %in% c("pass_forward","qb_sack","qb_strip_sack","pass_shovel","out_of_bounds")) %>%
    select(frameId) %>%
    distinct() %>%
    slice_min(order_by = frameId,n = 1) %>%
    as.matrix() %>%
    as.vector()
  
  frameId_snap <- week %>%
    filter(gameId == man_all[i,"gameId"] & playId == man_all[i,"playId"]) %>% 
    filter(event == "ball_snap") %>%
    select(frameId) %>%
    distinct() %>%
    slice_min(order_by = frameId,n = 1) %>%
    as.matrix() %>%
    as.vector()
  
  def <- week %>%
    filter(gameId == man_all[i,"gameId"] & playId == man_all[i,"playId"] & nflId == man_all[i,"nflId"]) %>%
    filter(frameId <= frameId_pass & frameId >= frameId_snap) %>%
    select(gameId,playId,frameId,event,x,y,nflId)
  
  off <- week %>%
    filter(gameId == man_all[i,"gameId"] & playId == man_all[i,"playId"] & nflId == man_all[i,"covering"]) %>%
    filter(frameId <= frameId_pass & frameId >= frameId_snap) %>%
    left_join(plays %>% select(gameId,playId,passResult,epa,isDefensivePI),by = c("gameId","playId")) %>%
    select(gameId,playId,frameId,event,route,passResult,epa,isDefensivePI,x,y,nflId)
  
  man_all$avg_dist[i] <- def %>%
    rename(def_x = x,
           def_y = y,
           def_nflId= nflId) %>%
    left_join(off %>% select(gameId,playId,frameId,nflId,x,y),by = c("gameId","playId","frameId")) %>%
    mutate(dist = sqrt((def_x - x)^2 + (def_y - y)^2)) %>%
    summarize(avg_dist = mean(dist)) %>%
    as.matrix %>%
    as.vector
  
  man_all$max_dist[i] <-def %>%
    rename(def_x = x,
           def_y = y,
           def_nflId= nflId) %>%
    left_join(off %>% select(gameId,playId,frameId,nflId,x,y),by = c("gameId","playId","frameId")) %>%
    mutate(dist = sqrt((def_x - x)^2 + (def_y - y)^2)) %>%
    summarize(max_dist = max(dist)) %>%
    as.matrix %>%
    as.vector
  
  man_all$dist_snap[i] <- def %>%
    rename(def_x = x,
           def_y = y,
           def_nflId= nflId) %>%
    left_join(off %>% select(gameId,playId,frameId,nflId,x,y),by = c("gameId","playId","frameId")) %>%
    mutate(dist = sqrt((def_x - x)^2 + (def_y - y)^2)) %>%
    slice_min(order_by = frameId,n = 1) %>%
    select(dist) %>%
    as.matrix %>%
    as.vector
  
  man_all$dist_pass[i] <- def %>%
    rename(def_x = x,
           def_y = y,
           def_nflId= nflId) %>%
    left_join(off %>% select(gameId,playId,frameId,nflId,x,y),by = c("gameId","playId","frameId")) %>%
    mutate(dist = sqrt((def_x - x)^2 + (def_y - y)^2)) %>%
    slice_max(order_by = frameId,n = 1) %>%
    select(dist) %>%
    as.matrix %>%
    as.vector
  
  man_all$route[i] <- off$route[1]
  
  man_all$targeted[i] <- targetedReceiver %>%
    filter(gameId == off$gameId[1] & playId == off$playId[1]) %>%
    mutate(targeted = ifelse(targetNflId == off$nflId[1],1,0)) %>%
    select(targeted) %>%
    as.matrix %>%
    as.vector
  
  man_all$completed[i] <- ifelse(man_all$targeted[i]==1,off$passResult[1],NA)
  
  man_all$epa[i] <- ifelse(man_all$targeted[i]==1,off$epa[1],NA)
  
  man_all$isDefensivePI[i] <- ifelse(man_all$targeted[i]==1,off$isDefensivePI[1],NA)
  
  week1_filtered <- week %>%
    filter(gameId == man_all[i,"gameId"] & playId == man_all[i,"playId"] & nflId == man_all[i,"nflId"])
  
  if (man_all$targeted[i] == 1 & any(week1_filtered$event == "pass_arrived")){
    
    def_arrival <- week %>%
      filter(gameId == man_all[i,"gameId"] & playId == man_all[i,"playId"] & nflId == man_all[i,"nflId"] & event =="pass_arrived") %>%
      select(gameId,playId,frameId,event,x,y,nflId)
    
    off_arrival <- week %>%
      filter(gameId == man_all[i,"gameId"] & playId == man_all[i,"playId"] & nflId == man_all[i,"covering"] & event =="pass_arrived") %>%
      select(gameId,playId,frameId,event,x,y,nflId)
    
    man_all$dist_pass_arrival[i] <- def_arrival %>%
      rename(def_x = x,
             def_y = y,
             def_nflId= nflId) %>%
      left_join(off_arrival %>% select(gameId,playId,frameId,nflId,x,y),by = c("gameId","playId","frameId")) %>%
      mutate(dist = sqrt((def_x - x)^2 + (def_y - y)^2)) %>%
      slice_max(order_by = frameId,n = 1) %>%
      select(dist) %>%
      as.matrix %>%
      as.vector
    
    man_all$break_diff[i] = man_all$dist_pass_arrival[i] - man_all$dist_pass[i]
    
  }
  
  setTxtProgressBar(pb, i)
  
}
#9419

man_all_hold <- man_all

man_all <- filter(man_all,!is.na(covering))

man_all <- man_all[1:22518,]

write.csv(man_all,"man_all_20201222.csv",row.names = F)

man_all <- read_csv("man_all_20201222.csv")
