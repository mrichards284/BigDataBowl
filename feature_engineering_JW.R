# Packages
library(tidyverse)

# Set Working Directory
setwd("/Users/jackwerner/Documents/My Stuff/Football/Big Data Bowl 2020")

# Data - this will take a minute. It loads all weeks

players <- read.csv("data/players.csv", stringsAsFactors=FALSE)
games <- read.csv("data/games.csv", stringsAsFactors=FALSE)
plays <- read.csv("data/plays.csv", stringsAsFactors=FALSE)

###
# Create reference of each player's closest teammate and opponent for each frame
###
i <- 17
week_data <- read_csv(paste0("data/week", i, ".csv"))

find_closest <- function(frame_df) {
  frame_df_2 <- frame_df %>% 
    filter(team != "football") %>%
    arrange(team, jerseyNumber)
  
  dist_mat <- frame_df_2 %>%
    select(x, y) %>% 
    as.matrix() %>%
    dist(upper = T) %>%
    as.matrix()
  
  num_away <- sum(frame_df_2$team == "away")
  
  dist_mat[dist_mat == 0] <- Inf
  dist_mat_opp <- dist_mat; dist_mat_team <- dist_mat
  
  dist_mat_opp[1:num_away,1:num_away] <- Inf
  dist_mat_opp[(num_away+1):nrow(dist_mat_opp),(num_away+1):ncol(dist_mat_opp)] <- Inf
  
  dist_mat_team[1:num_away,(num_away+1):ncol(dist_mat_opp)] <- Inf
  dist_mat_team[(num_away+1):nrow(dist_mat_opp),1:num_away] <- Inf
  
  frame_df_2$closest_teammate <- frame_df_2$nflId[apply(dist_mat_team, 1, which.min)]
  frame_df_2$closest_opponent <- frame_df_2$nflId[apply(dist_mat_opp, 1, which.min)]
  
  return(frame_df_2 %>% dplyr::select(nflId, closest_teammate, closest_opponent))
}

ptm <- proc.time()
closest_players_ref <- week_data %>%
  # Filter out 2018090912 2439 49-58: no players
  group_by(gameId, playId, frameId) %>%
  filter(sum(team == "away") > 0, sum(team == "home") > 0) %>%
  ungroup %>%
  group_by(gameId, playId, frameId) %>%
  group_modify(~find_closest(.x))
(proc.time() - ptm)/60

write.csv(closest_players_ref, 
          paste0("data/closest_players_ref_week", i, ".csv"), row.names = F)

rm(week_data)

###
# Feature creation
###

for (i in 3:17) {
  closest_players_ref <- read.csv(paste0("data/closest_players_ref_week", i, ".csv"))
  week_data <- read_csv(paste0("data/week", i, ".csv"))
  
  end_events <- c("pass_forward", "out_of_bounds", "qb_sack", "qb_strip_sack", "pass_shovel")
  
  week_features <- week_data %>%
    # Get start frame
    left_join(week_data %>%
                filter(event %in% end_events) %>%
                group_by(playId,gameId) %>%
                summarize(frameId_end = min(frameId)) %>%
                ungroup() ,by = c("gameId","playId")) %>%
    # Get end frame
    left_join(week_data %>%
                filter(event == "ball_snap") %>%
                group_by(playId,gameId) %>%
                summarize(frameId_start = min(frameId)) %>%
                ungroup() ,by = c("gameId","playId")) %>%
    # Join Line of Scrimmage
    left_join(plays %>%
                select(gameId, playId, absoluteYardlineNumber, yardlineNumber)
              , by = c("gameId", "playId")) %>%
    # Filter only events between snap and pass
    filter(frameId >= frameId_start, frameId <= frameId_end) %>%
    # Join player proximity data
    left_join(closest_players_ref, by = c("gameId", "playId", "frameId", "nflId")) %>%
    left_join(select(week_data, gameId, playId, frameId, nflId
                     , closest_teammate_x = x
                     , closest_teammate_y = y 
                     , closest_teammate_dir = dir),
              by = c("gameId", "playId", "frameId", "closest_teammate" = "nflId")) %>%
    left_join(select(week_data, gameId, playId, frameId, nflId
                     , closest_opponent_x = x
                     , closest_opponent_y = y 
                     , closest_opponent_dir = dir),
              by = c("gameId", "playId", "frameId", "closest_opponent" = "nflId")) %>%
    mutate(off = sqrt((x - closest_opponent_x)^2 + (y - closest_opponent_y)^2)
           , def = sqrt((x - closest_teammate_x)^2 + (y - closest_teammate_y)^2)
           , off_dir = pmin(abs(dir - closest_opponent_dir), 360 - abs(dir - closest_opponent_dir))
           , rat = off/sqrt((closest_opponent_x - closest_teammate_x)^2 + 
                              (closest_opponent_y - closest_teammate_y)^2)
           , o_adj = ifelse(playDirection == "left", o - 90, o-270)*2*pi/360
           , o_front = cos(o_adj)
           , x_los = case_when(!is.na(absoluteYardlineNumber) ~ abs(x - absoluteYardlineNumber)
                               , TRUE ~ pmin(abs(10 + yardlineNumber - x)
                                             , abs(110 - yardlineNumber - x)))
    ) %>%
    arrange(gameId, playId, nflId, frameId) %>%
    group_by(gameId, playId, nflId) %>%
    summarize(
      var_x = var(x) # var_x: Variance in X
      , var_y = var(y) # var_y: Variance in Y
      , speed_var = var(s) # speed_var: Variance in Speed
      , off_var = var(off) # off_var: Variance in distance to nearest offensive player
      , def_var = var(def) # def_var: Variance in distance to nearest defensive player
      , off_mean = mean(off) # off_mean: Mean distance to nearest offensive player
      , def_mean = mean(def) # def_mean: Mean distance to nearest defensive player
      , off_dir_var = var(off_dir) # off_dir_var: Variance in difference in direction from nearest offensive player
      , off_dir_mean = mean(off_dir) # off_dir_mean: Mean difference in direction from nearest offensive player
      , rat_mean = mean(rat) # rat_mean: Mean ratio of distance to nearest offensive player and between nearest offensive and defensive player
      , rat_var = var(rat) # rat_var: Variance of the ratio of distance to nearest offensive player and between nearest offensive and defensive player
      , o_front_mean = mean(o_front)
      , los_dist_snap = mean(x_los[event == "ball_snap"])
      , play_frames = n()
    ) %>%
    ungroup()
  
  write.csv(week_features, paste0("data/week", i, "_features.csv"), row.names = F)
  
  print(paste0("Week ", i, " done!"))
}


# Combine Weeks
week1_features <- read.csv("data/week1_features.csv", stringsAsFactors = FALSE)
week2_features <- read.csv("data/week2_features.csv", stringsAsFactors = FALSE)
week3_features <- read.csv("data/week3_features.csv", stringsAsFactors = FALSE)
week4_features <- read.csv("data/week4_features.csv", stringsAsFactors = FALSE)
week5_features <- read.csv("data/week5_features.csv", stringsAsFactors = FALSE)
week6_features <- read.csv("data/week6_features.csv", stringsAsFactors = FALSE)
week7_features <- read.csv("data/week7_features.csv", stringsAsFactors = FALSE)
week8_features <- read.csv("data/week8_features.csv", stringsAsFactors = FALSE)
week9_features <- read.csv("data/week9_features.csv", stringsAsFactors = FALSE)
week10_features <- read.csv("data/week10_features.csv", stringsAsFactors = FALSE)
week11_features <- read.csv("data/week11_features.csv", stringsAsFactors = FALSE)
week12_features <- read.csv("data/week12_features.csv", stringsAsFactors = FALSE)
week13_features <- read.csv("data/week13_features.csv", stringsAsFactors = FALSE)
week14_features <- read.csv("data/week14_features.csv", stringsAsFactors = FALSE)
week15_features <- read.csv("data/week15_features.csv", stringsAsFactors = FALSE)
week16_features <- read.csv("data/week16_features.csv", stringsAsFactors = FALSE)
week17_features <- read.csv("data/week17_features.csv", stringsAsFactors = FALSE)

features <- rbind(week1_features, week2_features, week3_features, week4_features
                  , week5_features, week6_features, week7_features, week8_features
                  , week9_features, week10_features, week11_features, week12_features
                  , week13_features, week14_features, week15_features, week16_features
                  , week17_features)

write.csv(features, "data/features_all.csv", row.names = F)



### Post Analysis

crazies <- week1_features %>% filter(los_dist_snap > 100)
playId_ex <- crazies$playId[1]
gameId_ex <- crazies$gameId[1]

sample_play <- week1 %>%
  filter(playId == playId_ex & gameId == gameId_ex) %>%
  inner_join(games) %>% inner_join(plays) 

sample_play %>% filter(nflId == nflId_ex) %>% select(displayName, jerseyNumber) %>% unique()

week1_features %>% 
  left_join(week1, by = c("gameId", "playId", "nflId")) %>%
  filter(playId == playId_ex & gameId == gameId_ex, event == "ball_snap") %>%
  select(gameId, playId, nflId, jerseyNumber, los_dist_snap)

animate_play_func(sample_play) 




       