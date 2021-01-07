# Packages
library(tidyverse)
library(gganimate)
library(ggmap)
library(mclust)

# Set Working Directory
setwd("/Users/jackwerner/Documents/My Stuff/Football/Big Data Bowl 2020")

source("src/zz_animate_play_func.R")

week1 <- read_csv("data/week1.csv")
features <- read.csv("data/features_all.csv", stringsAsFactors = FALSE)
players <- read.csv("data/players.csv", stringsAsFactors=FALSE)
games <- read.csv("data/games.csv", stringsAsFactors=FALSE)
plays <- read.csv("data/plays.csv", stringsAsFactors=FALSE)
coverages <- read.csv("data/coverages_wk1.csv", stringsAsFactors = FALSE)


cornerbacks <- features %>%
  left_join(players, by = "nflId") %>%
  #filter(position == "CB") # Just cornerbacks
  #filter(position %in% c("CB", "S", "SS", "FS", "DB")) %>%# All defensive backs
  filter(position %in% c("CB", "S", "SS", "FS", "DB", "DE", "DT", "ILB", "LB", "MLB", "NT")) %>%
  filter(rat_mean != Inf
         , !is.na(off_dir_var)
         , !is.na(off_dir_mean)
         , !is.na(o_front_mean)
         , !is.na(los_dist_snap)) %>%
  mutate(db_type = case_when(
    position == "CB" ~ "CB"
    , position %in% c("S", "SS", "FS") ~ "S"
    , position %in% c("ILB", "LB", "MLB") ~ "LB"
    , position %in% c("DE", "NT", "DT") ~ "DL"
    , TRUE ~ "NA"
  ))

cb_mat <- cornerbacks %>%
  #filter(!is.na(los_dist_snap)) %>%
  select(var_x, var_y, speed_var, off_var, def_var, off_mean, def_mean, off_dir_var,
         off_dir_mean, rat_mean, rat_var, o_front_mean
         #, los_dist_snap
         ) %>%
  #select(off_mean, rat_mean, rat_var, o_front_mean) %>%
  as.matrix() %>%
  scale()

#cor(cornerbacks$play_frames, cb_mat)

ggplot(data = cornerbacks, aes(x = off_mean, y = def_mean, color = db_type)) + geom_point(alpha = .2)

# Problems
# cornerbacks %>% filter(off_mean == def_mean) %>% dim()
# cornerbacks %>% filter(rat_mean == Inf) %>% dim()
# cornerbacks %>% filter(rat_mean == Inf) %>% dim()


##############
# CLUSTERING #
##############

# Cluster
cb_km <- kmeans(cb_mat, centers = 2)

# GMM
#gmm <- Mclust(cb_mat, G = 2)

# PCA (for visualization)
cb_pca <- prcomp(cb_mat)
cb_pca$rotation


# Plot
cb_full <- cbind(cornerbacks[!is.na(cornerbacks$los_dist_snap),], 
                 cb_pca$x, cluster_km = cb_km$cluster
                 #, cluster_gmm = gmm$classification
                 )


cb_out <- cb_full %>% 
  mutate(coverage_type = case_when(cluster_km == 2 ~ "zone"
                                   , TRUE ~ "man")) %>%
  select(gameId, playId, nflId, coverage_type)

#write.csv(cb_out, file = "data/all_weeks_zone_man.csv", row.names = F)


ggplot(data = cb_full, aes(x = PC1, y = PC2, color = factor(cb_km$cluster))) +
  facet_wrap(~db_type) +
  geom_point(alpha = .5) +
  coord_fixed() #+ theme(legend.position = "none")


cb_full %>%
  left_join(games %>%
              select(gameId, week),
            by = c("gameId")) %>%
  group_by(week, db_type) %>%
  summarize(clust1 = sum(cluster_km == 1)
            , clust2 = sum(cluster_km == 2)) %>%
  ungroup() %>%
  mutate(clust1_pct = clust1/(clust1 + clust2)) %>%
  ggplot(aes(x = week, y = clust1_pct, color = db_type)) +
    geom_point() + geom_line()

ggplot(data = cb_full, aes(x = PC1, y = PC2, color = factor(gmm$classification))) +
  geom_point(alpha = .75) +
  coord_fixed() +
  theme(legend.position = "none")


ggplot(data = filter(cb_full, position %in% c("CB", "FS", "SS")), 
                     aes(x = o_front_mean, color = position, fill = position)) + 
  geom_density(alpha = .2)

###
# Investigate specific cases
###

cb_full %>% 
  filter(PC2 < -3, db_type == "CB") %>% 
  select(gameId, playId, nflId, db_type, PC1, PC2)

cb_full %>% 
  filter(PC2 < -3, db_type == "S") %>% 
  arrange(desc(PC2)) %>%
  select(gameId, playId, nflId, db_type, PC1, PC2)


# Leftmost
playId_ex <- 4464; gameId_ex <- 2018090905; nflId_ex <- 2541243 # Hail Mary
playId_ex <- 3325; gameId_ex <- 2018090901; nflId_ex <- 2560712 # Zone?

# Uppermost (Right)
playId_ex <- 3607; gameId_ex <- 2018090900; nflId_ex <- 2543851 # Tight man across field
playId_ex <- 4896; gameId_ex <- 2018090901; nflId_ex <- 2555344 # Tight man across field

# Uppermost LB
playId_ex <- 829; gameId_ex <- 2018090905; nflId_ex <- 2561143 # Man coverage
playId_ex <- 2590; gameId_ex <- 2018090909; nflId_ex <- 2543769 # Rushing the QB

# Lowest
playId_ex <- 2710; gameId_ex <- 2018090905; nflId_ex <- 2558094 # Zone? Quick pass
playId_ex <- 3235; gameId_ex <- 2018090903; nflId_ex <- 2354 # Rushing passer

# Lowest CB
playId_ex <- 3645; gameId_ex <- 2018090905; nflId_ex <- 2552255 # Quick pass

# High-mid safety
playId_ex <- 939; gameId_ex <- 2018091001; nflId_ex <- 2556277 # Short play

sample_play <- week1 %>%
  filter(playId == playId_ex & gameId == gameId_ex) %>%
  inner_join(games) %>% inner_join(plays) 

sample_play %>% filter(nflId == nflId_ex) %>% select(displayName, jerseyNumber, position) %>% unique()

animate_play_func(sample_play) 


###
# Compare to Coverages
###

play_coverages <- cb_full %>%
  group_by(gameId, playId, cluster_km) %>%
  summarize(players = n()) %>%
  ungroup() %>%
  pivot_wider(names_from = cluster_km, values_from = players, values_fill = 0, 
              names_prefix = "cluster_") %>%
  left_join(coverages, by = c("gameId", "playId")) %>%
  group_by(coverage) %>%
  summarize(cluster_1 = mean(cluster_1), cluster_2 = mean(cluster_2),
            backs = cluster_1 + cluster_2, tot = n()) %>%
  ungroup()


################
# PREDICT CB/S #
################

# cornerbacks
# cb_mat



mod_df <- data.frame(s_ind = y_vec, cb_mat)
lr <- glm(s_ind ~ ., data = mod_df, family = binomial(link = "logit"))

y_vec <- ifelse(cornerbacks$db_type == "S", 1, 0)
folds <- 10
fold_ids <- sample(1:10, dim(cb_mat)[1], replace = T)
preds_glm <- rep(-1, length(fold_ids))

for (i in 1:folds) {
  x_train <- cb_mat[fold_ids != i,]
  y_train <- y_vec[fold_ids != i]
  train_df <- data.frame(s_ind = y_train, x_train)
  
  x_test <- cb_mat[fold_ids == i,]
  y_test <- y_vec[fold_ids == i]
  test_df <- data.frame(s_ind = y_test, x_test)
  
  lr_mod <- glm(s_ind ~ ., data = train_df, family = binomial(link = "logit"))
  
  preds_glm[fold_ids == i] <- predict(lr_mod, test_df, type = "response")
  
  print(paste0("Done with ", i))
}

preds_df <- data.frame(s_ind = y_vec, cb_mat, pred = preds_glm, cluster = cb_km$cluster)
table(preds_df$pred > .5, preds_df$cluster)

ggplot(data = preds_df, aes(x = pred)) + facet_grid(s_ind ~ .) + geom_histogram()



