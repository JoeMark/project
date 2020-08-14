library(tidyverse)
library(dslabs)
library(tidyverse)
data("movielens")

head(movielens)

movielens %>%
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

keep <- movielens %>%
  dplyr::count(movieId) %>%
  top_n(5) %>%
  pull(movieId)
tab <- movielens %>%
  filter(userId %in% c(13:20)) %>%
  filter(movieId %in% keep) %>%
  select(userId, title, rating) %>%
  spread(title, rating)
tab %>% knitr::kable()

users <- sample(unique(movielens$userId), 100)
rafalib::mypar()
movielens %>% filter(userId %in% users) %>%
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 100)) %>%
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")

movielens %>%
  dplyr::count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black") +
  scale_x_log10() +
  ggtitle("Movies")

movielens %>%
  dplyr::count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black") +
  scale_x_log10() +
  ggtitle("Users")

# Large Dataset
library(caret)
set.seed(755)
test_index <- createDataPartition(y = movielens$rating, times = 1,
                                  p = 0.2, list = FALSE)
train_set <- movielens[-test_index,]
test_set <- movielens[test_index,]

test_set <- test_set %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((predicted_ratings - true_ratings)^2))
}

# The average rating
mu_hat <- mean(train_set$rating)
mu_hat

# Initial error loss
naive_rmse <- RMSE(test_set$rating, mu_hat)
naive_rmse

# As we go along, we will be comparing different approaches.
# Let’s start by creating a results table with this naive approach:
rmse_results <- tibble(method = "Just the average", RMSE = naive_rmse)





##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Q2

edx %>% filter(rating == 3) %>% count()
edx %>% group_by(movieId) %>% count()
edx %>% group_by(title) %>% count()
edx %>% group_by(userId) %>% count()
edx %>% filter(genres %in% "Drama") %>% count()

edx %>% filter(genres %in% "Comedy") %>% count()

edx %>% filter(title %in% c("Forrest Gump", "Jurassic Park", "Pulp Fiction",
                             "The Shawshank Redemption", "Speed 2: Cruise Control"))

edx %>% filter(title %in% "The")
edx %>% filter(title == "Pulp Fiction")
edx %>% filter(title == "The Shawshank Redemption")
edx %>% filter(title == "Speed 2: Cruise Control")
edx %>% filter(title == "Boomerang")
edx %>% group_by(title) %>% filter(title=="Forrest Gump")

edx %>%
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId),
            n_rating = n_distinct(rating),
            n_title = n_distinct(title),
            n_genres = n_distinct(genres))


# Q5


drama <- edx %>% filter(str_detect(genres,"Drama"))

comedy <- edx %>% filter(str_detect(genres,"Comedy"))

thriller<- edx %>% filter(str_detect(genres,"Thriller"))

romance<- edx %>% filter(str_detect(genres,"Romance"))

nrow(drama)

nrow(comedy)

nrow(thriller)

nrow(romance)


# Q6
edx %>% filter(str_detect(title %in% c("Forrest Gump", "Pulp Fiction"))) %>%
  group_by(title) %>% summarise(n = n())

pulp <- edx %>% filter(str_detect(title,"Pulp Fiction"))
forrest <- edx %>% filter(str_detect(title,"Forrest Gump"))
jurassic <- edx %>% filter(str_detect(title,"Jurassic Park"))
shaw <- edx %>% filter(str_detect(title,"Shawshank Redemption"))
speed2 <- edx %>% filter(str_detect(title,"Speed 2: Cruise Control"))

nrow(pulp)
nrow(forrest)
nrow(jurassic)
nrow(shaw)
nrow(speed2)

# Q7
edx %>% group_by(rating) %>% summarise(n = n())





edx_nest <- edx %>%
  group_by(userId) %>%
  nest()


