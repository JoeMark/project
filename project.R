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

rm(dl, ratings, movies, test_index, movielens, temp, removed)
#####################################################################################

########################################## Look at the data ##########################
# take a look at the edx dataset; the set we will use as a train and test set
glimpse(edx)
edx %>% as_tibble()

# Let's determine the number of unique users, unique movies and number of classes
# of ratings
edx %>%
  summarise(users=n_distinct(userId),
            movies=n_distinct(movieId),
            ratings=n_distinct(rating))

# Now, show the maximum possible number of ratings if each user gives a single rating
# to each unique movie.
edx %>%
  summarise(users=n_distinct(userId),
            movies=n_distinct(movieId),
            total=users*movies)

# the following snippet of code generates a matrix that verifies not every user rated every movie
set.seed(29, sample.kind = "Rounding")
mat <- edx %>%
  slice_sample(n=6) %>% select(userId, title, rating) %>%
  spread(key = title, value = rating) %>% as.matrix()
colnames(mat) <- 1:ncol(mat)
mat[, 2:5]

# now let's visualise a matrix with 100 movies and 100 users
# First,
set.seed(30, sample.kind = "Rounding")
a <- edx %>% slice_sample(n =100) %>% select(userId, title, rating) %>%
  spread(key = title, value = rating) %>% as.matrix()

# then change the column names to numbers
colnames(a) <- 1:ncol(a)

# Now,take a look at a visual representation of the distribution of ratings for each movie
y <- 1:ncol(a)-1
x <- 1:nrow(a)
image(x, y, a[, -1], col = rainbow(1), xlab = "movies", ylab = "users")

# Let's confirm that some movies receive more ratings, like blockbusters
# and some users are busier than others and hence award more ratings. We'll
# so this by selecting the the first 30 movies by ID and the first 30 users by ID

# let's look at the movies with IDs 1-30 and confirm there are some movies that attract
# more attention than others.
edx %>% filter(movieId <= 30) %>%
ggplot(aes(x=fct_infreq(as.factor(movieId)))) +
  geom_bar() +
  labs(x = "movieId", y = "number of times rated", title = "Movies")

# and here let's look at users with IDs between 1-30. Let's compare how engaged these users
# are relative to one another
edx %>% filter(userId <= 30) %>%
ggplot(aes(x=fct_infreq(as.factor(userId)))) +
  geom_bar() +
  labs(x = "userId", y = "number of movies rated", title = "Users")

############################ create training and test sets ###############################
# Let's begin the process of building a predictive algorithm
# First, let's create a test set from edx
set.seed(755, sample.kind = "Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2,
                                  list = FALSE)
train_set <- edx[-test_index, ]
test_set <- edx[test_index, ]

# Now make sure we do not include users and movies in the test set that do not appear in
# the train set
test_set <- test_set %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

############################## the loss function ####################################
# The algorithm is evaluated and assessed with a loss function
# We will use the residual mean squared error (RMSE)
rmse <- function(true_rating, predicted_rating){
  sqrt(mean((true_rating-predicted_rating)^2))
}

######################## a simple model ##############################################
# Now let's begin with the simplest recommendation algorithm:
# Predict the same rating for every movie regardless of user
mu_hat <- mean(train_set$rating)

naive_rmse <- rmse(test_set$rating, mu_hat)
naive_rmse
# Let's compare different approaches.
# Start by creating a results table with the naive approach
rmse_results <- tibble(method="Just the average",
                       RMSE=naive_rmse)
rmse_results

################################ movie effects ######################################
# Now augment the model to accommodate for the fact that some movies are generally
# rated higher than others. Add another term b. b indicates 'bias'
# fit <- lm(rating ~ as.factor(movieId), data = edx)
# don't run the code above. It will take too long. Instead we know
# that the least squares estimate of b is the average of the difference between the
# actual rating and the predicted rating

# going forward we will drop the hat to indicate estimate
mu <- mean(train_set$rating)
mu

movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

qplot(b_i, data = movie_avgs, bins = 10, color = I("black"))

# mu is 3.5, therefore a movie with b = 1.5 has a perfect rating
# how much does our prediction improve when we compute y_hat = mu_hat + b_hat
predicted_ratings <- mu + test_set %>%
  left_join(movie_avgs, by="movieId") %>%
  pull(b_i)
# here we see that our loss function has improved
rmse(predicted_ratings, test_set$rating)

########################## user effects #################################################
# Augment the model to take into account USER effects
# can we do better?
# look at the average rating per user
train_set %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating)) %>%
  filter(n() >= 100) %>%
  ggplot(aes(b_u)) +
  geom_histogram(bins = 30, color = "black")
# The graphic shows us the variability across users.
# So we have to factor this user variabilty into our model
user_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarise(b_u=mean(rating-mu-b_i))
# now construct predictors and see if there is an improvement to RMSE.
predicted_ratings <- test_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

rmse(predicted_ratings, test_set$rating)

########################## Regularisation ######################################
# Augment the model to include regularisation techniques
# Can we do even better?
# Let's look at regularisation. A solution to overfitting
# What were the mistakes to our approach thus far?
# Below look at the 10 biggest errors
train_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(residual = rating - (mu + b_i)) %>%
  arrange(desc(abs(residual))) %>%
  slice(1:10) %>%
  pull(title)

# Now let's look at the 10 worst and 10 best movies based on our estimate b_i
# First, connect movieId to title.
movie_titles <- train_set %>%
  select(movieId, title) %>%
  distinct()

# 10 best
movie_avgs %>% left_join(movie_titles, by = "movieId") %>%
  arrange(desc(b_i)) %>%
  slice(1:10) %>%
  pull(title)

# 10 worst
movie_avgs %>% left_join(movie_titles, by = "movieId") %>%
  arrange(b_i) %>%
  slice(1:10) %>%
  pull(title)

# here is how often these were rated
train_set %>% count(movieId) %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(desc(b_i)) %>%
  slice(1:10) %>%
  pull(n)

train_set %>% count(movieId) %>%
  left_join(movie_avgs) %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(b_i) %>%
  slice(1:10) %>%
  pull(n)

############# Regularisation for movie effects ##########################################
lambda <- 3
mu <- mean(train_set$rating)
movie_reg_avgs <- train_set %>%
  group_by(movieId) %>%
  summarise(b_i = sum(rating - mu)/(n() + lambda), n_i = n())

# To see how the estimates shrink, below see a plot of the regularised estimates
# versus the least squares estimates.
tibble(original = movie_avgs$b_i, regularised = movie_reg_avgs$b_i,
       n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularised, size = sqrt(n))) +
  geom_point(shape = 1, alpha = 0.5)

# Now let's look at the top 10 movies based on the penalised estimates
train_set %>%
  count(movieId) %>%
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(desc(b_i)) %>%
  slice(1:10) %>%
  pull(title)

# And here are the 10 'worst' movies.
train_set %>%
  count(movieId) %>%
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(b_i) %>%
  slice(1:10) %>%
  pull(title)

# Does the penalised version improve our result?
predicted_ratings <- test_set %>%
  left_join(movie_reg_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

reg_movie <- RMSE(predicted_ratings, test_set$rating)
reg_movie

# The solution (below) shows us using cross-validation to pick the parameter lambda
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){

  mu <- mean(train_set$rating)

  b_i <- train_set %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu)/(n() + l))

  b_u <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - b_i - mu)/(n() + l))

  predicted_ratings <- test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)

  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(lambdas, rmses)

lambda <- lambdas[which.min(rmses)]
lambda

reg_movie_user <- min(rmses)
reg_movie_user

###### validation set #################################################################
validation_set <- validation %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# If we predict that all unknown ratings equals mu then the RMSE is:
naive_rmse <- RMSE(validation_set$rating, mu_hat)
naive_rmse

# model augmented to include movie effects
predicted_ratings <- mu + validation_set %>%
  left_join(movie_avgs, by="movieId") %>%
  pull(b_i)

movie_effect <- RMSE(predicted_ratings, validation_set$rating)
movie_effect

# model augmented to include user effects
user_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu - b_i))

# now construct predictors and see if there is an improvement to RMSE.
predicted_ratings <- validation_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

plus_user <- RMSE(predicted_ratings, validation_set$rating)
plus_user

# model augmented to include Regularisation for movie effects
lambda <- 3
mu <- mean(train_set$rating)
movie_reg_avgs <- train_set %>%
  group_by(movieId) %>%
  summarise(b_i = sum(rating - mu)/(n() + lambda), n_i = n())

predicted_ratings <- validation_set %>%
  left_join(movie_reg_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

reg_movie <- RMSE(predicted_ratings, validation_set$rating)
reg_movie

# model augmented to include cross validation and lambda
# The solution (below) shows us using cross-validation to pick the parameter lambda
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){

  mu <- mean(train_set$rating)

  b_i <- train_set %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu)/(n() + l))

  b_u <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - b_i - mu)/(n() + l))

  predicted_ratings <- validation_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)

  return(RMSE(predicted_ratings, validation_set$rating))
})

qplot(lambdas, rmses)

lambda <- lambdas[which.min(rmses)]
lambda

reg_movie_user <- min(rmses)
reg_movie_user

# Let's compare different approaches.
# Here is the results table.
(rmse_results <- tibble(method = c("Just the average", "Movie Effect Model",
                                   "Movie + User Effects Model",
                                   "Regularised Movie Effect Model",
                                   "Regularised Movie + User Effect Model"),
                        RMSE = c(naive_rmse, movie_effect, plus_user,
                                 reg_movie, reg_movie_user)))

##########################################################################################


