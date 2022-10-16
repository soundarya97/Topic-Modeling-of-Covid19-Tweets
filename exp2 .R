## installing all required packages
install.packages("tm")
install.packages("SnowballC")
install.packages("topicmodels")
install.packages("syuzhet")
install.packages("textreg")
install.packages("devtools")
install_github('trinker/sentimentr')
install.packages("textmineR")
install.packages("broom")
install.packages("dplyr")
install.packages("tidyverse")
install.packages("imputeTS")

## calling all required packages
library(tm)
library(SnowballC)
library(topicmodels)
library(syuzhet)
library(textreg)
library(devtools)
library(sentimentr)
library(textmineR)
library(ggplot2)
library(broom)
library(dplyr)
library(tidyverse)
library(imputeTS)
library(stringr)

options(scipen = 100)
#set working directory (modify path as needed)
wd <- "/Users/soundaryamantha/Documents/Penn State - Sem2 Spring1/Independent Study/AZ Tweets"
setwd(wd)

#load files into corpus
#get listing of .txt files in directory
filenames <- list.files(getwd(),pattern="*.txt")
#read files into a character vector
files <- lapply(filenames,readLines)
#create corpus from vector
docs <- Corpus(VectorSource(files))
#inspect a particular document in corpus
writeLines(as.character(docs[[30]]))

docs_raw <- Corpus(VectorSource(files))
writeLines(as.character(docs_raw[[30]]))
#setwd("C://Users//sus64//Desktop//Laptop2020//downloads//Tweets//TM")
setwd("/Users/soundaryamantha/Documents/Penn State - Sem2 Spring1/Independent Study/AZ TM")
#setwd(choose.dir())

### Data Preprocessing ###
#remove punctuation
docs <- tm_map(docs, removePunctuation)
#Strip digits
docs <- tm_map(docs, removeNumbers)
#remove stopwords
docs <- tm_map(docs, removeWords, stopwords('english'))
#remove whitespace
docs <- tm_map(docs, stripWhitespace)
#Good practice to check every now and then
writeLines(as.character(docs[[30]]))
#Stem document
docs <- tm_map(docs,stemDocument)

myStopwords <- stopwords("en")
docs <- tm_map(docs, removeWords, myStopwords)
#inspect a document as a check
writeLines(as.character(docs[[30]]))
#Transform to lower case
docs <-tm_map(docs,content_transformer(tolower))

## creating a document term matrix 
df <- data.frame(text = get("content", docs))
df$id <- as.integer(rownames(df))-1

write.csv(df,"Arizona.csv")

###################################
dtm <- CreateDtm(doc_vec = df$text,doc_names = df$id,cpus = 2) # default is all available cpus on the system

#convert rownames to filenames
rownames(dtm) <- filenames
#collapse matrix by summing over columns
freq <- colSums(as.matrix(dtm))
#length should be total number of terms
length(freq)
#create sort order (descending)
ord <- order(freq,decreasing=TRUE)
#List all terms in decreasing order of freq and write to disk
freq[ord]
write.csv(freq[ord],"word_freq.csv")

#load topic models library
set.seed(12345)

## checking model with k=14
model3 <- FitLdaModel(dtm = dtm, 
                      k = 14,
                      iterations = 500, 
                      burnin = 180,
                      alpha = 0.1,
                      beta = 0.05,
                      optimize_alpha = TRUE,
                      calc_likelihood = TRUE,
                      calc_coherence = TRUE,
                      calc_r2 = TRUE,
                      cpus = 2) 

View(CalcHellingerDist(model3$phi))
median(CalcHellingerDist(model3$phi))
#0.4105011
mean(CalcHellingerDist(model3$phi))
#0.3912592

model_theta <- as.data.frame(model3$theta)
docs_to_topics <- colnames(model_theta)[max.col(model_theta,ties.method="first")]
docs_to_topics <- as.matrix(docs_to_topics)
rownames(docs_to_topics) <- filenames

#write out results
#docs to topics
write.csv(docs_to_topics,file=paste("LDAGibbs",14,"DocsToTopics.csv"))

#top 100 terms in each topic
model_top_terms <- GetTopTerms(phi = model3$phi, M = 100)
model_top_terms <- as.data.frame(model_top_terms)

single_col <- model_top_terms %>% rownames_to_column('row') %>% pivot_longer(cols = -row)
freq <- single_col %>% group_by(value) %>% summarise(frequency_val = n())
repeated_words <- freq %>% filter(frequency_val >= 2)

spread_op <- spread(single_col, key=name, value=row)
value <- as.data.frame(spread_op$value)
names(value) <- "value"
spread_op <- sapply(spread_op %>% select(-c(value)), as.numeric)
spread_op[is.na(spread_op)] <- 0
spread_op[spread_op > 1] <- 1
spread_op <- as.data.frame(spread_op)
spread_op <- cbind(value,spread_op)

words_prob <- as.data.frame(model3$phi)
trans <- as.data.frame(t(words_prob))
trans <- tibble::rownames_to_column(trans, "value")
word_prob <- inner_join(trans,value,by = "value")
write.csv(spread_op,"all_words_topics_presence.csv")
write.csv(word_prob,"all_words_prob.csv")
### checking the repeated words ##

repeat_topics <- inner_join(repeated_words,spread_op,by = "value")
repeat_topic_freq <- inner_join(repeated_words,word_prob,by = "value")
repeat_topics$word_length = str_length(repeat_topics$value)
repeat_topic_freq$word_length = str_length(repeat_topic_freq$value)
repeat_topic_freq$weight <- log(14/repeat_topic_freq$frequency_val)/log(14)
repeat_topics$weight <- log(14/repeat_topic_freq$frequency_val)/log(14)

## checking for frequent words in english ##
repeating_words <- as.data.frame(repeat_topic_freq$value)
colnames(repeating_words) <- c('value')

english_freq <- read.csv("/Users/soundaryamantha/Documents/Penn State - Sem2 Spring1/Independent Study/unigram_freq.csv")
overlap_words <- inner_join(repeating_words,english_freq,by = c('value'='word'))
missing_words <- anti_join(repeating_words,english_freq,by = c('value'='word'))
missing_words$count <- max(overlap_words$count)
overlap_words <- rbind(overlap_words,missing_words)
#english_freq$stemmed_words <- tm_map(english_freq$word,stemDocument)
repeat_topics <- inner_join(repeat_topics,overlap_words,by = "value")
repeat_topics$final_weight <- (repeat_topics$word_length*repeat_topics$weight)/(repeat_topics$count)

## min-max normalization for final weight ##
norm_minmax <- function(x){
  (x- min(x)) /(max(x)-min(x))
}
repeat_topics$normal_weight <- norm_minmax(repeat_topics$final_weight)

repeat_topics_final <- repeat_topics %>% filter(normal_weight < 0.7)
repeat_topics_final <- repeat_topics_final %>% select(-c(frequency_val,word_length,weight,count,final_weight,normal_weight))

final_words_topics <- anti_join(spread_op,repeat_topics_final,by = "value")

topic_1 <- model_top_terms[rowSums(sapply(final_words_topics$value, grepl, model_top_terms$t_1)) > 0, , drop = FALSE]
topic_2 <- model_top_terms[rowSums(sapply(final_words_topics$value, grepl, model_top_terms$t_2)) > 0, , drop = FALSE]
topic_3 <- model_top_terms[rowSums(sapply(final_words_topics$value, grepl, model_top_terms$t_3)) > 0, , drop = FALSE]
topic_4 <- model_top_terms[rowSums(sapply(final_words_topics$value, grepl, model_top_terms$t_4)) > 0, , drop = FALSE]
topic_5 <- model_top_terms[rowSums(sapply(final_words_topics$value, grepl, model_top_terms$t_5)) > 0, , drop = FALSE]
topic_6 <- model_top_terms[rowSums(sapply(final_words_topics$value, grepl, model_top_terms$t_6)) > 0, , drop = FALSE]
topic_7 <- model_top_terms[rowSums(sapply(final_words_topics$value, grepl, model_top_terms$t_7)) > 0, , drop = FALSE]
topic_8 <- model_top_terms[rowSums(sapply(final_words_topics$value, grepl, model_top_terms$t_8)) > 0, , drop = FALSE]
topic_9 <- model_top_terms[rowSums(sapply(final_words_topics$value, grepl, model_top_terms$t_9)) > 0, , drop = FALSE]
topic_10 <- model_top_terms[rowSums(sapply(final_words_topics$value, grepl, model_top_terms$t_10)) > 0, , drop = FALSE]
topic_11 <- model_top_terms[rowSums(sapply(final_words_topics$value, grepl, model_top_terms$t_11)) > 0, , drop = FALSE]
topic_12 <- model_top_terms[rowSums(sapply(final_words_topics$value, grepl, model_top_terms$t_12)) > 0, , drop = FALSE]
topic_13 <- model_top_terms[rowSums(sapply(final_words_topics$value, grepl, model_top_terms$t_13)) > 0, , drop = FALSE]
topic_14 <- model_top_terms[rowSums(sapply(final_words_topics$value, grepl, model_top_terms$t_14)) > 0, , drop = FALSE]


install.packages("gdata")
library(gdata)

final_words_topics <- cbindX(topic_1,topic_2,topic_3,topic_4,topic_5,topic_6,topic_7,topic_8,topic_9,topic_10,topic_11,topic_12,topic_13,topic_14)

write.csv(repeat_topics,"repeated_words_1_0.csv")
write.csv(repeat_topic_freq,"repeated_words_prob.csv")
write.csv(repeated_words,"repeated_words.csv")
write.csv(model_top_terms,file=paste("LDAGibbs",14,"TopicsToTerms.csv"))
write.csv(final_words_topics,file=paste("LDAGibbs",14,"TopicsToTerms_weightage_op.csv"))

#############
#probabilities associated with each topic assignment
topicProbabilities <- as.data.frame(model3$gamma)
write.csv(topicProbabilities,file=paste("LDAGibbs",k,"TopicProbabilities.csv"))

