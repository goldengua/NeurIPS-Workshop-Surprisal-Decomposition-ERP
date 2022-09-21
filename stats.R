library(dplyr)
library(lmerTest)
library(brms)
library(ggplot2)
library(MuMIn)
library(simr)
library("rstan")
library(multcomp)
library(Rmisc)
library(emmeans)
library(lsmeans)
library(tidyverse)
rm(list = ls())
setwd('/directory')
data <- read.csv('result_roberta_base_400_human_merged.csv')
data <- subset(data, data$artefact!=1)
data$true_surprisal <- as.numeric(data$true_surprisal)
data$heuristic_surprisal <- as.numeric(data$heuristic_surprisal)
data$structural_update <- as.numeric(data$structural_update)

data$meanAmp <- as.numeric(data$meanAmp)
data$subject <- as.factor(data$subject)

data_n400 <- subset(data,data$time_window=="n4")
data_p600 <- subset(data,data$time_window=="p6")



m0 <- lmer(meanAmp ~ true_surprisal + (1 + true_surprisal|subject) + (1 + true_surprisal|item), data = data_n400)
summary(m0)

m1 <- lmer(meanAmp ~ heuristic_surprisal  + (1 + heuristic_surprisal|subject) + (1 + heuristic_surprisal|item), data = data_n400)
summary(m1)
anova(m0,m1)


m2 <- lmer(meanAmp ~ true_surprisal + (1 + true_surprisal|subject) + (1 + true_surprisal|item), data = data_p600)
summary(m2)

m3 <- lmer(meanAmp ~ structural_update +  (1 + structural_update|subject) + (1 + structural_update|item), data = data_p600)
summary(m3)
anova(m2,m3)


