library(R.matlab)
library(dplyr)
library(scales)
library(ggplot2)

behavior <- readMat("E:/BU Study_2022/679_Final Project/binned_behavior.mat")
behavior <- as.data.frame(behavior)

zscore <- readMat("E:/BU Study_2022/679_Final Project/binned_zscore.mat")
zscore <- as.data.frame(zscore)

range(behavior)
range(zscore) # -2.355535 to 15.632968

################################################################################
# zscore (draft)

colnames(zscore) <- paste0("V", c(1:47)) 
#colnames(zscore) <- paste0("V", seq.int(47)) 
#colnames(zscore) <- c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","V29","V30","V31","V32","V33","V34","V35","V36","V37","V38","V39","V40","V41","V42","V43","V44","V45","V46","V47")

z1 <- zscore %>% count(range= V1>10) %>% filter(!row_number() %in% c(1))
z2 <- zscore %>% count(range= 5<V2 & V2<=10) %>% filter(!row_number() %in% c(1))
z3 <- zscore %>% count(range= 3<V3 & V3<=5) %>% filter(!row_number() %in% c(1))
z4 <- zscore %>% count(range= 0<V4 & V4<=3) %>% filter(!row_number() %in% c(1))
z5 <- zscore %>% count(range= -3<V5 & V5<=0) %>% filter(!row_number() %in% c(1))
z <- rbind(z1,z2,z3,z4,z5) %>% mutate(ratio = percent(n/6300, accuracy=0.01)) %>% select(-c(1)) %>% mutate(range = c(">10","<5 & <=10","<3 & <=5","<0 & <=3","<-3 &<=0")) %>% mutate(cell = "1")

##############################################################################
# function (draft)

colnames(zscore) <- c("value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value")

my_function <- function(V) {
  z1 <- zscore %>% select(V) %>% count(range= value>10) %>% filter(!row_number() %in% c(1)) %>% mutate(range = ">10")
  z2 <- zscore %>% select(V) %>% count(range= 5<value & value<=10) %>% filter(!row_number() %in% c(1)) %>% mutate(range = "<5 & <=10")
  z3 <- zscore %>% select(V) %>% count(range= 3<value & value<=5) %>% filter(!row_number() %in% c(1)) %>% mutate(range = "<3 & <=5")
  z4 <- zscore %>% select(V) %>% count(range= 0<value & value<=3) %>% filter(!row_number() %in% c(1)) %>% mutate(range = "<0 & <=3")
  z5 <- zscore %>% select(V) %>% count(range= -3<value & value<=0) %>% filter(!row_number() %in% c(1)) %>% mutate(range = "<-3 &<=0")
  z <- rbind(z1,z2,z3,z4,z5) %>% mutate(ratio = percent(n/6300, accuracy=0.01)) %>% select(-c(1))  %>% mutate(cell = "V")
# print(z)
  return(z)
# assign(z, envir=.GlobalEnv)
}
my_function(2)

###############################################################################
# for loop 1 (final version)

colnames(zscore) <- c("value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value")

z <- data.frame()
z_final <- data.frame()

for (i in 1:47) {
  z_1 <- zscore %>% select(i) %>% count(range= value>3) %>% filter(!row_number() %in% c(1)) %>% mutate(range = ">3")
  z_2 <- zscore %>% select(i) %>% count(range= 2<value & value<=3) %>% filter(!row_number() %in% c(1)) %>% mutate(range = "<2 & <=3")
  z_3 <- zscore %>% select(i) %>% count(range= 1<value & value<=2) %>% filter(!row_number() %in% c(1)) %>% mutate(range = "<1 & <=2")
  z_4 <- zscore %>% select(i) %>% count(range= 0<value & value<=1) %>% filter(!row_number() %in% c(1)) %>% mutate(range = "<0 & <=1")
  z_5 <- zscore %>% select(i) %>% count(range= -1<value & value<=0) %>% filter(!row_number() %in% c(1)) %>% mutate(range = "<-1 &<=0")
  z_6 <- zscore %>% select(i) %>% count(range= -2<value & value<=-1) %>% filter(!row_number() %in% c(1)) %>% mutate(range = "<-2 &<=-1")
  z_7 <- zscore %>% select(i) %>% count(range= -3<value & value<=-2) %>% filter(!row_number() %in% c(1)) %>% mutate(range = "<-3 &<=-2")
  z <- dplyr::bind_rows(z_1,z_2,z_3,z_4,z_5,z_6,z_7) %>% mutate(ratio = n/6300) %>% mutate(cell = i)
  z_final <- dplyr::bind_rows(z_final,z)
  i = i+1
}

###############################################################################
# plot 1 (final version)

ggplot(z_final, aes(x = cell, y = ratio, fill = range))+
  geom_bar(stat = "identity", position = "stack")+
  geom_text(aes(label = percent(ratio,0.01)), position = position_stack(), vjust = 1, size = 2)

###############################################################################
# for loop 2 (final version)

colnames(zscore) <- c("value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value","value")

z <- data.frame()
z_final <- data.frame()

for (i in 1:47) {
  z_1 <- zscore %>% select(i) %>% count(range= 0.6<value & value<=1) %>% filter(!row_number() %in% c(1)) %>% mutate(range = "<0.6 & <=1")
  z_2 <- zscore %>% select(i) %>% count(range= 0.3<value & value<=0.6) %>% filter(!row_number() %in% c(1)) %>% mutate(range = "<0.3 & <=0.6")
  z_3 <- zscore %>% select(i) %>% count(range= 0<value & value<=0.3) %>% filter(!row_number() %in% c(1)) %>% mutate(range = "<0 & <=0.3")
  z_4 <- zscore %>% select(i) %>% count(range= -0.3<value & value<=0) %>% filter(!row_number() %in% c(1)) %>% mutate(range = "<-0.3 &<=0")
  z_5 <- zscore %>% select(i) %>% count(range= -0.6<value & value<=-0.3) %>% filter(!row_number() %in% c(1)) %>% mutate(range = "<-0.6 &<=-0.3")
  z_6 <- zscore %>% select(i) %>% count(range= -1<value & value<=-0.6) %>% filter(!row_number() %in% c(1)) %>% mutate(range = "<-1 &<=-0.6")
  z <- dplyr::bind_rows(z_1,z_2,z_3,z_4,z_5,z_6) %>% mutate(ratio = n/6300) %>% mutate(cell = i)
  z_final <- dplyr::bind_rows(z_final,z)
  i = i+1
}

###############################################################################
# plot 2 (final version)

ggplot(z_final, aes(x = cell, y = ratio, fill = range))+
  geom_bar(stat = "identity", position = "stack")+
  geom_text(aes(label = percent(ratio,0.01)), position = position_stack(), vjust = 1, size = 2)


