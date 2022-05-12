library(readxl)
data <- read_excel("E:/BU Study_2022/679_Final Project/accuarcy.xlsx")
mouse <- c("409","412","414","416","417","418","251","256","257","258","274","254","255")
data <- cbind(data,mouse)

library(ggplot2)
colors <- c("Logictic accuracy" = "blue", "RNN accuracy" = "red")
ggplot(data, aes(x = mouse)) +
  geom_point(aes(y = Logistic, color = "Logictic accuracy"), size = 3) +
  geom_point(aes(y = RNN, color = "RNN accuracy"), size = 3) +
  theme(axis.text.x=element_text(angle=45, vjust=0.6)) +
  labs(x = "Mouse",
       y = "Accuracy",
       color = "Legend",
       title = "Accuracy of Logistic vs. RNN") +
  scale_color_manual(values = colors)










