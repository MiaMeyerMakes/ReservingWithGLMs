library(data.table)
library(tidyverse)
library(DT)
library(patchwork)

# read the data from the website
msdata <- read.table("glms_meyershi.txt")
tail(msdata)
#print(msdata) for a simple look

# printing the table in a nicer format
datatable(msdata) |> 
  formatRound(c("cumulative", "incremental"), digits = 0)

p1 <- ggplot(data=msdata, aes(x=dev_year, y=cumulative, colour=as.factor(acc_year))) +
  geom_line(size=1) +
  scale_color_viridis_d(begin=0.9, end=0) + 
  ggtitle("Cumulative") + 
  theme_bw() + 
  theme(legend.position = "none", legend.title=element_blank(), legend.text=element_text(size=8))


p2 <- ggplot(data=msdata, aes(x=dev_year, y=incremental, colour=as.factor(acc_year))) +
  geom_line(size=1) +
  scale_color_viridis_d(begin=0.9, end=0) + 
  ggtitle("Incremental") + 
  theme_bw() + 
  theme(legend.position = "right", legend.title=element_blank(), legend.text=element_text(size=8))

p1 + p2   # combine the plots using patchwork


