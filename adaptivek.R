setwd("/Users/sathani/Desktop/ALDA Project")
#install.packages('akmeans')
library(akmeans)
#library(cluster)
#library(fpc)

#Read the output from the feature extraction python code into x
x <- read.table("Output.txt", header = FALSE, sep=" ")
x<-x[,1:1025]
x[1,]
xtag<-x[,1]
y<-as.matrix(x[,2:1025])

#threshold here is the dot product of the vector and
# the center of the cluster to which it is assigned.
adpkmeans<-akmeans(y,d.metric=2,max.k = 400, ths3=0.53,mode=3)
?adpkmeans
adpkmeans$cluster
adpkmeans$centers
MyData<-cbind(adpkmeans$cluster,xtag)
clustercenters<-sort(unique(adpkmeans$cluster))
centercluster<-cbind(centercluster,clustercenters)
dim(centercluster)
write.table(MyData, file = "clust_labels.txt",row.names=FALSE, col.names=FALSE, sep=",")
write.table(centercluster, file = "cluster_centers.txt",row.names=FALSE, col.names=FALSE, sep=",")
centercluster[,1025]
length(MyData)
##z<-skmeans(y,k=)

##c<-skmeans_xdist(y, y = NULL)

##clust<-kmeans(c,11)
##clust$centers
##plotcluster(c,clust$cluster)
##b<- (z,)
