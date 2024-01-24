# Type help(package="ADNIMERGE") to read more about the ADNIMERGE package.

library(ADNIMERGE)

df <- adnimerge

write.csv(df, file='data/adni_data.csv', row.names=FALSE)
