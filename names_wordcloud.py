import os
import cv2
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import datetime
import matplotlib.pyplot as plt
from matplotlib import pylab
from scipy.misc import imsave

#print([x.split("_") for x in os.listdir('lfw')])

names = [item for sublist in [x.split("_") for x in os.listdir(r'C:\dev\data\lfw')] for item in sublist]
names = [name.lower() for name in names]
names = [name.capitalize() for name in names]

person_count = len(os.listdir(r'C:\dev\data\lfw'))

runstr = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M") + '_wordcloud_from_' \
         + str(person_count) + "_names"

names_string = " ".join(names)

#wordcloud = WordCloud(max_words=1000, width=10800, height=3600).generate(names_string)
wordcloud = WordCloud(max_words=1000, width=1800, height=600).generate(names_string)

'''
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
#plt.show()

#cv2.imwrite(r'./wordcloud/'+runstr+'.jpg', wordcloud)
pylab.savefig(r'./wordcloud/'+runstr+'.jpg')
'''

imsave(os.path.join(r"C:\dev\data\lookalike_finder\wordcloud", runstr+'.jpg'), wordcloud)