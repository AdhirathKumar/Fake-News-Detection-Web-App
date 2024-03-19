# import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt
# import seaborn as sns
# import nltk
import re

# from wordcloud import WordCloud
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model

# from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score

fake = pd.read_csv("pending_updates\Fake.csv", encoding="utf-8")
# # fake.columns
# # fake['subject'].value_counts()
# # plt.figure(figsize=(10,6))
# # sns.countplot(x='subject',data=fake)

text = " ".join(fake["text"].tolist())

# # wordcloud=WordCloud(width=1920,height=1080).generate(text)
# # fig=plt.figure(figsize=(10,10))
# # plt.imshow(wordcloud)
# # plt.axis('off')
# # plt.tight_layout(pad=0)
# # plt.show()

real = pd.read_csv("pending_updates\True.csv", encoding="utf-8")
# # real.columns

text = " ".join(real["text"].tolist())

# # wordcloud=WordCloud(width=1920,height=1080).generate(text)
# # fig=plt.figure(figsize=(10,10))
# # plt.imshow(wordcloud)
# # plt.axis('off')
# # plt.tight_layout(pad=0)
# # plt.show()
# # real.sample(5)

unknown_publishers = []
for index, row in enumerate(real.text.values):
    try:
        record = row.split("-", maxsplit=1)
        record[1]
        assert len(record[0]) < 120
    except:
        unknown_publishers.append(index)

# len(unknown_publishers)
# real.iloc[unknown_publishers].text
# real.iloc[8970]

real = real.drop(8970, axis=0)

publisher = []
tmp_text = []
for index, row in enumerate(real.text.values):
    if index in unknown_publishers:
        tmp_text.append(row)
        publisher.append("Unknown")
    else:
        record = row.split("-", maxsplit=1)
        publisher.append(record[0].strip())
        tmp_text.append(record[1].strip())

real["publisher"] = publisher
real["text"] = tmp_text
# real.head()
# real.shape

empty_fake_index = [
    index for index, text in enumerate(fake.text.tolist()) if str(text).strip() == ""
]
fake.iloc[empty_fake_index]
real["text"] = real["title"] + " " + real["text"]
fake["text"] = fake["title"] + " " + fake["text"]
real["text"] = real["text"].apply(lambda x: str(x).lower())
fake["text"] = fake["text"].apply(lambda x: str(x).lower())
real["class"] = 1
fake["class"] = 0
# real.columns
real = real[["text", "class"]]
fake = fake[["text", "class"]]
# print(type(real),type(fake))
data = real._append(fake, ignore_index=True)
data.sample(5)


# https://github.com/laxmimerit/preprocess_kgptalkie
# !pip install spacy==2.2.3
# !python -m spacy download en_core_web_sm
# !pip install beautifulsoup4==4.9.1
# !pip install textblob==0.15.3
# !pip install git+https://github.com/laxmimerit/preprocess_kgptalkie.git --upgrade --force-reinstall
# import preprocess_kgptalkie as ps
# import gensim
# data.head()
def remove_special_chars(x):
    x = re.sub(r"[^\w ]+", "", x)
    x = " ".join(x.split())
    return x


data["text"].apply(lambda x: remove_special_chars(x))
# y=data['class'].values
x = [d.split() for d in data["text"].tolist()]
# type(x[0])
# DIM=100
# w2v_model=gensim.models.Word2Vec(sentences=x,vector_size=DIM, window=10, min_count=1)
# print(gensim.__version__)
# w2v_model.wv['india']
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x)
x = tokenizer.texts_to_sequences(x)
# tokenizer.word_index
# nos=np.array([len(X) for X in x])
# len(nos[nos>1000])
maxlen = 1000
# x=pad_sequences(x,maxlen=maxlen)
# len(x[101])
# vocab_size=len(tokenizer.word_index)+1
# vocab=tokenizer.word_index
# def get(model):
#   wm=np.zeros((vocab_size,DIM))
#   for word,i in vocab.items():
#     wm[i]=model.wv[word]
#   return wm
# embedding_vectors=get(w2v_model)
# embedding_vectors.shape
# model=Sequential()
# model.add(Embedding(vocab_size,output_dim=DIM,weights=[embedding_vectors],input_length=maxlen,trainable=False))
# model.add(LSTM(units=128))
# model.add(Dense(1,activation="sigmoid"))
# model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
# model.summary()
# X_train,X_test,y_train,y_test=train_test_split(x,y)
# model.fit(X_train,y_train,validation_split=0.3,epochs=6)
# y_pred=(model.predict(X_test)>=0.5).astype(int)
# accuracy_score(y_test,y_pred)
# print(classification_report(y_test,y_pred))
model = load_model(r"D:\Final Project\pending_updates\fake_news_bert.h5")
# m = [
#     " KING OF PRUSSIA, Pennsylvania/WASHINGTON (Reuters) - In the Fox & Hound sports bar, next to a shopping mall in suburban Philadelphia, four Democrats are giving speeches to potential voters as they begin their journey to try to unseat Republican congressman Pat Meehan in next yearâ€™s elections. Winning this congressional district - Pennsylvaniaâ€™s 7th - is key to Democratsâ€™ hopes of gaining the 24 seats they need to retake the U.S. House of Representatives next November. The stakes are high - control of the House would allow them to block President Donald Trumpâ€™s legislative agenda. On the surface, Democrats face a significant hurdle. In nearly two-thirds of 34 Republican-held districts that are top of the partyâ€™s target list, household income or job growth, and often both, have risen faster than state and national averages over the past two years, according to a Reuters analysis of census data.Â (Graphic:Â tmsnrt.rs/2Bgq29K) That is potentially vote-winning news for Republican incumbents, who in speeches and television ads can trumpet a strengthening economy as a product of Republican control of Washington, even though incomes and job growth began improving under former Democratic President Barack Obama. â€œThe good economy is really the only positive keeping Republicans afloat,â€ said David Wasserman, a congressional analyst with the non-partisan Cook Political Report. Still, trumpeting the good economy may have limited impact among voters in competitive districts like this mostly white southeast region of Pennsylvania bordering Delaware and New Jersey, which has switched between both parties twice in the past 15 years. Many of the two dozen voters that Reuters interviewed in the 6th and 7th districts agreed the economy was strong, that jobs were returning and wages were growing. A handful were committed Republicans and Democrats who always vote the party line. About half voted for Meehan last year, but most of those said they were unsure whether they would vote for him again in 2018. Some said they were disappointed with the Republican Partyâ€™s handling of healthcare and tax reform as well as Trumpâ€™s erratic performance. About half also felt that despite an improving economy, living costs are squeezing the middle class. Drew McGinty, one of the Democratic hopefuls at the Fox & Hound bar hoping to unseat Meehan, said the good economic numbers were misleading. â€œWhen I talk to people across the district, I hear about stagnant wages. I hear about massive debt young people are getting when they finish college. Thereâ€™s a lot out there not being told by the numbers,â€ he said. Still, Meehan, who won by 19 points in last Novemberâ€™s general election, is confident the strong economy will help him next year. He plans to run as a job creator and a champion of the middle class. â€œThe first thing people look at is whether they have got a job and income,â€ Meehan said in a telephone interview. Democratic presidential candidate Hillary Clinton carried the district by more than two points in the White House race, giving Democrats some hope that they can peel it away from Republicans next November. Kyle Kondik, a political analyst at the University of Virginia Center for Politics, said the election will essentially be a referendum on Trump. The economy might help Republicans, he said, but other issues will likely be uppermost in votersâ€™ minds, like the Republican tax overhaul - which is seen by some as favoring the rich over the middle class - and Trumpâ€™s dismantling of President Barack Obamaâ€™s initiative to expand healthcare to millions of Americans, popularly known as Obamacare. Indeed, healthcare is Americansâ€™ top concern, according to a Reuters/Ipsos poll conducted earlier this month. Next is terrorism and then the economy. â€œHealthcare will be the No. 1 issue,â€ in the election, predicted Molly Sheehan, another Democrat running to unseat Meehan. Democrats have warned that dismantling Obamacare will leave millions of Americans without health coverage, and political analysts say Republicans in vulnerable districts could be punished by angry voters. Republicans argue that Obamacare drives up costs for consumers and interferes with personal medical decisions. In Broomall, a hamlet in the 7th District, local builder Greg Dulgerian, 55, said he voted for Trump and Meehan. He still likes Trump because of his image as a political outsider, but he is less certain about Meehan. â€œIâ€™m busy, which is good,â€ Dulgerian said. â€œBut I actually make less than I did 10 years ago, because my living costs and costs of materials have gone up.â€ Dulgerian said he was not sure what Meehan was doing to address this, and he was open to a Democratic candidate with a plan to help the middle class. Ida McCausland, 65, is a registered Republican but said she is disappointed with the party. She views the overhaul of the tax system as a giveaway to the rich that will hit the middle class. â€œI will probably just go Democrat,â€ she said. Still, others interviewed said the good economy was the most important issue for them and would vote for Meehan. Â Â Â  Mike Allard, 35, a stocks day trader, voted for Clinton last year but did not cast a ballot in the congressional vote. He thinks the economy will help Meehan next year and is leaning toward voting for him. â€œLocal businesses like the way the economy is going right now,â€ he said. In the 7th district median household income jumped more than 10 percent from 2014 to 2016, from $78,000 to around $86,000, above the national average increase of 7.3 percent, while job growth held steady, the analysis of the census data shows. Overall, the U.S. economy has grown 3 percent in recent quarters, and some forecasters now think theÂ stimulus from the Republican tax cuts will sustain that rate of growth through next year. Unemployment has dropped to 4.1 percent, a 17-year low. In midterm congressional elections, history shows that voters often focus on issues other than the economy. In 1966 the economy was thriving, but President Lyndon B. Johnsonâ€™s Democrats suffered a net loss of 47 seats, partly because of growing unhappiness with the Vietnam War. In 2006, again the economy was humming, but Republicans lost a net 31 seats in the House, as voters focused on the Iraq war and the unpopularity of Republican President George W. Bush. In 2010, despite pulling the economy out of a major recession, Democrats lost control of the House to Republicans, mainly because of the passage of Obamacare, which at the time was highly unpopular with many voters. â€œWhen times are bad, the election is almost always about the economy. When the economy is good, people have the freedom and the ability to worry about other issues,â€ said Stu Rothenberg, a veteran political analyst. "
# ]
m = [
    """ Exclusive: New Study Finds Eating Chocolate Every Day Increases Lifespan by 10 Years!

In a groundbreaking study conducted by researchers at the University of Sweettooth, it has been revealed that consuming chocolate daily can significantly increase human lifespan. The study, which spanned over a decade and involved thousands of participants, has sent shockwaves through the medical community.

According to the lead researcher, Dr. Cocoa Jones, the key to longevity lies in the antioxidants and other beneficial compounds found in chocolate. Regular consumption of chocolate has been linked to improved heart health, reduced stress levels, and enhanced cognitive function.

"We were amazed by the results of our study," said Dr. Jones. "Participants who consumed chocolate every day showed a remarkable increase in lifespan compared to those who did not. It's truly a delicious way to live longer!"

The news has sparked a chocolate-buying frenzy worldwide, with sales of chocolate products skyrocketing in the wake of the study's publication. Chocolate manufacturers are reporting record-breaking profits as consumers rush to stock up on their favorite treats.

However, health experts urge caution, warning that excessive consumption of chocolate can lead to weight gain and other health issues. They advise moderation and recommend opting for dark chocolate, which has higher levels of antioxidants and less sugar.

Despite the warnings, chocolate lovers everywhere are celebrating the news and indulging in their favorite guilty pleasure with renewed enthusiasm. After all, who wouldn't want to live longer while enjoying the sweet taste of chocolate?" """
]
m = tokenizer.texts_to_sequences(m)
m = pad_sequences(m, maxlen=maxlen)
print((model.predict(m) >= 0.5).astype(int))

# # m=['Covid-19 cases are on the rise in India, with the country recording 614 new coronavirus infections in the past 24 hours, prompting the Union Health Minister to direct states to monitor emerging strains. This is the highest number of new cases detected since May 21. The comes even 21 cases of the JN.1 sub-variant have been detected in Goa, Kerala and Maharashtra.']
# m=['Pope Francis used his annual Christmas Day message to rebuke Donald Trump without even mentioning his name. The Pope delivered his message just days after members of the United Nations condemned Trump s move to recognize Jerusalem as the capital of Israel. The Pontiff prayed on Monday for the  peaceful coexistence of two states within mutually agreed and internationally recognized borders. We see Jesus in the children of the Middle East who continue to suffer because of growing tensions between Israelis and Palestinians,  Francis said.  On this festive day, let us ask the Lord for peace for Jerusalem and for all the Holy Land. Let us pray that the will to resume dialogue may prevail between the parties and that a negotiated solution can finally be reached. The Pope went on to plead for acceptance of refugees who have been forced from their homes, and that is an issue Trump continues to fight against. Francis used Jesus for which there was  no place in the inn  as an analogy. Today, as the winds of war are blowing in our world and an outdated model of development continues to produce human, societal and environmental decline, Christmas invites us to focus on the sign of the Child and to recognize him in the faces of little children, especially those for whom, like Jesus,  there is no place in the inn,  he said. Jesus knows well the pain of not being welcomed and how hard it is not to have a place to lay one s head,  he added.  May our hearts not be closed as they were in the homes of Bethlehem. The Pope said that Mary and Joseph were immigrants who struggled to find a safe place to stay in Bethlehem. They had to leave their people, their home, and their land,  Francis said.  This was no comfortable or easy journey for a young couple about to have a child.   At heart, they were full of hope and expectation because of the child about to be born; yet their steps were weighed down by the uncertainties and dangers that attend those who have to leave their home behind. So many other footsteps are hidden in the footsteps of Joseph and Mary,  Francis said Sunday. We see the tracks of entire families forced to set out in our own day. We see the tracks of millions of persons who do not choose to go away, but driven from their land, leave behind their dear ones. Amen to that.Photo by Christopher Furlong/Getty Images.']

# m=tokenizer.texts_to_sequences(m)
# m=pad_sequences(m,maxlen=maxlen)
# print((model.predict(m)>=0.5).astype(int))

# model.save('fake_news_bert.h5')
# model.save_weights("fake_news_bert_weights")
