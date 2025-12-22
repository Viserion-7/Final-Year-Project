# Improving-Neural-Machine-Translation-for-Sanskrit-English

The paper proposes new Neural Machine Translation approaches based on Reinforcement and Transfer Learning to translate Sanskrit to English and provides monolingual and parallel aligned corpora for the research community. Link for the paper: https://aclanthology.org/2020.icon-main.30/

Our latest paper used 9,000 parallel lines of data, but we are excited to announce that we have updated it to 93,030 lines extracted from Itihasa, a large-scale corpus for Sanskrit to English translation. This corpus contains 93,030 pairs of Sanskrit shlokas and their English translations extracted from two Indian epics viz., The Ramayana and The Mahabharata. This updated dataset has been carefully curated and verified by expert linguists.

In addition, the monolingual dataset that can be used for language modeling and other NLP tasks collected from the Romanized version of Mahabharata, consisting of 130,000 lines (approx) which can be accessed using the following link: https://www.sacred-texts.com/hin/mbs/index.htm.

Furthermore, if you require more data, you can also check out the Sanskrit Speech Corpus, which contains over 78 hours of audio data and includes recordings of 45,953 sentences with a sampling rate of 22KHz. The content includes readings of texts spanning over various Śāstras of Saṃskṛtam literature, as well as contemporary stories, radio programs, extempore discourse, and more. The corpus can be accessed using the following link: https://github.com/cyfer0618/Vaksanca.

We hope that these resources will be helpful for researchers and practitioners working on Sanskrit NLP and machine translation. We invite contributions and feedback from the community to help us improve the quality and accuracy of the translations.

### Citation
Thank you for your interest in our project.If you found this dataset to be useful, please consider citing the paper as follows:
```
@inproceedings{punia-etal-2020-improving,
    title = "Improving Neural Machine Translation for {S}anskrit-{E}nglish",
    author = "Punia, Ravneet  and
      Sharma, Aditya  and
      Pruthi, Sarthak  and
      Jain, Minni",
    booktitle = "Proceedings of the 17th International Conference on Natural Language Processing (ICON)",
    month = dec,
    year = "2020",
    address = "Indian Institute of Technology Patna, Patna, India",
    publisher = "NLP Association of India (NLPAI)",
    url = "https://aclanthology.org/2020.icon-main.30",
    pages = "234--238",
    abstract = "Sanskrit is one of the oldest languages of the Asian Subcontinent that fell out of common usage around 600 B.C. In this paper, we attempt to translate Sanskrit to English using Neural Machine Translation approaches based on Reinforcement Learning and Transfer learning that were never tried and tested on Sanskrit. Along with the paper, we also release monolingual Sanskrit and parallel aligned Sanskrit-English corpora for the research community. Our methodologies outperform the previous approaches applied to Sanskrit by various re- searchers and will further help the linguistic community to accelerate the costly and time consuming manual translation process.",
}
```