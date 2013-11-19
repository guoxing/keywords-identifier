import tfidf

class VocabularyBuilder:
    def __init__(self, documents, corpusFileOutputPath,stopWordsListPath):
        self.documents = documents
        self.documentsCounts = len(documents)
        self.corpusFileOutputPath = corpusFileOutputPath
        self.buildCorpus(corpusFileOutputPath) 
        self.tfIdf = tfidf.TfIdf(corpusFileOutputPath,stopWordsListPath)

    def buildCorpus(self, corpusFileOutputPath):
        dictionary = {}
        for document in self.documents:
            wordList = document.split()
            for word in wordList:
                if dictionary.has_key(word):
                    dictionary[word] = dictionary[word] + 1
                else:
                    dictionary[word] = 1

        with open(corpusFileOutputPath, 'w') as f:
            f.write(str(self.documentsCounts)+"\n")
            for key, value in dictionary.items():
                f.write(key+":"+str(value)+"\n")
            f.close()

    def buildVocabulary(self, threshold_percent):
        vocabulary = []
        for document in self.documents:
            sortedList = self.tfIdf.get_doc_keywords(document)
            index = 0 
            threshold = threshold_percent*len(sortedList)
            for key, value in sortedList:
                if key not in vocabulary and index <= threshold	:
                    vocabulary.append(key)
                    index = index +1
        return vocabulary	

