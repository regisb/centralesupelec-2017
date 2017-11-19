from nltk.corpus import stopwords
from pyspark import SparkContext

sc = SparkContext()
# We use a broadcast value to send the set of words to each executor
english_stop_words = sc.broadcast(set(stopwords.words("english")))

def filter_stop_words(word):
    return word not in english_stop_words.value

def load_text(text_path):
    # Split text in words
    # Remove empty word artefacts
    # Remove stop words ('I', 'you', 'a', 'the', ...)
    vocabulary = sc.textFile(text_path)\
        .flatMap(lambda lines: lines.lower().split())\
        .flatMap(lambda word: word.split("."))\
        .flatMap(lambda word: word.split(","))\
        .flatMap(lambda word: word.split("!"))\
        .flatMap(lambda word: word.split("?"))\
        .flatMap(lambda word: word.split("'"))\
        .flatMap(lambda word: word.split("\""))\
        .filter(lambda word: word is not None and len(word) > 0)\
        .filter(filter_stop_words)

    # cache RDD so that transformations are not applied twice
    vocabulary.persist()

    # Count the total number of words in the text
    word_count = vocabulary.count()

    # Compute the frequency of each word: frequency = #appearances/#word_count
    word_freq = vocabulary.map(lambda word: (word, 1))\
        .reduceByKey(lambda count1, count2: count1 + count2)\
        .map(lambda wc: (wc[0], wc[1]/float(word_count)))

    return word_freq

def main():
    iliad = load_text('../../data/iliad.mb.txt')
    odyssey = load_text('../../data/odyssey.mb.txt')

    # Join the two datasets and compute the difference in frequency
    # Note that we need to write (freq or 0) because some words do not appear
    # in one of the two books. Thus, some frequencies are equal to None after
    # the full outer join.
    join_words = iliad.fullOuterJoin(odyssey).map(lambda word_freq1_freq2: (
        word_freq1_freq2[0],
        (word_freq1_freq2[1][1] or 0) - (word_freq1_freq2[1][0] or 0)
    ))

    # 10 words that get a boost in frequency in the sequel
    emerging_words = join_words.takeOrdered(10, lambda word_freq_diff: -word_freq_diff[1])
    # 10 words that get a decrease in frequency in the sequel
    disappearing_words = join_words.takeOrdered(10, lambda word_freq_diff: word_freq_diff[1])

    # Print results
    for word, freq_diff in emerging_words:
        print("%.2f" % (freq_diff*10000), word)
    for word, freq_diff in disappearing_words[::-1]:
        print("%.2f" % (freq_diff*10000), word)

    # Decomment to activate UI when debugging
    # input("press ctrl+c to exit")

if __name__ == "__main__":
    main()
