import json
import os
import string


def tokenize(images):
    """ example method for tokenizing captions

    Args:
        images: a list of dictionary, and every dictionary(key: `captions`) contains some captions for an image.

    Returns:
        images: a list of dictionary, and every dictionary(key: `processed_tokens`) contains the tokenized  captions.
    """
    print('examples for tokenizing captions')
    for i, img in enumerate(images):
        images[i]['processed_tokens'] = []
        for caption in img['captions']:
            # remove punctuation and lower
            tokens = ''.join(c for c in caption.strip().lower() if c not in string.punctuation)
            tokens = tokens.split()
            if i == 0:
                print('origin: %s' % caption)
                print('tokenized: ', end='')
                print(tokens)
            images[i]['processed_tokens'].append(tokens)
    return images


def build_vocab(images_info, word_count_threshold):
    """build the vocabulary for the dataset

    Args:
        images_info: a list information of images
        word_count_threshold: the threshold to filter out the low frequency words

    Returns:
        vocab: the vocabulary for the dataset
        images_info: a list information of images(add a key: `final_captions`)
    """

    # count up the number of words
    counts = {}
    for img in images_info:
        for words in img['processed_tokens']:
            for w in words:
                counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    print('top words and their counts:')
    print('\n'.join(map(str, cw[:20])))
    total_words = sum(counts.values())
    print('total words:', total_words)
    bad_words = [w for w, n in counts.items() if n <= word_count_threshold]
    vocab = [w for w, n in counts.items() if n > word_count_threshold]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % len(vocab))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count * 100.0 / total_words))

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for img in images_info:
        for txt in img['processed_tokens']:
            n_words = len(txt)
            sent_lengths[n_words] = sent_lengths.get(n_words, 0) + 1
    max_len = max(sent_lengths.keys())
    print('max length sentence in raw data: ', max_len)
    print('sentence length distribution (count, number of words):')
    sum_len = sum(sent_lengths.values())
    for i in range(max_len + 1):
        print('%2d: %10d   %f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0) * 100.0 / sum_len))

    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('UNK')

    # filter out low frequency words
    for img in images_info:
        img['final_captions'] = []
        for txt in img['processed_tokens']:
            caption = [w if counts.get(w, 0) > word_count_threshold else 'UNK' for w in txt]
            img['final_captions'].append(caption)

    return vocab, images_info


def encode_captions(images_info, w2i):
    """encode captions

    Args:
        images_info: a list information of images
        w2i: the map of word to index

    Returns:
        train_set: a list information of images for training
        val_set: a list information of images for validation
        test_set: a list information of images for testing

    Raises:
        KeyError: unknown splitting type
    """
    # init the splitting set
    train_set, val_set, test_set = [], [], []
    for i, img in enumerate(images_info):
        n = len(img['final_captions'])
        assert n > 0, 'error: some image has no captions'
        index_captions = []
        # encode the captions
        for j, s in enumerate(img['final_captions']):
            index = []
            for k, w in enumerate(s):
                index.append(w2i[w])
            index_captions.append(index)
        img['index_captions'] = index_captions
        # spilt data
        if img['split'] == 'train':
            train_set.append(img)
        elif img['split'] == 'val':
            val_set.append(img)
        elif img['split'] == 'test':
            test_set.append(img)
        else:
            raise KeyError('no such split: %s' % img['split'])

    return train_set, val_set, test_set


def main():
    """function to test the pre-process
    """

    # load images list
    with open('data/reid_raw.json', 'r', encoding='utf8') as f:
        # print(f.read())
        images_info = json.load(f)
        # shuffle(images)  # shuffle images
    print('example:')
    print('\n'.join(k + ':' + str(v) for k, v in images_info[10].items()))

    #  tokenize captions
    # images_tokenized = tokenize(images)

    # create the vocab
    word_count_threshold = 16
    vocab, images_info = build_vocab(images_info, word_count_threshold)
    i2w = {i + 1: w for i, w in enumerate(vocab)}  # a 1-indexed vocab translation table
    w2i = {w: i + 1 for i, w in enumerate(vocab)}  # inverse table

    # save vocab-index map
    if not os.path.exists('vocab'):
        os.mkdir('vocab')
    with open('vocab/i2w.json', 'w', encoding='utf8') as f:
        json.dump(i2w, f, indent=2)
    with open('vocab/w2i.json', 'w', encoding='utf8') as f:
        json.dump(w2i, f, indent=2)

    # encode captions
    train_set, val_set, test_set = encode_captions(images_info, w2i)

    # save the splitting dataset
    if not os.path.exists('data'):
        os.mkdir('data')
    with open('data/train_set.json', 'w', encoding='utf8') as f:
        json.dump(train_set, f)
    with open('data/valid_set.json', 'w', encoding='utf8') as f:
        json.dump(val_set, f)
    with open('data/test_set.json', 'w', encoding='utf8') as f:
        json.dump(test_set, f)


if __name__ == '__main__':
    main()
