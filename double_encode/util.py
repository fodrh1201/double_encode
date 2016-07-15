import _pickle as cPickle
import numpy as np
import os
import collections
import itertools


class DataUtil:

    def __init__(self,
                 data_feature='googlenet',
                 data_set='imagenet',
                 clip_sentence_path='/data/movieQA/story/movie_clip_sentences_proc_sb.pkl',
                 feature_path='/data/movieQA/story/video_feature/'):

        self.data_feature = data_feature
        self.data_set = data_set
        self.clip_sentence_path = clip_sentence_path
        self.feature_path = feature_path
        self.load_movies_clips_sents()

    def load_movies_clips_sents(self):
        self.movies_clips_sents = {}
        with open(self.clip_sentence_path, 'rb') as f:
            self.movies_clips_sents = cPickle.load(f)
        self.movies = self.movies_clips_sents.keys()

    def get_movie_clips(self):
        movie_clips = {}
        for movie in self.movies:
            movie_clips[movie] = list(self.movies_clips_sents[movie].keys())
        return movie_clips

    def get_clips_path(self):
        movie_clips = self.get_movie_clips()
        clips_path = {}
        for movie in self.movies:
            for clip in movie_clips[movie]:
                clips_path[clip] = os.path.join(self.feature_path, self.data_feature, movie, self.data_set, clip)
        return collections.OrderedDict(sorted(clips_path.items()))

    def get_clips_sents(self):
        movie_clips = self.get_movie_clips()
        clips_sents = {}
        for movie in self.movies:
            for clip in movie_clips[movie]:
                clips_sents[clip] = self.movies_clips_sents[movie][clip]
        return collections.OrderedDict(sorted(clips_sents.items()))

    def get_sent_list_count(self):
        clips_sents = self.get_clips_sents()
        clips = clips_sents.keys()
        counts = []
        for clip in clips:
            counts.append(len(clips_sents[clip]))
        return counts

    def get_sents(self):
        sents2d = self.get_clips_sents().values()
        return list(itertools.chain(*sents2d))

    def get_pad_sents(self, pad_token=9485, pad_location="RIGHT", max_length=None):
        sequences = self.get_sents()

        if not max_length:
            max_length = max(len(x) for x in sequences)

        result = []
        for i in range(len(sequences)):
            sentence = sequences[i]
            num_padding = max_length - len(sentence)
            if num_padding == 0:
                new_sentence = sentence
            elif num_padding < 0:
                new_sentence = sentence[:max_length]
            elif pad_location == "RIGHT":
                new_sentence = sentence + [pad_token] * num_padding
            elif pad_location == "LEFT":
                new_sentence = [pad_token] * num_padding + sentence
            else:
                raise Error("Invalid pad_location. Specify LEFT or RIGHT.")
            result.append(new_sentence)
        return result

    def get_features(self):
        features = []
        clips_path = self.get_clips_path()
        for clip, path in clips_path.items():
            with open(path, 'rb') as f:
                feature = cPickle.load(f, encoding='latin1')
                features.append(self.modify_nan(feature.reshape(-1)))
        return np.array(features)

    def modify_nan(self, feature):
        nan = []
        for i, elem in enumerate(feature):
            if np.isnan(elem):
                nan.append(elem)
                feature[i] = 0
        if len(nan) > 0:
            print(len(nan))
        return feature

    def get_cluster_indices(self, max_length=None):
        counts = self.get_sent_list_count()
        end = 0
        ranges = []
        for i in counts:
            ranges.append((end, end+i))
            end += i
        return [list(range(start, end)) for start, end in ranges]

    def get_feature_data(self):
        features = self.get_features()
        cluster_indices = self.get_cluster_indices()
        avail_feat_indices = [i for i in range(len(cluster_indices)) if cluster_indices[i] != []]
        avail_features = features[avail_feat_indices]
        avail_cluster_indices = np.array(cluster_indices)[avail_feat_indices]

        num_of_train = int(2 * len(features) / 3)
        trF = avail_features[:num_of_train]
        trCI = avail_cluster_indices[:num_of_train]
        teF = avail_features[num_of_train:]
        teCI = avail_cluster_indices[num_of_train:]
        return trF, trCI, teF, teCI

    def batch_iter(self, avail_features, avail_cluster_indices, batch_size, num_epochs, seed=2, fill = False, max_length=25):
        sents = self.get_pad_sents(max_length=25)
        random = np.random.RandomState(seed)
        length = len(avail_features)
        num_batches_per_epoch = int(length/batch_size)
        if length % batch_size != 0:
            num_batches_per_epoch += 1
        for epoch in range(num_epochs):
            shuffle_indices = random.permutation(np.arange(length))
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, length)
                selected_indices = shuffle_indices[start_index:end_index]
                if fill is True and end_index >= length:
                    num_missing = batch_size - len(selected_indices)
                    selected_indices = np.concatenate([selected_indices, random.randint(0, length, num_missing)])
                selected_features = avail_features[selected_indices]
                selected_cluster_indices = avail_cluster_indices[selected_indices]
                selected_sents_indices = [random.choice(x, 1)[0] for x in selected_cluster_indices]
                selected_sents = np.array(sents)
                selected_sents = selected_sents[selected_sents_indices]
                yield (selected_features, selected_sents)

    def sfi_iter(self, num_epochs, max_length=25):
        sents = self.get_pad_sents(max_length=25)
        features = self.get_features()
        indices = self.get_cluster_indices()

        sents = np.hstack(sents).reshape([-1, max_length])

        for epoch in range(num_epochs):
            yield (sents, features, indices)
