import _pickle as cPickle
import os
import collections
import itertools

class DataUtil:

    def __init__(self,
                 data_feature = 'googlenet',
                 data_set = 'imagenet',
                 clip_sentence_path = '/data/movieQA/story/movie_clip_sentences_proc_sb.pkl',
                 feature_path = '/data/movieQA/story/video_feature/'):

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

    def pad_sents(self, pad_token=9485, pad_location="RIGHT", max_length=None):
        sequences = self.get_sents()

        if not max_length:
            max_length = max(len(x) for x in sequences)

        result = []
        for i in range(len(sequences)):
            sentence = sequences[i]
            num_padding = max_length - len(sentence)
            if num_padding == 0:
                new_sentence = sentence
            elif pad_location == "RIGHT":
                new_sentence = sentence + [pad_token] * num_padding
            elif pad_location == "LEFT":
                new_sentence = [pad_token] * num_padding + sentence
            else:
                raise Error("Invalid pad_location. Specify LEFT or RIGHT.")
            result.append(new_sentence)
        return result
