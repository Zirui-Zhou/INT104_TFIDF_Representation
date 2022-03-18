# -*- coding: UTF-8 -*-
"""
# @Author:  Zirui Zhou
# @Date:    2021/4/6 14:13:33
# @Contact: zirui.zhou19@student.xjtlu.edu.cn
"""

# This is my first formal python programme. I have tried to make everything perfect, but there
# are still many flaws. Although it is from an assessment, I would like to get some advice,
# whatever positive or negative, from anyone reading this source code file. Please contact me by
# email above, if you are interested in. Thank you for your time. -From a novice programmer.

import os
import re
import time

import nltk
import numpy as np
import scipy.sparse


class TFIDF:
    """A simple class to calculate the tf-idf matrix of a dataset.

    Attributes:
        stopwords: A set of stopwords.

        word_freq_dict: A dict-nested dictionary of term frequency (tf) of words in all documents,
        whose structure is `word_freq_dict[file_index][word_index] = tf`.
        document_freq_dict: A dictionary of word indexes (keys) and respective total document
        numbers of all the words (values).
        word_to_index_dict: A dictionary of word strings (keys) and word indexes (values).
        file_to_index_dict: A dictionary of file path strings (keys) and file indexes (values).

        Aik_matrix: A numpy matrix of A_ik matrix.

        is_sparse_matrix: A boolean of whether to convert numpy matrix into a sparse one.
        is_info_print: A boolean of whether to print information.
        selected_sample: A list of indexes of files to print.
        is_time_print: A boolean of whether to print duration of key processes.
    """

    def __init__(self):
        self.stopwords = set()

        self.word_freq_dict = dict()
        self.document_freq_dict = dict()
        self.word_to_index_dict = dict()
        self.file_to_index_dict = dict()

        self.Aik_matrix = np.empty((0, 0))

        self.is_sparse_matrix = False
        self.is_info_print = False
        self.selected_sample = list()
        self.is_time_print = False

    def load_stopwords(self, path):
        """Loads stopwords from the path.

        Args:
            path: A string of the path of a stopwords file.
        """
        with DebugTimer("stopwords loading", self.is_time_print):
            self.stopwords = self.get_stopwords(path)
        self.info_print("The stopwords set in \"{}\":\n{}\n".format(path, self.stopwords))

    def load_file(self, root):
        """Reads all the files as documents in the root path.

        Args:
            root: A string of the dataset path.
        """
        with DebugTimer("data analysis", self.is_time_print):
            self.traverse_file(root)

    def calc_matrix(self):
        """Calculates the Aik matrix of the given dataset after calling load_file().
        """
        word_num = len(self.word_to_index_dict)
        file_num = len(self.file_to_index_dict)

        with DebugTimer("Aik matrix calculation", self.is_time_print):
            tf_matrix = self.calc_tf_matrix(file_num, word_num, self.word_freq_dict)
            self.info_print("The term frequency (f_ik) matrix:\n{}\n".format(tf_matrix))

            nk_matrix = self.calc_nk_matrix(self.document_freq_dict)
            self.info_print("The document frequency (n_k) matrix:\n{}\n".format(nk_matrix))

            idf_matrix = self.calc_idf_matrix(file_num, nk_matrix)
            self.info_print("The inverse document frequency (idf) matrix:\n{}\n".format(idf_matrix))

            aik_matrix = self.calc_aik_matrix(tf_matrix, idf_matrix)
            self.info_print("The tf-idf (a_ik) matrix:\n{}\n".format(aik_matrix))

            Aik_matrix = self.calc_Aik_matrix(aik_matrix)
            self.info_print("The A_ik matrix:\n{}\n".format(Aik_matrix))

            self.Aik_matrix = Aik_matrix

    def load_Aik_matrix(self, path):
        """Loads the Aik matrix to a given path.

        Args:
            path: A string of the path to load the Aik matrix.
        """
        with DebugTimer("matrix loading", self.is_time_print):
            self.Aik_matrix = self.load_matrix(path, self.is_sparse_matrix)

    def save_Aik_matrix(self, path):
        """Saves the Aik matrix to a given path.

        Args:
            path: A string of the path to save the Aik matrix.
        """
        with DebugTimer("matrix saving", self.is_time_print):
            self.save_matrix(self.Aik_matrix, path, self.is_sparse_matrix)

    def traverse_file(self, root):
        """Traverses the whole root file to read all the files as documents.

        Args:
            root: A string of the dataset path.
        """

        for root, _, files in os.walk(root):
            for name in files:
                self.analyse_file(os.path.join(root, name))

    def analyse_file(self, path):
        """Analyses a single file from given path to calculate data which tf-idf matrix needs.

        Args:
            path: A string of the path of a file to analyse.
        """
        file_index = len(self.file_to_index_dict)
        self.file_to_index_dict[path] = file_index

        word_list = self.get_text(path)
        self.info_print("The raw document text in \"{}\":\n{}\n".format(path, word_list), file_index)

        word_list = self.remove_stopwords(word_list, self.stopwords)
        self.info_print("The document text removed stopwords in \"{}\":\n{}\n".format(path, word_list), file_index)

        word_list = self.stem_word(word_list)
        self.info_print("The document text stemmed in \"{}\":\n{}\n".format(path, word_list), file_index)

        # Here can have an optional secondary stopwords remove to ensure the filtration of noise.
        # word_list = self.remove_stopwords(word_list, self.stopwords)
        # self.info_print("The document text removed stopwords twice in \"{}\":\n{}\n".format(path, word_list))

        self.get_word_freq(word_list, file_index)

    def info_print(self, info, file_index=-1):
        """Prints the information according to configuration.

        Args:
            info: A string of the information to print.
            file_index: An integer of the index of the file of the word list.
        """
        if self.is_info_print and ((file_index in self.selected_sample) or file_index == -1):
            print(info)

    def get_word_freq(self, word_list, file_index):
        """Calculates term frequency (tf) and tf-idf-related data of a single given word_list.

        Calculates tf and tf-idf-related data, and integrates single file data into the data
        structure of the whole dataset. The tf_ik can be described as the frequency of word k in
        document i.

        Args:
            word_list: A list of words to calculate tf.
            file_index: An integer of the index of the file of the word list.
        """
        self.word_freq_dict[file_index] = dict()
        file_word_num = len(word_list)
        # `file_word_list` is used to keep the sequence of words and indexes.
        file_word_list = set()

        for elem in word_list:
            if elem not in self.word_to_index_dict:
                word_index = len(self.word_to_index_dict)
                self.word_to_index_dict[elem] = word_index
                self.document_freq_dict[word_index] = 0

            if elem not in file_word_list:
                word_index = self.word_to_index_dict[elem]
                file_word_list.add(elem)
                num = word_list.count(elem)
                # The former definition of document frequency (n_k) is "the total number of times
                # word k occurs in the dataset". Here remains the former code for further changes.
                # self.document_freq_dict[word_index] += num

                # The document frequency (n_k) is defined as "how many files (documents) contain
                # the word k".
                self.document_freq_dict[word_index] += 1
                self.word_freq_dict[file_index][word_index] = num / file_word_num

    @staticmethod
    def get_stopwords(path):
        """ Gets unique stopwords from a txt-file.

        Extracts stopwords line-by-line from an utf-8-encoding txt-file through the given path.
        All words are converted into their lower case form. All non-alphabet characters from the
        text are deleted.

        Args:
            path: A string of the path of a stopwords file.

        Returns:
            A set of lower-case and pure-letter stopwords.
        """
        stopwords = set()
        with open(path, mode='r', encoding='utf-8') as f:
            for line in f.readlines():
                stopwords.add(re.sub(r"[^a-z]", '', line.lower()).strip())
        return stopwords

    @staticmethod
    def get_text(path):
        """ Gets all words from a txt-file.

        Extract words from a Latin1-encoding txt-file through the given path. All words are
        converted into their lower case form. All non-alphabet characters from the text are
        deleted. The text is divided by space characters.

        Args:
            path: A string of the path of a words file.

        Returns:
            A list of lower-case and pure-letter words.
        """
        with open(path, mode='r', encoding='Latin1') as f:
            # The former standard is that "splitting separator is non-alphabet letters". Here
            # remains the former code for further changes.
            # word_list = re.findall(r"[a-z]+", f.read().lower())

            # Removes non-alphabet letters and splits with space characters.
            word_list = re.sub(r"[^a-z\s]", '', f.read().lower()).split()
        return word_list

    @staticmethod
    def remove_stopwords(word_list, stopwords):
        """Removes all the stopwords in a word list.

        Args:
            word_list: A list of the words to remove stopwords.
            stopwords: A set of stopwords.

        Returns:
            A list of the words without stopwords.
        """
        new_word_list = list(filter(lambda x: x not in stopwords, word_list))
        return new_word_list

    @staticmethod
    def stem_word(word_list):
        """Performs word stemming to remove the word suffix.

        Args:
            word_list: A list of the words to stem.

        Returns:
            A list of the words stemmed.
        """
        stemmer = nltk.stem.porter.PorterStemmer()
        new_word_list = [stemmer.stem(plural) for plural in word_list]
        return new_word_list

    @staticmethod
    def calc_tf_matrix(i, k, word_freq_dict):
        """Converts the term frequency (tf) dict into a matrix.

        Only copies non-zero values from dict to a matrix, because the matrix is probably a
        sparse matrix.

        Args:
            i: An integer of the number of documents.
            k: An integer of the number of words.
            word_freq_dict: A dict-nested dictionary of term frequency (tf) of words in all
            documents.

        Returns:
            A numpy matrix of the tf matrix.
        """
        tf_matrix = np.zeros([i, k])
        for file_index in range(i):
            for word_index in word_freq_dict[file_index].keys():
                tf_matrix[file_index, word_index] = word_freq_dict[file_index][word_index]
        return tf_matrix

    @staticmethod
    def calc_nk_matrix(document_freq_dict):
        """Converts the document frequency (n_k) dict into a matrix.

        Returns:
            A numpy matrix of the document frequency (n_k) matrix.
        """
        # The `document_freq_dict` is considered as a key-sorted dict in default. If is unsorted,
        # it needs to be sorted before call the `calc_idf_matrix method` as `nk_list`.
        # example: document_freq_dict = dict(sorted(document_freq_dict.items()))
        nk_list = list(document_freq_dict.values())
        nk_matrix = np.array(nk_list)
        nk_matrix = nk_matrix.reshape((1, len(nk_list)))
        return nk_matrix

    @staticmethod
    def calc_idf_matrix(N, nk_matrix):
        """Calculates the inverse document frequency (idf) matrix.

        The idf can be described as, where k is word index and N is the number of documents in
        the dataset:

        .. math::
            \mathrm{idf}(k, N) = \log \frac{N}{n_{k}}

        Args:
            N: An integer of the number of documents.
            nk_matrix: A matrix of the respective total numbers of all the words occur in the
            documents.

        Returns:
            A numpy matrix of the idf matrix.
        """
        # log in formula is considered as log_2 in the former code. Here remains the former code
        # for further changes.
        # idf_matrix = np.log2(N / nk_matrix)

        # log in formula is considered as log_10.
        idf_matrix = np.log(N / nk_matrix)
        return idf_matrix

    @staticmethod
    def calc_aik_matrix(tf_matrix, idf_matrix):
        """Calculates the tf-idf (a_ik) matrix, where k is word index and i is document index.

        The tf-idf (a_ik) can be described as:

        .. math::
            \mathrm{tfidf}(k,i,N) = \mathrm{tf}(k,i) \cdot \mathrm{idf}(k,N)

        Args:
            tf_matrix: A numpy matrix of the term frequency (tf) matrix.
            idf_matrix: A numpy matrix of the inverse document frequency (idf) matrix.

        Returns:
            A numpy matrix of the a_ik matrix.
        """
        aik_matrix = tf_matrix * idf_matrix
        return aik_matrix

    @staticmethod
    def calc_Aik_matrix(aik_matrix):
        """Calculates the A_ik matrix, where k is word index and i is document index.

        Takes the length of different documents into account. The A_ik can be described as:

        .. math::
            A_{ik} = \frac {a_{ik}} {\sqrt{ {\textstyle \sum_{j = 1}^{D}} {a}^{2}_{ij}}}

        Args:
            aik_matrix: A numpy matrix of the a_ik matrix.

        Returns:
            A numpy matrix of the A_ik matrix.
        """
        AND_matrix = np.sqrt((np.square(aik_matrix)).sum(axis=1))
        AND_matrix = AND_matrix.reshape((aik_matrix.shape[0], 1))
        Aik_matrix = aik_matrix / AND_matrix
        return Aik_matrix

    @staticmethod
    def load_matrix(path, sparse_matrix):
        """Loads a matrix from a given path.

        Args:
            path: A string of the target path to load the matrix.
            sparse_matrix: A boolean whether to save as a sparse matrix.

        Returns:
            A numpy matrix of the target matrix.
        """
        if sparse_matrix:
            matrix = scipy.sparse.load_npz(path).todense()
        else:
            matrix = np.load(path)['Aik']
        return matrix

    @staticmethod
    def save_matrix(matrix, path, sparse_matrix=False):
        """Saves a matrix to a given path.

        Args:
            matrix: A numpy matrix of the matrix to save.
            path: A string of the target path to save the matrix.
            sparse_matrix: A boolean of whether to save as a sparse matrix (default: False).
        """
        if sparse_matrix:
            scipy.sparse.save_npz(path, scipy.sparse.coo_matrix(matrix))
        else:
            np.savez(path, Aik=matrix)


class DebugTimer:
    """A simple class to record runtime of some codes.

    Attributes:
        start_time: A double of the timer's start time.
        end_time: A double of the timer's end time.
        desc: A string of the description of target process.
        print_format: A string of format of print().
        is_print: A boolean of whether time information is printed.
    """
    start_time = 0
    end_time = 0
    desc = str()
    print_format = str()
    is_print = True

    def __init__(self, desc="unknown process", is_print=True):
        self.desc = desc
        self.print_format = "The duration of {} is: {} s.\n"
        self.is_print = is_print

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end()

    def start(self):
        """Start the timer.
        """
        self.start_time = time.perf_counter()

    def end(self):
        """End the timer.
        """
        self.end_time = time.perf_counter()
        if self.is_print:
            print(self.print_format.format(self.desc, self.end_time - self.start_time))
