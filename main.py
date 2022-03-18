# -*- coding: UTF-8 -*-
"""
# @Author:  Zirui Zhou
# @Date:    2022/3/18 23:38:29
# @Contact: zirui.zhou19@student.xjtlu.edu.cn
"""

import myTfidf

DATASET_PATH = 'dataset'
STOPWORDS_PATH = 'stopwords.txt'
AIK_MATRIX_PATH = 'train-20ng.npz'

IS_SPARSE_MATRIX = True
IS_INFO_PRINT = True
SELECTED_SAMPLE = [0]
IS_TIME_PRINT = True


def main():
    new_tfidf = myTfidf.TFIDF()

    new_tfidf.is_sparse_matrix = IS_SPARSE_MATRIX
    new_tfidf.is_info_print = IS_INFO_PRINT
    new_tfidf.selected_sample = SELECTED_SAMPLE
    new_tfidf.is_time_print = IS_TIME_PRINT

    new_tfidf.load_stopwords(STOPWORDS_PATH)
    new_tfidf.load_file(DATASET_PATH)
    new_tfidf.calc_matrix()
    new_tfidf.save_Aik_matrix(AIK_MATRIX_PATH)


if __name__ == '__main__':
    main()
