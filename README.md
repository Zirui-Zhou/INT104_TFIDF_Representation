# TFIDF Representation

## Introduction

This is one simple project to finish the INT104 assignment. Fix the code style and git it in a lonely night. If you are confused by the similar assignment, feel free to get some inspiration ;)

## Testing and Results

```
The duration of stopwords loading is: 0.0013797000000002058 s.

The duration of data analysis is: 9.3061604 s.

The term frequency (f_ik) matrix:
[[0.00216685 0.00108342 0.00216685 ... 0.         0.         0.        ]
 [0.00156904 0.00104603 0.00156904 ... 0.         0.         0.        ]
 [0.         0.         0.00431034 ... 0.         0.         0.        ]
 ...
 [0.         0.         0.01388889 ... 0.         0.         0.        ]
 [0.         0.         0.01234568 ... 0.         0.         0.        ]
 [0.         0.         0.00636943 ... 0.00636943 0.00636943 0.00636943]]

The document frequency (n_k) matrix:
[[  36   30 2726 ...    1    1    1]]

The inverse document frequency (idf) matrix:
[[4.32707167 4.50939323 0.         ... 7.91059061 7.91059061 7.91059061]]

The tf-idf (a_ik) matrix:
[[0.0093761  0.00488558 0.         ... 0.         0.         0.        ]
 [0.00678934 0.00471694 0.         ... 0.         0.         0.        ]
 [0.         0.         0.         ... 0.         0.         0.        ]
 ...
 [0.         0.         0.         ... 0.         0.         0.        ]
 [0.         0.         0.         ... 0.         0.         0.        ]
 [0.         0.         0.         ... 0.05038593 0.05038593 0.05038593]]

The A_ik matrix:
[[0.04080429 0.02126179 0.         ... 0.         0.         0.        ]
 [0.02875316 0.01997645 0.         ... 0.         0.         0.        ]
 [0.         0.         0.         ... 0.         0.         0.        ]
 ...
 [0.         0.         0.         ... 0.         0.         0.        ]
 [0.         0.         0.         ... 0.         0.         0.        ]
 [0.         0.         0.         ... 0.10161307 0.10161307 0.10161307]]

The duration of Aik matrix calculation is: 0.8207867000000011 s.

The duration of matrix saving is: 0.6376372000000003 s.
```

