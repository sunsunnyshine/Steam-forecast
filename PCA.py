import numpy as np
from sklearn.decomposition import PCA

def main():
    # test data
    data = np.loadtxt('zhengqi_train.txt')
    mat= data[:, 0:38]
    # PCA by Scikit-learn
    pca = PCA(n_components=30) # n_components can be integer or float in (0,1)
    pca.fit(mat)  # fit the model
    print('\nMethod : PCA by Scikit-learn:')
    print('After PCA transformation, data becomes:')
    print(pca.fit_transform(mat))  # transformed data
    np.savetxt(r'zhengqi_train_PCA.txt', pca.fit_transform(mat))
    data = np.loadtxt('zhengqi_test.txt')
    mat = data[:, 0:38]
    # PCA by Scikit-learn
    pca = PCA(n_components=30)  # n_components can be integer or float in (0,1)
    pca.fit(mat)  # fit the model
    print('\nMethod : PCA by Scikit-learn:')
    print('After PCA transformation, data becomes:')
    print(pca.fit_transform(mat))  # transformed data
    np.savetxt(r'zhengqi_test_PCA.txt', pca.fit_transform(mat))
main()