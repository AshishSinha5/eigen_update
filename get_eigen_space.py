import numpy as np
import numpy.linalg as la
import pickle
import timeit


def svd_update(U, sigma, V, Ai):
    """
    :param U:  previous set of left singular vectors
    :param sigma: previous diagonal matrix containing singular values
    :param V: previous set of right singular vectors
    :param Ai: new image
    :return: updated SVD
    """
    x = np.transpose(U) @ Ai
    a_perp = Ai - U @ x
    a_perp = a_perp / la.norm(a_perp)
    # print("A_perp - ",a_perp)
    sigma_mat = np.identity(len(sigma)) * sigma
    zero = np.zeros((1, np.shape(sigma_mat)[1]))
    # print("Zero - ", zero)
    # print(np.transpose(a_perp)@Ai)
    new_mat = np.vstack((np.hstack((sigma_mat, np.transpose(U) @ Ai)),
                         np.hstack((zero, np.transpose(a_perp) @ Ai))))

    W, omega, Q_trans = la.svd(new_mat)
    U_ = np.hstack((U, a_perp)) @ W
    zero_col = np.zeros((np.shape(V)[0], 1))
    zero_row = np.zeros((1, np.shape(V)[1]))
    # print(np.shape(zero_row))
    V_ = np.hstack((V, zero_col))
    V_ = np.vstack((V_, np.hstack((zero_row, np.ones((1, 1))))))
    # print(V_)
    V_ = np.transpose(V_) @ np.transpose(Q_trans)
    # print(V_)
    return U_, omega, np.transpose(V_)


class get_eigen_space:
    def __init__(self, data: dict, eps=0.3, num_train=1):
        self.data = data
        self.eps = eps
        self.img_array = []
        self.num_train = num_train
        self.weights = None

    def gen_images(self):
        """
        :return: generates image array from dictionary
        """
        for k, v in self.data.items():
            for i, imgs in enumerate(v):
                if i == self.num_train:
                    break
                self.img_array.append(imgs)
        self.img_array = np.squeeze(np.array(self.img_array))
        self.img_array = np.reshape(self.img_array, (self.img_array.shape[0], -1))
        self.img_array = np.transpose(self.img_array)

    def eigenSpaceUpdate(self):
        """
        :return: SVD of image array with epsilon tolerance
        """
        self.gen_images()
        numImages = np.shape(self.img_array)[1]
        # print('number of Images - ', numImages)
        A1 = self.img_array[:, 0]
        A1 = np.reshape(A1, (len(A1), 1))
        U = A1 / la.norm(A1)
        V = np.ones((1, 1))
        sigma = [la.norm(A1)]
        for i in range(1, numImages):
            # print(i)
            Ai = self.img_array[:, i]
            Ai = np.reshape(Ai, (len(Ai), 1))
            _U, _sigma, _V_trans = svd_update(U, sigma, np.transpose(V), Ai)
            j = int(numImages * self.eps)
            """for j in range(len(_sigma)):
                if _sigma[j]<=self.eps:
                    break"""
            U = _U[:, :j + 1]
            sigma = _sigma[:j + 1]
            _V = np.transpose(_V_trans)
            # print(_V)
            V = _V[:, :j + 1]
        self.weights = U.transpose() @ self.img_array
        labels = np.ndarray.flatten(np.array([[i]*self.num_train for i in range(numImages)]))

        return U, sigma, V, list(zip(labels, self.weights.transpose()))

